"""Benchmark whisper models on command audio.

Cases are WAV files with a sidecar .txt holding the reference transcript. `make_cases` synthesises
them from bench/phrases.txt via the TTS gateway (several voices, clean + noisy); drop real
recordings (any rate, mono) with matching .txt files into the same directory to include them.

For each model a throwaway whisper-server is started on a spare port so the production service
is untouched. Scores: word error rate, and "intent match" - whether the transcript routes to the
same regex intent as the reference text, which is what actually matters for commands.
"""

from __future__ import annotations

import os
import re
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, parse_wav, pcm_to_wav
from .config import ROOT, Settings
from .handlers.clock import ClockHandler
from .handlers.music import MusicHandler
from .handlers.timers import TimerHandler
from .handlers.weather import WeatherHandler
from .http import get_json
from .normalize import normalize
from .router import Router, SystemHandler
from .stt import WhisperClient
from .tts import TtsClient

BENCH_PORT = 10030
VOICES = ("af_heart", "am_michael", "bf_emma", "bm_george")
NOISE_SNR_DB = 15.0


def resample(pcm: np.ndarray, rate: int, target: int = SAMPLE_RATE) -> np.ndarray:
    if rate == target or not len(pcm):
        return pcm.astype(np.int16)
    x = pcm.astype(np.float32)
    if rate > target:
        # Crude low-pass (moving average) before linear interpolation to limit aliasing.
        k = max(1, int(round(rate / target)))
        x = np.convolve(x, np.ones(k, dtype=np.float32) / k, mode="same")
    n_out = int(len(x) * target / rate)
    out = np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x)
    return np.clip(out, -32768, 32767).astype(np.int16)


def add_noise(pcm: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    x = pcm.astype(np.float32)
    power = float(np.mean(x ** 2)) or 1.0
    noise = np.random.default_rng(seed).normal(0, np.sqrt(power / 10 ** (snr_db / 10)), len(x))
    return np.clip(x + noise, -32768, 32767).astype(np.int16)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:48]


def load_phrases(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def make_cases(settings: Settings, out_dir: str) -> int:
    out = Path(out_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    phrases = load_phrases(ROOT / "bench" / "phrases.txt")
    written = 0
    for voice in VOICES:
        tts = TtsClient(settings.tts_url, voice, settings.tts_model, 1.0)
        for i, phrase in enumerate(phrases):
            try:
                pcm, rate, channels = tts.synth(phrase)
            except Exception as exc:
                print(f"make-cases: {voice}: {exc}; skipping voice")
                break
            if channels > 1:
                pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
            pcm = resample(pcm, rate)
            for variant, audio in (("clean", pcm), ("noisy", add_noise(pcm, NOISE_SNR_DB, seed=i))):
                stem = out / f"{slug(phrase)}__{voice}__{variant}"
                stem.with_suffix(".wav").write_bytes(pcm_to_wav(audio, SAMPLE_RATE))
                stem.with_suffix(".txt").write_text(phrase + "\n", encoding="utf-8")
                written += 1
    print(f"make-cases: wrote {written} cases to {out}")
    return 0 if written else 1


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    r, h = normalize(ref).split(), normalize(hyp).split()
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1], len(r)


def _intent_router(tmp: Path) -> Router:
    return Router([
        TimerHandler(tmp / "timers.json"),
        MusicHandler("http://127.0.0.1:0"),
        WeatherHandler(tmp),
        ClockHandler(),
        SystemHandler(),
    ])


def _intent_key(router: Router, text: str) -> tuple:
    intent = router.parse(text)
    if intent is None:
        return ("llm",)
    params = tuple(sorted((k, re.sub(r"[:\s]+", " ", str(v))) for k, v in intent.params.items()))
    return (intent.domain, intent.action, params)


def _rss_mb(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def _start_server(model_path: Path, settings: Settings) -> subprocess.Popen:
    env = dict(os.environ, WHISPER_MODEL_PATH=str(model_path), WHISPER_PORT=str(BENCH_PORT),
               WHISPER_HOST="127.0.0.1")
    proc = subprocess.Popen([str(ROOT / "scripts" / "start_whisper_server.sh")], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"whisper-server exited ({proc.returncode}) for {model_path.name}")
        try:
            if get_json(f"http://127.0.0.1:{BENCH_PORT}/health", timeout=1.0).get("status") == "ok":
                return proc
        except Exception:
            pass
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError(f"whisper-server did not become healthy for {model_path.name}")


def _load_cases(case_dir: Path) -> list[tuple[str, np.ndarray, str]]:
    cases = []
    for wav in sorted(case_dir.glob("*.wav")):
        ref_path = wav.with_suffix(".txt")
        if not ref_path.is_file():
            continue
        pcm, rate, channels = parse_wav(wav.read_bytes())
        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
        cases.append((wav.stem, resample(pcm, rate), ref_path.read_text(encoding="utf-8").strip()))
    return cases


def run_bench(settings: Settings, case_dir: str, models: list[str]) -> int:
    cdir = Path(case_dir)
    if not cdir.is_absolute():
        cdir = ROOT / cdir
    cases = _load_cases(cdir) if cdir.is_dir() else []
    if not cases:
        print(f"bench: no cases in {cdir}; run `python -m voicecmd --make-bench-cases` first")
        return 1
    print(f"bench: {len(cases)} cases from {cdir}")
    tmp = Path(tempfile.mkdtemp(prefix="voicecmd-bench-"))
    router = _intent_router(tmp)
    expected = {name: _intent_key(router, ref) for name, _, ref in cases}

    rows = []
    for model in models:
        path = ROOT / "models" / f"ggml-{model}.bin"
        if not path.is_file():
            print(f"bench: {path.name} missing, skipping")
            continue
        proc = _start_server(path, settings)
        try:
            client = WhisperClient(f"http://127.0.0.1:{BENCH_PORT}", settings.whisper_prompt, settings.stt_pad_ms)
            client.transcribe(cases[0][1])  # warm-up
            errs = words = matches = 0
            lat: list[int] = []
            misses: list[str] = []
            for name, pcm, ref in cases:
                t0 = time.monotonic()
                tr = client.transcribe(pcm)
                lat.append(int((time.monotonic() - t0) * 1000))
                e, n = word_errors(ref, tr.text)
                errs, words = errs + e, words + n
                if _intent_key(router, tr.text) == expected[name]:
                    matches += 1
                else:
                    misses.append(f"    {name}: {tr.text!r}")
            rss = _rss_mb(proc.pid)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        rows.append((model, errs / max(words, 1), matches / len(cases), statistics.median(lat),
                     sorted(lat)[int(len(lat) * 0.95) - 1], rss))
        print(f"\n{model}: intent {matches}/{len(cases)}  misses:")
        print("\n".join(misses[:15]) or "    none")

    print(f"\n{'model':<16}{'WER':>7}{'intent':>9}{'p50 ms':>9}{'p95 ms':>9}{'RSS MB':>9}")
    for model, wer, acc, p50, p95, rss in rows:
        print(f"{model:<16}{wer:>7.1%}{acc:>9.1%}{p50:>9.0f}{p95:>9.0f}{rss:>9.0f}")
    return 0 if rows else 1
