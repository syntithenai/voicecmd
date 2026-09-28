"""Client for the resident whisper.cpp server (POST /inference)."""

from __future__ import annotations

import json
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field

import numpy as np

from .audio import SAMPLE_RATE, pcm_to_wav
from .http import post_multipart

DEFAULT_PROMPT = (
    "Voice commands: play, pause, resume, stop, next track, volume up, volume down, "
    "what's playing, set a timer for 10 minutes, cancel all timers, set an alarm for 7:30 am, "
    "remind me to check the oven, how long is left, what's the temperature outside, how humid is it, "
    "weather today, yesterday."
)


@dataclass
class Transcript:
    text: str
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0
    duration_s: float = 0.0
    elapsed_ms: int = 0
    segments: list[dict] = field(default_factory=list)
    server: str = ""


_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="stt")


class WhisperClient:
    """Primary (GPU) server, hedged by an optional CPU server when the primary is slow.

    If the primary hasn't answered within `hedge_after_ms` (e.g. the GPU is busy with a ComfyUI
    job), the same audio goes to the fallback and whichever finishes first wins.
    """

    def __init__(self, url: str, prompt: str = "", pad_ms: int = 300, timeout: float = 15.0,
                 fallback_url: str = "", hedge_after_ms: int = 500):
        self.url = url.rstrip("/")
        self.fallback_url = fallback_url.rstrip("/")
        self.hedge_after_s = hedge_after_ms / 1000.0
        self.base_prompt = prompt or DEFAULT_PROMPT
        self.prompt = self.base_prompt
        self.pad_ms = pad_ms
        self.timeout = timeout
        self.hedged = 0
        self.fallback_wins = 0

    def set_vocabulary(self, names: list[str]) -> None:
        """Append device names so whisper spells them consistently.

        whisper keeps the end of an over-long prompt, so the names go last.
        """
        names = [n for n in names if n][:40]
        self.prompt = f"{self.base_prompt} Devices: {', '.join(names)}." if names else self.base_prompt

    def transcribe(self, pcm: np.ndarray, prompt: str | None = None) -> Transcript:
        pad = np.zeros(self.pad_ms * SAMPLE_RATE // 1000, dtype=np.int16)
        audio = np.concatenate([pad[: len(pad) // 3], pcm.astype(np.int16), pad])
        return self.transcribe_wav(pcm_to_wav(audio), prompt=prompt)

    def transcribe_wav(self, wav: bytes, prompt: str | None = None) -> Transcript:
        if not self.fallback_url:
            return self._infer(self.url, wav, prompt)
        started = time.monotonic()
        primary = _POOL.submit(self._infer, self.url, wav, prompt)
        try:
            return primary.result(timeout=self.hedge_after_s)
        except FutureTimeout:
            pass
        except Exception as exc:
            print(f"stt: primary failed ({exc}); using fallback", flush=True)
        self.hedged += 1
        fallback = _POOL.submit(self._infer, self.fallback_url, wav, prompt)
        pending = {primary, fallback}
        error: BaseException | None = None
        while pending:
            done, pending = wait(pending, timeout=self.timeout, return_when=FIRST_COMPLETED)
            if not done:
                break
            for fut in done:
                if fut.exception() is None:
                    tr = fut.result()
                    if fut is fallback:
                        self.fallback_wins += 1
                    tr.elapsed_ms = int((time.monotonic() - started) * 1000)
                    return tr
                error = fut.exception()
        raise error or TimeoutError("whisper: no server answered")

    def _infer(self, url: str, wav: bytes, prompt: str | None) -> Transcript:
        started = time.monotonic()
        raw = post_multipart(
            f"{url}/inference",
            {
                "response_format": "verbose_json",
                "temperature": "0.0",
                "prompt": self.prompt if prompt is None else prompt,
            },
            {"file": ("audio.wav", wav, "audio/wav")},
            timeout=self.timeout,
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)
        data = json.loads(raw)
        segments = data.get("segments") or []
        text = " ".join(str(s.get("text") or "").strip() for s in segments).strip()
        if not text:
            text = str(data.get("text") or "").strip()
        no_speech = max((float(s.get("no_speech_prob") or 0.0) for s in segments), default=0.0)
        logprobs = [float(s["avg_logprob"]) for s in segments if "avg_logprob" in s]
        return Transcript(
            text=text,
            no_speech_prob=no_speech,
            avg_logprob=sum(logprobs) / len(logprobs) if logprobs else 0.0,
            duration_s=float(data.get("duration") or 0.0),
            elapsed_ms=elapsed_ms,
            segments=segments,
            server=url,
        )

    def healthy(self, url: str | None = None) -> bool:
        from .http import get_json

        try:
            return (get_json(f"{url or self.url}/health", timeout=2.0) or {}).get("status") == "ok"
        except Exception:
            return False
