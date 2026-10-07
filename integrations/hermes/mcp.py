#!/usr/bin/env python3
"""voicecmd MCP server: Whisper STT (+ optional TTS/status) for Hermes Agent.

Wraps the HTTP services of voicecmd (https://github.com/syntithenai/voicecmd) as an
MCP stdio server, so an agent can transcribe audio through a resident whisper.cpp
server and check the stack's health. Nothing else from voicecmd is exposed: no wake
word, microphone, PipeWire, router or home-device control.

Endpoints (all loopback-friendly, URLs via env):
  VOICECMD_WHISPER_URL   default http://127.0.0.1:10020   whisper.cpp server
  VOICECMD_CONTROL_URL   default http://127.0.0.1:10021   voicecmd daemon status
  VOICECMD_TTS_URL       default http://127.0.0.1:8789    Kokoro TTS gateway (v1.1)

Tools:
  transcribe  audio file path or URL -> text (ffmpeg converts as needed; 16 kHz wav
              is what whisper.cpp expects, see voicecmd/audio.py SAMPLE_RATE)
  status      reachability of whisper (+ control and TTS when present)
  speak       text -> wav file (v1.1: hidden unless VOICECMD_TTS_ENABLED=1)

Stdlib-only (urllib), so it runs under any Python >= 3.11 with just the `mcp` package.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path

# When run as a script, Python puts this file's directory first on sys.path, which
# would shadow the installed `mcp` package with this very module (mcp.py). Drop it.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

try:
    from mcp.server.mcpserver import MCPServer  # mcp >= 2
except ImportError:
    from mcp.server.fastmcp import FastMCP as MCPServer  # mcp 1.x: same API, old name

WHISPER_URL = os.environ.get("VOICECMD_WHISPER_URL", "http://127.0.0.1:10020").rstrip("/")
CONTROL_URL = os.environ.get("VOICECMD_CONTROL_URL", "http://127.0.0.1:10021").rstrip("/")
TTS_URL = os.environ.get("VOICECMD_TTS_URL", "http://127.0.0.1:8789").rstrip("/")
TTS_VOICE = os.environ.get("VOICECMD_TTS_VOICE", "af_heart")
TTS_MODEL = os.environ.get("VOICECMD_TTS_MODEL", "kokoro")
TTS_SPEED = float(os.environ.get("VOICECMD_TTS_SPEED", "1.0"))
HTTP_TIMEOUT_S = float(os.environ.get("VOICECMD_HTTP_TIMEOUT_S", "60"))
STT_PAD_MS = int(os.environ.get("VOICECMD_STT_PAD_MS", "300"))

AUDIO_EXTS = {".ogg", ".oga", ".opus", ".mp3", ".m4a", ".aac", ".wav", ".flac",
              ".webm", ".amr", ".3gp", ".wma", ".aiff", ".aif", ".caf"}

server = MCPServer(
    name="voicecmd",
    instructions=("Voice tools for Hermes: `transcribe` audio files via a resident "
                  "whisper.cpp server, `status` to check the voice stack. Audio arrives "
                  "as a local path (e.g. a Telegram download); pass it straight through."),
)


def _http(url: str, data: bytes | None = None, headers: dict | None = None,
          timeout: float | None = None) -> tuple[int, bytes]:
    hdrs = {"User-Agent": "voicecmd-mcp/1.0 (Hermes voice tool)"}
    hdrs.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout or HTTP_TIMEOUT_S) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _multipart(field: str, filename: str, payload: bytes, extra: dict[str, str]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for key, value in extra.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode())
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{filename}\"\r\n"
        f"Content-Type: application/octet-stream\r\n\r\n".encode())
    parts.append(payload + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _resolve_audio(src: str) -> Path:
    """Accept a local path or http(s) URL; return a file on disk."""
    if src.startswith(("http://", "https://")):
        status, body = _http(src)
        if status != 200:
            raise RuntimeError(f"download failed ({status}) for {src}")
        suffix = Path(src.split("?")[0]).suffix.lower()
        if suffix not in AUDIO_EXTS:
            suffix = ".audio"
        out = Path(tempfile.gettempdir()) / f"voicecmd_dl_{uuid.uuid4().hex[:8]}{suffix}"
        out.write_bytes(body)
        return out
    p = Path(src).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"no such audio file: {src}")
    return p


def _ensure_wav16k(src: Path) -> Path:
    """whisper.cpp wants 16 kHz mono PCM; ffmpeg handles everything else."""
    if src.suffix.lower() == ".wav" and _probe_rate(src) == 16000:
        return src
    if not shutil.which("ffmpeg"):
        if src.suffix.lower() != ".wav":
            raise RuntimeError("ffmpeg not found and input is not a 16 kHz wav")
        return src  # hope whisper's loader copes; it reads wav natively
    out = src.with_name(src.stem + "_16k.wav")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
           "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[:400]}")
    return out


def _probe_rate(path: Path) -> int | None:
    if not shutil.which("ffprobe"):
        return None
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=10)
        return int(proc.stdout.strip()) if proc.stdout.strip() else None
    except Exception:
        return None


def _padded_wav(wav: Path) -> bytes:
    """Wav bytes with STT_PAD_MS of trailing silence, as voicecmd's WhisperClient does.

    The pad goes inside the data chunk (appending bytes after it would be ignored by
    whisper's wav reader); anything that isn't PCM16 is sent untouched."""
    payload = wav.read_bytes()
    if not STT_PAD_MS:
        return payload
    try:
        with wave.open(io.BytesIO(payload)) as r:
            params = r.getparams()
            frames = r.readframes(r.getnframes())
    except (wave.Error, EOFError):
        return payload
    if params.sampwidth != 2:
        return payload
    silence = b"\x00" * (params.framerate * STT_PAD_MS // 1000 * params.nchannels * 2)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setparams(params)
        w.writeframes(frames + silence)
    return buf.getvalue()


def _post_inference(wav: Path, prompt: str, language: str) -> dict:
    payload = _padded_wav(wav)
    # verbose_json, as voicecmd/stt.py uses: plain `json` returns only {"text"}, with
    # no segments (so no no_speech_prob) and no duration.
    fields = {"response_format": "verbose_json", "language": language or "auto",
              "temperature": "0", "no_context": "true"}
    if prompt:
        fields["prompt"] = prompt
    body, ctype = _multipart("file", wav.name, payload, fields)
    status, raw = _http(f"{WHISPER_URL}/inference", data=body,
                        headers={"Content-Type": ctype})
    if status != 200:
        raise RuntimeError(f"whisper /inference -> HTTP {status}: {raw[:300]!r}")
    return json.loads(raw)


@server.tool(name="transcribe",
             description="Transcribe a voice audio file (path or URL) via the voicecmd "
                         "whisper.cpp server. Returns text. Handles ogg/mp3/m4a/opus/wav "
                         "etc by converting to 16 kHz mono wav with ffmpeg.")
def transcribe(audio_path: str, prompt: str = "", language: str = "auto") -> dict:
    temps: list[Path] = []
    try:
        src = _resolve_audio(audio_path)
        if src != Path(audio_path).expanduser():
            temps.append(src)  # downloaded from a URL
        wav = _ensure_wav16k(src)
        if wav != src:
            temps.append(wav)  # ffmpeg conversion
        result = _post_inference(wav, prompt, language)
        # Same parsing as voicecmd/stt.py: join segments, fall back to `text`.
        segments = result.get("segments") or []
        text = " ".join(str(s.get("text") or "").strip() for s in segments).strip()
        if not text:
            text = str(result.get("text") or "").strip()
        no_speech = max((float(s.get("no_speech_prob") or 0.0) for s in segments), default=0.0)
        return {"transcription": text,
                "no_speech_prob": no_speech,
                "duration_s": float(result.get("duration") or 0.0)}
    except Exception as e:
        return {"error": str(e)}
    finally:
        for t in temps:
            t.unlink(missing_ok=True)


@server.tool(name="status",
             description="Check the voice stack: whisper.cpp /health and voicecmd "
                         "daemon /health. Call before transcribing to give a clear "
                         "error when the services are down.")
def status() -> dict:
    out: dict = {"whisper_url": WHISPER_URL}
    try:
        code, _ = _http(f"{WHISPER_URL}/health", timeout=5)
        out["whisper"] = {"ok": code == 200, "http": code}
    except Exception as e:
        out["whisper"] = {"ok": False, "error": str(e)}
    try:
        code, body = _http(f"{CONTROL_URL}/health", timeout=5)
        if code == 200:
            out["daemon"] = {"ok": True, **json.loads(body)}
        else:
            out["daemon"] = {"ok": False, "http": code}
    except Exception as e:
        out["daemon"] = {"ok": False, "error": str(e)}
    return out


def _speak_impl(text: str, out_path: str = "", voice: str = "", speed: float = 0) -> dict:
    try:
        body = json.dumps({"model": TTS_MODEL, "voice": voice or TTS_VOICE,
                           "input": text, "response_format": "wav",
                           "speed": speed or TTS_SPEED}).encode()
        code, wav = _http(f"{TTS_URL}/v1/audio/speech", data=body,
                          headers={"Content-Type": "application/json"})
        if code != 200:
            return {"error": f"TTS -> HTTP {code}: {wav[:200]!r}"}
        target = Path(out_path) if out_path else Path(tempfile.gettempdir()) / f"voicecmd_tts_{uuid.uuid4().hex[:8]}.wav"
        target.write_bytes(wav)
        return {"audio_path": str(target), "bytes": len(wav)}
    except Exception as e:
        return {"error": str(e)}


# v1.1: `speak` is only registered when explicitly enabled, so a default v1 install
# exposes transcribe + status and nothing that could surprise anyone.
if os.environ.get("VOICECMD_TTS_ENABLED", "") == "1":
    server.tool(name="speak",
                description="Synthesise speech from text via the voicecmd Kokoro TTS "
                            "gateway; writes a wav and returns its path. (v1.1 feature.)")(
        _speak_impl)


if __name__ == "__main__":
    server.run(transport="stdio")
