"""Tests for the Hermes voicecmd MCP server.

Self-contained: a threaded mock of the whisper.cpp HTTP API answers on loopback, so
these run anywhere Python + ffmpeg exist, with no GPU and no voicecmd install.
Run from the voicecmd repo root:  pytest integrations/hermes/tests -v
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import io
import struct
import subprocess
import sys
import tempfile
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
MCP_PY = HERE.parent / "mcp.py"

CANNED_TEXT = "hey jarvis what is the weather today"
LAST_REQUEST: dict = {}
SERVED_WAV: bytes = b""  # what GET /clip.wav returns, for the URL-input test


class MockWhisper(BaseHTTPRequestHandler):
    """Answers like whisper.cpp's server: `json` gives only text, `verbose_json` adds
    segments and duration. /broken/inference fails, /clip.wav serves SERVED_WAV."""

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            self._reply(200, json.dumps({"status": "ok"}).encode())
        elif self.path == "/clip.wav":
            self._reply(200, SERVED_WAV)
        else:
            self._reply(404, b"not found")

    def do_POST(self):
        global LAST_REQUEST
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        if self.path == "/broken/inference":
            self._reply(500, b"model exploded")
            return
        if self.path != "/inference":
            self._reply(404, b"not found")
            return
        LAST_REQUEST = {"ctype": self.headers.get("Content-Type", ""), "raw": raw}
        if b'name="response_format"\r\n\r\nverbose_json\r\n' in raw:
            resp = {"task": "transcribe", "language": "en", "duration": 2.5,
                    "text": CANNED_TEXT,
                    "segments": [{"text": " " + CANNED_TEXT, "no_speech_prob": 0.01}]}
        else:
            resp = {"text": CANNED_TEXT}
        self._reply(200, json.dumps(resp).encode())

    def _reply(self, code: int, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def mock_url():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), MockWhisper)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def make_wav(path: Path, rate: int = 16000, seconds: float = 0.2) -> None:
    """Minimal valid mono PCM16 wav (stdlib sine, no numpy needed)."""
    n = int(rate * seconds)
    data = b"".join(struct.pack("<h", int(9000 * math.sin(2 * math.pi * 300 * i / rate)))
                    for i in range(n))
    hdr = (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
           + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
           + b"data" + struct.pack("<I", len(data)))
    path.write_bytes(hdr + data)


async def _tool_call(url: str, tool: str, args: dict, tts_enabled: bool = False):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = dict(os.environ)
    env.update({"VOICECMD_WHISPER_URL": url,
                "VOICECMD_CONTROL_URL": url,
                "VOICECMD_TTS_ENABLED": "1" if tts_enabled else ""})
    params = StdioServerParameters(command=sys.executable, args=[str(MCP_PY)], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            if tool == "__list__":
                return [t.name for t in (await session.list_tools()).tools]
            result = await session.call_tool(tool, args)
            return json.loads(result.content[0].text)


def run(url: str, tool: str, args: dict, **kw) -> dict | list:
    return asyncio.run(_tool_call(url, tool, args, **kw))


def test_tools_registered_v1_default(mock_url):
    names = run(mock_url, "__list__", {})
    assert "transcribe" in names and "status" in names
    assert "speak" not in names  # v1 default hides the v1.1 tool


def test_speak_registered_when_enabled(mock_url):
    names = run(mock_url, "__list__", {}, tts_enabled=True)
    assert "speak" in names


def test_transcribe_wav(mock_url, tmp_path):
    wav = tmp_path / "clip.wav"
    make_wav(wav)
    payload = run(mock_url, "transcribe", {"audio_path": str(wav)})
    assert payload["transcription"] == CANNED_TEXT
    # only verbose_json carries these; the mock omits them for plain json
    assert payload["duration_s"] == 2.5
    assert payload["no_speech_prob"] == 0.01
    assert LAST_REQUEST["ctype"].startswith("multipart/form-data")
    assert wav.exists()  # caller's file is never deleted


def test_silence_pad_is_inside_wav_data(mock_url, tmp_path):
    wav = tmp_path / "clip.wav"
    make_wav(wav, seconds=0.2)
    run(mock_url, "transcribe", {"audio_path": str(wav)})
    raw = LAST_REQUEST["raw"]
    with wave.open(io.BytesIO(raw[raw.index(b"RIFF"):])) as r:
        seconds = r.getnframes() / r.getframerate()
    assert seconds == pytest.approx(0.2 + 0.3)  # VOICECMD_STT_PAD_MS default 300


def test_transcribe_url_cleans_up_download(mock_url, tmp_path):
    global SERVED_WAV
    wav = tmp_path / "src.wav"
    make_wav(wav)
    SERVED_WAV = wav.read_bytes()
    before = set(Path(tempfile.gettempdir()).glob("voicecmd_dl_*"))
    payload = run(mock_url, "transcribe", {"audio_path": f"{mock_url}/clip.wav"})
    assert payload["transcription"] == CANNED_TEXT
    assert set(Path(tempfile.gettempdir()).glob("voicecmd_dl_*")) == before


def test_whisper_http_error_is_reported(mock_url, tmp_path):
    wav = tmp_path / "clip.wav"
    make_wav(wav)
    payload = run(mock_url + "/broken", "transcribe", {"audio_path": str(wav)})
    assert "HTTP 500" in payload["error"]


def test_transcribe_ogg_goes_through_ffmpeg(mock_url, tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    src = tmp_path / "tone.wav"
    make_wav(src)
    ogg = tmp_path / "voice.ogg"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                    "-c:a", "libopus", str(ogg)], check=True)
    payload = run(mock_url, "transcribe", {"audio_path": str(ogg)})
    assert payload["transcription"] == CANNED_TEXT
    # converted file cleaned up
    assert not (tmp_path / "voice_16k.wav").exists()


def test_status_tool(mock_url):
    payload = run(mock_url, "status", {})
    assert payload["whisper"]["ok"] is True
    assert payload["daemon"]["ok"] is True


def test_missing_file_returns_error_not_crash(mock_url):
    payload = run(mock_url, "transcribe", {"audio_path": "/nonexistent/nope.wav"})
    assert "error" in payload and "no such audio file" in payload["error"]
