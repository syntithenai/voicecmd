"""Audio I/O via PipeWire's pulse tools (parecord/paplay), ring buffer, earcons."""

from __future__ import annotations

import collections
import io
import queue
import struct
import subprocess
import threading
import time
import wave
from typing import Iterator

import numpy as np

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # 80 ms: openWakeWord's native chunk, 4 x 20 ms VAD frames
FRAME_BYTES = FRAME_SAMPLES * 2
FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE


def rms(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    f = frame.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(f * f)))


class MicStream:
    """Continuous 16 kHz mono int16 capture in a reader thread; restarts parecord if it dies."""

    def __init__(self, source: str = "", sample_rate: int = SAMPLE_RATE):
        self.source = source
        self.sample_rate = sample_rate
        self._frames: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)
        self._proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.restarts = 0

    def _spawn(self) -> subprocess.Popen:
        cmd = [
            "parecord", "--raw", "--format=s16le", f"--rate={self.sample_rate}", "--channels=1",
            "--latency-msec=20", "--client-name=voicecmd", "--stream-name=voicecmd-mic",
        ]
        if self.source:
            cmd.append(f"--device={self.source}")
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)

    def _reader(self) -> None:
        while not self._stop.is_set():
            self._proc = self._spawn()
            stdout = self._proc.stdout
            assert stdout is not None
            buf = b""
            while not self._stop.is_set():
                chunk = stdout.read(FRAME_BYTES - len(buf))
                if not chunk:
                    break
                buf += chunk
                if len(buf) < FRAME_BYTES:
                    continue
                frame = np.frombuffer(buf, dtype=np.int16).copy()
                buf = b""
                try:
                    self._frames.put_nowait(frame)
                except queue.Full:
                    try:
                        self._frames.get_nowait()
                    except queue.Empty:
                        pass
                    self._frames.put_nowait(frame)
            if self._proc.poll() is None:
                self._proc.terminate()
            if not self._stop.is_set():
                self.restarts += 1
                print(f"mic: parecord exited, restarting (source={self.source or 'default'})", flush=True)
                time.sleep(1.0)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._reader, name="mic", daemon=True)
        self._thread.start()

    def frames(self) -> Iterator[np.ndarray]:
        while not self._stop.is_set():
            try:
                yield self._frames.get(timeout=1.0)
            except queue.Empty:
                continue

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()


class RingBuffer:
    """Keeps the most recent `ms` of frames for pre-roll."""

    def __init__(self, ms: int = 1500):
        self._frames: collections.deque[np.ndarray] = collections.deque(maxlen=max(1, ms // FRAME_MS))

    def push(self, frame: np.ndarray) -> None:
        self._frames.append(frame)

    def tail(self, ms: int) -> np.ndarray:
        if ms <= 0 or not self._frames:
            return np.zeros(0, dtype=np.int16)
        joined = np.concatenate(list(self._frames))
        n = ms * SAMPLE_RATE // 1000
        return joined[-n:]

    def clear(self) -> None:
        self._frames.clear()


def pcm_to_wav(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.astype(np.int16).tobytes())
    return out.getvalue()


def parse_wav(data: bytes) -> tuple[np.ndarray, int, int]:
    """Parse a PCM16 WAV, tolerating streaming headers with bogus sizes (Kokoro).

    Returns (samples int16 interleaved, sample_rate, channels).
    """
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    pos = 12
    rate, channels, bits = 16000, 1, 16
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = pos + 8
        if cid == b"fmt ":
            _fmt, channels, rate, _br, _ba, bits = struct.unpack("<HHIIHH", data[body:body + 16])
        elif cid == b"data":
            end = len(data) if size == 0 or body + size > len(data) else body + size
            raw = data[body:end]
            if bits != 16:
                raise ValueError(f"unsupported WAV bit depth {bits}")
            raw = raw[: len(raw) - (len(raw) % (2 * channels))]
            return np.frombuffer(raw, dtype=np.int16).copy(), rate, channels
        pos = body + size + (size & 1)
    raise ValueError("WAV has no data chunk")


class Player:
    """Plays int16 PCM through paplay; `stop()` interrupts immediately (barge-in)."""

    def __init__(self, sink: str = ""):
        self.sink = sink
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._interrupted = False

    @property
    def playing(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    def play(self, pcm: np.ndarray, sample_rate: int, channels: int = 1) -> bool:
        """Blocking play. Returns False if interrupted."""
        cmd = [
            "paplay", "--raw", "--format=s16le", f"--rate={sample_rate}", f"--channels={channels}",
            "--client-name=voicecmd", "--stream-name=voicecmd-tts", "--latency-msec=40",
        ]
        if self.sink:
            cmd.append(f"--device={self.sink}")
        with self._lock:
            self._interrupted = False
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            proc = self._proc
        try:
            assert proc.stdin is not None
            proc.stdin.write(pcm.astype(np.int16).tobytes())
            proc.stdin.close()
        except BrokenPipeError:
            pass
        proc.wait()
        with self._lock:
            interrupted = self._interrupted
            self._proc = None
        return not interrupted

    def stop(self) -> None:
        with self._lock:
            proc = self._proc
            if proc is not None and proc.poll() is None:
                self._interrupted = True
                proc.kill()


# --- earcons (generated, no asset files) ---

def _tone(freqs: list[float], ms: int, rate: int = 24000, gain: float = 0.25) -> np.ndarray:
    n = rate * ms // 1000
    t = np.arange(n) / rate
    wave_ = sum(np.sin(2 * np.pi * f * t) for f in freqs) / max(1, len(freqs))
    env = np.minimum(1.0, np.minimum(t / 0.008, (t[-1] - t + 1e-9) / 0.04))
    return (wave_ * env * gain * 32767).astype(np.int16)


EARCON_RATE = 24000


def earcon(name: str) -> np.ndarray:
    if name == "wake":
        return np.concatenate([_tone([880], 60), _tone([1320], 70)])
    if name == "done":
        return _tone([660, 990], 90, gain=0.18)
    if name == "error":
        return np.concatenate([_tone([440], 110, gain=0.2), _tone([330], 160, gain=0.2)])
    if name == "alarm":
        beep = _tone([1000, 1500], 180, gain=0.35)
        gap = np.zeros(EARCON_RATE * 120 // 1000, dtype=np.int16)
        return np.concatenate([beep, gap, beep, gap, beep, np.zeros(EARCON_RATE * 500 // 1000, dtype=np.int16)])
    raise KeyError(name)
