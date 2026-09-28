"""TTS via the abc2book gateway (OpenAI /v1/audio/speech), phrase cache, pipelined speaker."""

from __future__ import annotations

import hashlib
import json
import queue
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

from .audio import EARCON_RATE, Player, earcon, parse_wav
from .handlers.llm import spoken_text
from .http import request

CACHE_MAX_CHARS = 90
CACHE_MAX_FILES = 300


def split_sentences(text: str, min_chars: int = 25) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?;])\s+", text.strip()) if p.strip()]
    out: list[str] = []
    for part in parts:
        if out and len(out[-1]) < min_chars:
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return out


class TtsClient:
    def __init__(self, url: str, voice: str, model: str = "kokoro", speed: float = 1.0,
                 cache_dir: Path | None = None, timeout: float = 20.0):
        self.url = url.rstrip("/")
        self.voice = voice
        self.model = model
        self.speed = speed
        self.cache_dir = cache_dir
        self.timeout = timeout
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, text: str) -> Path | None:
        if not self.cache_dir or len(text) > CACHE_MAX_CHARS:
            return None
        key = hashlib.sha1(f"{self.model}|{self.voice}|{self.speed}|{text}".encode()).hexdigest()[:20]
        return self.cache_dir / f"{key}.wav"

    def synth(self, text: str) -> tuple[np.ndarray, int, int]:
        path = self._cache_path(text)
        if path and path.is_file():
            path.touch()
            return parse_wav(path.read_bytes())
        raw = request(
            "POST",
            f"{self.url}/v1/audio/speech",
            body=json.dumps({
                "model": self.model, "input": text, "voice": self.voice,
                "response_format": "wav", "speed": self.speed,
            }).encode(),
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        pcm, rate, channels = parse_wav(raw)
        if path:
            from .audio import pcm_to_wav

            path.write_bytes(pcm_to_wav(pcm, rate) if channels == 1 else raw)
            self._prune()
        return pcm, rate, channels

    def _prune(self) -> None:
        files = sorted(self.cache_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime) if self.cache_dir else []
        for old in files[:-CACHE_MAX_FILES]:
            old.unlink(missing_ok=True)

    def healthy(self) -> bool:
        from .http import get_json

        try:
            return bool((get_json(f"{self.url}/health", timeout=2.0) or {}).get("ok"))
        except Exception:
            return False


class Speaker:
    """Serialised speech output. `say` queues; `stop` cuts off current + queued speech (barge-in)."""

    def __init__(self, tts: TtsClient, player: Player,
                 on_state: Callable[[bool], None] | None = None,
                 on_text: Callable[[str], None] | None = None):
        self.tts = tts
        self.player = player
        self.on_state = on_state
        self.on_text = on_text
        self._queue: "queue.Queue[tuple[int, str, object]]" = queue.Queue()
        self._gen = 0
        self._busy = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-synth")
        self.last_first_audio_at = 0.0
        self.errors = 0
        threading.Thread(target=self._worker, name="speaker", daemon=True).start()

    @property
    def speaking(self) -> bool:
        return self._busy.is_set()

    def say(self, text: str) -> None:
        text = spoken_text(text)
        if text:
            self._idle.clear()
            self._queue.put((self._gen, "text", text))

    def chime(self, name: str) -> None:
        self._idle.clear()
        self._queue.put((self._gen, "earcon", name))

    def play_now(self, name: str) -> None:
        """Earcon outside the queue (e.g. wake click while listening); never blocks the caller."""
        threading.Thread(target=self.player.play, args=(earcon(name), EARCON_RATE), daemon=True).start()

    def stop(self) -> None:
        self._gen += 1
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        self.player.stop()

    def wait_idle(self, timeout: float | None = None) -> bool:
        return self._idle.wait(timeout)

    def _set_busy(self, busy: bool) -> None:
        if busy == self._busy.is_set():
            return
        if busy:
            self._busy.set()
        else:
            self._busy.clear()
        if self.on_state:
            self.on_state(busy)

    def _worker(self) -> None:
        while True:
            try:
                gen, kind, payload = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._queue.empty() and not self.player.playing:
                    self._set_busy(False)
                    self._idle.set()
                continue
            if gen != self._gen:
                continue
            self._set_busy(True)
            if kind == "earcon":
                self.player.play(earcon(str(payload)), EARCON_RATE)
                continue
            text = str(payload)
            if self.on_text:
                self.on_text(text)
            sentences = split_sentences(text)
            pending: Future | None = self._pool.submit(self.tts.synth, sentences[0]) if sentences else None
            for i in range(len(sentences)):
                if gen != self._gen or pending is None:
                    break
                try:
                    pcm, rate, channels = pending.result()
                except Exception as exc:
                    self.errors += 1
                    print(f"tts: synth failed: {exc}", flush=True)
                    self.player.play(earcon("error"), EARCON_RATE)
                    break
                pending = self._pool.submit(self.tts.synth, sentences[i + 1]) if i + 1 < len(sentences) else None
                if gen != self._gen:
                    break
                if i == 0:
                    self.last_first_audio_at = time.monotonic()
                if not self.player.play(pcm, rate, channels):
                    break
