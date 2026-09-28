"""WebRTC VAD endpointing: decides when a spoken command starts and ends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import webrtcvad

from .audio import SAMPLE_RATE

VAD_FRAME_MS = 20
VAD_FRAME_SAMPLES = SAMPLE_RATE * VAD_FRAME_MS // 1000


@dataclass
class EndpointResult:
    status: str  # "listening" | "done" | "timeout" | "max"
    audio: np.ndarray | None = None


class Endpointer:
    """Collects frames after a wake word (or in the follow-up window) until trailing silence.

    - `listen_timeout_ms`: give up if speech never starts.
    - `min_speech_ms`: voiced audio needed before we consider speech started.
    - `min_silence_ms`: trailing silence that ends the utterance.
    - `max_utterance_ms`: hard cap (noisy rooms can prevent silence).
    """

    def __init__(
        self,
        mode: int = 2,
        min_speech_ms: int = 100,
        min_silence_ms: int = 400,
        listen_timeout_ms: int = 5000,
        max_utterance_ms: int = 8000,
    ):
        self._vad = webrtcvad.Vad(mode)
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.listen_timeout_ms = listen_timeout_ms
        self.max_utterance_ms = max_utterance_ms
        self.reset()

    def reset(self, preroll: np.ndarray | None = None, listen_timeout_ms: int | None = None) -> None:
        self._chunks: list[np.ndarray] = [preroll] if preroll is not None and preroll.size else []
        self._elapsed_ms = 0
        self._speech_ms = 0
        self._voiced_run_ms = 0
        self._silence_ms = 0
        self.started = False
        self._timeout_ms = listen_timeout_ms if listen_timeout_ms is not None else self.listen_timeout_ms
        self._speech_start_ms = 0

    def is_speech(self, frame20: np.ndarray) -> bool:
        return self._vad.is_speech(frame20.astype(np.int16).tobytes(), SAMPLE_RATE)

    def feed(self, frame: np.ndarray) -> EndpointResult:
        self._chunks.append(frame)
        for i in range(0, len(frame) - VAD_FRAME_SAMPLES + 1, VAD_FRAME_SAMPLES):
            voiced = self.is_speech(frame[i:i + VAD_FRAME_SAMPLES])
            self._elapsed_ms += VAD_FRAME_MS
            if voiced:
                self._voiced_run_ms += VAD_FRAME_MS
                self._speech_ms += VAD_FRAME_MS
                self._silence_ms = 0
                if not self.started and self._voiced_run_ms >= self.min_speech_ms:
                    self.started = True
                    self._speech_start_ms = self._elapsed_ms - self._voiced_run_ms
            else:
                self._voiced_run_ms = 0
                if self.started:
                    self._silence_ms += VAD_FRAME_MS

            if self.started and self._silence_ms >= self.min_silence_ms:
                return EndpointResult("done", self._audio())
            if self.started and self._elapsed_ms - self._speech_start_ms >= self.max_utterance_ms:
                return EndpointResult("max", self._audio())
            if not self.started and self._elapsed_ms >= self._timeout_ms:
                return EndpointResult("timeout")
        return EndpointResult("listening")

    def _audio(self) -> np.ndarray:
        audio = np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.int16)
        # Drop most of the trailing silence; the STT client pads a fixed amount back.
        trim = max(0, self._silence_ms - 100) * SAMPLE_RATE // 1000
        return audio[: len(audio) - trim] if trim else audio
