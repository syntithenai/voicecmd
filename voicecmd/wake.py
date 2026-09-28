"""openWakeWord wrapper with near-silence guard and cooldown."""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np

from .audio import rms


def resolve_model_path(name: str) -> str:
    if Path(name).is_file():
        return name
    import openwakeword

    models = openwakeword.models
    if name in models:
        return models[name]["model_path"]
    raise ValueError(f"unknown wake model {name!r}; bundled: {', '.join(sorted(models))}")


class WakeWord:
    def __init__(self, model: str, threshold: float, min_rms: float, cooldown_ms: int):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from openwakeword.model import Model

            self._model = Model(wakeword_model_paths=[resolve_model_path(model)])
        self.threshold = threshold
        self.min_rms = min_rms
        self.cooldown_s = cooldown_ms / 1000.0
        self._last_fire = 0.0
        self.last_score = 0.0

    def process(self, frame: np.ndarray) -> bool:
        """Feed one 80 ms int16 frame; True when the wake word fires."""
        scores = self._model.predict(frame)
        score = float(max(scores.values())) if scores else 0.0
        self.last_score = score
        if score < self.threshold:
            return False
        now = time.monotonic()
        if now - self._last_fire < self.cooldown_s:
            return False
        if rms(frame) < self.min_rms:
            return False
        self._last_fire = now
        self.reset()
        return True

    def reset(self) -> None:
        """Clear the model's rolling feature buffer so one utterance can't fire twice."""
        try:
            self._model.reset()
        except AttributeError:
            for key in getattr(self._model, "prediction_buffer", {}):
                self._model.prediction_buffer[key].clear()
