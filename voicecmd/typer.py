"""Types text into the focused window via ydotool (/dev/uinput; works on GNOME Wayland and X11)."""

from __future__ import annotations

import shutil
import subprocess
import unicodedata
from typing import Callable

ASCII_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "–": "-", "—": "-", "‐": "-", "−": "-",
    "…": "...", "\u00a0": " ", "\u2009": " ", "\u202f": " ",
})

KEYS = {
    "enter": ["enter"],
    "shift+enter": ["shift+enter"],
    "backspace": ["backspace"],
}

Runner = Callable[[list[str], bytes | None], int]


def to_ascii(text: str) -> str:
    """ydotool types US key codes, so anything outside ASCII would come out wrong or not at all."""
    text = (text or "").translate(ASCII_MAP)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


def _run(argv: list[str], stdin: bytes | None) -> int:
    return subprocess.run(argv, input=stdin, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          timeout=60, check=False).returncode


class Typer:
    def __init__(self, tool: str = "ydotool", key_delay_ms: int = 4, runner: Runner | None = None):
        self.tool = tool
        self.key_delay_ms = key_delay_ms
        self._runner = runner or _run
        self._path = shutil.which(tool) if runner is None else tool

    @property
    def available(self) -> bool:
        return bool(self._path)

    def type_text(self, text: str) -> int:
        """Types `text`; returns the number of characters sent (0 if nothing to type or tool missing)."""
        out = to_ascii(text)
        if not out or not self._path:
            return 0
        rc = self._runner([self._path, "type", "--key-delay", str(self.key_delay_ms), "--file", "-"],
                          out.encode("ascii"))
        if rc != 0:
            raise RuntimeError(f"{self.tool} type exited {rc}")
        return len(out)

    def key(self, name: str, times: int = 1) -> None:
        seq = KEYS.get(name)
        if seq is None:
            raise ValueError(f"unknown key {name!r}")
        if times <= 0 or not self._path:
            return
        argv = [self._path, "key", "--key-delay", str(self.key_delay_ms)]
        if times > 1:
            argv += ["--repeat", str(times)]
        rc = self._runner(argv + seq, None)
        if rc != 0:
            raise RuntimeError(f"{self.tool} key exited {rc}")
