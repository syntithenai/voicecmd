"""Ghost-transcript gate: rejects Whisper hallucinations, self-echo and stray blips.

Trimmed port of openclaw-voice `decide_ghost_transcript` plus the whisper-side
blocklist / repetition-loop defences.
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .normalize import words_to_numbers

LOOP_RE = re.compile(r"(.{10,}?)\1{7,}", re.S)
BRACKETED_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*|♪+")
ARTIFACT_PATTERNS = (
    re.compile(r"\b(?:we(?:'ll|'re| will| are)?|i(?:'ll| will)?)\s+(?:be\s+)?right\s+back\b"),
    re.compile(r"\bstay\s+tuned\b"),
    re.compile(r"\bthanks?\s+(?:you\s+)?for\s+watching\b"),
    re.compile(r"\bsubtitles\s+by\b"),
    re.compile(r"\blike\s+and\s+subscribe\b"),
)
ACK_TOKENS = {"yes", "yeah", "yep", "no", "nope", "ok", "okay", "sure", "correct", "cancel", "stop"}


def canonical(text: str) -> str:
    lowered = (text or "").strip().lower().replace("’", "'")
    lowered = re.sub(r"[^a-z0-9\s']+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def clean_transcript(text: str) -> str:
    text = BRACKETED_RE.sub(" ", text or "")
    match = LOOP_RE.search(text)
    if match:
        text = text[: match.start()] + match.group(1) + text[match.end():]
    return re.sub(r"\s+", " ", text).strip()


def load_blocklist(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(canonical(line))
    return out


def _echo_canonical(text: str) -> str:
    # TTS says "14.6" as "fourteen point six"; whisper may write either form.
    text = re.sub(r"\bpoint\b", " ", canonical(words_to_numbers(canonical(text))))
    return re.sub(r"\s+", " ", text).strip()


def self_echo_similarity(transcript: str, recent: list[str]) -> float:
    tx = _echo_canonical(transcript)
    if not tx:
        return 0.0
    tx_set = set(tx.split())
    best = 0.0
    for raw in recent:
        cand = _echo_canonical(raw)
        if not cand:
            continue
        if tx == cand:
            return 1.0
        if len(tx) >= 8 and (tx in cand or cand in tx):
            best = max(best, 0.92)
        cand_set = set(cand.split())
        shared = len(tx_set & cand_set)
        combined = 0.65 * shared / max(1, len(tx_set)) + 0.35 * shared / max(1, len(cand_set))
        best = max(best, combined)
    return min(1.0, best)


@dataclass
class GhostDecision:
    accept: bool
    reason: str
    text: str


class GhostGate:
    def __init__(
        self,
        blocklist_path: Path,
        is_supported_command: Callable[[str], bool] | None = None,
        echo_threshold: float = 0.75,
        playback_tail_ms: int = 1500,
        no_speech_threshold: float = 0.6,
    ):
        self.blocklist = load_blocklist(blocklist_path)
        self.is_supported_command = is_supported_command or (lambda _t: False)
        self.echo_threshold = echo_threshold
        self.playback_tail_s = playback_tail_ms / 1000.0
        self.no_speech_threshold = no_speech_threshold
        self._recent_tts: deque[tuple[str, float]] = deque(maxlen=6)
        self.tts_playing = False
        self._tts_ended_at = 0.0

    def note_tts(self, text: str) -> None:
        self._recent_tts.append((text, time.monotonic()))

    def note_tts_state(self, playing: bool) -> None:
        if self.tts_playing and not playing:
            self._tts_ended_at = time.monotonic()
        self.tts_playing = playing

    def decide(
        self,
        text: str,
        *,
        no_speech_prob: float = 0.0,
        avg_logprob: float = 0.0,
        expecting_reply: bool = False,
        now: float | None = None,
    ) -> GhostDecision:
        now = time.monotonic() if now is None else now
        cleaned = clean_transcript(text)
        canon = canonical(cleaned)
        if not canon or not re.search(r"[a-z0-9]", canon):
            return GhostDecision(False, "empty", cleaned)

        tokens = canon.split()
        supported = self.is_supported_command(cleaned)
        is_ack = canon in ACK_TOKENS

        if no_speech_prob >= self.no_speech_threshold and avg_logprob < -0.8 and not supported:
            return GhostDecision(False, "no_speech", cleaned)

        if any(p.search(canon) for p in ARTIFACT_PATTERNS):
            return GhostDecision(False, "artifact_phrase", cleaned)

        recent = [t for t, _ts in self._recent_tts]
        in_tail = self.tts_playing or (now - self._tts_ended_at) <= self.playback_tail_s
        if in_tail and recent and self_echo_similarity(cleaned, recent) >= self.echo_threshold:
            return GhostDecision(False, "self_echo", cleaned)

        if canon in self.blocklist:
            if expecting_reply and is_ack:
                return GhostDecision(True, "ack_expected", cleaned)
            if supported:
                return GhostDecision(True, "supported_command", cleaned)
            return GhostDecision(False, "blocklist", cleaned)

        if len(tokens) <= 2:
            if supported:
                return GhostDecision(True, "supported_command", cleaned)
            if expecting_reply:
                return GhostDecision(True, "expected_reply", cleaned)
            return GhostDecision(False, "short_unsupported", cleaned)

        return GhostDecision(True, "accept", cleaned)
