"""Dictation phrase parsing on raw (case- and punctuation-preserving) whisper text."""

from __future__ import annotations

import re

WAKE_LEAK_RE = re.compile(
    r"^\s*(?:(?:hey|hi|hello|ok(?:ay)?)[\s,]+)?(?:jarvis|mycroft|marvin|alexa|computer)\b[\s,.!?:;-]*",
    re.I,
)
ONCE_RE = re.compile(r"^(?:please\s+)?dictate\b[\s,.:;!?-]*(?P<text>.*?)\s*$", re.I | re.S)
START_RE = re.compile(
    r"^(?:please\s+)?(?:(?:start|begin|enable|turn\s+on)\s+(?:dictation|dictating|dictate)(?:\s+mode)?"
    r"|dictation\s+mode(?:\s+on)?)$"
)
STOP_PHRASE = r"(?:(?:stop|end|finish|exit|quit)\s+(?:the\s+)?(?:dictation|dictating|dictate)(?:\s+mode)?|dictation\s+(?:mode\s+)?off)"
STOP_RE = re.compile(rf"^(?:please\s+)?{STOP_PHRASE}$")
STOP_TAIL_RE = re.compile(rf"[\s,;:-]*\b(?:please\s+)?{STOP_PHRASE.replace(r'\s+', r'[\s,.]+')}[\s.!?]*$", re.I)
BARE_STOP_RE = re.compile(r"^(?:stop|stop it|stop that|that's enough|that is enough|enough|done|i'm done)$")
SPOKEN_KEYS = {
    "new line": "newline", "newline": "newline", "next line": "newline",
    "new paragraph": "paragraph",
    "send it": "send", "submit": "send", "submit it": "send", "press enter": "send", "hit enter": "send",
    "scratch that": "scratch", "delete that": "scratch", "undo that": "scratch",
}


def canonical(text: str) -> str:
    t = (text or "").strip().lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9\s']+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def strip_wake(raw: str) -> str:
    return WAKE_LEAK_RE.sub("", raw or "", count=1).strip()


def clean(raw: str) -> str:
    return re.sub(r"\s+", " ", raw or "").strip()


def match_command(raw: str) -> tuple[str, str] | None:
    """Outside dictation: ("once", text) for "dictate <text>", ("start", "") to enter continuous mode."""
    body = strip_wake(clean(raw))
    if not body:
        return None
    if START_RE.match(canonical(body)):
        return ("start", "")
    m = ONCE_RE.match(body)
    if m and m.group("text").strip(" .,"):
        return ("once", m.group("text"))
    return None


def is_stop(raw: str) -> bool:
    return bool(STOP_RE.match(canonical(strip_wake(clean(raw)))))


def is_bare_stop(raw: str) -> bool:
    return bool(BARE_STOP_RE.match(canonical(strip_wake(clean(raw)))))


def match_in_mode(raw: str) -> tuple[str, str]:
    """Inside dictation: (action, text_to_type). Actions: text | newline | paragraph | send | scratch | stop."""
    text = clean(raw)
    canon = canonical(text)
    if not canon:
        return ("text", "")
    if STOP_RE.match(canonical(strip_wake(text))):
        return ("stop", "")
    key = SPOKEN_KEYS.get(canon)
    if key:
        return (key, "")
    m = STOP_TAIL_RE.search(text)
    if m:
        return ("stop", text[: m.start()].strip())
    return ("text", text)


def is_command(raw: str) -> bool:
    """Lets the ghost gate accept short dictation commands ("start dictation", "dictate hi")."""
    return match_command(raw) is not None or is_stop(raw)
