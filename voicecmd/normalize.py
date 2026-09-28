"""Transcript normalisation before fast-path matching.

Strips wake-word leakage, polite framing and filler tails, and turns spoken
numbers into digits (ideas from openclaw-voice music/parser.py).
"""

from __future__ import annotations

import re

UNITS = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}

WAKE_PREFIX_RE = re.compile(
    r"^(?:(?:hey|hi|hello|ok(?:ay)?)\s+)?(?:jarvis|mycroft|marvin|alexa|computer)\b[\s,.!?]*"
)
POLITE_PREFIX_RE = re.compile(
    r"^(?:please\s+)?(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?|"
    r"i(?:\s+would|'d)\s+like\s+(?:you\s+)?to\s+|i\s+want\s+(?:you\s+)?to\s+|"
    r"(?:would\s+you\s+)?mind\s+|go\s+ahead\s+and\s+|just\s+)"
)
FILLER_TAIL_RE = re.compile(
    r"(?:[\s,.;:!?-]+(?:all\s+right|alright|okay|ok|please|thanks|thank\s+you|um+|uh+|hmm+|mm+|now))+[\s,.;:!?-]*$"
)
FILLER_HEAD_RE = re.compile(r"^(?:um+|uh+|so|okay|ok|right|and)[\s,]+")


def words_to_numbers(text: str) -> str:
    """'twenty five minutes' -> '25 minutes'; 'one hundred' -> '100'. Leaves other words alone."""
    tokens = text.split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in TENS:
            value = TENS[tok]
            if i + 1 < len(tokens) and tokens[i + 1] in UNITS and 0 < UNITS[tokens[i + 1]] < 10:
                value += UNITS[tokens[i + 1]]
                i += 1
            elif i + 1 < len(tokens) and re.fullmatch(r"[1-9]", tokens[i + 1]):
                value += int(tokens[i + 1])
                i += 1
            out.append(str(value))
        elif tok in UNITS and tok != "oh":
            value = UNITS[tok]
            if i + 1 < len(tokens) and tokens[i + 1] == "hundred":
                value *= 100
                i += 1
            out.append(str(value))
        elif tok == "hundred" and out and out[-1].isdigit():
            out[-1] = str(int(out[-1]) * 100)
        else:
            out.append(tok)
        i += 1
    joined = " ".join(out)
    # "100 and 20" -> "120"
    joined = re.sub(r"\b(\d+)00 and (\d{1,2})\b", lambda m: str(int(m.group(1)) * 100 + int(m.group(2))), joined)
    return joined


def normalize(text: str, numbers: bool = True) -> str:
    """Lowercase command form. `numbers=False` keeps number words (for song titles)."""
    t = (text or "").strip().lower().replace("’", "'")
    t = re.sub(r"\ba\.m\.?", "am", t)
    t = re.sub(r"\bp\.m\.?", "pm", t)
    t = re.sub(r"(\d)\.(\d\d)\b", r"\1:\2", t)  # whisper writes "7.30"
    t = re.sub(r"[\"“”]", "", t)
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    t = re.sub(r"[,!?;]+", " ", t)
    t = re.sub(r"\.(?=\s|$)", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    for _ in range(2):
        t = WAKE_PREFIX_RE.sub("", t).strip()
        t = FILLER_HEAD_RE.sub("", t).strip()
        t = POLITE_PREFIX_RE.sub("", t).strip()
    prev = None
    while prev != t:
        prev = t
        t = FILLER_TAIL_RE.sub("", t).strip()
    t = re.sub(r"^please\s+", "", t)
    if numbers:
        t = words_to_numbers(t)
    t = re.sub(r"(\d)\s*%", r"\1 percent", t)
    t = re.sub(r"\s+", " ", t).strip(" .-")
    return t
