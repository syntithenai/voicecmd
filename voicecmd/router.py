"""Regex fast-path router; anything unmatched goes to the LLM fallback."""

from __future__ import annotations

import re
import threading
import time
from typing import Callable, Protocol

from .intents import Intent, Reply
from .normalize import normalize


class Handler(Protocol):
    domain: str

    def parse(self, n: str, nw: str) -> Intent | None: ...

    def execute(self, intent: Intent) -> Reply: ...


DISMISS_RE = re.compile(
    r"^(?:never\s*mind|nevermind|forget\s+it|cancel|nothing|no\s+thanks?|that's\s+all|that\s+is\s+all|"
    r"thanks?|thank\s+you|cheers|ok|okay|go\s+away|shut\s+up)$"
)
HELP_RE = re.compile(r"^(?:help|what\s+can\s+you\s+do|what\s+do\s+you\s+do|what\s+are\s+your\s+commands)$")
HELP_TEXT = (
    "I can play music, pause, skip or change the volume, set timers and alarms, "
    "tell you the weather outside, and type what you say with dictate or start dictation. "
    "Anything else I'll try to answer."
)


def help_text(device_names: list[str] | None = None) -> str:
    names = list(device_names or [])
    if not names:
        return HELP_TEXT
    listed = ", ".join(names[:-1]) + f" and {names[-1]}" if len(names) > 1 else names[0]
    return HELP_TEXT.replace(" Anything else", f" I can also switch {listed}. Anything else")


class SystemHandler:
    domain = "system"

    def __init__(self, device_names: Callable[[], list[str]] | None = None):
        self.device_names = device_names

    def parse(self, n: str, nw: str) -> Intent | None:
        if DISMISS_RE.match(n):
            return Intent("system", "dismiss")
        if HELP_RE.match(n):
            return Intent("system", "help")
        return None

    def execute(self, intent: Intent) -> Reply:
        if intent.action == "help":
            return Reply(help_text(self.device_names() if self.device_names else None), intent=intent)
        return Reply("", intent=intent, silent=True)


class Router:
    def __init__(self, handlers: list[Handler], llm=None):
        self.handlers = handlers
        self.by_domain = {h.domain: h for h in handlers}
        self.llm = llm

    def parse(self, text: str) -> Intent | None:
        n = normalize(text)
        if not n:
            return None
        nw = normalize(text, numbers=False)
        for handler in self.handlers:
            intent = handler.parse(n, nw)
            if intent is not None:
                return intent
        return None

    def is_supported(self, text: str) -> bool:
        try:
            return self.parse(text) is not None
        except Exception:
            return False

    def execute(self, intent: Intent) -> Reply:
        handler = self.by_domain.get(intent.domain)
        if handler is None:
            return Reply(f"I can't do {intent.domain} things yet.", ok=False, intent=intent)
        try:
            reply = handler.execute(intent)
        except Exception as exc:
            print(f"router: {intent.domain}.{intent.action} failed: {exc}", flush=True)
            return Reply("Sorry, that didn't work.", ok=False, intent=intent, data={"error": str(exc)})
        if reply.intent is None:
            reply.intent = intent
        return reply

    def route(self, text: str, on_slow: Callable[[], None] | None = None, slow_after_s: float = 1.5) -> Reply:
        intent = self.parse(text)
        if intent is not None:
            return self.execute(intent)
        if self.llm is None:
            return Reply("Sorry, I didn't catch that.", ok=False)

        timer = None
        if on_slow is not None:
            timer = threading.Timer(slow_after_s, on_slow)
            timer.daemon = True
            timer.start()
        started = time.monotonic()
        try:
            reply = self.llm.handle(text, self.execute)
        finally:
            if timer is not None:
                timer.cancel()
        reply.data.setdefault("llm_ms", int((time.monotonic() - started) * 1000))
        return reply
