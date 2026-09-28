"""Local time/date answers."""

from __future__ import annotations

import re
from datetime import datetime

from ..intents import Intent, Reply

TIME_RE = re.compile(r"^(?:what(?:'s| is) the time|what time is it|what's the time now|time|tell me the time|do you have the time)$")
DATE_RE = re.compile(r"^(?:what(?:'s| is) (?:the date|today's date|the day|today)|what day is (?:it|today)|what's the date today|date)$")


class ClockHandler:
    domain = "clock"

    def __init__(self, now=datetime.now):
        self.now = now

    def parse(self, n: str, nw: str) -> Intent | None:
        if TIME_RE.match(n):
            return Intent("clock", "time")
        if DATE_RE.match(n):
            return Intent("clock", "date")
        return None

    def execute(self, intent: Intent) -> Reply:
        now = self.now()
        if intent.action == "time":
            return Reply(f"It's {now.strftime('%-I:%M %p').lower().replace('am', 'a.m.').replace('pm', 'p.m.')}")
        return Reply(f"It's {now.strftime('%A, %-d %B')}.")
