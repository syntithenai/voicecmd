"""Timers and alarms: regex parsing, persisted JSON store, scheduler thread, ringing state."""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from ..intents import Intent, Reply

UNIT_SECONDS = {"h": 3600, "m": 60, "s": 1}
STALE_S = 1800
UNIT_RE = r"(hours?|hrs?|minutes?|mins?|seconds?|secs?)"


def _unit(word: str) -> int:
    return UNIT_SECONDS[word[0]] if word[0] in "hs" else 60


def parse_duration(text: str) -> tuple[int, str]:
    """Sum every duration phrase in `text`. Returns (seconds, text with durations removed)."""
    total = 0.0
    rest = f" {text} "

    def take(pattern: str, fn) -> None:
        nonlocal total, rest
        for m in list(re.finditer(pattern, rest)):
            total += fn(m)
        rest = re.sub(pattern, " ", rest)

    take(rf"\b(\d+(?:\.\d+)?) and a half {UNIT_RE}\b", lambda m: (float(m.group(1)) + 0.5) * _unit(m.group(2)))
    take(rf"\b(?:an?|1) (hour|minute) and a half\b", lambda m: 1.5 * _unit(m.group(1)))
    take(r"\b(?:a )?half (?:an? )?(hour|minute)\b", lambda m: 0.5 * _unit(m.group(1)))
    take(r"\b3 quarters of an hour\b", lambda m: 2700)
    take(r"\b(?:a )?quarter (?:of )?(?:an )?hour\b", lambda m: 900)
    take(rf"\b(?:a )?couple (?:of )?{UNIT_RE}\b", lambda m: 2 * _unit(m.group(1)))
    take(rf"\b(?:a )?few {UNIT_RE}\b", lambda m: 3 * _unit(m.group(1)))
    take(rf"\b(\d+(?:\.\d+)?|an?)[ -]?{UNIT_RE}\b",
         lambda m: (1.0 if m.group(1) in ("a", "an") else float(m.group(1))) * _unit(m.group(2)))
    rest = re.sub(r"\band\b(?=\s*$)", " ", rest)
    return int(round(total)), re.sub(r"\s+", " ", rest).strip()


def parse_clock(text: str, now: datetime | None = None) -> datetime | None:
    """Clock phrases -> next matching datetime: '7:30 am', '7 30', 'half past 6', 'noon', '18:30'."""
    now = now or datetime.now()
    t = f" {text.strip()} "
    tomorrow = " tomorrow" in t
    ampm = None
    if re.search(r"\b(am|in the morning|this morning|tomorrow morning)\b", t):
        ampm = "am"
    elif re.search(r"\b(pm|in the (?:afternoon|evening)|this (?:afternoon|evening)|tonight|at night)\b", t):
        ampm = "pm"

    hour = minute = None
    if re.search(r"\bnoon\b|\bmidday\b", t):
        hour, minute, ampm = 12, 0, "fixed"
    elif re.search(r"\bmidnight\b", t):
        hour, minute, ampm = 0, 0, "fixed"
    else:
        m = re.search(r"\bhalf past (\d{1,2})\b", t)
        if m:
            hour, minute = int(m.group(1)), 30
        if hour is None and (m := re.search(r"\bquarter past (\d{1,2})\b", t)):
            hour, minute = int(m.group(1)), 15
        if hour is None and (m := re.search(r"\bquarter to (\d{1,2})\b", t)):
            hour, minute = (int(m.group(1)) - 1) % 24, 45
        if hour is None and (m := re.search(r"\b(\d{1,2}) (?:minutes )?past (\d{1,2})\b", t)):
            hour, minute = int(m.group(2)), int(m.group(1))
        if hour is None and (m := re.search(r"\b(\d{1,2}) (?:minutes )?to (\d{1,2})\b", t)):
            hour, minute = (int(m.group(2)) - 1) % 24, 60 - int(m.group(1))
        if hour is None and (m := re.search(r"\b(\d{1,2})(?::| )(\d{2})\b", t)):
            hour, minute = int(m.group(1)), int(m.group(2))
        if hour is None and (m := re.search(r"\b(\d{3,4})\s*(?:am|pm|hours|tomorrow|tonight|in the|this)\b", t)):
            v = m.group(1)
            hour, minute = int(v[:-2]), int(v[-2:])
        if hour is None and (m := re.search(r"\b(\d{1,2})\s*(?:o'?clock|am|pm)?\b", t)):
            hour, minute = int(m.group(1)), 0
    if hour is None or minute is None or hour > 23 or minute > 59:
        return None

    candidates: list[int] = []
    if ampm == "fixed" or hour > 12 or hour == 0:
        candidates = [hour]
    elif ampm == "am":
        candidates = [0 if hour == 12 else hour]
    elif ampm == "pm":
        candidates = [12 if hour == 12 else hour + 12]
    else:
        candidates = [hour % 12, hour % 12 + 12]

    base = now.replace(second=0, microsecond=0)
    options = []
    for h in candidates:
        when = base.replace(hour=h, minute=minute)
        if tomorrow:
            when = when + timedelta(days=1) if when.date() == now.date() else when
            if when <= now:
                when += timedelta(days=1)
        elif when <= now:
            when += timedelta(days=1)
        options.append(when)
    return min(options)


def speak_duration(seconds: float) -> str:
    seconds = int(round(max(0, seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h} hour{'s' if h != 1 else ''}")
    if m:
        parts.append(f"{m} minute{'s' if m != 1 else ''}")
    if s and not h:
        parts.append(f"{s} second{'s' if s != 1 else ''}")
    if not parts:
        return "0 seconds"
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def speak_clock(when: datetime, now: datetime | None = None) -> str:
    now = now or datetime.now()
    h12 = when.hour % 12 or 12
    label = f"{h12}:{when.minute:02d}" if when.minute else f"{h12}"
    suffix = "am" if when.hour < 12 else "pm"
    day = ""
    if when.date() == (now + timedelta(days=1)).date():
        day = " tomorrow"
    elif when.date() != now.date():
        day = " on " + when.strftime("%A")
    return f"{label} {suffix}{day}"


LABEL_STOPWORDS = {
    "set", "start", "create", "make", "a", "an", "the", "timer", "timers", "for", "called", "named",
    "labelled", "labeled", "please", "me", "new", "and", "of", "countdown", "on", "up", "my", "in",
    "remind", "to", "alarm", "at", "put", "time",
}

RINGING_STOP_RE = re.compile(
    r"^(?:stop|ok(?:ay)?|dismiss|enough|shut up|quiet|cancel|off|turn (?:it )?off|i'm up|i am up|"
    r"got it|thanks?|thank you|stop (?:the )?(?:alarm|timer|ringing|beeping|noise|it)|"
    r"(?:turn off|cancel|dismiss) (?:the )?(?:alarm|timer))$"
)
SNOOZE_RE = re.compile(r"^snooze(?: (?:it|the alarm))?(?: for (\d+) minutes?)?$")
CANCEL_RE = re.compile(
    r"^(?:cancel|stop|delete|clear|remove|turn off|kill)\s+(?:the\s+|my\s+)?(?:(all)\s+(?:the\s+|my\s+|of\s+the\s+)?)?"
    r"(?:(.+?)\s+)?(timers?|alarms?|reminders?)$"
)
STATUS_RE = re.compile(
    r"^(?:how (?:long|much time|much longer|many minutes)(?: is| is there|'s| do i have)?(?: left)?"
    r"(?: on (?:the |my )?(?:(.+?) )?timers?)?|how long (?:is )?left\b.*\btimers?|time left|"
    r"what timers?(?: do i have| are (?:set|running))?|"
    r"(?:check|list) (?:my |the )?(?:timers?|alarms?)|what alarms?(?: do i have| are set)?|"
    r"when(?:'s| is) (?:my |the )?(?:next )?alarm|(?:are there |do i have )?any (?:timers|alarms)(?: set| running)?|"
    r"timer status)$"
)
ALARM_RE = re.compile(r"^(?:set (?:an? |my )?alarm|alarm|wake me(?: up)?|create (?:an? )?alarm)\b\s*(?:for |at )?(.*)$")
REMIND_RE = re.compile(r"^remind me in (.+?)(?: to (.+))?$|^in (.+?) remind me to (.+)$")


@dataclass
class Timer:
    id: str
    kind: str  # timer | alarm | reminder
    due: float  # epoch seconds
    label: str = ""
    created: float = 0.0

    def name(self) -> str:
        if self.kind == "reminder":
            return f"reminder to {self.label}" if self.label else "reminder"
        if self.kind == "alarm":
            return f"{self.label} alarm" if self.label else "alarm"
        return f"{self.label} timer" if self.label else "timer"


class TimerHandler:
    domain = "timer"

    def __init__(self, store_path: Path, on_fire: Callable[[Timer], None] | None = None, clock=time.time):
        self.store_path = store_path
        self.on_fire = on_fire
        self.clock = clock
        self._lock = threading.RLock()
        self.timers: list[Timer] = []
        self.ringing: list[Timer] = []
        self._stop = threading.Event()
        self._load()

    # --- persistence ---

    def _load(self) -> None:
        if not self.store_path.is_file():
            return
        try:
            raw = json.loads(self.store_path.read_text(encoding="utf-8"))
            self.timers = [Timer(**t) for t in raw if isinstance(t, dict)]
        except (json.JSONDecodeError, TypeError) as exc:
            print(f"timers: ignoring unreadable store: {exc}", flush=True)

    def _save(self) -> None:
        tmp = self.store_path.with_suffix(".tmp")
        tmp.write_text(json.dumps([asdict(t) for t in self.timers], indent=2), encoding="utf-8")
        tmp.replace(self.store_path)

    # --- scheduler ---

    def start(self) -> None:
        threading.Thread(target=self._loop, name="timers", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(0.25):
            self.tick()

    def tick(self) -> list[Timer]:
        now = self.clock()
        with self._lock:
            due = [t for t in self.timers if t.due <= now]
            if not due:
                return []
            self.timers = [t for t in self.timers if t.due > now]
            self._save()
            stale = [t for t in due if now - t.due > STALE_S]
            for t in stale:
                print(f"timers: dropping {t.name()} ({now - t.due:.0f}s overdue, missed while down)", flush=True)
            due = [t for t in due if now - t.due <= STALE_S]
            self.ringing.extend(due)
        for t in due:
            late = now - t.due
            print(f"timers: {t.name()} fired ({late:.0f}s late)", flush=True)
            if self.on_fire:
                try:
                    self.on_fire(t)
                except Exception as exc:
                    print(f"timers: on_fire failed: {exc}", flush=True)
        return due

    def add(self, kind: str, due: float, label: str = "") -> Timer:
        t = Timer(id=uuid.uuid4().hex[:8], kind=kind, due=due, label=label, created=self.clock())
        with self._lock:
            self.timers.append(t)
            self.timers.sort(key=lambda x: x.due)
            self._save()
        return t

    def stop_ringing(self) -> list[Timer]:
        with self._lock:
            stopped, self.ringing = self.ringing, []
            if stopped:
                self._recently_stopped = (stopped, self.clock())
        return stopped

    def _can_snooze(self) -> list[Timer]:
        if self.ringing:
            return self.ringing
        stopped, at = getattr(self, "_recently_stopped", ([], 0.0))
        return stopped if self.clock() - at <= 120 else []

    # --- parsing ---

    def parse(self, n: str, nw: str) -> Intent | None:
        if self.ringing and RINGING_STOP_RE.match(n):
            return Intent("timer", "stop_ringing")
        if self._can_snooze():
            m = SNOOZE_RE.match(n)
            if m:
                return Intent("timer", "snooze", {"minutes": int(m.group(1) or 9)})
        m = STATUS_RE.match(n)
        if m:
            return Intent("timer", "status", {"label": (m.group(1) or "").strip()})
        m = CANCEL_RE.match(n)
        if m:
            kind = {"a": "alarm", "r": "reminder"}.get(m.group(3)[0], "timer")
            label = (m.group(2) or "").strip()
            everything = bool(m.group(1)) or label in ("old", "all of the", "all of my")  # whisper hears "all" as "old"
            return Intent("timer", "cancel", {"all": everything, "label": "" if everything else label, "kind": kind})
        m = ALARM_RE.match(n)
        if m and m.group(1).strip():
            return Intent("timer", "alarm", {"time": m.group(1).strip()})
        m = REMIND_RE.match(n)
        if m:
            dur_text = m.group(1) or m.group(3) or ""
            label = (m.group(2) or m.group(4) or "").strip()
            seconds, _rest = parse_duration(dur_text)
            if seconds > 0:
                return Intent("timer", "set", {"seconds": seconds, "label": label, "reminder": bool(label)})
        if re.search(r"\btimer\b", n) or n.startswith("time for "):
            seconds, rest = parse_duration(n)
            if seconds > 0:
                label = " ".join(w for w in rest.split() if w not in LABEL_STOPWORDS)
                return Intent("timer", "set", {"seconds": seconds, "label": label})
        return None

    # --- execute ---

    def _match(self, label: str, kind: str | None) -> list[Timer]:
        pool = [t for t in self.timers if kind is None or t.kind == kind]
        if not label:
            return pool
        return [t for t in pool if label in t.label or t.label in label and t.label]

    def execute(self, intent: Intent) -> Reply:
        a, p = intent.action, intent.params
        now = self.clock()
        if a == "set":
            seconds = int(p.get("seconds") or 0)
            if seconds <= 0:
                return Reply("How long should the timer be?", ok=False, expects_reply=True)
            label = str(p.get("label") or "").strip()
            if p.get("reminder") and label:
                t = self.add("reminder", now + seconds, label)
                return Reply(f"I'll remind you to {label} in {speak_duration(seconds)}.", data={"id": t.id})
            t = self.add("timer", now + seconds, label)
            return Reply(f"{t.name().capitalize()} set for {speak_duration(seconds)}.", data={"id": t.id})
        if a == "alarm":
            when = parse_clock(str(p.get("time") or ""), datetime.fromtimestamp(now))
            if when is None:
                return Reply("What time should the alarm be?", ok=False, expects_reply=True)
            t = self.add("alarm", when.timestamp(), str(p.get("label") or "").strip())
            return Reply(f"Alarm set for {speak_clock(when, datetime.fromtimestamp(now))}.", data={"id": t.id})
        if a == "cancel":
            kind = p.get("kind")
            with self._lock:
                targets = self._match(str(p.get("label") or ""), kind) if not p.get("all") else [
                    t for t in self.timers if kind is None or t.kind == kind]
                if not targets:
                    word = f"{kind}s" if kind else "timers or alarms"
                    return Reply(f"There are no {word} to cancel.", ok=False)
                if len(targets) > 1 and not p.get("all") and not p.get("label"):
                    targets = [min(targets, key=lambda t: t.due)]
                ids = {t.id for t in targets}
                self.timers = [t for t in self.timers if t.id not in ids]
                self._save()
            if len(targets) == 1:
                return Reply(f"Cancelled the {targets[0].name()}.")
            return Reply(f"Cancelled {len(targets)} {kind or 'timer'}s.")
        if a == "status":
            with self._lock:
                items = self._match(str(p.get("label") or ""), None)
            if not items:
                return Reply("You have no timers or alarms set.")
            parts = []
            for t in items[:4]:
                if t.kind == "timer":
                    parts.append(f"{speak_duration(t.due - now)} left on the {t.name()}")
                elif t.kind == "reminder":
                    parts.append(f"{speak_duration(t.due - now)} until the {t.name()}")
                else:
                    parts.append(f"the {t.name()} is set for {speak_clock(datetime.fromtimestamp(t.due), datetime.fromtimestamp(now))}")
            text = "; ".join(parts)
            return Reply(text[0].upper() + text[1:] + ".")
        if a == "stop_ringing":
            self.stop_ringing()
            return Reply("", silent=True)
        if a == "snooze":
            stopped = self._can_snooze()
            self.stop_ringing()
            self._recently_stopped = ([], 0.0)
            minutes = int(p.get("minutes") or 9)
            label = stopped[0].label if stopped else ""
            kind = stopped[0].kind if stopped else "alarm"
            self.add(kind, now + minutes * 60, label)
            return Reply(f"Snoozing for {minutes} minutes.")
        return Reply("Sorry, I can't do that with timers.", ok=False)
