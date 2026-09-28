"""Weather answers straight from the weatherstation JSONL files (no LLM)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

from ..intents import Intent, Reply

TOPIC_PATTERNS = [
    ("yesterday", re.compile(r"\byesterday\b")),
    ("soil", re.compile(r"\b(soil|garden|moisture|plants? need water|water the (?:garden|plants))\b")),
    ("humidity", re.compile(r"\bhumid(?:ity)?\b")),
    ("pressure", re.compile(r"\b(pressure|barometer|barometric)\b")),
    ("today", re.compile(r"\b(high|low|highs|lows|max(?:imum)?|min(?:imum)?|hottest|coldest|warmest|today's)\b")),
    ("temperature", re.compile(
        r"\b(temp(?:erature)?|degrees|how (?:hot|cold|warm|chilly)|is it (?:hot|cold|warm|chilly|freezing))\b")),
    ("summary", re.compile(r"\b(weather|outside|conditions)\b")),
]
QUESTION_HINT = re.compile(r"^(?:what|what's|how|is|are|tell|give|weather|temperature|humidity|pressure|soil|check|do|does|should)\b")


def _read_last_json_line(path: Path, max_bytes: int = 65536) -> dict | None:
    if not path.is_file():
        return None
    with path.open("rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - max_bytes))
        lines = fh.read().splitlines()
    for raw in reversed(lines):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            continue
    return None


def _read_daily(path: Path, day: str) -> dict | None:
    if not path.is_file():
        return None
    with path.open("rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - 16384))
        lines = fh.read().splitlines()
    for raw in reversed(lines):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if row.get("date") == day:
            return row
    return None


def _deg(v: float) -> str:
    return f"{round(v)} degrees" if abs(v - round(v)) < 0.05 else f"{v:.1f} degrees"


class WeatherHandler:
    domain = "weather"

    def __init__(self, data_dir: Path, now=datetime.now):
        self.data_dir = Path(data_dir)
        self.now = now

    def parse(self, n: str, nw: str) -> Intent | None:
        if n.startswith(("play ", "put on ", "queue ")):
            return None
        if not QUESTION_HINT.match(n) and len(n.split()) > 3:
            return None
        for topic, pattern in TOPIC_PATTERNS:
            if pattern.search(n):
                return Intent("weather", topic)
        return None

    def _latest(self) -> tuple[dict | None, str]:
        row = _read_last_json_line(self.data_dir / "readings.jsonl")
        if not row:
            return None, ""
        stale = ""
        try:
            ts = datetime.fromisoformat(str(row.get("ts")))
            if self.now().timestamp() - ts.timestamp() > 45 * 60:
                stale = f" as of {ts.strftime('%-I:%M %p').lower()}"
        except (TypeError, ValueError):
            pass
        return row, stale

    def execute(self, intent: Intent) -> Reply:
        topic = intent.action
        if topic in ("today", "yesterday"):
            day = self.now().date() - timedelta(days=1 if topic == "yesterday" else 0)
            daily = _read_daily(self.data_dir / "daily.jsonl", day.isoformat())
            if not daily or daily.get("temp_max") is None:
                return Reply(f"I don't have {topic}'s figures yet.", ok=False)
            when = "Yesterday" if topic == "yesterday" else "Today so far"
            return Reply(
                f"{when}, the high was {_deg(daily['temp_max'])} and the low was {_deg(daily['temp_min'])}."
            )

        row, stale = self._latest()
        if not row:
            return Reply("I can't read the weather station right now.", ok=False)
        temp = row.get("outdoor_temp_c")
        hum = row.get("outdoor_humidity_pct")
        pres = row.get("rel_pressure_hpa") or row.get("abs_pressure_hpa")

        if topic == "temperature" and temp is not None:
            return Reply(f"It's {_deg(temp)} outside{stale}.")
        if topic == "humidity" and hum is not None:
            return Reply(f"Humidity is {round(hum)} percent{stale}.")
        if topic == "pressure" and pres is not None:
            return Reply(f"Pressure is {round(pres)} hectopascals{stale}.")
        if topic == "soil":
            soil = {k: v for k, v in (row.get("soil_moisture_pct") or {}).items() if v}
            if not soil:
                return Reply("I don't have any soil moisture readings.", ok=False)
            driest = min(soil, key=soil.get)
            lo, hi = round(min(soil.values())), round(max(soil.values()))
            return Reply(
                f"Soil moisture ranges from {lo} to {hi} percent across {len(soil)} sensors; "
                f"the driest is sensor {driest}{stale}."
            )

        parts = []
        if temp is not None:
            parts.append(f"it's {_deg(temp)}")
        if hum is not None:
            parts.append(f"{round(hum)} percent humidity")
        daily = _read_daily(self.data_dir / "daily.jsonl", self.now().date().isoformat())
        extra = ""
        if daily and daily.get("temp_max") is not None:
            extra = f" Today's high so far is {_deg(daily['temp_max'])}, low {_deg(daily['temp_min'])}."
        if not parts:
            return Reply("I can't read the weather station right now.", ok=False)
        return Reply(f"Outside {' with '.join(parts)}{stale}.{extra}")
