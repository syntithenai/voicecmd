import json
from datetime import datetime

import pytest

from voicecmd.handlers.timers import STALE_S, TimerHandler, parse_clock, parse_duration, speak_duration
from voicecmd.normalize import normalize


class FakeClock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.mark.parametrize(
    "text, seconds, rest",
    [
        ("5 minutes", 300, ""),
        ("an hour and a half", 5400, ""),
        ("half an hour", 1800, ""),
        ("2 and a half minutes", 150, ""),
        ("90 seconds", 90, ""),
        ("1 hour 20 minutes pasta", 4800, "pasta"),
        ("a quarter of an hour", 900, ""),
        ("a couple of minutes", 120, ""),
    ],
)
def test_parse_duration(text, seconds, rest):
    assert parse_duration(text) == (seconds, rest)


NOW = datetime(2026, 9, 28, 21, 0)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("6:30 tomorrow morning", datetime(2026, 9, 29, 6, 30)),
        ("half past 7", datetime(2026, 9, 29, 7, 30)),
        ("quarter to 8", datetime(2026, 9, 29, 7, 45)),
        ("7 30 pm", datetime(2026, 9, 29, 19, 30)),
        ("noon", datetime(2026, 9, 29, 12, 0)),
        ("9:30 pm", datetime(2026, 9, 28, 21, 30)),
        ("630 tomorrow morning", datetime(2026, 9, 29, 6, 30)),
    ],
)
def test_parse_clock_picks_next_future_time(text, expected):
    assert parse_clock(text, NOW) == expected


def test_speak_duration():
    assert speak_duration(5400) == "1 hour and 30 minutes"
    assert speak_duration(61) == "1 minute and 1 second"


def _handler(tmp_path, clock=None, on_fire=None):
    return TimerHandler(tmp_path / "timers.json", on_fire=on_fire, clock=clock or FakeClock())


def _run(h, text):
    intent = h.parse(normalize(text), normalize(text, numbers=False))
    assert intent is not None, text
    return intent, h.execute(intent)


def test_set_timer_persists_and_reloads(tmp_path):
    clock = FakeClock()
    h = _handler(tmp_path, clock)
    intent, reply = _run(h, "set a pasta timer for twelve minutes")
    assert intent.params == {"seconds": 720, "label": "pasta"}
    assert reply.ok and "12 minutes" in reply.text
    saved = json.loads((tmp_path / "timers.json").read_text())
    assert saved[0]["label"] == "pasta" and saved[0]["due"] == clock.t + 720

    again = _handler(tmp_path, clock)
    assert [t.label for t in again.timers] == ["pasta"]


def test_timer_fires_rings_and_stops(tmp_path):
    clock = FakeClock()
    fired = []
    h = _handler(tmp_path, clock, on_fire=fired.append)
    _run(h, "set a timer for 90 seconds")
    clock.t += 89
    assert h.tick() == []
    clock.t += 2
    assert len(h.tick()) == 1 and len(fired) == 1
    assert h.ringing and not h.timers
    intent, reply = _run(h, "stop")
    assert intent.action == "stop_ringing" and reply.silent
    assert not h.ringing


def test_snooze_after_wake_word_stopped_the_ringing(tmp_path):
    clock = FakeClock()
    h = _handler(tmp_path, clock)
    _run(h, "set an alarm for 6:30")
    clock.t = h.timers[0].due + 1
    h.tick()
    h.stop_ringing()  # the wake word silences the alarm before the command is heard
    intent, reply = _run(h, "snooze for ten minutes")
    assert intent.action == "snooze"
    assert [t.kind for t in h.timers] == ["alarm"]
    assert h.timers[0].due == pytest.approx(clock.t + 600)


def test_stale_timers_are_dropped_not_rung(tmp_path):
    clock = FakeClock()
    fired = []
    h = _handler(tmp_path, clock, on_fire=fired.append)
    _run(h, "set a timer for 5 minutes")
    clock.t += 300 + STALE_S + 1
    assert h.tick() == [] and fired == [] and not h.timers


def test_cancel_named_timer(tmp_path):
    h = _handler(tmp_path)
    _run(h, "set a pasta timer for 12 minutes")
    _run(h, "set an egg timer for 5 minutes")
    intent, reply = _run(h, "cancel the pasta timer")
    assert intent.action == "cancel"
    assert [t.label for t in h.timers] == ["egg"]


@pytest.mark.parametrize(
    "heard, action, params",
    [
        ("How long left are my timer?", "status", None),
        ("Cancel old timers.", "cancel", {"all": True, "label": "", "kind": "timer"}),
        ("Time for an hour and a half.", "set", {"seconds": 5400, "label": ""}),
    ],
)
def test_common_mishearings(tmp_path, heard, action, params):
    h = _handler(tmp_path)
    intent = h.parse(normalize(heard), normalize(heard, numbers=False))
    assert intent is not None and intent.action == action
    if params is not None:
        assert intent.params == params


def test_status_and_reminder(tmp_path):
    h = _handler(tmp_path)
    _, reply = _run(h, "remind me in twenty minutes to check the oven")
    assert "remind you to check the oven" in reply.text
    intent, reply = _run(h, "how long left on my timers")
    assert intent.action == "status"
    assert "check the oven" in reply.text
