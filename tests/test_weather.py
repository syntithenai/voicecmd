import json
from datetime import datetime

import pytest

from voicecmd.handlers.weather import WeatherHandler
from voicecmd.intents import Intent

NOW = datetime(2026, 9, 28, 14, 0)


@pytest.fixture
def data_dir(tmp_path):
    readings = [
        {"ts": "2026-09-28T13:50:00", "outdoor_temp_c": 14.0, "outdoor_humidity_pct": 71.4,
         "rel_pressure_hpa": 1012.6, "soil_moisture_pct": {"1": 32.0, "2": 18.4, "3": 0}},
    ]
    (tmp_path / "readings.jsonl").write_text("\n".join(json.dumps(r) for r in readings) + "\n")
    daily = [
        {"date": "2026-09-27", "temp_min": 8.24, "temp_max": 17.0},
        {"date": "2026-09-28", "temp_min": 9.5, "temp_max": 15.1},
    ]
    (tmp_path / "daily.jsonl").write_text("\n".join(json.dumps(r) for r in daily) + "\n")
    return tmp_path


def run(h, topic):
    return h.execute(Intent("weather", topic)).text


def test_topics(data_dir):
    h = WeatherHandler(data_dir, now=lambda: NOW)
    assert run(h, "temperature") == "It's 14 degrees outside."
    assert run(h, "humidity") == "Humidity is 71 percent."
    assert run(h, "pressure") == "Pressure is 1013 hectopascals."
    assert run(h, "soil") == "Soil moisture ranges from 18 to 32 percent across 2 sensors; the driest is sensor 2."
    assert run(h, "yesterday") == "Yesterday, the high was 17 degrees and the low was 8.2 degrees."
    summary = run(h, "summary")
    assert summary.startswith("Outside it's 14 degrees with 71 percent humidity.")
    assert "high so far is 15.1 degrees" in summary


def test_stale_reading_mentions_time(data_dir):
    h = WeatherHandler(data_dir, now=lambda: datetime(2026, 9, 28, 16, 0))
    assert run(h, "temperature") == "It's 14 degrees outside as of 1:50 pm."


def test_missing_data(tmp_path):
    h = WeatherHandler(tmp_path, now=lambda: NOW)
    reply = h.execute(Intent("weather", "temperature"))
    assert not reply.ok


def test_parse_ignores_music(data_dir):
    h = WeatherHandler(data_dir)
    assert h.parse("play stormy weather", "play stormy weather") is None
    assert h.parse("is it cold outside", "is it cold outside").action == "temperature"
