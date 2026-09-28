import pytest

from voicecmd.handlers.clock import ClockHandler
from voicecmd.handlers.llm import spoken_text, tool_call_to_intent
from voicecmd.handlers.music import MusicHandler
from voicecmd.handlers.timers import TimerHandler
from voicecmd.handlers.weather import WeatherHandler
from voicecmd.router import Router, SystemHandler


@pytest.fixture
def router(tmp_path):
    return Router([
        TimerHandler(tmp_path / "timers.json"),
        MusicHandler("http://127.0.0.1:0"),
        WeatherHandler(tmp_path),
        ClockHandler(),
        SystemHandler(),
    ])


@pytest.mark.parametrize(
    "text, domain, action",
    [
        ("pause the music", "music", "pause"),
        ("Resume.", "music", "resume"),
        ("skip this song", "music", "next"),
        ("stop the music", "music", "stop"),
        ("turn it up", "music", "volume_up"),
        ("turn the volume down", "music", "volume_down"),
        ("what's playing", "music", "now_playing"),
        ("play some music", "music", "resume"),
        ("set a timer for five minutes", "timer", "set"),
        ("set an alarm for 6:30 tomorrow morning", "timer", "alarm"),
        ("cancel all timers", "timer", "cancel"),
        ("what's the temperature outside", "weather", "temperature"),
        ("what's the weather like", "weather", "summary"),
        ("how humid is it", "weather", "humidity"),
        ("how wet is the soil", "weather", "soil"),
        ("how cold did it get yesterday", "weather", "yesterday"),
        ("what time is it", "clock", "time"),
        ("never mind", "system", "dismiss"),
        ("what can you do", "system", "help"),
    ],
)
def test_fast_path(router, text, domain, action):
    intent = router.parse(text)
    assert intent is not None, text
    assert (intent.domain, intent.action) == (domain, action)


def test_play_query_params(router):
    intent = router.parse("Play Kind of Blue by Miles Davis")
    assert intent.action == "play_query"
    assert intent.params == {"query": "kind of blue by miles davis", "title": "kind of blue", "artist": "miles davis"}


def test_volume_level(router):
    assert router.parse("set the volume to forty percent").params == {"level": 40}


@pytest.mark.parametrize("text", ["who wrote the moonlight sonata", "play the weather channel theme on repeat"])
def test_falls_through(router, text):
    intent = router.parse(text)
    assert intent is None or intent.domain == "music"


def test_route_without_llm(router):
    reply = router.route("explain quantum physics")
    assert not reply.ok


def test_tool_call_mapping():
    intent = tool_call_to_intent("timer_set", {"seconds": 900, "label": "tea"})
    assert (intent.domain, intent.action, intent.params["seconds"]) == ("timer", "set", 900)
    assert tool_call_to_intent("music_control", {"action": "pause"}).action == "pause"


def test_spoken_text_strips_markup():
    assert spoken_text("<think>hmm</think>**Owls** can see [here](http://x.y) well.") == "Owls can see here well."
