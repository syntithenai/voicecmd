import pytest

from voicecmd.normalize import normalize, words_to_numbers


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Hey Jarvis, set a timer for twenty five minutes please.", "set a timer for 25 minutes"),
        ("Could you turn it up a bit?", "turn it up a bit"),
        ("Set an alarm for 7.30 a.m.", "set an alarm for 7:30 am"),
        ("Set the volume to 40%.", "set the volume to 40 percent"),
        ("  ", ""),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_normalize_keeps_words_when_asked():
    assert normalize("play twenty one pilots", numbers=False) == "play twenty one pilots"


@pytest.mark.parametrize(
    "raw, expected",
    [("one hundred and twenty", "120"), ("twenty five", "25"), ("five", "5"), ("no numbers here", "no numbers here")],
)
def test_words_to_numbers(raw, expected):
    assert words_to_numbers(raw) == expected
