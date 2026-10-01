import numpy as np
import pytest

from voicecmd import dictation
from voicecmd.typer import Typer, to_ascii
from voicecmd.vad import Endpointer


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Dictate let's refactor the router.", ("once", "let's refactor the router.")),
        ("Hey Jarvis, dictate: Hello from voice!", ("once", "Hello from voice!")),
        ("jarvis dictate  Call Mum at 5pm", ("once", "Call Mum at 5pm")),
        ("Dictate, The API returns JSON.", ("once", "The API returns JSON.")),
        ("Please dictate hi", ("once", "hi")),
        ("Dictate.", None),
        ("Start dictation.", ("start", "")),
        ("Hey Jarvis, start dictating", ("start", "")),
        ("Begin dictation mode", ("start", "")),
        ("Dictation mode.", ("start", "")),
        ("turn off the jug", None),
        ("what's the dictator's name", None),
    ],
)
def test_match_command(raw, expected):
    assert dictation.match_command(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("This is a sentence.", ("text", "This is a sentence.")),
        ("  two   spaces  ", ("text", "two spaces")),
        ("Stop dictation.", ("stop", "")),
        ("Stop dictating!", ("stop", "")),
        ("End dictation", ("stop", "")),
        ("Jarvis, stop dictation.", ("stop", "")),
        ("That's all for now. Stop dictation.", ("stop", "That's all for now.")),
        ("Ship it, stop dictating", ("stop", "Ship it")),
        ("New line.", ("newline", "")),
        ("Newline", ("newline", "")),
        ("New paragraph.", ("paragraph", "")),
        ("Send it.", ("send", "")),
        ("Submit.", ("send", "")),
        ("Scratch that.", ("scratch", "")),
        ("I want a new line of products.", ("text", "I want a new line of products.")),
        ("", ("text", "")),
    ],
)
def test_match_in_mode(raw, expected):
    assert dictation.match_in_mode(raw) == expected


def test_stop_helpers():
    assert dictation.is_stop("Stop dictation.")
    assert not dictation.is_stop("stop")
    assert dictation.is_bare_stop("Stop.")
    assert dictation.is_bare_stop("Hey Jarvis, stop")
    assert not dictation.is_bare_stop("stop the music")
    assert dictation.is_command("start dictation")
    assert dictation.is_command("dictate hi")
    assert not dictation.is_command("play some jazz")


def test_to_ascii():
    assert to_ascii("It’s “fine” — really…") == 'It\'s "fine" - really...'
    assert to_ascii("café") == "cafe"


class FakeRunner:
    def __init__(self, rc=0):
        self.calls = []
        self.rc = rc

    def __call__(self, argv, stdin):
        self.calls.append((argv, stdin))
        return self.rc


def test_typer_type_text_uses_stdin():
    run = FakeRunner()
    t = Typer(key_delay_ms=7, runner=run)
    assert t.available
    assert t.type_text("-n “hi”") == 7
    argv, stdin = run.calls[0]
    assert argv == ["ydotool", "type", "--key-delay", "7", "--file", "-"]
    assert stdin == b'-n "hi"'


def test_typer_skips_empty_and_keys():
    run = FakeRunner()
    t = Typer(runner=run)
    assert t.type_text("   ".strip()) == 0
    t.key("shift+enter")
    t.key("backspace", 5)
    t.key("backspace", 0)
    assert run.calls == [
        (["ydotool", "key", "--key-delay", "4", "shift+enter"], None),
        (["ydotool", "key", "--key-delay", "4", "--repeat", "5", "backspace"], None),
    ]
    with pytest.raises(ValueError):
        t.key("ctrl+alt+delete")


def test_typer_raises_on_failure():
    with pytest.raises(RuntimeError):
        Typer(runner=FakeRunner(rc=1)).type_text("hi")


def _dictating_app(transcripts, run):
    import threading
    from collections import deque
    from types import SimpleNamespace

    from voicecmd.app import VoiceApp

    app = object.__new__(VoiceApp)
    queue_ = list(transcripts)
    app.whisper = SimpleNamespace(transcribe=lambda pcm, prompt=None: SimpleNamespace(
        text=queue_.pop(0), no_speech_prob=0.0, avg_logprob=0.0, elapsed_ms=1, server=""))
    app.ghost = SimpleNamespace(decide=lambda text, **_k: SimpleNamespace(accept=True, reason="accept", text=text))
    app.typer = Typer(runner=run)
    app.s = SimpleNamespace(dictate_prompt="", earcons=False)
    app.dictating = True
    app._dict_chunks = deque(maxlen=200)
    app.history = deque(maxlen=20)
    app._lock = threading.RLock()
    return app


def test_scratch_that_walks_back_through_chunks():
    run = FakeRunner()
    said = ["First sentence.", "Second one.", "New line.", "Third.", "Scratch that.", "Scratch that.",
            "Scratch that.", "Scratch that.", "Scratch that."]
    app = _dictating_app(said, run)
    for _ in said:
        app._process_dictation(np.zeros(160, dtype=np.int16))
    backspaces = [c[0] for c in run.calls if "backspace" in c[0]]
    counts = [int(a[a.index("--repeat") + 1]) if "--repeat" in a else 1 for a in backspaces]
    # "Third. " (7), the newline (1), "Second one. " (12), "First sentence. " (16); the 5th has nothing left.
    assert counts == [7, 1, 12, 16]
    assert not app._dict_chunks


def test_send_clears_scratch_history():
    run = FakeRunner()
    said = ["Hello.", "Send it.", "Scratch that."]
    app = _dictating_app(said, run)
    for _ in said:
        app._process_dictation(np.zeros(160, dtype=np.int16))
    assert not any("backspace" in c[0] for c in run.calls)


def test_endpointer_reset_overrides():
    ep = Endpointer(min_silence_ms=400, max_utterance_ms=8000, listen_timeout_ms=5000)
    ep.reset(listen_timeout_ms=100, min_silence_ms=900, max_utterance_ms=30000)
    assert (ep._timeout_ms, ep._min_silence, ep._max_utterance) == (100, 900, 30000)
    silence = np.zeros(1600, dtype=np.int16)
    assert ep.feed(silence).status == "timeout"
    ep.reset()
    assert (ep._timeout_ms, ep._min_silence, ep._max_utterance) == (5000, 400, 8000)
