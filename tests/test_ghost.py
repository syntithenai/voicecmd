from voicecmd.config import ROOT
from voicecmd.ghost import GhostGate, clean_transcript


def gate(supported=()):
    return GhostGate(ROOT / "hallucinations" / "en.txt", is_supported_command=lambda t: t.lower().strip(".") in supported)


def test_empty_and_brackets():
    g = gate()
    assert g.decide("").reason == "empty"
    assert g.decide("[BLANK_AUDIO]").reason == "empty"
    assert g.decide("(music playing) ♪♪").reason == "empty"


def test_blocklist_rejects_unless_reply_expected():
    g = gate()
    assert not g.decide("Thank you.").accept
    assert g.decide("Yes.", expecting_reply=True).accept


def test_short_supported_command_passes_blocklist():
    g = gate(supported={"stop", "resume"})
    assert g.decide("Stop.").accept
    assert g.decide("Resume.").accept
    assert g.decide("Banana.").reason == "short_unsupported"


def test_artifact_phrases():
    assert gate().decide("Thanks for watching!").reason == "artifact_phrase"


def test_no_speech():
    d = gate().decide("play some jazz tonight", no_speech_prob=0.9, avg_logprob=-1.2)
    assert d.reason == "no_speech"


def test_self_echo_only_during_playback_tail():
    g = gate()
    g.note_tts("It's 14.6 degrees outside.")
    g.note_tts_state(True)
    assert g.decide("it's fourteen point six degrees outside").reason == "self_echo"
    g.note_tts_state(False)
    assert g.decide("it's 14.6 degrees outside", now=g._tts_ended_at + 5).accept


def test_loop_collapse():
    text = "set a timer " * 10
    assert clean_transcript(text).count("set a timer") <= 2


def test_normal_command_accepted():
    assert gate().decide("what's the weather like outside").accept
