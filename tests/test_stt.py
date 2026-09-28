import time

import numpy as np

from voicecmd.stt import Transcript, WhisperClient


def client(delays: dict[str, float], fail: set[str] = frozenset()):
    c = WhisperClient("http://gpu", fallback_url="http://cpu", hedge_after_ms=50)

    def fake_infer(url, wav, prompt):
        time.sleep(delays[url])
        if url in fail:
            raise OSError(f"{url} down")
        return Transcript(text=url, server=url)

    c._infer = fake_infer
    return c


PCM = np.zeros(1600, dtype=np.int16)


def test_fast_primary_is_not_hedged():
    c = client({"http://gpu": 0.01, "http://cpu": 0.01})
    assert c.transcribe(PCM).server == "http://gpu"
    assert c.hedged == 0


def test_slow_primary_hedges_to_fallback():
    c = client({"http://gpu": 1.0, "http://cpu": 0.05})
    started = time.monotonic()
    tr = c.transcribe(PCM)
    assert tr.server == "http://cpu" and c.fallback_wins == 1
    assert time.monotonic() - started < 0.5


def test_failed_primary_uses_fallback():
    c = client({"http://gpu": 0.0, "http://cpu": 0.01}, fail={"http://gpu"})
    assert c.transcribe(PCM).server == "http://cpu"


def test_primary_still_wins_if_it_finishes_first_after_hedging():
    c = client({"http://gpu": 0.1, "http://cpu": 1.0})
    assert c.transcribe(PCM).server == "http://gpu"
    assert c.hedged == 1 and c.fallback_wins == 0
