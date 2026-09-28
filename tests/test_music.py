import random
import time

from voicecmd.handlers import music
from voicecmd.handlers.music import MusicHandler, pick_tracks


def cand(title, artist="", duration=200.0, link=True):
    return {"title": title, "artist": artist, "duration": duration, "link": f"http://x/{title}" if link else ""}


def test_pick_tracks_drops_stubs_duplicates_and_unplayable():
    cands = [
        cand("Jazz", "dizzy gillespie", 7.9),
        cand("Jazz", "dizzy gillespie", 7.9),
        cand("All Blues", "Miles Davis"),
        cand("all blues", "miles davis"),
        cand("So What", "Miles Davis", link=False),
        cand("Unknown length", duration=0),
    ]
    assert [c["title"] for c in pick_tracks(cands, broad=False, limit=10)] == ["All Blues", "Unknown length"]


def test_pick_tracks_shuffles_broad_requests_only():
    cands = [cand(f"t{i}") for i in range(20)]
    ordered = pick_tracks(cands, broad=False, limit=5)
    assert [c["title"] for c in ordered] == ["t0", "t1", "t2", "t3", "t4"]
    shuffled = pick_tracks(cands, broad=True, limit=20, rng=random.Random(1))
    assert sorted(c["title"] for c in shuffled) == sorted(c["title"] for c in cands)
    assert [c["title"] for c in shuffled] != [c["title"] for c in cands]


def _handler(monkeypatch, status):
    h = MusicHandler("http://resolver")
    h.session_id = "s1"
    posted = []
    monkeypatch.setattr(music, "get_json", lambda url, **kw: status)
    monkeypatch.setattr(music, "post_json", lambda url, payload, **kw: posted.append(url) or {})
    return h, posted


def test_advances_when_track_finished(monkeypatch):
    h, posted = _handler(monkeypatch, {"isPlaying": False, "currentTime": 199.0, "duration": 200.0,
                                       "canGoNext": True, "queueIndex": 0, "queueLength": 5})
    assert h.advance_if_finished()
    time.sleep(0.05)  # prefetch runs on its own thread
    assert posted == ["http://resolver/snapcast-playback/session/s1/next",
                      "http://resolver/snapcast-playback/session/s1/prefetch"]


def test_advances_when_vbr_duration_overstates_length(monkeypatch):
    h, posted = _handler(monkeypatch, {"isPlaying": False, "currentTime": 347.0, "duration": 351.2,
                                       "canGoNext": True, "queueIndex": 0, "queueLength": 25})
    assert h.advance_if_finished() and posted


def test_does_not_advance_when_paused_mid_track(monkeypatch):
    h, posted = _handler(monkeypatch, {"isPlaying": False, "currentTime": 60.0, "duration": 200.0, "canGoNext": True})
    assert not h.advance_if_finished() and not posted


def test_does_not_advance_after_our_own_pause(monkeypatch):
    h, posted = _handler(monkeypatch, {"isPlaying": False, "currentTime": 199.5, "duration": 200.0, "canGoNext": True})
    h._paused_by_us = True
    assert not h.advance_if_finished() and not posted


def test_end_of_queue_clears_session(monkeypatch):
    h, posted = _handler(monkeypatch, {"isPlaying": False, "currentTime": 200.0, "duration": 200.0, "canGoNext": False})
    assert not h.advance_if_finished()
    assert h.session_id is None and not posted
