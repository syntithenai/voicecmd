import random

import pytest

from voicecmd.handlers.music import MusicHandler, parse_collection
from voicecmd.handlers.tunebook import Tunebook, playable_item
from voicecmd.intents import Intent
from voicecmd.normalize import normalize


def tune(tid, name, books=(), tags=(), link=None, composer=""):
    links = [{"link": link}] if link else []
    return {"id": tid, "name": name, "books": list(books), "tags": list(tags), "links": links, "composer": composer}


TUNES = [
    tune("1", "Blackbird", books=["songs"], tags=["Charlotte Setlist"], link="https://www.youtube.com/watch?v=a"),
    tune("2", "Chilly Winds", books=["songs"], tags=["Charlotte Setlist", "charlotte backups"],
         link="http://127.0.0.1:8787/music-collection/x.mp3"),
    tune("3", "Fly Away", books=["songs"], tags=["charlotte setlist"], link="abcbook-recording:abc"),
    tune("4", "Bourree d'Aurore Sand", books=["eurosession"], link="https://youtu.be/b"),
    tune("5", "Andro", books=["eurosession"], link="https://youtu.be/c"),
    tune("6", "Notation Only", books=["eurosession"]),
    tune("7", "Salty Dog", books=["old time american"], tags=["jam"], link="https://example.org/salty.mid"),
]


def make_tunebook(tunes=TUNES, updated=1.0):
    state = {"tunes": tunes, "updatedAt": updated}

    def fetch(summary):
        return {"updatedAt": state["updatedAt"]} if summary else dict(state)

    tb = Tunebook("http://resolver", fetch=fetch, refresh_s=0)
    tb.refresh(force=True)
    return tb, state


@pytest.mark.parametrize("text, kind, name", [
    ("play the tag charlotte setlist", "tag", "charlotte setlist"),
    ("Play the Eurosession book.", "book", "eurosession"),
    ("play the charlotte setlist", "tag", "charlotte setlist"),
    ("play the charlotte set list", "tag", "charlotte setlist"),
    ("play tunes from the old time american book", "book", "old time american"),
    ("put on book eurosession", "book", "eurosession"),
    ("play songs tagged jam", "tag", "jam"),
])
def test_parse_collection(text, kind, name):
    got = parse_collection(normalize(text, numbers=False))
    assert (got["kind"], got["name"]) == (kind, name)


@pytest.mark.parametrize("text", ["play some jazz", "play kind of blue by miles davis", "play music"])
def test_parse_collection_ignores_normal_requests(text):
    assert parse_collection(normalize(text, numbers=False)) is None


def test_resolve_names():
    tb, _ = make_tunebook()
    assert tb.resolve("charlotte setlist", "tag") == ("tag", "charlotte setlist")
    assert tb.resolve("euro session", "book") == ("book", "eurosession")
    assert tb.resolve("eurosession", "") == ("book", "eurosession")
    assert tb.resolve("old time", "book") == ("book", "old time american")
    assert tb.resolve("charlote setlist", "tag") == ("tag", "charlotte setlist")
    assert tb.resolve("jungle", "book") is None


def test_queue_tag_in_order_skips_unplayable():
    tb, _ = make_tunebook()
    items, total = tb.queue_for("tag", "charlotte setlist")
    assert total == 3
    assert [i["title"] for i in items] == ["Blackbird", "Chilly Winds"]
    assert [i["sourceType"] for i in items] == ["youtube", ""]


def test_queue_book_shuffled_and_notation_skipped():
    tb, _ = make_tunebook()
    items, total = tb.queue_for("book", "eurosession", rng=random.Random(1))
    assert total == 3 and len(items) == 2
    assert {i["title"] for i in items} == {"Bourree d'Aurore Sand", "Andro"}


def test_playable_item_midi_and_start():
    item = playable_item({"id": "x", "name": "X", "links": [{"link": "https://e.org/a.mid", "startAt": "12"}]})
    assert item["sourceType"] == "" and item["startSeconds"] == 12.0
    assert playable_item({"id": "y", "name": "Y", "links": [{"link": "abcbook-recording:1"}]}) is None


def test_refresh_only_downloads_on_change():
    tb, state = make_tunebook()
    calls = []
    orig = tb._fetch
    tb._fetch = lambda summary: calls.append(summary) or orig(summary)
    tb.refresh(force=True)
    assert calls == [True]
    state["tunes"] = TUNES + [tune("8", "New Tune", tags=["new tag"], link="https://youtu.be/d")]
    state["updatedAt"] = 2.0
    tb.refresh(force=True)
    assert calls == [True, True, False]
    assert tb.resolve("new tag", "tag") == ("tag", "new tag")


def test_vocabulary_lists_books_and_bigger_tags():
    seen = []
    tb, state = make_tunebook(tunes=[])
    tb.on_vocabulary_changed(seen.append)
    state["tunes"], state["updatedAt"] = TUNES, 2.0
    tb.refresh(force=True)
    assert "Eurosession" in seen[-1] and "Charlotte Setlist" in seen[-1]
    assert "jam" not in seen[-1]  # fewer than 3 tunes


def test_music_handler_plays_collection_and_falls_back(monkeypatch):
    music = MusicHandler("http://resolver")
    music.tunebook, _ = make_tunebook()
    played, searched = [], []
    monkeypatch.setattr(music, "play_items", lambda items: played.append(items) or {"sessionId": "s"})
    monkeypatch.setattr(music, "search", lambda q, *a, **k: searched.append(q) or [])

    reply = music.execute(music.parse(normalize("play the tag charlotte setlist"),
                                      normalize("play the tag charlotte setlist", numbers=False)))
    assert reply.text == "Playing the Charlotte Setlist tag, 2 tunes, starting with Blackbird. 1 without recordings skipped."
    assert [i["title"] for i in played[-1]] == ["Blackbird", "Chilly Winds"]

    reply = music.execute(Intent("music", "play_collection", {"kind": "book", "name": "jungle", "query": "the jungle book"}))
    assert searched == ["the jungle book"] and not reply.ok
