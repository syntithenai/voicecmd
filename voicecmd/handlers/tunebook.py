"""The user's abc2book tunebook (synced to the resolver by the web app): play a tag or a book."""

from __future__ import annotations

import difflib
import random
import re
import threading
import time
from typing import Any, Callable

from ..http import get_json

REFRESH_S = 60.0
FUZZY_MIN = 0.8
MAX_QUEUE = 200
VOCAB_MIN_TUNES = 3
VOCAB_MAX = 25
# Links the resolver can fetch by itself (recordings and Drive files live only in the browser).
PLAYABLE_RE = re.compile(r"^https?://", re.I)
YOUTUBE_RE = re.compile(r"(?:youtube\.com|youtu\.be)/", re.I)


def collection_key(text: str) -> str:
    t = text.lower().replace("’", "'").replace("'", "")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\bset list\b", "setlist", t)
    return re.sub(r"\s+", " ", t).strip()


def playable_item(tune: dict) -> dict | None:
    for link in tune.get("links") or []:
        url = str(link.get("link") or "")
        if not PLAYABLE_RE.match(url):
            continue
        # http .mid URLs are fetched and rendered by the resolver; the "midifile" type expects inline MIDI.
        source_type = "youtube" if YOUTUBE_RE.search(url) else ""
        item = {
            "source": url,
            "sourceType": source_type,
            "title": tune.get("name") or "",
            "artist": tune.get("composer") or "",
            "duration": 0,
            "tuneId": tune.get("id"),
        }
        try:
            start = float(link.get("startAt") or 0)
        except (TypeError, ValueError):
            start = 0
        if start > 0:
            item["startSeconds"] = start
        return item
    return None


class Tunebook:
    def __init__(self, base_url: str, headers: Callable[[], dict[str, str]] | None = None,
                 fetch: Callable[[bool], dict] | None = None, refresh_s: float = REFRESH_S):
        self.base = base_url.rstrip("/")
        self._headers = headers or (lambda: {})
        self._fetch = fetch or self._http_fetch
        self.refresh_s = refresh_s
        self.tunes: list[dict] = []
        self.books: dict[str, int] = {}
        self.tags: dict[str, str] = {}  # key -> display name
        self.tag_counts: dict[str, int] = {}
        self.updated_at: float | None = None
        self.checked_at = 0.0
        self.available = False
        self._lock = threading.Lock()
        self._listeners: list[Callable[[list[str]], None]] = []

    def _http_fetch(self, summary: bool) -> dict:
        url = f"{self.base}/snapcast-playback/tunebook" + ("?summary=1" if summary else "")
        return get_json(url, headers=self._headers(), timeout=10.0) or {}

    def on_vocabulary_changed(self, callback: Callable[[list[str]], None]) -> None:
        self._listeners.append(callback)

    def vocabulary(self) -> list[str]:
        books = sorted(self.books, key=lambda b: -self.books[b])
        tags = sorted((k for k, c in self.tag_counts.items() if c >= VOCAB_MIN_TUNES), key=lambda k: -self.tag_counts[k])
        return ([b.title() for b in books] + [self.tags[k] for k in tags])[:VOCAB_MAX]

    def refresh(self, force: bool = False) -> bool:
        """Download the tunebook when the resolver's copy changed; cheap summary check otherwise."""
        if not force and time.time() - self.checked_at < self.refresh_s:
            return self.available
        self.checked_at = time.time()
        try:
            summary = self._fetch(True)
            if summary.get("updatedAt") == self.updated_at and self.tunes:
                self.available = True
                return True
            data = self._fetch(False)
        except Exception as exc:
            if self.available:
                print(f"tunebook: unavailable: {exc}", flush=True)
            self.available = bool(self.tunes)
            return self.available
        tunes = list(data.get("tunes") or [])
        books: dict[str, int] = {}
        tags: dict[str, str] = {}
        counts: dict[str, int] = {}
        for tune in tunes:
            for book in tune.get("books") or []:
                books[book.lower()] = books.get(book.lower(), 0) + 1
            for tag in tune.get("tags") or []:
                key = tag.lower()
                tags.setdefault(key, tag)
                counts[key] = counts.get(key, 0) + 1
        old_vocab = self.vocabulary()
        with self._lock:
            self.tunes, self.books, self.tags, self.tag_counts = tunes, books, tags, counts
            self.updated_at = data.get("updatedAt")
        self.available = True
        print(f"tunebook: {len(tunes)} tunes, {len(books)} books, {len(tags)} tags", flush=True)
        vocab = self.vocabulary()
        if vocab != old_vocab:
            for cb in self._listeners:
                try:
                    cb(vocab)
                except Exception as exc:
                    print(f"tunebook: vocabulary listener failed: {exc}", flush=True)
        return True

    def resolve(self, phrase: str, kind: str = "") -> tuple[str, str] | None:
        """('tag'|'book', name) for a spoken collection name, preferring `kind`."""
        key = collection_key(phrase)
        key = re.sub(r"^(?:the|my|our)\s+", "", key)
        if not key:
            return None
        pools: list[tuple[str, dict[str, str]]] = [
            ("book", {collection_key(b): b for b in self.books}),
            ("tag", {collection_key(k): k for k in self.tags}),
        ]
        if kind == "tag":
            pools.reverse()
        squash = key.replace(" ", "")
        for match in (
            lambda cands: [c for c in cands if c == key or c.replace(" ", "") == squash],
            lambda cands: [c for c in cands if set(key.split()) <= set(c.split())],
        ):
            for pool_kind, pool in pools:
                hits = match(pool)
                if len(hits) == 1:
                    return pool_kind, pool[hits[0]]
                if hits and kind in ("", pool_kind):
                    return pool_kind, pool[min(hits, key=len)]
        best: tuple[float, str, str] | None = None
        for pool_kind, pool in pools:
            for cand, name in pool.items():
                score = max(difflib.SequenceMatcher(None, key, cand).ratio(),
                            difflib.SequenceMatcher(None, squash, cand.replace(" ", "")).ratio())
                if best is None or score > best[0]:
                    best = (score, pool_kind, name)
        if best and best[0] >= FUZZY_MIN:
            return best[1], best[2]
        return None

    def queue_for(self, kind: str, name: str, rng: random.Random | None = None) -> tuple[list[dict], int]:
        """Playable queue items for a tag (tunebook order) or book (shuffled), and how many tunes matched."""
        want = name.lower()
        with self._lock:
            if kind == "book":
                tunes = [t for t in self.tunes if want in [b.lower() for b in t.get("books") or []]]
            else:
                tunes = [t for t in self.tunes if want in [x.lower() for x in t.get("tags") or []]]
        items = [i for i in (playable_item(t) for t in tunes) if i]
        if kind == "book":
            (rng or random).shuffle(items)
        return items[:MAX_QUEUE], len(tunes)

    def status(self) -> dict[str, Any]:
        return {"available": self.available, "tunes": len(self.tunes), "books": len(self.books),
                "tags": len(self.tags), "updated_at": self.updated_at}
