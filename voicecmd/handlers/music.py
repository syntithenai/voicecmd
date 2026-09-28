"""Music via abc2book: collection search + snapcast playback sessions; volume via snapserver."""

from __future__ import annotations

import json
import random
import re
import socket
import threading
from pathlib import Path
from typing import Any

from ..http import HttpError, get_json, post_json, request
from ..intents import Intent, Reply

PAUSE_RE = re.compile(r"^(?:pause|hold)(?: (?:the )?(?:music|song|track|it|playback))?$")
RESUME_RE = re.compile(
    r"^(?:resume|continue|unpause|play)(?: (?:the )?(?:music|song|track|it|playing|playback))?$|"
    r"^(?:start|keep) (?:the )?(?:music|playing)$"
)
STOP_RE = re.compile(r"^stop(?: (?:the )?(?:music|song|track|playing|playback))?$|^turn (?:the )?music off$|^turn off (?:the )?music$")
NEXT_RE = re.compile(
    r"^(?:next|skip)(?: (?:the |this )?(?:song|track|one|tune))?$|^play (?:the )?next(?: (?:song|track|one))?$|"
    r"^(?:go to |go )?(?:the )?next (?:song|track|one)$"
)
PREVIOUS_RE = re.compile(r"^(?:previous|go back|back)(?: (?:a |one )?(?:song|track))?$|^(?:play )?(?:the )?(?:previous|last) (?:song|track)$")
VOLUME_SET_RE = re.compile(r"^(?:set |turn )?(?:the )?volume (?:to |at )?(\d{1,3})(?: percent)?$")
VOLUME_UP_RE = re.compile(
    r"^(?:turn (?:it |the volume |the music )?up|turn up (?:the )?(?:volume|music)|volume up|louder|"
    r"(?:a (?:bit|little) )?louder(?: please)?|make it louder|increase (?:the )?volume|raise (?:the )?volume)$"
)
VOLUME_DOWN_RE = re.compile(
    r"^(?:turn (?:it |the volume |the music )?down|turn down (?:the )?(?:volume|music)|volume down|quieter|softer|"
    r"(?:a (?:bit|little) )?quieter|make it quieter|decrease (?:the )?volume|lower (?:the )?volume)$"
)
MUTE_RE = re.compile(r"^mute(?: (?:the )?(?:music|volume|sound))?$")
UNMUTE_RE = re.compile(r"^unmute(?: (?:the )?(?:music|volume|sound))?$")
NOW_PLAYING_RE = re.compile(
    r"^(?:what(?:'s| is) (?:playing|this(?: song| track| tune)?|the (?:song|track|tune))(?: called)?|"
    r"what song is (?:this|playing)|now playing|who (?:is this|sings this|is singing|is playing))$"
)
PLAY_QUERY_RE = re.compile(
    r"^(?:play|put on|queue(?: up)?|listen to|i want to hear)\s+(?:me\s+)?(?:some\s+)?"
    r"(?:music by |songs? by |stuff by |something by |the album |album |the song |song |the track |track )?(.+?)$"
)
GENERIC_QUERIES = {"music", "some music", "something", "anything", "a song", "songs", "tunes", "some tunes"}
MIN_TRACK_S = 45
BROAD_SEARCH_LIMIT = 60
BROAD_QUEUE_SIZE = 25
ADVANCE_POLL_S = 2.0


def pick_tracks(candidates: list[dict], broad: bool, limit: int, rng: random.Random | None = None) -> list[dict]:
    """Drop stubs (intros, skits) and duplicate copies; shuffle genre/artist requests."""
    seen: set[tuple[str, str]] = set()
    picked = []
    for c in candidates:
        if not c.get("link"):
            continue
        duration = float(c.get("duration") or 0)
        if 0 < duration < MIN_TRACK_S:
            continue
        key = (str(c.get("title") or "").strip().lower(), str(c.get("artist") or "").strip().lower())
        if key in seen:
            continue
        seen.add(key)
        picked.append(c)
    if broad:
        (rng or random).shuffle(picked)
    return picked[:limit]


class SnapcastRpc:
    """Minimal JSON-RPC over TCP (snapserver :1705)."""

    def __init__(self, host: str, port: int, timeout: float = 3.0):
        self.host, self.port, self.timeout = host, port, timeout
        self._id = 0

    def call(self, method: str, params: dict | None = None) -> Any:
        self._id += 1
        msg = {"id": self._id, "jsonrpc": "2.0", "method": method}
        if params:
            msg["params"] = params
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
            sock.sendall(json.dumps(msg).encode() + b"\r\n")
            buf = b""
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
                for line in buf.split(b"\n"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if data.get("id") == self._id:
                        if "error" in data:
                            raise RuntimeError(f"snapserver {method}: {data['error']}")
                        return data.get("result")
        raise RuntimeError(f"snapserver {method}: no response")

    def clients(self) -> list[dict]:
        status = self.call("Server.GetStatus") or {}
        out = []
        for group in (status.get("server") or {}).get("groups") or []:
            out.extend(group.get("clients") or [])
        return out

    def set_volume(self, client_id: str, percent: int, muted: bool | None = None) -> None:
        volume: dict[str, Any] = {"percent": max(0, min(100, int(percent)))}
        if muted is not None:
            volume["muted"] = muted
        self.call("Client.SetVolume", {"id": client_id, "volume": volume})


class MusicHandler:
    domain = "music"

    def __init__(
        self,
        abc2book_url: str,
        service_token: str = "",
        snapserver_host: str = "127.0.0.1",
        snapserver_port: int = 1705,
        snapcast_client: str = "",
        queue_size: int = 10,
        duck_ratio: float = 0.3,
        state_path: Path | None = None,
    ):
        self.base = abc2book_url.rstrip("/")
        self.token = service_token
        self.rpc = SnapcastRpc(snapserver_host, snapserver_port)
        self.snapcast_client = snapcast_client
        self.queue_size = queue_size
        self.duck_ratio = duck_ratio
        self.state_path = state_path
        self._session_id: str | None = None
        if state_path and state_path.is_file():
            self._session_id = state_path.read_text(encoding="utf-8").strip() or None
        self._paused_by_us = False
        self._stop = threading.Event()
        self._duck_lock = threading.Lock()
        self._ducked: dict[str, int] = {}

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @session_id.setter
    def session_id(self, value: str | None) -> None:
        """Persisted so a daemon restart keeps auto-advancing the queue it started."""
        if value == self._session_id:
            return
        self._session_id = value
        if self.state_path:
            try:
                if value:
                    self.state_path.write_text(value, encoding="utf-8")
                else:
                    self.state_path.unlink(missing_ok=True)
            except OSError as exc:
                print(f"music: can't save session state: {exc}", flush=True)

    # --- parsing ---

    def parse(self, n: str, nw: str) -> Intent | None:
        if PAUSE_RE.match(n):
            return Intent("music", "pause")
        if STOP_RE.match(n):
            return Intent("music", "stop")
        if NEXT_RE.match(n):
            return Intent("music", "next")
        if PREVIOUS_RE.match(n):
            return Intent("music", "previous")
        m = VOLUME_SET_RE.match(n)
        if m:
            return Intent("music", "set_volume", {"level": int(m.group(1))})
        if VOLUME_UP_RE.match(n):
            return Intent("music", "volume_up")
        if VOLUME_DOWN_RE.match(n):
            return Intent("music", "volume_down")
        if MUTE_RE.match(n):
            return Intent("music", "mute")
        if UNMUTE_RE.match(n):
            return Intent("music", "unmute")
        if NOW_PLAYING_RE.match(n):
            return Intent("music", "now_playing")
        if RESUME_RE.match(n):
            return Intent("music", "resume")
        m = PLAY_QUERY_RE.match(nw)
        if m:
            query = m.group(1).strip()
            query = re.sub(r"\s+(?:music|songs|please)$", "", query).strip()
            if not query or query in GENERIC_QUERIES:
                return Intent("music", "resume")
            params: dict[str, str] = {"query": query}
            by = re.match(r"^(.+?) by (.+)$", query)
            if by:
                params.update(title=by.group(1).strip(), artist=by.group(2).strip())
            return Intent("music", "play_query", params)
        return None

    # --- abc2book / snapcast ---

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def plugin_state(self) -> dict:
        return get_json(f"{self.base}/snapcast-playback/plugin", timeout=3.0) or {}

    def _plugin(self, action: str) -> dict:
        return post_json(f"{self.base}/snapcast-playback/plugin", {"action": action}, timeout=15.0) or {}

    def search(self, query: str, title: str = "", artist: str = "", limit: int | None = None) -> list[dict]:
        payload: dict[str, Any] = {"query": query, "limit": limit or max(self.queue_size * 2, 10)}
        if title:
            payload["title"] = title
        if artist:
            payload["artist"] = artist
        data = post_json(f"{self.base}/search-music-collection", payload, headers=self._headers(), timeout=10.0)
        return list((data or {}).get("candidates") or [])

    def play_candidates(self, candidates: list[dict]) -> dict:
        items = [
            {
                "source": c.get("link"),
                "sourceType": "",
                "title": c.get("title") or "",
                "artist": c.get("artist") or "",
                "duration": float(c.get("duration") or 0),
            }
            for c in candidates
            if c.get("link")
        ]
        if not items:
            raise RuntimeError("no playable candidates")
        first = items[0]
        body = dict(first)
        body["queue"] = items
        data = post_json(f"{self.base}/snapcast-playback/session", body, headers=self._headers(), timeout=60.0) or {}
        self.session_id = data.get("sessionId") or self.session_id
        self._paused_by_us = False
        return data

    # --- queue auto-advance (the resolver leaves advancing to its client) ---

    def start(self) -> None:
        threading.Thread(target=self._advance_loop, name="music-advance", daemon=True).start()

    def _advance_loop(self) -> None:
        while not self._stop.wait(ADVANCE_POLL_S):
            try:
                self.advance_if_finished()
            except Exception as exc:
                print(f"music: auto-advance check failed: {exc}", flush=True)

    def advance_if_finished(self) -> bool:
        sid = self.session_id
        if not sid or self._paused_by_us:
            return False
        try:
            status = get_json(f"{self.base}/snapcast-playback/session/{sid}/status",
                              headers=self._headers(), timeout=5.0) or {}
        except HttpError as exc:
            if exc.status == 404 and self.session_id == sid:
                self.session_id = None
            return False
        duration = float(status.get("duration") or 0)
        position = float(status.get("currentTime") or 0)
        # Paused elsewhere (e.g. the tune book UI) leaves position short of the end; the margin
        # absorbs VBR mp3 durations that overstate the real length by a few seconds.
        margin = max(2.5, duration * 0.05)
        finished = not status.get("isPlaying") and duration > 0 and position >= duration - margin
        if not finished:
            return False
        if not status.get("canGoNext"):
            print("music: queue finished", flush=True)
            self.session_id = None
            return False
        post_json(f"{self.base}/snapcast-playback/session/{sid}/next", {}, headers=self._headers(), timeout=60.0)
        print(f"music: advanced to queue item {int(status.get('queueIndex') or 0) + 2}/{status.get('queueLength')}",
              flush=True)
        return True

    # --- volume / ducking ---

    def _target_clients(self) -> list[dict]:
        clients = [c for c in self.rpc.clients() if c.get("connected")]
        if not self.snapcast_client:
            return clients
        want = self.snapcast_client.lower()
        return [
            c for c in clients
            if want in {str(c.get("id", "")).lower(), str((c.get("host") or {}).get("name", "")).lower(),
                        str((c.get("config") or {}).get("name", "")).lower()}
        ]

    def _change_volume(self, delta: int | None = None, level: int | None = None, muted: bool | None = None) -> int | None:
        last = None
        for c in self._target_clients():
            current = int(((c.get("config") or {}).get("volume") or {}).get("percent", 100))
            new = current if level is None and delta is None else (level if level is not None else current + (delta or 0))
            new = max(0, min(100, new))
            self.rpc.set_volume(c["id"], new, muted)
            last = new
        return last

    def duck(self) -> None:
        """Lower music while listening/speaking; `unduck` restores. Safe to call when idle."""
        with self._duck_lock:
            if self._ducked:
                return
            try:
                if not self.plugin_state().get("isPlaying"):
                    return
                for c in self._target_clients():
                    current = int(((c.get("config") or {}).get("volume") or {}).get("percent", 100))
                    self._ducked[c["id"]] = current
                    self.rpc.set_volume(c["id"], int(current * self.duck_ratio))
            except Exception as exc:
                print(f"music: duck failed: {exc}", flush=True)

    def unduck(self) -> None:
        with self._duck_lock:
            ducked, self._ducked = self._ducked, {}
            for client_id, percent in ducked.items():
                try:
                    self.rpc.set_volume(client_id, percent)
                except Exception as exc:
                    print(f"music: unduck failed: {exc}", flush=True)

    def _ducked_level(self) -> int | None:
        """If a volume command arrives while ducked, apply it to the saved level instead."""
        return next(iter(self._ducked.values()), None) if self._ducked else None

    # --- execute ---

    def execute(self, intent: Intent) -> Reply:
        a, p = intent.action, intent.params
        if a == "play_query":
            query = str(p.get("query") or "")
            broad = not p.get("title") and not p.get("artist") and len(query.split()) <= 2
            try:
                candidates = self.search(query, p.get("title", ""), p.get("artist", ""),
                                         limit=BROAD_SEARCH_LIMIT if broad else None)
            except HttpError as exc:
                if exc.status in (401, 403):
                    return Reply("I'm not allowed into the music collection. Check the abc2book service token.", ok=False)
                raise
            tracks = pick_tracks(candidates, broad, max(self.queue_size, BROAD_QUEUE_SIZE) if broad else self.queue_size)
            if not tracks:
                return Reply(f"I couldn't find {query} in the music collection.", ok=False)
            self.play_candidates(tracks)
            top = tracks[0]
            song = f"{top.get('title') or 'something'}" + (f" by {top['artist']}" if top.get("artist") else "")
            if broad:
                return Reply(f"Playing {query}, starting with {song}.")
            return Reply(f"Playing {song}.")
        if a in ("pause", "resume", "next"):
            state = self.plugin_state()
            if not state.get("canPlay") and not state.get("canPause") and a != "resume":
                return Reply("Nothing is playing.", ok=False)
            if a == "resume" and not (state.get("canPlay") or state.get("canPause")):
                return Reply("What would you like to hear?", expects_reply=True)
            result = self._plugin("play" if a == "resume" else a)
            self._paused_by_us = a == "pause"
            if not result.get("ok", False):
                return Reply("There's no next track." if a == "next" else "Nothing is playing.", ok=False)
            if a == "next":
                status = result.get("status") or {}
                title = status.get("title") or ""
                return Reply(f"Next up, {title}." if title else "Skipping.")
            return Reply("", silent=True)
        if a == "stop":
            if self.session_id:
                try:
                    request("DELETE", f"{self.base}/snapcast-playback/session/{self.session_id}",
                            headers=self._headers(), timeout=10.0)
                    self.session_id = None
                    return Reply("", silent=True)
                except HttpError:
                    self.session_id = None
            result = self._plugin("pause")
            self._paused_by_us = True
            return Reply("", silent=True) if result.get("ok") else Reply("Nothing is playing.", ok=False)
        if a == "previous":
            return Reply("Sorry, I can't go back a track yet.", ok=False)
        if a in ("volume_up", "volume_down", "set_volume"):
            saved = self._ducked_level()
            if saved is not None:
                new = int(p["level"]) if a == "set_volume" else saved + (10 if a == "volume_up" else -10)
                new = max(0, min(100, new))
                self._ducked = {k: new for k in self._ducked}
                level = new
            elif a == "set_volume":
                level = self._change_volume(level=int(p.get("level") or 0))
            else:
                level = self._change_volume(delta=10 if a == "volume_up" else -10)
            if level is None:
                return Reply("I can't find the speaker to change the volume.", ok=False)
            return Reply(f"Volume {level} percent." if a == "set_volume" else "", silent=a != "set_volume")
        if a in ("mute", "unmute"):
            self._change_volume(muted=a == "mute")
            return Reply("", silent=True)
        if a == "now_playing":
            state = self.plugin_state()
            if not state.get("title"):
                return Reply("Nothing is playing right now.")
            who = f" by {state['artist']}" if state.get("artist") else ""
            verb = "This is" if state.get("isPlaying") else "Paused on"
            return Reply(f"{verb} {state['title']}{who}.")
        return Reply("Sorry, I can't do that with the music.", ok=False)
