"""Loopback control/status HTTP server (stdlib) for testing and the watchdog."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .app import VoiceApp


def _make_handler(app: "VoiceApp"):
    class Handler(BaseHTTPRequestHandler):
        server_version = "voicecmd/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # keep journald quiet
            pass

        def _send(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                data = json.loads(self.rfile.read(length))
            except json.JSONDecodeError:
                return {}
            return data if isinstance(data, dict) else {}

        def do_GET(self) -> None:
            if self.path == "/status":
                self._send(200, app.status())
            elif self.path == "/health":
                health = app.health()
                self._send(200 if health["ok"] else 503, health)
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            body = self._body()
            if self.path == "/say":
                text = str(body.get("text") or "").strip()
                if not text:
                    self._send(400, {"error": "missing text"})
                    return
                app.speaker.say(text)
                self._send(200, {"ok": True})
            elif self.path == "/command":
                text = str(body.get("text") or "").strip()
                if not text:
                    self._send(400, {"error": "missing text"})
                    return
                self._send(200, app.handle_text(text, speak=bool(body.get("speak", True)), source="http"))
            elif self.path == "/stop":
                app.stop_audio_output()
                self._send(200, {"ok": True})
            elif self.path == "/dictate":
                action = str(body.get("action") or "toggle").strip().lower()
                if action == "toggle":
                    action = "stop" if app.dictating else "start"
                if action == "start":
                    ok = app.start_dictation()
                    self._send(200 if ok else 409, {"ok": ok, "dictating": app.dictating})
                elif action == "stop":
                    app.stop_dictation("http")
                    self._send(200, {"ok": True, "dictating": app.dictating})
                else:
                    self._send(400, {"error": "action must be start, stop or toggle"})
            elif self.path == "/type":
                text = str(body.get("text") or "")
                if not text.strip():
                    self._send(400, {"error": "missing text"})
                    return
                if not app.typer.available:
                    self._send(503, {"error": "ydotool not installed"})
                    return
                self._send(200, {"ok": True, "chars": app.type_text(text)})
            else:
                self._send(404, {"error": "not found"})

    return Handler


def start_control_server(app: "VoiceApp", host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), _make_handler(app))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="control-http", daemon=True).start()
    return server
