"""The always-on daemon: mic -> wake word -> endpointing -> whisper -> ghost gate -> router -> TTS."""

from __future__ import annotations

import signal
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np

from .audio import FRAME_MS, MicStream, Player, RingBuffer
from .config import ROOT, Settings
from .control import start_control_server
from .ghost import GhostGate
from .handlers.clock import ClockHandler
from .handlers.devices import DeviceHandler
from .handlers.llm import LlmFallback, device_tools
from .handlers.music import MusicHandler
from .handlers.timers import Timer, TimerHandler
from .handlers.weather import WeatherHandler
from .intents import Reply
from .router import Router, SystemHandler
from .stt import WhisperClient
from .tts import Speaker, TtsClient
from .vad import Endpointer

KEEPWARM_S = 240
MIC_STALL_S = 15
FILLER_PHRASE = "One sec."


class VoiceApp:
    def __init__(self, settings: Settings):
        self.s = settings
        self.whisper = WhisperClient(settings.whisper_url, settings.whisper_prompt, settings.stt_pad_ms,
                                     fallback_url=settings.whisper_fallback_url,
                                     hedge_after_ms=settings.whisper_hedge_ms)
        self.timers = TimerHandler(settings.data_dir / "timers.json", on_fire=self._on_timer_fire)
        self.music = MusicHandler(
            settings.abc2book_url, settings.abc2book_service_token, settings.snapserver_host,
            settings.snapserver_port, settings.snapcast_client, settings.music_queue_size, settings.music_duck_ratio,
            state_path=settings.data_dir / "music_session",
        )
        self.devices = (DeviceHandler(settings.devices_url, settings.devices_refresh_s, settings.device_aliases)
                        if settings.devices_url else None)
        self.llm = LlmFallback(settings.llm_url, settings.llm_model, settings.llm_timeout_s,
                               tools_provider=(lambda: device_tools(self.devices.names())) if self.devices else None)
        handlers = [self.timers, self.devices, self.music, WeatherHandler(settings.weather_data_dir), ClockHandler(),
                    SystemHandler(self.devices.names if self.devices else None)]
        self.router = Router([h for h in handlers if h is not None], self.llm)
        if self.devices:
            self.devices.on_names_changed(self.whisper.set_vocabulary)
        self.music.tunebook.on_vocabulary_changed(lambda names: self.whisper.set_vocabulary(names, "Tunebook"))
        self.ghost = GhostGate(ROOT / "hallucinations" / "en.txt", is_supported_command=self.router.is_supported)
        self.player = Player(settings.audio_sink)
        self.tts = TtsClient(settings.tts_url, settings.tts_voice, settings.tts_model, settings.tts_speed,
                             cache_dir=settings.data_dir / "tts_cache")
        self.speaker = Speaker(self.tts, self.player, on_state=self.ghost.note_tts_state, on_text=self.ghost.note_tts)

        self._lock = threading.RLock()
        self.state = "idle"  # idle | listening | processing
        self.listen_via = ""
        self.expecting_reply = False
        self._turn = 0
        self.started_at = time.time()
        self.history: deque[dict] = deque(maxlen=20)
        self.counters = {"wakes": 0, "utterances": 0, "rejected": 0, "commands": 0, "llm": 0}
        self.mic: MicStream | None = None
        self.last_frame_at = 0.0
        self.wake = None
        self.endpointer: Endpointer | None = None
        self._ring_thread: threading.Thread | None = None
        self._stop = threading.Event()

    # --- text path (also used by --text and the control API) ---

    def handle_text(self, text: str, speak: bool = True, source: str = "text") -> dict:
        started = time.monotonic()

        def filler() -> None:
            if speak:
                self.speaker.say(FILLER_PHRASE)

        reply = self.router.route(text, on_slow=filler, slow_after_s=self.s.llm_filler_after_s)
        route_ms = int((time.monotonic() - started) * 1000)
        if speak:
            if reply.text and not reply.silent:
                self.speaker.say(reply.text)
            elif self.s.earcons:
                self.speaker.chime("done" if reply.ok else "error")
        self.counters["commands"] += 1
        if reply.intent is not None and reply.intent.source == "llm":
            self.counters["llm"] += 1
        self.expecting_reply = reply.expects_reply
        entry = {
            "at": datetime.now().isoformat(timespec="seconds"),
            "source": source,
            "text": text,
            "intent": f"{reply.intent.domain}.{reply.intent.action}" if reply.intent else None,
            "via": reply.intent.source if reply.intent else None,
            "params": reply.intent.params if reply.intent else {},
            "reply": reply.text,
            "ok": reply.ok,
            "route_ms": route_ms,
        }
        entry.update({k: v for k, v in reply.data.items() if k.endswith("_ms")})
        self.history.appendleft(entry)
        print(f"cmd[{source}]: {text!r} -> {entry['intent']} ({entry['via']}, {route_ms} ms) {reply.text!r}", flush=True)
        entry["_reply"] = reply
        out = {k: v for k, v in entry.items() if not k.startswith("_")}
        out["expects_reply"] = reply.expects_reply
        return out

    # --- timers ringing ---

    def _fire_message(self, t: Timer) -> str:
        if t.kind == "reminder":
            return f"Reminder: {t.label}."
        if t.kind == "alarm":
            return f"It's {datetime.now().strftime('%-I:%M')}. Your {t.name()} is going off."
        return f"Your {t.name()} is done."

    def _on_timer_fire(self, t: Timer) -> None:
        if self._ring_thread and self._ring_thread.is_alive():
            return
        self._ring_thread = threading.Thread(target=self._ring, args=(t,), name="ring", daemon=True)
        self._ring_thread.start()

    def _ring(self, t: Timer) -> None:
        threading.Thread(target=self.music.duck, daemon=True).start()
        deadline = time.monotonic() + self.s.alarm_ring_s
        cycle = 0
        try:
            while self.timers.ringing and time.monotonic() < deadline and not self._stop.is_set():
                self.speaker.chime("alarm")
                if cycle % 3 == 0:
                    names = {x.id: x for x in self.timers.ringing}
                    for x in names.values():
                        self.speaker.say(self._fire_message(x))
                self.speaker.wait_idle(timeout=15)
                cycle += 1
        finally:
            if self.timers.ringing:
                self.timers.stop_ringing()
            if self.state == "idle":
                self.music.unduck()

    def stop_audio_output(self) -> None:
        self.timers.stop_ringing()
        self.speaker.stop()

    # --- status ---

    def health(self) -> dict:
        """Cheap liveness: the mic loop must be receiving frames (when running as the daemon)."""
        age = time.monotonic() - self.last_frame_at if self.last_frame_at else None
        ok = self.mic is None or (age is not None and age < MIC_STALL_S)
        return {"ok": ok, "state": self.state, "last_frame_age_s": None if age is None else round(age, 2)}

    def status(self) -> dict:
        now = time.time()
        return {
            "state": self.state,
            "listen_via": self.listen_via,
            "uptime_s": int(now - self.started_at),
            "speaking": self.speaker.speaking,
            "wake_model": self.s.wake_model,
            "wake_score": round(getattr(self.wake, "last_score", 0.0), 3),
            "mic_source": self.s.audio_source or "default",
            "mic_restarts": self.mic.restarts if self.mic else None,
            "whisper_ok": self.whisper.healthy(),
            "whisper_fallback_ok": self.whisper.healthy(self.s.whisper_fallback_url) if self.s.whisper_fallback_url else None,
            "whisper_hedged": self.whisper.hedged,
            "whisper_fallback_wins": self.whisper.fallback_wins,
            "tts_ok": self.tts.healthy(),
            "devices": self.devices.status() if self.devices else None,
            "tunebook": self.music.tunebook.status(),
            "timers": [
                {"kind": t.kind, "label": t.label, "due_in_s": int(t.due - now)} for t in self.timers.timers
            ],
            "ringing": [t.name() for t in self.timers.ringing],
            "counters": self.counters,
            "recent": [{k: v for k, v in e.items() if not k.startswith("_")} for e in list(self.history)[:10]],
        }

    # --- audio loop ---

    def _set_state(self, state: str, via: str = "") -> None:
        with self._lock:
            self.state = state
            self.listen_via = via

    def _begin_listening(self, via: str, preroll: np.ndarray | None, timeout_ms: int) -> None:
        with self._lock:
            assert self.endpointer is not None
            self.endpointer.reset(preroll=preroll, listen_timeout_ms=timeout_ms)
            self._set_state("listening", via)

    def _end_interaction(self, turn: int | None = None) -> None:
        if turn is not None and turn != self._turn:
            return
        self._set_state("idle")
        self.expecting_reply = False
        if not self.timers.ringing:
            threading.Thread(target=self.music.unduck, daemon=True).start()

    def _on_wake(self, ring: RingBuffer) -> None:
        self.counters["wakes"] += 1
        self._turn += 1
        print(f"wake: {self.s.wake_model} score={self.wake.last_score:.2f}", flush=True)
        if self.timers.ringing:
            self.stop_audio_output()
        if self.speaker.speaking:
            self.speaker.stop()
        threading.Thread(target=self.music.duck, daemon=True).start()
        if self.s.earcons:
            self.speaker.play_now("wake")
        self._begin_listening("wake", ring.tail(self.s.wake_preroll_ms), self.s.listen_timeout_ms)

    def _process(self, pcm: np.ndarray, via: str, endpoint_at: float, turn: int) -> None:
        try:
            self.counters["utterances"] += 1
            try:
                tr = self.whisper.transcribe(pcm)
            except Exception as exc:
                print(f"stt: failed: {exc}", flush=True)
                if self.s.earcons:
                    self.speaker.chime("error")
                self._end_interaction(turn)
                return
            decision = self.ghost.decide(
                tr.text, no_speech_prob=tr.no_speech_prob, avg_logprob=tr.avg_logprob,
                expecting_reply=self.expecting_reply,
            )
            print(
                f"stt[{via}]: {tr.text!r} {tr.elapsed_ms} ms"
                f"{' (cpu fallback)' if tr.server == self.s.whisper_fallback_url else ''} audio={len(pcm) / 16000:.1f}s "
                f"gate={decision.reason}{'' if decision.accept else ' (rejected)'}",
                flush=True,
            )
            if not decision.accept:
                self.counters["rejected"] += 1
                if via == "wake" and self.s.earcons:
                    self.speaker.chime("error")
                self._end_interaction(turn)
                return

            result = self.handle_text(decision.text, speak=True, source="voice")
            reply: Reply | None = None
            if self.history:
                reply = self.history[0].get("_reply")
            self.speaker.wait_idle(timeout=60)
            first_audio = self.speaker.last_first_audio_at
            if self.history:
                self.history[0]["stt_ms"] = tr.elapsed_ms
                if first_audio > endpoint_at:
                    self.history[0]["endpoint_to_audio_ms"] = int((first_audio - endpoint_at) * 1000)
            if turn != self._turn:
                return
            spoken = bool(result.get("reply")) and not (reply and reply.silent)
            if self._want_followup(reply, spoken):
                self._begin_listening("followup", None, self.s.followup_window_ms)
            else:
                self._end_interaction(turn)
        except Exception as exc:
            print(f"process: unexpected error: {exc}", flush=True)
            self._end_interaction(turn)

    def _want_followup(self, reply: Reply | None, spoken: bool) -> bool:
        if reply is None or not spoken:
            return False
        if reply.expects_reply:
            return True
        if reply.intent is not None and reply.intent.domain in ("system",):
            return False
        # Open-mic follow-ups while music plays would transcribe lyrics.
        try:
            return not self.music.plugin_state().get("isPlaying")
        except Exception:
            return True

    def _keepwarm(self) -> None:
        silence = np.zeros(16000 // 2, dtype=np.int16)
        while not self._stop.wait(KEEPWARM_S):
            if self.state != "idle":
                continue
            try:
                self.whisper.transcribe(silence, prompt="")
            except Exception:
                pass
            try:
                self.tts.synth("Okay.")
            except Exception:
                pass

    def run(self) -> None:
        from .wake import WakeWord

        s = self.s
        self.wake = WakeWord(s.wake_model, s.wake_threshold, s.wake_min_rms, s.wake_cooldown_ms)
        self.endpointer = Endpointer(s.vad_mode, s.vad_min_speech_ms, s.vad_min_silence_ms,
                                     s.listen_timeout_ms, s.max_utterance_ms)
        self.timers.start()
        self.music.start()
        if self.devices:
            self.devices.start()
        control = start_control_server(self, s.control_host, s.control_port)
        threading.Thread(target=self._keepwarm, name="keepwarm", daemon=True).start()
        self.mic = MicStream(s.audio_source)
        self.mic.start()
        ring = RingBuffer(1500)

        def shutdown(*_a) -> None:
            self._stop.set()
            if self.mic:
                self.mic.stop()

        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)
        print(
            f"voicecmd: listening on {s.audio_source or 'default source'} for '{s.wake_model}' "
            f"(threshold {s.wake_threshold}); whisper={s.whisper_url} tts={s.tts_url} llm={s.llm_model}; "
            f"control http://{s.control_host}:{s.control_port}",
            flush=True,
        )

        for frame in self.mic.frames():
            if self._stop.is_set():
                break
            self.last_frame_at = time.monotonic()
            ring.push(frame)
            fired = self.wake.process(frame)
            state = self.state
            if fired:
                barge_in = self.speaker.speaking or bool(self.timers.ringing)
                if state == "idle" or self.listen_via == "followup" or barge_in:
                    self._on_wake(ring)
                    continue
            if state != "listening" or self.endpointer is None:
                continue
            if self.speaker.speaking:
                continue
            result = self.endpointer.feed(frame)
            if result.status == "listening":
                continue
            via = self.listen_via
            if result.status == "timeout" or result.audio is None:
                self._end_interaction()
                continue
            self._set_state("processing", via)
            threading.Thread(
                target=self._process, args=(result.audio, via, time.monotonic(), self._turn),
                name="process", daemon=True,
            ).start()

        control.shutdown()
        self.timers.stop()
        print("voicecmd: stopped", flush=True)


__all__ = ["VoiceApp", "FRAME_MS"]
