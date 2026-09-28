"""Settings from .env (hand-rolled loader, same approach as weatherstation)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path, *, override: bool = False) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = val


def _str(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name) or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_str(name) or default)
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = _str(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: ROOT / "data")

    # Audio (PipeWire node names; empty = default device)
    audio_source: str = ""
    audio_sink: str = ""
    sample_rate: int = 16000

    # Wake word
    wake_model: str = "hey_jarvis"
    wake_threshold: float = 0.42
    wake_min_rms: float = 0.0015
    wake_cooldown_ms: int = 2500
    wake_preroll_ms: int = 80

    # VAD / endpointing
    vad_mode: int = 2
    vad_min_speech_ms: int = 100
    vad_min_silence_ms: int = 400
    listen_timeout_ms: int = 5000
    max_utterance_ms: int = 8000
    followup_window_ms: int = 8000
    stt_pad_ms: int = 300

    # Services
    whisper_url: str = "http://127.0.0.1:10020"
    whisper_fallback_url: str = "http://127.0.0.1:10022"
    whisper_hedge_ms: int = 500
    whisper_prompt: str = ""
    tts_url: str = "http://127.0.0.1:8789"
    tts_voice: str = "af_heart"
    tts_model: str = "kokoro"
    tts_speed: float = 1.1
    llm_url: str = "http://127.0.0.1:1234/v1/chat/completions"
    llm_model: str = "openai/gpt-oss-20b"
    llm_timeout_s: float = 20.0
    llm_filler_after_s: float = 1.5
    abc2book_url: str = "http://127.0.0.1:8787"
    abc2book_service_token: str = ""
    snapserver_host: str = "127.0.0.1"
    snapserver_port: int = 1705
    snapcast_client: str = ""
    music_duck_ratio: float = 0.3
    music_queue_size: int = 10
    weather_data_dir: Path = field(
        default_factory=lambda: Path.home() / "projects" / "weatherstation" / "data"
    )

    # ESPHome devices (kogan_switches panel)
    devices_url: str = "http://127.0.0.1:8790"
    devices_refresh_s: float = 20.0
    device_aliases: str = ""

    # Timers
    alarm_ring_s: int = 60

    # Control HTTP
    control_host: str = "127.0.0.1"
    control_port: int = 10021

    earcons: bool = True


def load_settings(env_path: Path | None = None) -> Settings:
    _load_dotenv(env_path or ROOT / ".env")
    s = Settings()
    s.data_dir = Path(_str("VOICECMD_DATA_DIR") or s.data_dir)
    s.audio_source = _str("AUDIO_SOURCE", s.audio_source)
    s.audio_sink = _str("AUDIO_SINK", s.audio_sink)
    s.wake_model = _str("WAKE_MODEL", s.wake_model)
    s.wake_threshold = _float("WAKE_THRESHOLD", s.wake_threshold)
    s.wake_min_rms = _float("WAKE_MIN_RMS", s.wake_min_rms)
    s.wake_cooldown_ms = _int("WAKE_COOLDOWN_MS", s.wake_cooldown_ms)
    s.wake_preroll_ms = _int("WAKE_PREROLL_MS", s.wake_preroll_ms)
    s.vad_mode = _int("VAD_MODE", s.vad_mode)
    s.vad_min_speech_ms = _int("VAD_MIN_SPEECH_MS", s.vad_min_speech_ms)
    s.vad_min_silence_ms = _int("VAD_MIN_SILENCE_MS", s.vad_min_silence_ms)
    s.listen_timeout_ms = _int("LISTEN_TIMEOUT_MS", s.listen_timeout_ms)
    s.max_utterance_ms = _int("MAX_UTTERANCE_MS", s.max_utterance_ms)
    s.followup_window_ms = _int("FOLLOWUP_WINDOW_MS", s.followup_window_ms)
    s.stt_pad_ms = _int("STT_PAD_MS", s.stt_pad_ms)
    s.whisper_url = _str("WHISPER_URL", s.whisper_url).rstrip("/")
    s.whisper_fallback_url = _str("WHISPER_FALLBACK_URL", s.whisper_fallback_url).rstrip("/")
    s.whisper_hedge_ms = _int("WHISPER_HEDGE_MS", s.whisper_hedge_ms)
    s.whisper_prompt = _str("WHISPER_PROMPT", s.whisper_prompt)
    s.tts_url = _str("TTS_URL", s.tts_url).rstrip("/")
    s.tts_voice = _str("TTS_VOICE", s.tts_voice)
    s.tts_model = _str("TTS_MODEL", s.tts_model)
    s.tts_speed = _float("TTS_SPEED", s.tts_speed)
    s.llm_url = _str("LLM_URL", s.llm_url)
    s.llm_model = _str("LLM_MODEL", s.llm_model)
    s.llm_timeout_s = _float("LLM_TIMEOUT_S", s.llm_timeout_s)
    s.llm_filler_after_s = _float("LLM_FILLER_AFTER_S", s.llm_filler_after_s)
    s.abc2book_url = _str("ABC2BOOK_URL", s.abc2book_url).rstrip("/")
    s.abc2book_service_token = _str("ABC2BOOK_SERVICE_TOKEN", s.abc2book_service_token)
    s.snapserver_host = _str("SNAPSERVER_HOST", s.snapserver_host)
    s.snapserver_port = _int("SNAPSERVER_PORT", s.snapserver_port)
    s.snapcast_client = _str("SNAPCAST_CLIENT", s.snapcast_client)
    s.music_duck_ratio = _float("MUSIC_DUCK_RATIO", s.music_duck_ratio)
    s.music_queue_size = _int("MUSIC_QUEUE_SIZE", s.music_queue_size)
    s.weather_data_dir = Path(_str("WEATHER_DATA_DIR") or s.weather_data_dir)
    s.devices_url = _str("DEVICES_URL", s.devices_url).rstrip("/")
    s.devices_refresh_s = _float("DEVICES_REFRESH_S", s.devices_refresh_s)
    s.device_aliases = _str("DEVICE_ALIASES", s.device_aliases)
    s.alarm_ring_s = _int("ALARM_RING_S", s.alarm_ring_s)
    s.control_host = _str("CONTROL_HOST", s.control_host)
    s.control_port = _int("CONTROL_PORT", s.control_port)
    s.earcons = _bool("EARCONS", s.earcons)
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s
