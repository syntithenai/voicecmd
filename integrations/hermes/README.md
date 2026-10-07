# Hermes integration for voicecmd

Expose voicecmd's resident whisper.cpp server (and optionally the TTS gateway and
daemon status) to [Hermes Agent](https://hermes-agent.nousresearch.com) as MCP tools,
so an agent can transcribe incoming voice messages (Telegram, etc.) through the same
local Whisper the voice daemon uses.

This wraps HTTP endpoints only. It deliberately does not touch the wake word,
microphone, PipeWire, router or home-device handlers: an agent gets ears (and, if you
enable it, a voice), not the keys to the house.

## Files

- `mcp.py` — stdio MCP server. Tools: `transcribe` (audio path/URL → text, via ffmpeg
  to 16 kHz wav + `POST /inference`), `status` (whisper `/health`, daemon `/health`).
  `speak` (text → wav via `/v1/audio/speech`) is v1.1 and hidden unless
  `VOICECMD_TTS_ENABLED=1`. Stdlib-only otherwise; needs Python ≥ 3.11 and `mcp`.
- `SKILL.md` — the Hermes skill: when and how the agent uses the tools.
- `tests/` — pytest suite against a mock whisper server. No GPU, no voicecmd install
  needed. ffmpeg is used when present.

## Setup

```bash
cd <voicecmd repo>
python -m venv .venv && .venv/bin/pip install "mcp>=2" pytest   # mcp 1.x also works
.venv/bin/python -m pytest integrations/hermes/tests -v
```

Add to `~/.hermes/config.yaml` (Hermes 2.x ships `mcp==2.0.0` in its venv):

```yaml
mcp_servers:
  voicecmd:
    command: "/path/to/voicecmd/.venv/bin/python"   # or hermes venv python
    args: ["<repo>/integrations/hermes/mcp.py"]
    env:
      VOICECMD_WHISPER_URL: "http://127.0.0.1:10020"
      VOICECMD_CONTROL_URL: "http://127.0.0.1:10021"
      # VOICECMD_TTS_ENABLED: "1"                  # also expose v1.1 `speak`
```

Restart Hermes. Tools register as `mcp_voicecmd_transcribe`, `mcp_voicecmd_status`
(and `mcp_voicecmd_speak`). If voicecmd runs on another machine, point the URLs at it,
over a tailnet or `ssh -L 10020:127.0.0.1:10020 user@host`.

## Configuration

| env | default | meaning |
|---|---|---|
| `VOICECMD_WHISPER_URL` | `http://127.0.0.1:10020` | whisper.cpp server base URL |
| `VOICECMD_CONTROL_URL` | `http://127.0.0.1:10021` | voicecmd daemon control API |
| `VOICECMD_TTS_URL` | `http://127.0.0.1:8789` | TTS gateway (v1.1) |
| `VOICECMD_TTS_ENABLED` | unset | set `1` to register the `speak` tool |
| `VOICECMD_TTS_VOICE` / `_MODEL` / `_SPEED` | `af_heart` / `kokoro` / `1.0` | TTS params |
| `VOICECMD_HTTP_TIMEOUT_S` | `60` | inference timeout |
| `VOICECMD_STT_PAD_MS` | `300` | trailing silence pad, as in `voicecmd/stt.py` |
