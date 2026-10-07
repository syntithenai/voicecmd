# Hermes integration for voicecmd

Expose voicecmd's resident whisper.cpp server (and optionally the TTS gateway and
daemon status) to [Hermes Agent](https://github.com/NousResearch/hermes-agent) (Nous Research's open-source agent) as MCP tools,
so an agent can transcribe incoming voice messages (Telegram, etc.) through the same
local Whisper the voice daemon uses.

This wraps HTTP endpoints only. It deliberately does not touch the wake word,
microphone, PipeWire, router or home-device handlers: an agent gets ears (and, if you
enable it, a voice), not the keys to the house.

## Files

### `mcp.py`: the MCP server

A small stdio [MCP](https://modelcontextprotocol.io) server that Hermes starts as a
subprocess. It only talks HTTP to services voicecmd already runs and imports nothing from
the `voicecmd` package. Its only dependency outside the standard library is `mcp`
(2.x preferred; 1.x also works).

- `transcribe(audio_path, prompt="", language="auto")`: takes a local file or http(s)
  URL, converts it with ffmpeg to 16 kHz mono wav if needed, adds the same short silence
  pad as `voicecmd/stt.py`, and `POST`s to whisper.cpp `/inference` (`verbose_json`).
  Returns `{transcription, no_speech_prob, duration_s}`, or `{error}` instead of
  raising. Temporary downloads and conversions are deleted afterwards.
- `status()`: whisper `/health` and daemon `/health`, so the agent can tell "service
  down" from "bad audio".
- `speak(text)`: text → wav via the TTS gateway's `/v1/audio/speech`. Only registered
  when `VOICECMD_TTS_ENABLED=1`.

### `SKILL.md`: the Hermes skill

A Hermes [skill](https://github.com/NousResearch/hermes-agent) is a markdown file with
YAML front matter that Hermes loads into the agent's context. This one tells the agent
when to reach for the tools (a Telegram voice note, any audio file or URL), how to call
them (pass the path straight through, use `prompt` for names and jargon), and what to
watch for (treat whisper output as fallible, run `status` when a call fails, and read a
high `no_speech_prob` as noise rather than speech). Copy it to
`~/.hermes/skills/voicecmd/SKILL.md`.

### `tests/`

A pytest suite that runs the real server over stdio against a mock whisper.cpp server on
loopback. It needs no GPU and no voicecmd install; the ffmpeg test is skipped when
ffmpeg is missing. It isn't in the repo's default `testpaths`, so the main suite never
needs `mcp`.

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

Copy `SKILL.md` to `~/.hermes/skills/voicecmd/SKILL.md` and restart Hermes. Tools register as `mcp_voicecmd_transcribe`, `mcp_voicecmd_status`
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
