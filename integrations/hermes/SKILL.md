---
name: voicecmd
description: Transcribe voice/audio via voicecmd's resident whisper.cpp server; check the voice stack.
version: 1.0.0
license: MIT
metadata:
  hermes:
    tags: [voice, stt, whisper, mcp]
    related_skills: []
---

# voicecmd (Whisper STT via MCP)

## When to Use

When an audio file needs transcribing: a Telegram voice note, any local `.ogg/.opus/
.mp3/.m4a/.wav`, or a URL to one. Also when you want to confirm the voice stack is
alive before promising a transcription.

## Tools (MCP server `voicecmd`)

- `mcp_voicecmd_transcribe(audio_path, prompt="", language="auto")` →
  `{transcription, no_speech_prob, duration_s}`. Pass the local path of the audio; the
  tool converts with ffmpeg to 16 kHz mono wav and POSTs to whisper.cpp `/inference`.
  URLs work too.
- `mcp_voicecmd_status()` → reachability of whisper `/health` and daemon `/health`.
  Call this first if a transcription failed oddly.
- `mcp_voicecmd_speak(text)` → v1.1; only present when `VOICECMD_TTS_ENABLED=1`.

## Workflow

1. Receive audio (e.g. Telegram saves to `~/.hermes/cache/audio/*.ogg`); note the path.
2. `transcribe` with that path. Use `prompt` for known spellings (names, jargon) —
   whisper uses it as initial context, exactly like voicecmd's own wake vocabulary.
3. Return the text. Whisper is `small.en`/`base.en`: expect minor mis-transcriptions
   of rare words; never present them as certainty about what was said.
4. If the tool errors, run `status` to distinguish service-down from bad-audio.

## Pitfalls

- The whisper server binds loopback on the voicecmd host; remote use needs a tailnet
  or SSH forward. A refused connection is config, not code.
- Long audio (minutes) can approach the 60 s inference timeout on a loaded GPU.
- `no_speech_prob` near 1.0 with empty text means music/noise, not silence to report.

## Setup

See `README.md` next to this file: `pip install "mcp>=2"` (1.x also works), add the `mcp_servers.voicecmd`
block to `~/.hermes/config.yaml`, restart Hermes.
