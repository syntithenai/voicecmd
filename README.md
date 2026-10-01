# voicecmd

Lightweight always-on voice commands for this box: wake word → whisper → regex fast paths
(music, timers, weather, clock) → LM Studio tool-calling fallback → spoken reply. No Home
Assistant; a handful of stdlib-first Python modules plus systemd user units, in the same spirit as
`~/projects/weatherstation`.

```
USB mic ─► PipeWire echo-cancel (voicecmd_ec_source, cancels TTS + snapcast music)
        ─► parecord 16 kHz ─► openWakeWord "hey jarvis" ─► WebRTC VAD endpointing
        ─► whisper-server small.en on GPU :10020  (hedged to CPU base.en :10022 if the GPU is busy)
        ─► ghost gate (hallucination blocklist, repetition loops, self-echo)
        ─► router: regex fast paths ─► handlers ─► TTS gateway :8789 (Kokoro) ─► speakers
                   └─ no match ─► LM Studio :1234 (gpt-oss-20b, tool calls map back onto handlers)
```

## Services

| unit | what |
|---|---|
| `voicecmd.service` | the daemon (`python -m voicecmd --run`), control/status HTTP on 127.0.0.1:10021 |
| `voicecmd-whisper.service` | resident whisper.cpp server, Vulkan, `small.en-q5_1`, 127.0.0.1:10020 |
| `voicecmd-whisper-cpu.service` | CPU-only `base.en-q5_1` fallback on 127.0.0.1:10022 |
| `voicecmd-echo-cancel.service` | standalone PipeWire instance providing `voicecmd_ec_source` |
| `voicecmd-watchdog.timer` | every minute; restarts a component after two consecutive failed health checks |

The whisper server is also a general STT API for other apps on the box:

```bash
curl -F file=@clip.wav -F response_format=json http://127.0.0.1:10020/inference
```

Logs: `journalctl --user -u voicecmd -f` (each command logs transcript, gate decision, intent and timings).

## Setup

```bash
cd ~/projects/voicecmd
scripts/install.sh            # venv + deps, builds whisper.cpp (Vulkan) and fetches models if missing,
                              # renders echo-cancel for the current default mic, installs + starts units
scripts/install.sh <mic-node> # pick a specific mic (pactl list short sources)
```

Build prerequisites for whisper.cpp: `cmake`, a C++ compiler, Vulkan headers/loader and `glslc`
(SPIRV-Headers are fetched into `vendor/` automatically). Config lives in `.env` (see
`.env.example`); restart `voicecmd` after editing.

Music needs the abc2book local-resolver to accept `LOCAL_SERVICE_TOKEN` (loopback-only bypass for
`/search-music-collection`, `/music-collection` and `/snapcast-playback/`). Set the same secret as
`LOCAL_SERVICE_TOKEN` in `abc2book/local-resolver/.env` and `ABC2BOOK_SERVICE_TOKEN` here, then
rebuild the resolver container.

## What it understands

Say **"hey Jarvis"**, wait for the chime (or just keep talking), then:

- **Music**: "play some jazz", "play Kind of Blue by Miles Davis", "pause", "resume", "skip",
  "stop the music", "turn it up/down", "volume 40 percent", "mute", "what's playing".
  Music is ducked while listening and speaking.
- **Tunebook** (your abc2book tags and books): "play the tag charlotte setlist", "play the charlotte
  setlist", "play the eurosession book", "play tunes from the celtic book". Tags play in tunebook
  (alphabetical) order and books are shuffled. Only tunes with a YouTube or audio link play; tunes
  with just notation are skipped. The tunebook web app pushes a compact copy to the home resolver
  (`/snapcast-playback/tunebook`) when it loads or saves. If no tag or book matches, the request
  becomes an ordinary music-collection search.
- **Timers/alarms/reminders** (persisted in `data/timers.json`): "set a pasta timer for 12 minutes",
  "timer for an hour and a half", "set an alarm for 6:30 tomorrow morning", "wake me at half past
  seven", "remind me in 20 minutes to check the oven", "how long left", "cancel the pasta timer",
  "cancel all timers". While ringing: "hey Jarvis" silences it; "stop" / "snooze for 10 minutes".
- **Weather** (from `weatherstation/data`): temperature, humidity, pressure, soil, today's high/low,
  yesterday, summary.
- **Devices** (ESPHome plugs and lights, see below): "turn off the jug", "Taras TV on", "toggle the
  garden pump", "set the office light to 30 percent", "dim the office light", "turn off the
  lights" (lights only), "is the stereo on", "how much power is the washing machine using".
- **Clock**: "what time is it", "what's the date".
- **Dictation** (types into whichever window has focus, e.g. Cursor chat; see below): "dictate let's
  refactor the router", "start dictation" … "stop dictation".
- Anything else goes to the LLM (answers in a sentence or two; can also call the tools above, e.g.
  "put on something relaxing", "a timer for a quarter of an hour").
- "never mind" / "thanks" ends quietly. After a spoken answer the mic stays open for 8 s for a
  follow-up without the wake word (not while music is playing).

"hey Jarvis" during a reply interrupts it (barge-in).

## Devices

Device control goes through the switch panel in `~/projects/yogapp/tools/kogan_switches`
(`kogan-switches.service`, http://127.0.0.1:8790; install with `deploy/install.sh` there). The
panel is the device registry. It finds ESPHome `web_server` devices by mDNS and the neighbour
table, rescans every 5 minutes and right after a new device announces itself, and keeps a live
event stream to each one.

voicecmd polls `GET /api/devices` every `DEVICES_REFRESH_S` (20 s) and switches devices with
`POST /api/command`. Nothing is hard-coded: every switch or light entity becomes controllable by
its friendly name, and the whisper prompt (for spelling), the LLM `device_control` /
`device_status` tools and the "what can you do" answer all follow the live list. A newly flashed
plug is usable by voice within about half a minute of joining Wi-Fi. If the panel is down, the
last known list is kept.

Names match exactly, then by variants ("Tara's TV", "dining room" for "Dining Room Light",
"t v"), then `DEVICE_ALIASES` from `.env` (e.g. `kettle=Jug; pool=Swimming Machine`), then a
unique subset of words ("washing machine"), then fuzzily. If several devices match, it asks
which one. If nothing matches, the utterance falls through to the music handler and the LLM, so
"turn off the music" is unaffected. Offline devices get "The X is offline."

## Dictation

Spoken text is typed into the focused window with `ydotool` (`sudo apt install ydotool`). It drives
`/dev/uinput`, so it works in native Wayland windows on GNOME, where `xdotool` and `wtype` don't; the
service user must be in the `input` group. ydotool types US key codes, so curly quotes, dashes and
accents are folded to ASCII first.

- **One-shot**: "hey Jarvis, dictate let's refactor the router" types `let's refactor the router`
  (Whisper's capitalisation and punctuation, no Enter). Long one-shot sentences are capped by
  `MAX_UTTERANCE_MS`; use continuous mode for anything longer.
- **Continuous**: "hey Jarvis, start dictation" (or "start dictating", "dictation mode") chimes, then
  every pause-separated chunk is typed with a trailing space, no wake word needed. Music is ducked
  and replies stay silent. While dictating, these utterances on their own are keys, not text:

  | say | does |
  |---|---|
  | "new line" | Shift+Enter (plain Enter sends in Cursor chat) |
  | "new paragraph" | Shift+Enter twice |
  | "send it" / "submit" / "press enter" | Enter |
  | "scratch that" | backspaces over the last typed chunk; say it again to keep going back, chunk by chunk (new lines count as chunks). Error chime when there's nothing left; history clears on "send it" |
  | "stop dictation" / "stop dictating" / "end dictation" | ends dictation (also at the end of a chunk: "… that's it, stop dictation") |

  "hey Jarvis" during dictation ends it and listens for a normal command ("hey Jarvis, stop" just
  ends it). Dictation also ends after `DICTATE_IDLE_S` without speech.
- **Keyboard shortcut**: bind a GNOME custom shortcut to
  `curl -s -X POST -d '{"action":"toggle"}' http://127.0.0.1:10021/dictate`.

Dictation chunks are transcribed with `DICTATE_PROMPT` (prose) instead of the command vocabulary,
and end after `DICTATE_SILENCE_MS` (900) of silence or `DICTATE_MAX_UTTERANCE_MS` (30 s). Logs show
`dict: '<transcript>' … action=…` per chunk.

## CLI

```bash
.venv/bin/python -m voicecmd --text "set a timer for 5 minutes"   # routed via the daemon if it's up
.venv/bin/python -m voicecmd --text "what's the weather" --no-speak
.venv/bin/python -m voicecmd --say "hello"
.venv/bin/python -m voicecmd --status                             # state, health, timers, recent commands
.venv/bin/python -m voicecmd --make-bench-cases                   # synthesise bench/cases from bench/phrases.txt
.venv/bin/python -m voicecmd --bench --models tiny.en-q5_1,base.en-q5_1,small.en-q5_1
.venv/bin/python -m pytest
```

Control HTTP (loopback only): `GET /health`, `GET /status`, `POST /say {"text"}`,
`POST /command {"text", "speak"}`, `POST /stop` (silence TTS and ringing alarms),
`POST /dictate {"action": "start"|"stop"|"toggle"}`, `POST /type {"text"}` (type into the focused window).

## Model choice

`--bench` on 280 synthetic cases (35 commands × 4 Kokoro voices × clean/15 dB noise), scoring
whether the transcript routes to the same intent as the reference text:

| model (GPU, idle) | WER | intent match | p50 | RSS |
|---|---|---|---|---|
| tiny.en-q5_1 | 2.3% | 95.4% | ~30–65 ms | ~155 MB |
| base.en-q5_1 | 3.3% | 96.4% | ~48 ms | ~140 MB |
| small.en-q5_1 | 1.3% | 98.6% | ~120–130 ms | ~150 MB |

`small.en` is the default: the extra ~80 ms buys noticeably better command accuracy. When the GPU
is saturated (e.g. a ComfyUI job) GPU latency rose to 3.5–4 s, whereas CPU `base.en` stays at
~0.26 s (CPU `small.en` ~1 s) — hence the hedged CPU fallback. Add real recordings to
`bench/cases/` as `name.wav` + `name.txt` to benchmark your own voice and room.

## Tuning

- False wakes: raise `WAKE_THRESHOLD` (0.42 default) or `WAKE_MIN_RMS`; missed wakes: lower it.
  `--status` shows the current `wake_score`.
- Cut off mid-sentence: raise `VAD_MIN_SILENCE_MS` (400). Slow to respond after you stop: lower it.
- Other bundled wake words: `hey_mycroft`, `alexa`, `hey_marvin` (or a path to a custom `.onnx`).
- Hallucination blocklist: `hallucinations/en.txt` (one phrase per line, matched after canonicalising).
