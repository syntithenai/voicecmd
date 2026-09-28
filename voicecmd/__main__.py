"""voicecmd CLI.

  python -m voicecmd --run                 always-on daemon (mic, wake word, control HTTP)
  python -m voicecmd --text "set a timer for 5 minutes"   route a command without the mic
  python -m voicecmd --say "hello"         speak via the TTS gateway
  python -m voicecmd --bench bench/cases   benchmark whisper models on recorded WAVs
  python -m voicecmd --status              query a running daemon
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import load_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voicecmd")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="run the always-on daemon")
    mode.add_argument("--text", metavar="COMMAND", help="route a text command (no mic)")
    mode.add_argument("--say", metavar="TEXT", help="speak text via TTS")
    mode.add_argument("--bench", metavar="DIR", nargs="?", const="bench/cases", help="benchmark whisper models")
    mode.add_argument("--make-bench-cases", metavar="DIR", nargs="?", const="bench/cases",
                      help="synthesise benchmark WAVs from bench/phrases.txt via the TTS gateway")
    mode.add_argument("--status", action="store_true", help="status of a running daemon")
    parser.add_argument("--no-speak", action="store_true", help="with --text: print reply only")
    parser.add_argument("--local", action="store_true",
                        help="with --text/--say: run in this process even if the daemon is up")
    parser.add_argument("--models", default="tiny.en-q5_1,base.en-q5_1,small.en-q5_1",
                        help="with --bench: comma-separated model names in models/")
    args = parser.parse_args(argv)

    settings = load_settings()

    if args.status:
        from .http import get_json

        try:
            print(json.dumps(get_json(f"http://{settings.control_host}:{settings.control_port}/status"), indent=2))
        except OSError as exc:
            print(f"daemon not reachable: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.bench is not None:
        from .bench import run_bench

        return run_bench(settings, args.bench, [m.strip() for m in args.models.split(",") if m.strip()])

    if args.make_bench_cases is not None:
        from .bench import make_cases

        return make_cases(settings, args.make_bench_cases)

    if (args.say or args.text) and not args.local:
        forwarded = _forward_to_daemon(settings, args)
        if forwarded is not None:
            return forwarded

    from .app import VoiceApp

    app = VoiceApp(settings)

    if args.say:
        app.speaker.say(args.say)
        app.speaker.wait_idle()
        return 0

    if args.text:
        result = app.handle_text(args.text, speak=not args.no_speak)
        print(json.dumps(result, indent=2, default=str))
        if not args.no_speak:
            app.speaker.wait_idle()
        return 0

    app.run()
    return 0


def _forward_to_daemon(settings, args) -> int | None:
    """Send --text/--say to a running daemon so timers and playback state live in one place."""
    from .http import get_json, post_json

    base = f"http://{settings.control_host}:{settings.control_port}"
    try:
        get_json(f"{base}/health", timeout=1.0)
    except OSError:
        return None
    if args.say:
        post_json(f"{base}/say", {"text": args.say}, timeout=10)
        return 0
    result = post_json(f"{base}/command", {"text": args.text, "speak": not args.no_speak}, timeout=60)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
