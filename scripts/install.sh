#!/usr/bin/env bash
# Install/refresh the voicecmd systemd user units and Python package. Safe to re-run.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_DIR="$HOME/.config/systemd/user"
cd "$ROOT"

[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install -q -e '.[dev]'
[ -f .env ] || { cp .env.example .env; chmod 600 .env; echo "created .env from .env.example - review it"; }

[ -x vendor/whisper.cpp/build/bin/whisper-server ] || scripts/build_whisper_server.sh
ls models/ggml-*.bin >/dev/null 2>&1 || scripts/download_models.sh

mkdir -p "$UNIT_DIR"
for unit in voicecmd-whisper.service voicecmd-whisper-cpu.service voicecmd.service voicecmd-watchdog.service voicecmd-watchdog.timer; do
  install -m 644 "deploy/$unit" "$UNIT_DIR/$unit"
done
systemctl --user daemon-reload

# Renders data/echo-cancel.conf for the current default mic and (re)starts voicecmd-echo-cancel.
scripts/setup_echo_cancel.sh "${1:-}"

systemctl --user enable --now voicecmd-whisper.service voicecmd-whisper-cpu.service
systemctl --user enable voicecmd.service
systemctl --user restart voicecmd.service
systemctl --user enable --now voicecmd-watchdog.timer

sleep 4
systemctl --user --no-pager --lines=0 status voicecmd-echo-cancel voicecmd-whisper voicecmd-whisper-cpu voicecmd | grep -E '●|Active:'
curl -fsS -m 5 "http://127.0.0.1:${CONTROL_PORT:-10021}/health" && echo
