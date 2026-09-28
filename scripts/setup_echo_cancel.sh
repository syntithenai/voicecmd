#!/usr/bin/env bash
# Render deploy/echo-cancel.conf.template for a chosen mic and (re)start the
# voicecmd-echo-cancel user service. Usage:
#   scripts/setup_echo_cancel.sh                 # uses the current default source
#   scripts/setup_echo_cancel.sh <source-node>   # e.g. alsa_input.usb-...-input
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MIC="${1:-$(pactl get-default-source)}"
if [[ "$MIC" == voicecmd_ec_source ]]; then
  echo "Default source is already the echo-cancel source; pass the physical mic name." >&2
  exit 1
fi
mkdir -p "$ROOT/data"
sed "s|@MIC@|$MIC|g" "$ROOT/deploy/echo-cancel.conf.template" > "$ROOT/data/echo-cancel.conf"
echo "echo-cancel mic: $MIC -> voicecmd_ec_source"

install -m 644 "$ROOT/deploy/voicecmd-echo-cancel.service" "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable voicecmd-echo-cancel.service >/dev/null
systemctl --user restart voicecmd-echo-cancel.service
sleep 2
pactl list short sources | grep voicecmd_ec_source || {
  echo "voicecmd_ec_source did not appear; see: journalctl --user -u voicecmd-echo-cancel" >&2
  exit 1
}
