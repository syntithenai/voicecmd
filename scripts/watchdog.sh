#!/usr/bin/env bash
# Health watchdog (run every minute by voicecmd-watchdog.timer).
# A component is restarted only after two consecutive failed checks, and only if its unit is
# meant to be running, so a deliberate `systemctl --user stop` is respected.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$ROOT/.env" ] && set -a && . "$ROOT/.env" && set +a

WHISPER_PORT="${WHISPER_PORT:-10020}"
CONTROL_PORT="${CONTROL_PORT:-10021}"
EC_SOURCE="voicecmd_ec_source"
STATE_DIR="$ROOT/data/watchdog"
mkdir -p "$STATE_DIR"

check() {  # check <unit> <command...>
  local unit="$1"; shift
  local flag="$STATE_DIR/$unit.fail"
  if ! systemctl --user is-enabled --quiet "$unit" || ! systemctl --user is-active --quiet "$unit"; then
    rm -f "$flag"
    return
  fi
  if "$@"; then
    rm -f "$flag"
  elif [ -f "$flag" ]; then
    echo "watchdog: $unit failed twice, restarting"
    rm -f "$flag"
    systemctl --user restart "$unit"
  else
    echo "watchdog: $unit check failed (will restart if it fails again)"
    touch "$flag"
  fi
}

whisper_ok() { curl -fsS -m 5 "http://127.0.0.1:$WHISPER_PORT/health" 2>/dev/null | grep -q '"ok"'; }
whisper_cpu_ok() { curl -fsS -m 5 "http://127.0.0.1:${WHISPER_CPU_PORT:-10022}/health" 2>/dev/null | grep -q '"ok"'; }
daemon_ok() { curl -fsS -m 5 "http://127.0.0.1:$CONTROL_PORT/health" >/dev/null 2>&1; }
ec_ok() { pactl list short sources 2>/dev/null | grep -q "$EC_SOURCE"; }

check voicecmd-echo-cancel.service ec_ok
check voicecmd-whisper.service whisper_ok
check voicecmd-whisper-cpu.service whisper_cpu_ok
check voicecmd.service daemon_ok
exit 0
