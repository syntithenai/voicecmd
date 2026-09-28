#!/usr/bin/env bash
# Resident whisper.cpp server for voicecmd (model stays loaded between requests).
#   start_whisper_server.sh          GPU (Vulkan) primary, small.en on :10020
#   start_whisper_server.sh --cpu    CPU-only fallback, base.en on :10022 (used when the GPU is busy)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${WHISPER_SERVER_BIN:-$ROOT/vendor/whisper.cpp/build/bin/whisper-server}"
HOST="${WHISPER_HOST:-127.0.0.1}"
EXTRA=()
if [[ "${1:-}" == "--cpu" ]]; then
  MODEL="${WHISPER_CPU_MODEL_PATH:-$ROOT/models/ggml-base.en-q5_1.bin}"
  PORT="${WHISPER_CPU_PORT:-10022}"
  THREADS="${WHISPER_CPU_THREADS:-8}"
  EXTRA=(-ng)
else
  MODEL="${WHISPER_MODEL_PATH:-$ROOT/models/ggml-small.en-q5_1.bin}"
  PORT="${WHISPER_PORT:-10020}"
  THREADS="${WHISPER_THREADS:-8}"
fi

if [[ ! -x "$BIN" ]]; then
  echo "Missing $BIN — run scripts/build_whisper_server.sh" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "Missing $MODEL — run scripts/download_models.sh" >&2
  exit 1
fi

# Greedy decode, no temperature fallback; no_context is the server default.
exec "$BIN" -m "$MODEL" --host "$HOST" --port "$PORT" -l en -bo 1 -bs 1 -nf -t "$THREADS" "${EXTRA[@]}"
