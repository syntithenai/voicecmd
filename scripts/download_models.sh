#!/usr/bin/env bash
# Fetch English-only quantised whisper.cpp models used by voicecmd.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/models"
cd "$ROOT/models"

for name in ${WHISPER_MODELS:-small.en-q5_1 base.en-q5_1 tiny.en-q5_1}; do
  file="ggml-$name.bin"
  if [[ -s "$file" ]]; then
    echo "have $file"
    continue
  fi
  echo "downloading $file"
  curl -fSL -o "$file.part" "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$file"
  mv "$file.part" "$file"
done
ls -la
