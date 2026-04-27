#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA="${LLAMA:-/home/charlie/Documents/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-$SCRIPT_DIR/Checkpoint-375-Ollama-Clean-7.5B-Q8_0.gguf}"
MMPROJ="${MMPROJ:-$SCRIPT_DIR/mmproj-Checkpoint-375-Ollama-Clean-BF16.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
CTX="${CTX:-16384}"
NGL="${NGL:-99}"

for f in "$LLAMA" "$MODEL" "$MMPROJ"; do
  if [[ ! -f "$f" ]]; then
    echo "No existe: $f" >&2
    exit 1
  fi
done

exec "$LLAMA" \
  -m "$MODEL" \
  --mmproj "$MMPROJ" \
  --host "$HOST" \
  --port "$PORT" \
  -c "$CTX" \
  -ngl "$NGL" \
  --jinja \
  "$@"
