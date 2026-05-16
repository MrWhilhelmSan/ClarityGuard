#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LLAMA="${LLAMA:-/home/charlie/Documents/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-$SCRIPT_DIR/ClarityGuard-v2.gguf}"
MMPROJ="${MMPROJ:-$SCRIPT_DIR/mmproj-ClarityGuard-v2.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8081}"
CTX="${CTX:-12288}"
NGL="${NGL:-999}"

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
