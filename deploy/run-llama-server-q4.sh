#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default: busca llama-server en PATH o en el directorio actual.
# Sobrescribe con: export LLAMA=/ruta/a/tu/llama-server
LLAMA="${LLAMA:-$(command -v llama-server 2>/dev/null || echo './llama-server')}"
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
