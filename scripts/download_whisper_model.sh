#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHISPER_DIR="${WHISPER_CPP_DIR:-$ROOT/vendor/whisper.cpp}"
MODEL_NAME="${WHISPER_MODEL_NAME:-large-v3-turbo}"
MODEL_DIR="${WHISPER_MODEL_DIR:-$ROOT/models}"
mkdir -p "$MODEL_DIR"
"$WHISPER_DIR/models/download-ggml-model.sh" "$MODEL_NAME" "$MODEL_DIR"
printf 'Model: %s/ggml-%s.bin\n' "$MODEL_DIR" "$MODEL_NAME"
