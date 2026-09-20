#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${WHISPER_CPP_DIR:-$ROOT/vendor/whisper.cpp}"
REF="${WHISPER_CPP_REF:-v1.9.4}"

mkdir -p "$(dirname "$DEST")"
if [[ ! -d "$DEST/.git" ]]; then
  git clone https://github.com/ggml-org/whisper.cpp.git "$DEST"
fi

git -C "$DEST" fetch --tags --force
git -C "$DEST" checkout --detach "$REF"
cmake -S "$DEST" -B "$DEST/build" -DGGML_VULKAN=1 -DCMAKE_BUILD_TYPE=Release
cmake --build "$DEST/build" -j"$(nproc)" --config Release

"$DEST/build/bin/whisper-cli" --help >/dev/null
printf 'Built whisper.cpp %s at %s\n' "$REF" "$DEST/build/bin/whisper-cli"
