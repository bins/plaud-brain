#!/usr/bin/env bash
set -euo pipefail

fail=0
check() {
  if command -v "$1" >/dev/null 2>&1; then
    printf 'OK   %-12s %s\n' "$1" "$(command -v "$1")"
  else
    printf 'MISS %-12s\n' "$1"
    fail=1
  fi
}

for cmd in python3 node npm docker ffmpeg ffprobe git cmake; do check "$cmd"; done

if command -v vulkaninfo >/dev/null 2>&1; then
  echo "OK   vulkaninfo"
  vulkaninfo --summary 2>/dev/null | sed -n '1,35p' || true
else
  echo "MISS vulkaninfo"
  fail=1
fi

if [[ -e /dev/dri/renderD128 ]]; then
  echo "OK   /dev/dri/renderD128"
else
  echo "WARN /dev/dri/renderD128 not found"
fi

if command -v plaud >/dev/null 2>&1; then
  echo "OK   plaud CLI"
  plaud version || true
else
  echo "MISS plaud CLI (install: npm install -g @plaud-ai/cli)"
  fail=1
fi

exit "$fail"
