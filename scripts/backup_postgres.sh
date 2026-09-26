#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_ROOT/data/backups}"
BACKUP_RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"

if [[ ! "$BACKUP_RETENTION_DAYS" =~ ^[0-9]+$ ]] ||
   (( BACKUP_RETENTION_DAYS < 1 || BACKUP_RETENTION_DAYS > 3650 )); then
    echo "BACKUP_RETENTION_DAYS must be between 1 and 3650" >&2
    exit 2
fi

umask 077
mkdir -p "$BACKUP_DIR"
exec 9>"$BACKUP_DIR/.backup.lock"
if ! flock -n 9; then
    echo "another backup is already running" >&2
    exit 3
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
target="$BACKUP_DIR/plaud_brain-$timestamp.dump"
temporary="$(mktemp "$BACKUP_DIR/.plaud_brain-XXXXXXXX.dump")"
cleanup() {
    rm -f -- "$temporary"
}
trap cleanup EXIT

cd "$PROJECT_ROOT"
docker compose exec -T postgres pg_dump -U plaud -Fc plaud_brain >"$temporary"
if [[ ! -s "$temporary" ]]; then
    echo "database backup is empty" >&2
    exit 4
fi
docker compose exec -T postgres pg_restore --list <"$temporary" >/dev/null
chmod 600 "$temporary"
mv -- "$temporary" "$target"
trap - EXIT

find "$BACKUP_DIR" -maxdepth 1 -type f -name 'plaud_brain-*.dump' -mtime "+$BACKUP_RETENTION_DAYS" -delete

echo "backup created: $target"
