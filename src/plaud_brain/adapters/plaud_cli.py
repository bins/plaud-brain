from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from plaud_brain.types import PlaudRecording

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_URL = re.compile(r"https?://[^\s]+")
_ID = re.compile(r"^[A-Za-z0-9_-]{10,}$")


class PlaudCliError(RuntimeError):
    def __init__(self, message: str, *, returncode: int | None = None):
        super().__init__(message)
        self.returncode = returncode


@dataclass
class PlaudCliAdapter:
    binary: str = "plaud"
    timeout_seconds: int = 60

    def _run(self, *args: str) -> str:
        try:
            proc = subprocess.run(
                [self.binary, *args],
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            raise PlaudCliError(f"PLAUD CLI not found: {self.binary}") from exc
        except subprocess.TimeoutExpired as exc:
            raise PlaudCliError(f"PLAUD CLI timed out: {' '.join(args)}") from exc
        if proc.returncode != 0:
            stderr = proc.stderr.strip() or proc.stdout.strip()
            raise PlaudCliError(stderr or f"PLAUD command failed: {args}", returncode=proc.returncode)
        return proc.stdout

    def list_recent(self, days: int = 3, page_size: int = 100) -> list[PlaudRecording]:
        # `recent` is the official client-side date filter. It may emit a pretty table;
        # keep that parser confined to this adapter.
        out = self._run("recent", "--days", str(days))
        return parse_recording_table(out)

    def audio_url(self, plaud_id: str) -> str:
        out = self._run("audio", plaud_id)
        match = _URL.search(_ANSI.sub("", out))
        if not match:
            raise PlaudCliError(f"No audio URL returned for {plaud_id}")
        return match.group(0).rstrip(".,)")

    def file_metadata(self, plaud_id: str) -> dict[str, str]:
        out = self._run("file", plaud_id)
        return parse_key_values(out)


def _cells(line: str) -> list[str]:
    line = _ANSI.sub("", line).strip()
    if not line:
        return []
    # Unicode/ASCII table boundaries.
    for ch in ("│", "|"):
        if ch in line:
            return [x.strip() for x in line.strip(ch).split(ch) if x.strip()]
    return [x.strip() for x in re.split(r"\s{2,}", line) if x.strip()]


def _parse_datetime(value: str) -> datetime | None:
    value = value.strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_duration(value: str) -> int | None:
    v = value.strip().lower()
    if not v:
        return None
    if re.fullmatch(r"\d+", v):
        return int(v)
    if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", v):
        parts = [int(x) for x in v.split(":")]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    h = re.search(r"(\d+)\s*h", v)
    m = re.search(r"(\d+)\s*m", v)
    s = re.search(r"(\d+)\s*s", v)
    if h or m or s:
        return (int(h.group(1)) if h else 0) * 3600 + (int(m.group(1)) if m else 0) * 60 + (int(s.group(1)) if s else 0)
    return None


def parse_recording_table(text: str) -> list[PlaudRecording]:
    rows = [_cells(line) for line in text.splitlines()]
    rows = [r for r in rows if r]
    if not rows:
        return []

    header = None
    header_idx = -1
    for i, row in enumerate(rows):
        normalized = [c.strip().lower().replace(" ", "_") for c in row]
        if "id" in normalized and ("name" in normalized or "created_at" in normalized):
            header, header_idx = normalized, i
            break

    result: list[PlaudRecording] = []
    if header:
        for row in rows[header_idx + 1:]:
            if len(row) < len(header):
                continue
            data = dict(zip(header, row))
            pid = data.get("id", "")
            if not _ID.match(pid):
                continue
            result.append(PlaudRecording(
                plaud_id=pid,
                name=data.get("name"),
                created_at=_parse_datetime(data.get("created_at", "")),
                duration_seconds=_parse_duration(data.get("duration", "")),
            ))
        return result

    # Conservative fallback: official docs state the ID is the left-most column.
    for row in rows:
        if not row or not _ID.match(row[0]):
            continue
        result.append(PlaudRecording(
            plaud_id=row[0],
            name=row[1] if len(row) > 1 else None,
        ))
    return result


def parse_key_values(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in text.splitlines():
        line = _ANSI.sub("", raw).strip().strip("│|")
        if not line:
            continue
        parts = [p.strip() for p in re.split(r"\s{2,}|:\s+|│|\|", line, maxsplit=1) if p.strip()]
        if len(parts) == 2:
            key = parts[0].lower().replace(" ", "_")
            result[key] = parts[1]
    return result
