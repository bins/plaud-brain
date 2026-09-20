from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PlaudRecording:
    plaud_id: str
    name: str | None = None
    created_at: datetime | None = None
    duration_seconds: int | None = None


@dataclass(frozen=True)
class TranscriptSegment:
    seq: int
    start_ms: int
    end_ms: int
    text: str
    speaker: str | None = None


@dataclass(frozen=True)
class TranscriptResult:
    engine: str
    model: str
    language: str | None
    duration_ms: int | None
    raw_json: dict[str, Any]
    segments: list[TranscriptSegment]
    normalized_audio: Path
