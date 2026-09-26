from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol


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


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path, work_dir: Path) -> TranscriptResult: ...


@dataclass(frozen=True)
class DiarizationTurn:
    start_ms: int
    end_ms: int
    speaker: str
    confidence: float | None = None


@dataclass(frozen=True)
class DiarizationResult:
    provider: str
    model: str
    raw_json: dict[str, Any]
    turns: list[DiarizationTurn]


class Diarizer(Protocol):
    provider: str
    model: str

    def diarize(
        self, audio_path: Path, work_dir: Path
    ) -> DiarizationResult: ...
