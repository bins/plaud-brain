from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from plaud_brain.types import DiarizationResult, DiarizationTurn


class PyannoteDiarizationError(RuntimeError):
    pass


class PyannoteDiarizer:
    provider = "pyannote"

    def __init__(
        self,
        model: str = "pyannote/speaker-diarization-community-1",
        token_env: str = "HF_TOKEN",
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        pipeline_loader: Callable[[str, str], object] | None = None,
    ) -> None:
        if min_speakers is not None and min_speakers < 1:
            raise ValueError("minimum speakers must be positive")
        if max_speakers is not None and max_speakers < 1:
            raise ValueError("maximum speakers must be positive")
        if min_speakers and max_speakers and min_speakers > max_speakers:
            raise ValueError("minimum speakers must not exceed maximum speakers")
        self.model = model
        self.token_env = token_env
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.pipeline_loader = pipeline_loader
        self._pipeline = None

    def diarize(self, audio_path: Path, work_dir: Path) -> DiarizationResult:
        if not audio_path.is_file():
            raise PyannoteDiarizationError(f"audio file not found: {audio_path}")
        token = os.getenv(self.token_env, "").strip()
        if not token:
            raise PyannoteDiarizationError(
                f"{self.token_env} is required after accepting the pyannote model conditions"
            )
        pipeline = self._load_pipeline(token)
        options = {}
        if self.min_speakers is not None:
            options["min_speakers"] = self.min_speakers
        if self.max_speakers is not None:
            options["max_speakers"] = self.max_speakers
        try:
            output = pipeline(str(audio_path), **options)
            annotation = (
                getattr(output, "exclusive_speaker_diarization", None)
                or output.speaker_diarization
            )
            turns = [
                DiarizationTurn(
                    start_ms=round(segment.start * 1000),
                    end_ms=round(segment.end * 1000),
                    speaker=str(speaker),
                )
                for segment, _, speaker in annotation.itertracks(yield_label=True)
            ]
        except Exception as exc:
            raise PyannoteDiarizationError("pyannote diarization failed") from exc
        if not turns:
            raise PyannoteDiarizationError("pyannote returned no speaker turns")
        raw_json = {
            "turns": [
                {
                    "start_ms": turn.start_ms,
                    "end_ms": turn.end_ms,
                    "speaker": turn.speaker,
                }
                for turn in turns
            ]
        }
        return DiarizationResult(
            provider=self.provider,
            model=self.model,
            raw_json=raw_json,
            turns=turns,
        )

    def _load_pipeline(self, token: str):
        if self._pipeline is not None:
            return self._pipeline
        if self.pipeline_loader is not None:
            self._pipeline = self.pipeline_loader(self.model, token)
            return self._pipeline
        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise PyannoteDiarizationError(
                "pyannote.audio is not installed; install the diarization extra"
            ) from exc
        self._pipeline = Pipeline.from_pretrained(self.model, token=token)
        return self._pipeline
