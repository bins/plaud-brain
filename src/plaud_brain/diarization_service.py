from __future__ import annotations

import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path

from plaud_brain.types import DiarizationResult, DiarizationTurn, Diarizer

log = logging.getLogger(__name__)

# The ID becomes a directory that is deleted after each run.
_SAFE_ID = re.compile(r"[A-Za-z0-9_-]+")


def align_speakers(segments: list[dict], turns: list[DiarizationTurn]) -> dict[int, str]:
    """Assign each transcript segment to the speaker with greatest overlap."""
    normalized = sorted(turns, key=lambda turn: (turn.start_ms, turn.end_ms, turn.speaker))
    for turn in normalized:
        if turn.start_ms < 0 or turn.end_ms <= turn.start_ms:
            raise ValueError("diarization turn timestamps are invalid")
        if not turn.speaker.strip():
            raise ValueError("diarization speaker must not be empty")
        if turn.confidence is not None and not 0.0 <= turn.confidence <= 1.0:
            raise ValueError("diarization confidence must be between 0 and 1")

    assignments: dict[int, str] = {}
    for segment in segments:
        start_ms = int(segment["start_ms"])
        end_ms = int(segment["end_ms"])
        if start_ms < 0 or end_ms <= start_ms:
            raise ValueError("transcript segment timestamps are invalid")
        overlap_by_speaker: dict[str, int] = defaultdict(int)
        first_start: dict[str, int] = {}
        for turn in normalized:
            overlap = min(end_ms, turn.end_ms) - max(start_ms, turn.start_ms)
            if overlap <= 0:
                continue
            speaker = turn.speaker.strip()
            overlap_by_speaker[speaker] += overlap
            first_start[speaker] = min(first_start.get(speaker, turn.start_ms), turn.start_ms)
        if overlap_by_speaker:
            assignments[int(segment["seq"])] = min(
                overlap_by_speaker,
                key=lambda speaker: (
                    -overlap_by_speaker[speaker],
                    first_start[speaker],
                    speaker,
                ),
            )
    return assignments


class DiarizationService:
    def __init__(self, store, diarizer: Diarizer, data_dir: Path) -> None:
        self.store = store
        self.diarizer = diarizer
        self.data_dir = data_dir

    def diarize_pending(self, limit: int, max_attempts: int) -> dict[str, int]:
        """Diarize recordings that have none yet; one failure never stops the batch."""
        counts = {"diarized": 0, "diarization_cached": 0, "diarization_failed": 0}
        for row in self.store.pending_diarizations(max_attempts, limit):
            plaud_id = row["plaud_id"]
            try:
                _, cached = self.diarize_one(plaud_id)
            except Exception:
                log.exception("diarization failed for %s", plaud_id)
                counts["diarization_failed"] += 1
                continue
            counts["diarization_cached" if cached else "diarized"] += 1
        return counts

    def diarize_one(self, plaud_id: str, refresh: bool = False) -> tuple[dict, bool]:
        # The lock makes a manual run and a timer run on the same recording
        # wait for each other; the second then hits the cache instead of
        # running the model again.
        with self.store.diarization_lock(plaud_id):
            return self._diarize_locked(plaud_id, refresh)

    def _diarize_locked(self, plaud_id: str, refresh: bool) -> tuple[dict, bool]:
        if not _SAFE_ID.fullmatch(plaud_id):
            raise ValueError("recording identifier contains unsupported characters")
        source = self.store.get_diarization_input(plaud_id)
        if source is None:
            raise ValueError(f"recording or transcript not found: {plaud_id}")
        if not source.get("audio_path") or not source.get("audio_sha256"):
            raise ValueError(f"recording audio is unavailable: {plaud_id}")

        cached = self.store.get_diarization(plaud_id)
        if (
            not refresh
            and cached is not None
            and cached["provider"] == self.diarizer.provider
            and cached["model"] == self.diarizer.model
            and cached["audio_sha256"] == source["audio_sha256"]
        ):
            return cached, True

        run_id = self.store.start_run(source["recording_id"], "DIARIZATION")
        work_dir = self.data_dir / "work" / plaud_id / "diarization"
        try:
            work_dir.mkdir(parents=True, exist_ok=True)
            result = self.diarizer.diarize(Path(source["audio_path"]), work_dir)
            self._validate_result(result)
            assignments = align_speakers(source["segments"], result.turns)
            self.store.save_diarization(
                source["transcript_id"],
                source["audio_sha256"],
                result,
                assignments,
            )
            self.store.finish_run(run_id, True)
        except Exception as exc:
            self.store.finish_run(
                run_id,
                False,
                type(exc).__name__,
                str(exc),
            )
            raise
        finally:
            # The normalized WAV is a full private copy of the audio; keep none.
            shutil.rmtree(work_dir, ignore_errors=True)
        stored = self.store.get_diarization(plaud_id)
        if stored is None:
            raise RuntimeError("saved diarization could not be loaded")
        return stored, False

    def _validate_result(self, result: DiarizationResult) -> None:
        if result.provider != self.diarizer.provider or result.model != self.diarizer.model:
            raise ValueError("diarization provider or model mismatch")
        if not result.turns:
            raise ValueError("diarization returned no speaker turns")
