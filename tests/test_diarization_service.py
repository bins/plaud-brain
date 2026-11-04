from contextlib import contextmanager
from pathlib import Path

import pytest

from plaud_brain.diarization_service import DiarizationService, align_speakers
from plaud_brain.types import DiarizationResult, DiarizationTurn


def test_aligns_by_greatest_overlap_and_breaks_ties_deterministically():
    segments = [
        {"seq": 0, "start_ms": 0, "end_ms": 1000},
        {"seq": 1, "start_ms": 1000, "end_ms": 2000},
        {"seq": 2, "start_ms": 3000, "end_ms": 4000},
    ]
    turns = [
        DiarizationTurn(0, 700, "SPEAKER_00"),
        DiarizationTurn(700, 1500, "SPEAKER_01"),
        DiarizationTurn(1500, 2000, "SPEAKER_00"),
    ]

    assert align_speakers(segments, turns) == {
        0: "SPEAKER_00",
        1: "SPEAKER_01",
    }


@pytest.mark.parametrize(
    "turn",
    [
        DiarizationTurn(-1, 10, "SPEAKER_00"),
        DiarizationTurn(10, 10, "SPEAKER_00"),
        DiarizationTurn(0, 10, " "),
        DiarizationTurn(0, 10, "SPEAKER_00", 1.1),
    ],
)
def test_rejects_invalid_turns(turn):
    with pytest.raises(ValueError):
        align_speakers([{"seq": 0, "start_ms": 0, "end_ms": 10}], [turn])


class FakeStore:
    def __init__(self):
        self.saved = None
        self.runs = []
        self.cached = {}
        self.events = []
        self.pending = []

    @contextmanager
    def diarization_lock(self, plaud_id):
        self.events.append(("lock", plaud_id))
        try:
            yield
        finally:
            self.events.append(("unlock", plaud_id))

    def pending_diarizations(self, max_attempts, limit):
        self.events.append(("pending", max_attempts, limit))
        return [{"plaud_id": plaud_id} for plaud_id in self.pending[:limit]]

    def get_diarization_input(self, plaud_id):
        return {
            "recording_id": "recording-id",
            "transcript_id": f"transcript-{plaud_id}",
            "audio_path": "/tmp/audio.wav",
            "audio_sha256": "audio-hash",
            "segments": [{"seq": 0, "start_ms": 0, "end_ms": 1000}],
        }

    def get_diarization(self, plaud_id):
        return self.cached.get(f"transcript-{plaud_id}")

    def start_run(self, recording_id, stage):
        self.runs.append((recording_id, stage, "started"))
        return "run-id"

    def finish_run(self, run_id, success, code=None, message=None):
        self.runs.append((run_id, success, code, message))

    def save_diarization(self, transcript_id, audio_sha256, result, assignments):
        self.saved = (transcript_id, audio_sha256, result, assignments)
        self.cached[transcript_id] = {
            "provider": result.provider,
            "model": result.model,
            "audio_sha256": audio_sha256,
            "turns": result.turns,
        }


class FakeDiarizer:
    provider = "fake"
    model = "speaker-v1"

    def __init__(self):
        self.calls = 0

    def diarize(self, audio_path, work_dir):
        self.calls += 1
        self.work_dir = work_dir
        (work_dir / "normalized.16k.wav").write_bytes(b"private audio")
        return DiarizationResult(
            provider=self.provider,
            model=self.model,
            raw_json={"safe": True},
            turns=[DiarizationTurn(0, 1000, "SPEAKER_00", 0.9)],
        )


def test_service_saves_then_reuses_matching_audio_cache(tmp_path):
    store = FakeStore()
    diarizer = FakeDiarizer()
    service = DiarizationService(store, diarizer, tmp_path)

    result, cached = service.diarize_one("recording")
    assert cached is False
    assert result["turns"][0].speaker == "SPEAKER_00"
    assert store.saved[3] == {0: "SPEAKER_00"}
    assert diarizer.calls == 1

    _, cached = service.diarize_one("recording")
    assert cached is True
    assert diarizer.calls == 1


class FailingDiarizer(FakeDiarizer):
    def __init__(self, failing_ids):
        super().__init__()
        self.failing_ids = failing_ids

    def diarize(self, audio_path, work_dir):
        result = super().diarize(audio_path, work_dir)
        if work_dir.parent.name in self.failing_ids:
            raise RuntimeError("model crashed")
        return result


def test_run_holds_recording_lock(tmp_path):
    store = FakeStore()
    DiarizationService(store, FakeDiarizer(), tmp_path).diarize_one("recording")
    assert store.events == [("lock", "recording"), ("unlock", "recording")]


def test_work_directory_is_removed_after_success_and_failure(tmp_path):
    store = FakeStore()
    diarizer = FailingDiarizer({"broken"})
    service = DiarizationService(store, diarizer, tmp_path)

    service.diarize_one("recording")
    assert not diarizer.work_dir.exists()

    with pytest.raises(RuntimeError):
        service.diarize_one("broken")
    assert not diarizer.work_dir.exists()
    assert store.runs[-1][1:3] == (False, "RuntimeError")


@pytest.mark.parametrize("plaud_id", ["../escape", "a/b", "", "id with space"])
def test_rejects_unsafe_recording_identifier(tmp_path, plaud_id):
    store = FakeStore()
    diarizer = FakeDiarizer()
    with pytest.raises(ValueError):
        DiarizationService(store, diarizer, tmp_path).diarize_one(plaud_id)
    assert diarizer.calls == 0
    assert store.runs == []


def test_pending_batch_continues_after_a_failure(tmp_path):
    store = FakeStore()
    store.pending = ["first", "broken", "third"]
    diarizer = FailingDiarizer({"broken"})
    service = DiarizationService(store, diarizer, tmp_path)

    counts = service.diarize_pending(limit=5, max_attempts=3)

    assert counts == {"diarized": 2, "diarization_cached": 0, "diarization_failed": 1}
    assert store.events[0] == ("pending", 3, 5)
    assert diarizer.calls == 3


def test_pending_batch_respects_limit(tmp_path):
    store = FakeStore()
    store.pending = ["first", "second", "third"]
    diarizer = FakeDiarizer()
    counts = DiarizationService(store, diarizer, tmp_path).diarize_pending(limit=1, max_attempts=3)
    assert counts["diarized"] == 1
    assert diarizer.calls == 1
