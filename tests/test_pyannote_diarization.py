from pathlib import Path

import pytest

from plaud_brain.adapters.pyannote_diarization import (
    PyannoteDiarizationError,
    PyannoteDiarizer,
)


class FakeSegment:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class FakeAnnotation:
    def itertracks(self, yield_label=False):
        assert yield_label is True
        yield FakeSegment(0.0, 1.25), None, "SPEAKER_00"
        yield FakeSegment(1.25, 2.5), None, "SPEAKER_01"


class FakeOutput:
    exclusive_speaker_diarization = FakeAnnotation()
    speaker_diarization = FakeAnnotation()


class FakePipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, path, **options):
        self.calls.append((path, options))
        return FakeOutput()


def test_pyannote_adapter_uses_exclusive_turns(monkeypatch, tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    pipeline = FakePipeline()
    loaded = []

    def loader(model, token):
        loaded.append((model, token))
        return pipeline

    monkeypatch.setenv("TEST_HF_TOKEN", "secret-token")
    diarizer = PyannoteDiarizer(
        token_env="TEST_HF_TOKEN",
        min_speakers=2,
        max_speakers=3,
        pipeline_loader=loader,
    )
    result = diarizer.diarize(audio, tmp_path)

    assert loaded == [
        ("pyannote/speaker-diarization-community-1", "secret-token")
    ]
    assert [(turn.start_ms, turn.end_ms, turn.speaker) for turn in result.turns] == [
        (0, 1250, "SPEAKER_00"),
        (1250, 2500, "SPEAKER_01"),
    ]
    assert pipeline.calls == [(str(audio), {"min_speakers": 2, "max_speakers": 3})]
    assert "secret-token" not in str(result.raw_json)


def test_pyannote_adapter_requires_token(monkeypatch, tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    monkeypatch.delenv("TEST_HF_TOKEN", raising=False)
    diarizer = PyannoteDiarizer(
        token_env="TEST_HF_TOKEN",
        pipeline_loader=lambda model, token: FakePipeline(),
    )
    with pytest.raises(PyannoteDiarizationError, match="TEST_HF_TOKEN"):
        diarizer.diarize(audio, tmp_path)


def test_pyannote_adapter_validates_speaker_bounds():
    with pytest.raises(ValueError, match="must not exceed"):
        PyannoteDiarizer(min_speakers=3, max_speakers=2)
