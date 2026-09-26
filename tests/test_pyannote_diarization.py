from pathlib import Path
import wave
from subprocess import CompletedProcess

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
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * 16_000)
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
    assert len(pipeline.calls) == 1
    pipeline_input, options = pipeline.calls[0]
    assert pipeline_input["sample_rate"] == 16_000
    assert tuple(pipeline_input["waveform"].shape) == (1, 16_000)
    assert options == {"min_speakers": 2, "max_speakers": 3}
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


def test_pyannote_adapter_reports_model_access_failure_without_token():
    def failing_loader(model, token):
        raise RuntimeError(f"access rejected for {token}")

    diarizer = PyannoteDiarizer(pipeline_loader=failing_loader)
    with pytest.raises(PyannoteDiarizationError, match="token is valid") as exc_info:
        diarizer._load_pipeline("secret-token")

    assert "secret-token" not in str(exc_info.value)


def test_pyannote_adapter_normalizes_compressed_audio(monkeypatch, tmp_path):
    audio = tmp_path / "audio.source"
    audio.write_bytes(b"compressed")
    pipeline = FakePipeline()

    def fake_run(command, **kwargs):
        assert command[0] == "/opt/ffmpeg"
        target = Path(command[-1])
        with wave.open(str(target), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16_000)
            wav.writeframes(b"\x00\x00" * 16_000)
        return CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setenv("TEST_HF_TOKEN", "secret-token")
    diarizer = PyannoteDiarizer(
        token_env="TEST_HF_TOKEN",
        ffmpeg_bin="/opt/ffmpeg",
        pipeline_loader=lambda model, token: pipeline,
    )

    result = diarizer.diarize(audio, tmp_path / "work")

    assert len(result.turns) == 2
    assert pipeline.calls[0][0]["sample_rate"] == 16_000
