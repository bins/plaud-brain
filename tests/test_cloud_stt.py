import json
import wave
from pathlib import Path

import pytest

from plaud_brain.adapters.cloud_stt import (
    CloudSTTError,
    FallbackTranscriber,
    OpenAICompatibleSTTTranscriber,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def _write_wav(path: Path, seconds: int) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 16000 * seconds)


def test_cloud_stt_merges_chunk_timestamps_and_reuses_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_STT_KEY", "secret-value")
    source = tmp_path / "audio.source"
    source.write_bytes(b"source")
    work = tmp_path / "work"
    chunks = work / "audio.cloud-chunks-480"
    chunks.mkdir(parents=True)
    _write_wav(chunks / "chunk-0000.wav", 2)
    _write_wav(chunks / "chunk-0001.wav", 3)
    (chunks / ".complete").write_text("ok\n")
    responses = iter([
        {"language": "en", "text": "Hello", "segments": [{"start": 0.2, "end": 1.5, "text": " Hello"}]},
        {"language": "en", "text": "World", "segments": [{"start": 0.1, "end": 2.0, "text": " World"}]},
    ])
    requests = []

    def opener(request, timeout):
        requests.append((request, timeout))
        return FakeResponse(next(responses))

    transcriber = OpenAICompatibleSTTTranscriber(
        "openrouter",
        "https://openrouter.ai/api/v1/",
        "openai/whisper-large-v3-turbo",
        "TEST_STT_KEY",
        opener=opener,
        sleeper=lambda _: None,
    )
    result = transcriber.transcribe(source, work)

    assert [(s.start_ms, s.end_ms, s.text) for s in result.segments] == [
        (200, 1500, "Hello"),
        (2100, 4000, "World"),
    ]
    assert result.duration_ms == 5000
    assert result.engine == "openrouter"
    assert all(request.full_url.endswith("/audio/transcriptions") for request, _ in requests)
    assert all(request.headers["Authorization"] == "Bearer secret-value" for request, _ in requests)
    assert all(b"openai/whisper-large-v3-turbo" in request.data for request, _ in requests)

    cached = transcriber.transcribe(source, work)
    assert [s.text for s in cached.segments] == ["Hello", "World"]
    assert len(requests) == 2


def test_cloud_stt_requires_key(tmp_path, monkeypatch):
    monkeypatch.delenv("MISSING_STT_KEY", raising=False)
    transcriber = OpenAICompatibleSTTTranscriber(
        "openrouter", "https://openrouter.ai/api/v1", "model", "MISSING_STT_KEY"
    )
    with pytest.raises(CloudSTTError, match="MISSING_STT_KEY"):
        transcriber.transcribe(tmp_path / "audio", tmp_path / "work")


def test_fallback_transcriber_uses_local_on_cloud_error(tmp_path):
    expected = object()

    class Primary:
        def transcribe(self, audio, work):
            raise CloudSTTError("offline")

    class Local:
        def transcribe(self, audio, work):
            return expected

    result = FallbackTranscriber(Primary(), Local()).transcribe(
        tmp_path / "audio", tmp_path / "work"
    )
    assert result is expected


@pytest.mark.parametrize(
    "payload",
    [
        {"segments": "not-a-list"},
        {"segments": ["not-an-object"]},
        {"segments": [{"text": "hello", "start": "invalid", "end": 1}]},
    ],
)
def test_cloud_stt_rejects_malformed_segments(payload):
    with pytest.raises(CloudSTTError):
        OpenAICompatibleSTTTranscriber._parse_chunk(payload, 1000)
