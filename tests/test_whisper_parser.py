import json
import subprocess
from pathlib import Path

import pytest

from plaud_brain.adapters.whisper_cpp import WhisperCppTranscriber, parse_whisper_segments
from plaud_brain.config import _env_bool


def test_parse_segments():
    raw = {
        "result": {"language": "fr"},
        "transcription": [
            {"offsets": {"from": 0, "to": 1200}, "text": " Bonjour."},
            {"offsets": {"from": 1200, "to": 3100}, "text": " On commence."},
        ],
    }
    segs = parse_whisper_segments(raw)
    assert len(segs) == 2
    assert segs[0].start_ms == 0
    assert segs[1].end_ms == 3100
    assert segs[0].text == "Bonjour."


def _valid_result() -> dict:
    return {
        "result": {"language": "fr"},
        "transcription": [
            {"offsets": {"from": 0, "to": 1200}, "text": " Bonjour."},
        ],
    }


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_env_bool_true(monkeypatch, value):
    monkeypatch.setenv("WHISPER_NO_GPU", value)
    assert _env_bool("WHISPER_NO_GPU") is True


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off"])
def test_env_bool_false(monkeypatch, value):
    monkeypatch.setenv("WHISPER_NO_GPU", value)
    assert _env_bool("WHISPER_NO_GPU", default=True) is False


def test_env_bool_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("WHISPER_NO_GPU", "sometimes")
    with pytest.raises(ValueError, match="WHISPER_NO_GPU"):
        _env_bool("WHISPER_NO_GPU")


def test_transcribe_reuses_valid_json_without_subprocess(tmp_path, monkeypatch):
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    audio = tmp_path / "recording.source"
    audio.write_bytes(b"audio")
    work = tmp_path / "work"
    work.mkdir()
    output_json = work / "recording.whisper.json"
    output_json.write_text(json.dumps(_valid_result()), encoding="utf-8")

    def unexpected_run(*args, **kwargs):
        pytest.fail("valid cached JSON must not invoke a subprocess")

    monkeypatch.setattr(subprocess, "run", unexpected_run)
    transcriber = WhisperCppTranscriber("whisper-cli", model, no_gpu=True)

    result = transcriber.transcribe(audio, work)

    assert result.language == "fr"
    assert [segment.text for segment in result.segments] == ["Bonjour."]


@pytest.mark.parametrize("cached", ["", "not-json", "{}"])
def test_transcribe_replaces_invalid_cache(tmp_path, monkeypatch, cached):
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    audio = tmp_path / "recording.source"
    audio.write_bytes(b"audio")
    work = tmp_path / "work"
    work.mkdir()
    (work / "recording.16k.wav").write_bytes(b"wav")
    output_json = work / "recording.whisper.json"
    output_json.write_text(cached, encoding="utf-8")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        output_base = Path(cmd[cmd.index("-of") + 1])
        Path(f"{output_base}.json").write_text(
            json.dumps(_valid_result()),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    transcriber = WhisperCppTranscriber("whisper-cli", model, no_gpu=True)

    result = transcriber.transcribe(audio, work)

    assert len(calls) == 1
    assert "-ng" in calls[0]
    assert calls[0][calls[0].index("-t") + 1] == "8"
    assert result.segments[0].text == "Bonjour."


def test_transcribe_uses_configured_thread_count(tmp_path, monkeypatch):
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    audio = tmp_path / "recording.source"
    audio.write_bytes(b"audio")
    work = tmp_path / "work"
    work.mkdir()
    (work / "recording.16k.wav").write_bytes(b"wav")
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        output_base = Path(cmd[cmd.index("-of") + 1])
        Path(f"{output_base}.json").write_text(
            json.dumps(_valid_result()),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    transcriber = WhisperCppTranscriber(
        "whisper-cli",
        model,
        threads=6,
        no_gpu=True,
    )

    transcriber.transcribe(audio, work)

    assert commands[0][commands[0].index("-t") + 1] == "6"


def test_gpu_mode_does_not_add_no_gpu_flag(tmp_path, monkeypatch):
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    audio = tmp_path / "recording.source"
    audio.write_bytes(b"audio")
    work = tmp_path / "work"
    work.mkdir()
    (work / "recording.16k.wav").write_bytes(b"wav")
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        output_base = Path(cmd[cmd.index("-of") + 1])
        Path(f"{output_base}.json").write_text(
            json.dumps(_valid_result()),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    transcriber = WhisperCppTranscriber(
        "whisper-cli",
        model,
        no_gpu=False,
        flash_attn=False,
    )

    transcriber.transcribe(audio, work)

    assert len(commands) == 1
    assert "-ng" not in commands[0]
    assert "-nfa" in commands[0]


def test_flash_attention_mode_does_not_add_disable_flag(tmp_path, monkeypatch):
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    audio = tmp_path / "recording.source"
    audio.write_bytes(b"audio")
    work = tmp_path / "work"
    work.mkdir()
    (work / "recording.16k.wav").write_bytes(b"wav")
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        output_base = Path(cmd[cmd.index("-of") + 1])
        Path(f"{output_base}.json").write_text(
            json.dumps(_valid_result()),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    transcriber = WhisperCppTranscriber(
        "whisper-cli",
        model,
        no_gpu=False,
        flash_attn=True,
    )

    transcriber.transcribe(audio, work)

    assert "-ng" not in commands[0]
    assert "-nfa" not in commands[0]
