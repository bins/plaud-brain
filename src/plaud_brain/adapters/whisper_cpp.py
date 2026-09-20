from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from plaud_brain.types import TranscriptResult, TranscriptSegment


class WhisperCppError(RuntimeError):
    pass


class WhisperCppTranscriber:
    def __init__(
        self,
        whisper_bin: str,
        model_path: Path,
        ffmpeg_bin: str = "ffmpeg",
        language: str = "auto",
        timeout_seconds: int = 7200,
    ) -> None:
        self.whisper_bin = whisper_bin
        self.model_path = model_path
        self.ffmpeg_bin = ffmpeg_bin
        self.language = language
        self.timeout_seconds = timeout_seconds

    def transcribe(self, audio_path: Path, work_dir: Path) -> TranscriptResult:
        if not self.model_path.exists():
            raise WhisperCppError(f"Whisper model not found: {self.model_path}")
        work_dir.mkdir(parents=True, exist_ok=True)
        wav = work_dir / f"{audio_path.stem}.16k.wav"
        self._normalize(audio_path, wav)
        output_base = work_dir / f"{audio_path.stem}.whisper"
        output_json = output_base.with_suffix(".json")
        output_json.unlink(missing_ok=True)

        cmd = [
            self.whisper_bin,
            "-m", str(self.model_path),
            "-f", str(wav),
            "-l", self.language,
            "-oj",
            "-of", str(output_base),
            "-np",
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=self.timeout_seconds, check=False)
        # Some whisper.cpp versions have returned exit code 0 after input decode failure.
        # Require a real, non-empty, parseable JSON artifact as the success contract.
        if proc.returncode != 0 or not output_json.exists() or output_json.stat().st_size == 0:
            raise WhisperCppError(
                f"whisper-cli failed rc={proc.returncode}; stderr={proc.stderr[-2000:]}"
            )
        try:
            raw = json.loads(output_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WhisperCppError(f"Invalid whisper JSON: {output_json}") from exc

        segments = parse_whisper_segments(raw)
        if not segments:
            raise WhisperCppError("Whisper produced no transcript segments")
        language = (raw.get("result") or {}).get("language")
        duration_ms = max((s.end_ms for s in segments), default=None)
        return TranscriptResult(
            engine="whisper.cpp",
            model=self.model_path.name,
            language=language,
            duration_ms=duration_ms,
            raw_json=raw,
            segments=segments,
            normalized_audio=wav,
        )

    def _normalize(self, source: Path, target: Path) -> None:
        cmd = [
            self.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(target),
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=1800, check=False)
        if proc.returncode != 0 or not target.exists() or target.stat().st_size == 0:
            raise WhisperCppError(f"ffmpeg normalization failed: {proc.stderr[-2000:]}")


def _offset_ms(item: dict[str, Any], key: str) -> int:
    offsets = item.get("offsets") or {}
    value = offsets.get(key)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def parse_whisper_segments(raw: dict[str, Any]) -> list[TranscriptSegment]:
    result: list[TranscriptSegment] = []
    for seq, item in enumerate(raw.get("transcription") or []):
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        result.append(TranscriptSegment(
            seq=seq,
            start_ms=_offset_ms(item, "from"),
            end_ms=_offset_ms(item, "to"),
            text=text,
        ))
    return result
