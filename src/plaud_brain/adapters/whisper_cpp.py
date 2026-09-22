from __future__ import annotations

import json
import subprocess
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
        threads: int = 8,
        no_gpu: bool = False,
        flash_attn: bool = True,
        timeout_seconds: int = 7200,
    ) -> None:
        self.whisper_bin = whisper_bin
        self.model_path = model_path
        self.ffmpeg_bin = ffmpeg_bin
        self.language = language
        if threads < 1:
            raise ValueError("threads must be at least 1")
        self.threads = threads
        self.no_gpu = no_gpu
        self.flash_attn = flash_attn
        self.timeout_seconds = timeout_seconds

    def transcribe(self, audio_path: Path, work_dir: Path) -> TranscriptResult:
        if not self.model_path.exists():
            raise WhisperCppError(f"Whisper model not found: {self.model_path}")

        work_dir.mkdir(parents=True, exist_ok=True)

        wav = work_dir / f"{audio_path.stem}.16k.wav"
        output_base = work_dir / f"{audio_path.stem}.whisper"
        output_json = Path(f"{output_base}.json")

        cached = self._load_result(output_json, wav)
        if cached is not None:
            return cached

        if not wav.exists() or wav.stat().st_size == 0:
            self._normalize(audio_path, wav)

        output_json.unlink(missing_ok=True)

        cmd = [
            self.whisper_bin,
            "-m", str(self.model_path),
            "-f", str(wav),
            "-l", self.language,
            "-t", str(self.threads),
            "-oj",
            "-of", str(output_base),
            "-np",
        ]
        if self.no_gpu:
            cmd.append("-ng")
        elif not self.flash_attn:
            cmd.append("-nfa")

        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
            check=False,
        )

        if (
            proc.returncode != 0
            or not output_json.exists()
            or output_json.stat().st_size == 0
        ):
            raise WhisperCppError(
                f"whisper-cli failed rc={proc.returncode}; "
                f"stderr={proc.stderr[-2000:]}"
            )

        result = self._load_result(output_json, wav)
        if result is None:
            raise WhisperCppError(f"Invalid whisper JSON: {output_json}")

        return result

    def _load_result(
        self,
        output_json: Path,
        wav: Path,
    ) -> TranscriptResult | None:
        if not output_json.exists() or output_json.stat().st_size == 0:
            return None

        try:
            raw = json.loads(output_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        segments = parse_whisper_segments(raw)
        if not segments:
            return None

        language = (raw.get("result") or {}).get("language")
        duration_ms = max((segment.end_ms for segment in segments), default=None)

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
