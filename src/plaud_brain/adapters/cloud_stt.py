from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Callable

from plaud_brain.types import TranscriptResult, TranscriptSegment

log = logging.getLogger(__name__)


class CloudSTTError(RuntimeError):
    pass


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-") or "stt"


def _wav_duration_ms(path: Path) -> int:
    try:
        with wave.open(str(path), "rb") as audio:
            return round(audio.getnframes() * 1000 / audio.getframerate())
    except (OSError, EOFError, wave.Error, ZeroDivisionError) as exc:
        raise CloudSTTError(f"Invalid WAV chunk: {path.name}") from exc


class OpenAICompatibleSTTTranscriber:
    def __init__(
        self,
        provider: str,
        base_url: str,
        model: str,
        api_key_env: str,
        ffmpeg_bin: str = "ffmpeg",
        language: str = "auto",
        chunk_seconds: int = 480,
        timeout_seconds: int = 75,
        max_retries: int = 3,
        opener: Callable = urllib.request.urlopen,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if chunk_seconds < 30:
            raise ValueError("chunk_seconds must be at least 30")
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.ffmpeg_bin = ffmpeg_bin
        self.language = language
        self.chunk_seconds = chunk_seconds
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.opener = opener
        self.sleeper = sleeper

    def transcribe(self, audio_path: Path, work_dir: Path) -> TranscriptResult:
        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            raise CloudSTTError(f"Missing API key environment variable: {self.api_key_env}")

        work_dir.mkdir(parents=True, exist_ok=True)
        cache_tag = f"{_safe_slug(self.provider)}-{_safe_slug(self.model)}"
        merged_path = work_dir / f"{audio_path.stem}.{cache_tag}.json"
        cached = self._load_merged(merged_path, work_dir)
        if cached is not None:
            return cached

        chunks = self._prepare_chunks(audio_path, work_dir)
        all_segments: list[TranscriptSegment] = []
        chunk_records: list[dict] = []
        offset_ms = 0
        detected_language: str | None = None

        for index, chunk in enumerate(chunks):
            chunk_cache = work_dir / f"{audio_path.stem}.{cache_tag}.chunk-{index:04d}.json"
            response = self._load_chunk(chunk_cache)
            if response is None:
                response = self._request_chunk(chunk, api_key)
                parsed = self._parse_chunk(response, _wav_duration_ms(chunk))
                if not parsed:
                    raise CloudSTTError(f"Cloud STT returned no transcript for chunk {index}")
                chunk_cache.write_text(json.dumps(response, ensure_ascii=False), encoding="utf-8")

            chunk_duration_ms = _wav_duration_ms(chunk)
            parsed = self._parse_chunk(response, chunk_duration_ms)
            if not parsed:
                raise CloudSTTError(f"Invalid cached cloud STT response for chunk {index}")
            detected_language = detected_language or response.get("language")
            for segment in parsed:
                all_segments.append(
                    TranscriptSegment(
                        seq=len(all_segments),
                        start_ms=offset_ms + segment.start_ms,
                        end_ms=offset_ms + segment.end_ms,
                        text=segment.text,
                        speaker=segment.speaker,
                    )
                )
            chunk_records.append({"index": index, "duration_ms": chunk_duration_ms, "response": response})
            offset_ms += chunk_duration_ms

        raw = {
            "provider": self.provider,
            "model": self.model,
            "language": detected_language,
            "duration_ms": offset_ms,
            "chunks": chunk_records,
            "transcription": [
                {
                    "offsets": {"from": segment.start_ms, "to": segment.end_ms},
                    "text": segment.text,
                    "speaker": segment.speaker,
                }
                for segment in all_segments
            ],
        }
        merged_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        return TranscriptResult(
            engine=self.provider,
            model=self.model,
            language=detected_language,
            duration_ms=offset_ms,
            raw_json=raw,
            segments=all_segments,
            normalized_audio=chunks[0],
        )

    def _prepare_chunks(self, audio_path: Path, work_dir: Path) -> list[Path]:
        chunks_dir = work_dir / f"{audio_path.stem}.cloud-chunks-{self.chunk_seconds}"
        marker = chunks_dir / ".complete"
        existing = sorted(chunks_dir.glob("chunk-*.wav")) if marker.exists() else []
        if existing and all(path.stat().st_size > 44 for path in existing):
            return existing

        chunks_dir.mkdir(parents=True, exist_ok=True)
        for path in chunks_dir.glob("chunk-*.wav"):
            path.unlink()
        marker.unlink(missing_ok=True)
        pattern = chunks_dir / "chunk-%04d.wav"
        cmd = [
            self.ffmpeg_bin,
            "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(audio_path), "-map", "0:a:0", "-vn",
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            "-f", "segment", "-segment_time", str(self.chunk_seconds),
            "-reset_timestamps", "1", str(pattern),
        ]
        try:
            proc = subprocess.run(cmd, text=True, capture_output=True, timeout=1800, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CloudSTTError("ffmpeg cloud chunking could not start or timed out") from exc
        chunks = sorted(chunks_dir.glob("chunk-*.wav"))
        if proc.returncode != 0 or not chunks or any(path.stat().st_size <= 44 for path in chunks):
            raise CloudSTTError(f"ffmpeg cloud chunking failed: {proc.stderr[-1000:]}")
        marker.write_text("ok\n", encoding="utf-8")
        return chunks

    def _request_chunk(self, chunk: Path, api_key: str) -> dict:
        fields = {"model": self.model, "response_format": "verbose_json"}
        if self.language and self.language != "auto":
            fields["language"] = self.language
        body, content_type = self._multipart(fields, chunk)
        request = urllib.request.Request(
            f"{self.base_url}/audio/transcriptions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": content_type,
                "User-Agent": "plaud-brain/0.1",
            },
            method="POST",
        )
        for attempt in range(self.max_retries):
            try:
                with self.opener(request, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise CloudSTTError("Cloud STT returned a non-object response")
                return payload
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable or attempt + 1 == self.max_retries:
                    raise CloudSTTError(f"{self.provider} STT failed with HTTP {exc.code}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt + 1 == self.max_retries:
                    raise CloudSTTError(f"{self.provider} STT network failure") from exc
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CloudSTTError(f"{self.provider} STT returned invalid JSON") from exc
            self.sleeper(2**attempt)
        raise CloudSTTError(f"{self.provider} STT failed")

    @staticmethod
    def _multipart(fields: dict[str, str], file_path: Path) -> tuple[bytes, str]:
        boundary = f"plaud-brain-{uuid.uuid4().hex}"
        parts: list[bytes] = []
        for name, value in fields.items():
            parts.extend([
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            ])
        parts.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'.encode(),
            b"Content-Type: audio/wav\r\n\r\n",
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ])
        return b"".join(parts), f"multipart/form-data; boundary={boundary}"

    @staticmethod
    def _load_chunk(path: Path) -> dict | None:
        if not path.exists() or path.stat().st_size == 0:
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _parse_chunk(raw: dict, duration_ms: int) -> list[TranscriptSegment]:
        result: list[TranscriptSegment] = []
        items = raw.get("segments") or []
        if not isinstance(items, list):
            raise CloudSTTError("Cloud STT returned invalid segments")
        for item in items:
            if not isinstance(item, dict):
                raise CloudSTTError("Cloud STT returned an invalid segment")
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            try:
                start_ms = round(float(item.get("start", 0)) * 1000)
                end_ms = round(float(item.get("end", duration_ms / 1000)) * 1000)
            except (TypeError, ValueError) as exc:
                raise CloudSTTError("Cloud STT returned invalid segment timestamps") from exc
            start_ms = min(max(start_ms, 0), duration_ms)
            end_ms = min(max(end_ms, start_ms), duration_ms)
            result.append(TranscriptSegment(len(result), start_ms, end_ms, text))
        if not result:
            text = str(raw.get("text") or "").strip()
            if text:
                result.append(TranscriptSegment(0, 0, duration_ms, text))
        return result

    def _load_merged(self, path: Path, work_dir: Path) -> TranscriptResult | None:
        raw = self._load_chunk(path)
        if raw is None:
            return None
        segments: list[TranscriptSegment] = []
        for item in raw.get("transcription") or []:
            text = str(item.get("text") or "").strip()
            offsets = item.get("offsets") or {}
            if text:
                segments.append(
                    TranscriptSegment(
                        seq=len(segments),
                        start_ms=int(offsets.get("from", 0)),
                        end_ms=int(offsets.get("to", 0)),
                        text=text,
                        speaker=item.get("speaker"),
                    )
                )
        if not segments:
            return None
        return TranscriptResult(
            engine=self.provider,
            model=self.model,
            language=raw.get("language"),
            duration_ms=raw.get("duration_ms"),
            raw_json=raw,
            segments=segments,
            normalized_audio=work_dir,
        )


class FallbackTranscriber:
    def __init__(self, primary, fallback) -> None:
        self.primary = primary
        self.fallback = fallback

    def transcribe(self, audio_path: Path, work_dir: Path) -> TranscriptResult:
        try:
            return self.primary.transcribe(audio_path, work_dir)
        except CloudSTTError as exc:
            log.warning("cloud STT unavailable; falling back to local STT: %s", exc)
            return self.fallback.transcribe(audio_path, work_dir)
