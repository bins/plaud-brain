from __future__ import annotations

import os
import subprocess
import tempfile
import wave
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
        ffmpeg_bin: str = "ffmpeg",
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
        self.ffmpeg_bin = ffmpeg_bin
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
        audio_input = self._prepare_audio(audio_path, work_dir)
        try:
            output = pipeline(audio_input, **options)
            annotation = getattr(output, "exclusive_speaker_diarization", None)
            if annotation is None:
                annotation = output.speaker_diarization
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
        if self.pipeline_loader is None:
            try:
                from pyannote.audio import Pipeline
            except ImportError as exc:
                raise PyannoteDiarizationError(
                    "pyannote.audio is not installed; install the diarization extra"
                ) from exc
            def loader(model, access_token):
                return Pipeline.from_pretrained(model, token=access_token)
        else:
            loader = self.pipeline_loader
        try:
            self._pipeline = loader(self.model, token)
        except Exception as exc:
            raise PyannoteDiarizationError(
                "unable to load the pyannote model; verify that the Hugging Face "
                "token is valid, has read access, and the model conditions were accepted"
            ) from exc
        return self._pipeline

    def _prepare_audio(self, audio_path: Path, work_dir: Path) -> dict:
        try:
            return self._load_pcm_wav(audio_path)
        except PyannoteDiarizationError:
            pass

        work_dir.mkdir(parents=True, exist_ok=True)
        normalized = work_dir / f"{audio_path.stem}.16k.wav"
        if (
            not normalized.is_file()
            or normalized.stat().st_size == 0
            or normalized.stat().st_mtime < audio_path.stat().st_mtime
        ):
            with tempfile.NamedTemporaryFile(
                prefix=f".{audio_path.stem}.",
                suffix=".wav",
                dir=work_dir,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            try:
                command = [
                    self.ffmpeg_bin,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(audio_path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(temporary_path),
                ]
                process = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    timeout=1800,
                    check=False,
                )
                if process.returncode != 0 or temporary_path.stat().st_size == 0:
                    raise PyannoteDiarizationError("audio normalization failed")
                temporary_path.replace(normalized)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise PyannoteDiarizationError(
                    "audio normalization could not start or timed out"
                ) from exc
            finally:
                temporary_path.unlink(missing_ok=True)
        return self._load_pcm_wav(normalized)

    @staticmethod
    def _load_pcm_wav(audio_path: Path) -> dict:
        """Preload normalized PCM WAV input so pyannote does not require TorchCodec."""
        try:
            import torch
        except ImportError as exc:
            raise PyannoteDiarizationError(
                "PyTorch is not installed; install the diarization extra"
            ) from exc

        try:
            with wave.open(str(audio_path), "rb") as wav:
                if wav.getcomptype() != "NONE":
                    raise PyannoteDiarizationError("audio must be an uncompressed PCM WAV")
                channels = wav.getnchannels()
                sample_rate = wav.getframerate()
                sample_width = wav.getsampwidth()
                frame_count = wav.getnframes()
                frames = wav.readframes(frame_count)
        except (wave.Error, OSError) as exc:
            raise PyannoteDiarizationError(
                "audio must be a readable PCM WAV"
            ) from exc

        if channels < 1 or sample_rate < 1 or frame_count < 1:
            raise PyannoteDiarizationError("audio WAV has invalid metadata")
        dtype_and_scale = {
            1: (torch.uint8, 128.0),
            2: (torch.int16, 32768.0),
            4: (torch.int32, 2147483648.0),
        }
        if sample_width not in dtype_and_scale:
            raise PyannoteDiarizationError(
                f"unsupported PCM WAV sample width: {sample_width * 8} bits"
            )
        dtype, scale = dtype_and_scale[sample_width]
        waveform = torch.frombuffer(bytearray(frames), dtype=dtype).reshape(
            frame_count, channels
        )
        waveform = waveform.transpose(0, 1).to(torch.float32)
        if sample_width == 1:
            waveform = (waveform - 128.0) / scale
        else:
            waveform = waveform / scale
        return {"waveform": waveform, "sample_rate": sample_rate}
