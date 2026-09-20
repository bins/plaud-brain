from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    database_url: str
    data_dir: Path
    plaud_bin: str
    plaud_lookback_days: int
    plaud_page_size: int
    whisper_bin: str
    whisper_model: Path
    whisper_language: str
    ffmpeg_bin: str
    ffprobe_bin: str

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        return cls(
            database_url=os.getenv("DATABASE_URL", "postgresql://plaud:plaud@127.0.0.1:5432/plaud_brain"),
            data_dir=Path(os.getenv("DATA_DIR", "./data")).expanduser().resolve(),
            plaud_bin=os.getenv("PLAUD_BIN", "plaud"),
            plaud_lookback_days=int(os.getenv("PLAUD_LOOKBACK_DAYS", "3")),
            plaud_page_size=int(os.getenv("PLAUD_PAGE_SIZE", "100")),
            whisper_bin=os.getenv("WHISPER_BIN", "./vendor/whisper.cpp/build/bin/whisper-cli"),
            whisper_model=Path(os.getenv("WHISPER_MODEL", "./models/ggml-large-v3-turbo.bin")).expanduser().resolve(),
            whisper_language=os.getenv("WHISPER_LANGUAGE", "auto"),
            ffmpeg_bin=os.getenv("FFMPEG_BIN", "ffmpeg"),
            ffprobe_bin=os.getenv("FFPROBE_BIN", "ffprobe"),
        )
