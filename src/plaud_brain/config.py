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


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


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
    whisper_threads: int
    whisper_no_gpu: bool
    whisper_flash_attn: bool
    stt_provider: str
    stt_base_url: str
    stt_model: str
    stt_api_key_env: str
    stt_chunk_seconds: int
    stt_timeout_seconds: int
    stt_fallback_provider: str
    summary_provider: str
    summary_base_url: str
    summary_model: str
    summary_api_key_env: str
    summary_timeout_seconds: int
    summary_auto: bool
    embedding_provider: str
    embedding_base_url: str
    embedding_model: str
    embedding_api_key_env: str
    embedding_dimensions: int
    embedding_batch_size: int
    embedding_timeout_seconds: int
    embedding_document_input_type: str
    embedding_query_input_type: str
    embedding_auto: bool
    qa_provider: str
    qa_base_url: str
    qa_model: str
    qa_api_key_env: str
    qa_timeout_seconds: int
    qa_top_k: int
    qa_min_score: float
    web_host: str
    web_port: int
    web_api_token_env: str
    ffmpeg_bin: str
    ffprobe_bin: str

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        return cls(
            database_url=os.getenv("DATABASE_URL", "postgresql://plaud:plaud@127.0.0.1:5433/plaud_brain"),
            data_dir=Path(os.getenv("DATA_DIR", "./data")).expanduser().resolve(),
            plaud_bin=os.getenv("PLAUD_BIN", "plaud"),
            plaud_lookback_days=int(os.getenv("PLAUD_LOOKBACK_DAYS", "3")),
            plaud_page_size=int(os.getenv("PLAUD_PAGE_SIZE", "100")),
            whisper_bin=os.getenv("WHISPER_BIN", "./vendor/whisper.cpp/build/bin/whisper-cli"),
            whisper_model=Path(os.getenv("WHISPER_MODEL", "./models/ggml-large-v3-turbo.bin")).expanduser().resolve(),
            whisper_language=os.getenv("WHISPER_LANGUAGE", "auto"),
            whisper_threads=int(os.getenv("WHISPER_THREADS", "8")),
            whisper_no_gpu=_env_bool("WHISPER_NO_GPU"),
            whisper_flash_attn=_env_bool("WHISPER_FLASH_ATTN", default=True),
            stt_provider=os.getenv("STT_PROVIDER", "local").strip().lower(),
            stt_base_url=os.getenv("STT_BASE_URL", "https://openrouter.ai/api/v1"),
            stt_model=os.getenv("STT_MODEL", "openai/whisper-large-v3-turbo"),
            stt_api_key_env=os.getenv("STT_API_KEY_ENV", "OPENROUTER_API_KEY"),
            stt_chunk_seconds=int(os.getenv("STT_CHUNK_SECONDS", "480")),
            stt_timeout_seconds=int(os.getenv("STT_TIMEOUT_SECONDS", "75")),
            stt_fallback_provider=os.getenv("STT_FALLBACK_PROVIDER", "local").strip().lower(),
            summary_provider=os.getenv("SUMMARY_PROVIDER", "deepseek"),
            summary_base_url=os.getenv("SUMMARY_BASE_URL", "https://api.deepseek.com"),
            summary_model=os.getenv("SUMMARY_MODEL", "deepseek-flash"),
            summary_api_key_env=os.getenv("SUMMARY_API_KEY_ENV", "DEEPSEEK_API_KEY"),
            summary_timeout_seconds=int(os.getenv("SUMMARY_TIMEOUT_SECONDS", "300")),
            summary_auto=_env_bool("SUMMARY_AUTO"),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openrouter").strip().lower(),
            embedding_base_url=os.getenv("EMBEDDING_BASE_URL", "https://openrouter.ai/api/v1"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b"),
            embedding_api_key_env=os.getenv("EMBEDDING_API_KEY_ENV", "OPENROUTER_API_KEY"),
            embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1024")),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
            embedding_timeout_seconds=int(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "60")),
            embedding_document_input_type=os.getenv("EMBEDDING_DOCUMENT_INPUT_TYPE", ""),
            embedding_query_input_type=os.getenv("EMBEDDING_QUERY_INPUT_TYPE", ""),
            embedding_auto=_env_bool("EMBEDDING_AUTO"),
            qa_provider=os.getenv("QA_PROVIDER", "deepseek").strip().lower(),
            qa_base_url=os.getenv("QA_BASE_URL", "https://api.deepseek.com"),
            qa_model=os.getenv("QA_MODEL", "deepseek-flash"),
            qa_api_key_env=os.getenv("QA_API_KEY_ENV", "DEEPSEEK_API_KEY"),
            qa_timeout_seconds=int(os.getenv("QA_TIMEOUT_SECONDS", "300")),
            qa_top_k=int(os.getenv("QA_TOP_K", "8")),
            qa_min_score=float(os.getenv("QA_MIN_SCORE", "0.45")),
            web_host=os.getenv("WEB_HOST", "127.0.0.1"),
            web_port=int(os.getenv("WEB_PORT", "8787")),
            web_api_token_env=os.getenv("WEB_API_TOKEN_ENV", "PLAUD_BRAIN_WEB_TOKEN"),
            ffmpeg_bin=os.getenv("FFMPEG_BIN", "ffmpeg"),
            ffprobe_bin=os.getenv("FFPROBE_BIN", "ffprobe"),
        )
