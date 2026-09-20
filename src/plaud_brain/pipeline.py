from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from plaud_brain.adapters.downloader import AudioDownloader
from plaud_brain.adapters.plaud_cli import PlaudCliAdapter
from plaud_brain.adapters.whisper_cpp import WhisperCppTranscriber
from plaud_brain.store import PostgresStore

log = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        store: PostgresStore,
        plaud: PlaudCliAdapter,
        downloader: AudioDownloader,
        transcriber: WhisperCppTranscriber,
        data_dir: Path,
    ) -> None:
        self.store = store
        self.plaud = plaud
        self.downloader = downloader
        self.transcriber = transcriber
        self.data_dir = data_dir

    def discover(self, days: int, page_size: int) -> int:
        recordings = self.plaud.list_recent(days=days, page_size=page_size)
        for rec in recordings:
            self.store.upsert_recording(rec)
        log.info("discovered/upserted %d recordings", len(recordings))
        return len(recordings)

    def process(self, limit: int = 20) -> dict[str, int]:
        counts = {"transcribed": 0, "download_failed": 0, "stt_failed": 0}
        for rec in self.store.pending(limit=limit):
            status = rec["status"]
            if status in {"DISCOVERED", "FAILED_DOWNLOAD"}:
                if not self._download(rec):
                    counts["download_failed"] += 1
                    continue
                rec = {**rec, "status": "DOWNLOADED", "audio_path": str(self._audio_path(rec["plaud_id"]))}
            if rec["status"] in {"DOWNLOADED", "FAILED_STT"}:
                if self._transcribe(rec):
                    counts["transcribed"] += 1
                else:
                    counts["stt_failed"] += 1
        return counts

    def _audio_path(self, plaud_id: str) -> Path:
        return self.data_dir / "audio" / f"{plaud_id}.source"

    def _download(self, rec: dict) -> bool:
        run = self.store.start_run(rec["id"], "DOWNLOAD")
        try:
            path = self._audio_path(rec["plaud_id"])
            if path.exists() and path.stat().st_size > 0:
                digest_obj = hashlib.sha256()
                with path.open("rb") as fh:
                    while chunk := fh.read(1024 * 1024):
                        digest_obj.update(chunk)
                digest = digest_obj.hexdigest()
            else:
                url = self.plaud.audio_url(rec["plaud_id"])
                result = self.downloader.download(url, path)
                digest = result.sha256
            self.store.update_downloaded(rec["id"], path, digest)
            self.store.finish_run(run, True)
            return True
        except Exception as exc:
            log.exception("download failed for %s", rec["plaud_id"])
            self.store.update_failed(rec["id"], "FAILED_DOWNLOAD", exc.__class__.__name__, str(exc))
            self.store.finish_run(run, False, exc.__class__.__name__, str(exc))
            return False

    def _transcribe(self, rec: dict) -> bool:
        run = self.store.start_run(rec["id"], "STT")
        try:
            audio = Path(rec.get("audio_path") or self._audio_path(rec["plaud_id"]))
            if not audio.exists():
                raise FileNotFoundError(audio)
            work = self.data_dir / "work" / rec["plaud_id"]
            result = self.transcriber.transcribe(audio, work)
            self.store.save_transcript(rec["id"], result)
            self.store.finish_run(run, True)
            return True
        except Exception as exc:
            log.exception("STT failed for %s", rec["plaud_id"])
            self.store.update_failed(rec["id"], "FAILED_STT", exc.__class__.__name__, str(exc))
            self.store.finish_run(run, False, exc.__class__.__name__, str(exc))
            return False
