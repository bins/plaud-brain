from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterable

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # permits pure parser unit tests without runtime deps installed
    psycopg = None
    dict_row = None

from plaud_brain.types import PlaudRecording, TranscriptResult


class PostgresStore:
    def __init__(self, database_url: str):
        if psycopg is None:
            raise RuntimeError("psycopg is required; run pip install -e .")
        self.database_url = database_url

    @contextmanager
    def connect(self):
        with psycopg.connect(self.database_url, row_factory=dict_row) as conn:
            yield conn

    def upsert_recording(self, rec: PlaudRecording) -> dict:
        rid = uuid.uuid4()
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO recordings(id, plaud_id, name, created_at, duration_seconds, status)
                VALUES (%s,%s,%s,%s,%s,'DISCOVERED')
                ON CONFLICT(plaud_id) DO UPDATE SET
                    name = COALESCE(EXCLUDED.name, recordings.name),
                    created_at = COALESCE(EXCLUDED.created_at, recordings.created_at),
                    duration_seconds = COALESCE(EXCLUDED.duration_seconds, recordings.duration_seconds),
                    updated_at = now()
                RETURNING *
                """,
                (rid, rec.plaud_id, rec.name, rec.created_at, rec.duration_seconds),
            )
            return cur.fetchone()

    def pending(self, limit: int = 20) -> list[dict]:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM recordings
                WHERE status IN ('DISCOVERED','FAILED_DOWNLOAD','DOWNLOADED','FAILED_STT')
                ORDER BY COALESCE(created_at, discovered_at), discovered_at
                LIMIT %s
                """,
                (limit,),
            )
            return list(cur.fetchall())

    def update_downloaded(self, recording_id: uuid.UUID, path: Path, sha256: str) -> None:
        self._update_status(recording_id, "DOWNLOADED", audio_path=str(path), audio_sha256=sha256)

    def update_failed(self, recording_id: uuid.UUID, status: str, code: str, message: str) -> None:
        if status not in {"FAILED_DOWNLOAD", "FAILED_STT"}:
            raise ValueError(status)
        self._update_status(recording_id, status, error_code=code, error_message=message[:4000])

    def _update_status(self, recording_id: uuid.UUID, status: str, **fields) -> None:
        allowed = {"audio_path", "audio_sha256", "error_code", "error_message"}
        if not set(fields).issubset(allowed):
            raise ValueError("invalid update field")
        assignments = ["status=%s", "updated_at=now()"]
        values: list[object] = [status]
        for key, value in fields.items():
            assignments.append(f"{key}=%s")
            values.append(value)
        if status not in {"FAILED_DOWNLOAD", "FAILED_STT"}:
            assignments.extend(["error_code=NULL", "error_message=NULL"])
        values.append(recording_id)
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(f"UPDATE recordings SET {', '.join(assignments)} WHERE id=%s", values)

    def save_transcript(self, recording_id: uuid.UUID, result: TranscriptResult) -> uuid.UUID:
        tid = uuid.uuid4()
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO transcripts(id, recording_id, engine, model, language, duration_ms, raw_json)
                VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT(recording_id) DO UPDATE SET
                    engine=EXCLUDED.engine, model=EXCLUDED.model, language=EXCLUDED.language,
                    duration_ms=EXCLUDED.duration_ms, raw_json=EXCLUDED.raw_json, created_at=now()
                RETURNING id
                """,
                (tid, recording_id, result.engine, result.model, result.language, result.duration_ms, json.dumps(result.raw_json)),
            )
            tid = cur.fetchone()["id"]
            cur.execute("DELETE FROM transcript_segments WHERE transcript_id=%s", (tid,))
            for seg in result.segments:
                cur.execute(
                    """
                    INSERT INTO transcript_segments(transcript_id, seq, start_ms, end_ms, speaker, text)
                    VALUES (%s,%s,%s,%s,%s,%s)
                    """,
                    (tid, seg.seq, seg.start_ms, seg.end_ms, seg.speaker, seg.text),
                )
            cur.execute(
                """UPDATE recordings SET status='TRANSCRIBED', error_code=NULL, error_message=NULL, updated_at=now()
                   WHERE id=%s""",
                (recording_id,),
            )
        return tid

    def start_run(self, recording_id: uuid.UUID, stage: str) -> uuid.UUID:
        run_id = uuid.uuid4()
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO processing_runs(id,recording_id,stage,status) VALUES (%s,%s,%s,'RUNNING')",
                (run_id, recording_id, stage),
            )
        return run_id

    def finish_run(self, run_id: uuid.UUID, success: bool, code: str | None = None, message: str | None = None) -> None:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE processing_runs SET status=%s, ended_at=now(), error_code=%s, error_message=%s
                WHERE id=%s
                """,
                ("SUCCEEDED" if success else "FAILED", code, (message or "")[:4000] or None, run_id),
            )

    def status_counts(self) -> list[dict]:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT status, count(*) AS count FROM recordings GROUP BY status ORDER BY status")
            return list(cur.fetchall())
