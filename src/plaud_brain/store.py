from __future__ import annotations

import json
import hashlib
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
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

    @contextmanager
    def embedding_sync_lock(self):
        """Serialize paid embedding syncs across timers and manual commands."""
        lock_name = "plaud-brain:embedding-sync"
        with self.connect() as conn:
            conn.execute("SELECT pg_advisory_lock(hashtext(%s))", (lock_name,))
            try:
                yield
            finally:
                conn.execute("SELECT pg_advisory_unlock(hashtext(%s))", (lock_name,))

    @contextmanager
    def rag_answer_lock(self, cache_key: str):
        """Serialize generation for one cache key to avoid duplicate paid calls."""
        lock_name = f"plaud-brain:rag:{cache_key}"
        with self.connect() as conn:
            conn.execute("SELECT pg_advisory_lock(hashtext(%s))", (lock_name,))
            try:
                yield
            finally:
                conn.execute("SELECT pg_advisory_unlock(hashtext(%s))", (lock_name,))

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

    def list_recordings(self, limit: int = 20) -> list[dict]:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.plaud_id, r.name, r.created_at, r.duration_seconds, r.status,
                       t.language, count(s.*)::int AS segment_count,
                       (su.id IS NOT NULL) AS summarized
                FROM recordings r
                LEFT JOIN transcripts t ON t.recording_id = r.id
                LEFT JOIN transcript_segments s ON s.transcript_id = t.id
                LEFT JOIN summaries su ON su.transcript_id = t.id
                GROUP BY r.id, t.id, su.id
                ORDER BY COALESCE(r.created_at, r.discovered_at) DESC
                LIMIT %s
                """,
                (limit,),
            )
            return list(cur.fetchall())

    def get_transcript(self, plaud_id: str) -> dict | None:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.id AS recording_id, t.id AS transcript_id,
                       r.plaud_id, r.name, r.created_at, r.duration_seconds, r.status,
                       t.engine, t.model, t.language, t.duration_ms
                FROM recordings r
                JOIN transcripts t ON t.recording_id = r.id
                WHERE r.plaud_id = %s
                """,
                (plaud_id,),
            )
            transcript = cur.fetchone()
            if transcript is None:
                return None
            cur.execute(
                """
                SELECT s.seq, s.start_ms, s.end_ms, s.speaker, s.text
                FROM transcript_segments s
                JOIN transcripts t ON t.id = s.transcript_id
                JOIN recordings r ON r.id = t.recording_id
                WHERE r.plaud_id = %s
                ORDER BY s.seq
                """,
                (plaud_id,),
            )
            return {**transcript, "segments": list(cur.fetchall())}

    def pending_summaries(self, limit: int = 20) -> list[dict]:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.plaud_id
                FROM recordings r
                JOIN transcripts t ON t.recording_id = r.id
                LEFT JOIN summaries s ON s.transcript_id = t.id
                WHERE r.status = 'TRANSCRIBED' AND s.id IS NULL
                ORDER BY COALESCE(r.created_at, r.discovered_at), r.discovered_at
                LIMIT %s
                """,
                (limit,),
            )
            return list(cur.fetchall())

    def save_summary(
        self,
        transcript_id: uuid.UUID,
        provider: str,
        model: str,
        prompt_version: str,
        content: dict,
    ) -> uuid.UUID:
        summary_id = uuid.uuid4()
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO summaries(id, transcript_id, provider, model, prompt_version, content)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT(transcript_id) DO UPDATE SET
                    provider=EXCLUDED.provider,
                    model=EXCLUDED.model,
                    prompt_version=EXCLUDED.prompt_version,
                    content=EXCLUDED.content,
                    updated_at=now()
                RETURNING id
                """,
                (summary_id, transcript_id, provider, model, prompt_version, json.dumps(content)),
            )
            stored_id = cur.fetchone()["id"]
            self._sync_summary_items(cur, stored_id, content)
            return stored_id

    @staticmethod
    def _fingerprint(value: dict) -> str:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _sync_summary_items(self, cur, summary_id: uuid.UUID, content: dict) -> None:
        cur.execute(
            "SELECT id, fingerprint, created_at FROM decisions WHERE summary_id=%s",
            (summary_id,),
        )
        existing_decisions = {row["fingerprint"]: row for row in cur.fetchall()}
        cur.execute("DELETE FROM decisions WHERE summary_id=%s", (summary_id,))
        for seq, item in enumerate(content.get("decisions") or []):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            timestamp_ms = item.get("timestamp_ms")
            timestamp_ms = int(timestamp_ms) if isinstance(timestamp_ms, (int, float)) else None
            fingerprint = self._fingerprint({"text": text, "timestamp_ms": timestamp_ms})
            previous = existing_decisions.get(fingerprint)
            cur.execute(
                """
                INSERT INTO decisions(id, summary_id, seq, fingerprint, text, timestamp_ms, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    previous["id"] if previous else uuid.uuid4(),
                    summary_id,
                    seq,
                    fingerprint,
                    text,
                    timestamp_ms,
                    previous["created_at"] if previous else datetime.now(timezone.utc),
                ),
            )

        cur.execute(
            """
            SELECT id, fingerprint, status, completed_at, created_at
            FROM action_items WHERE summary_id=%s
            """,
            (summary_id,),
        )
        existing = {row["fingerprint"]: row for row in cur.fetchall()}
        cur.execute("DELETE FROM action_items WHERE summary_id=%s", (summary_id,))
        for seq, item in enumerate(content.get("action_items") or []):
            if not isinstance(item, dict):
                continue
            task = str(item.get("task") or "").strip()
            if not task:
                continue
            owner = str(item["owner"]).strip() if item.get("owner") else None
            due_text = str(item["due"]).strip() if item.get("due") else None
            timestamp_ms = item.get("timestamp_ms")
            timestamp_ms = int(timestamp_ms) if isinstance(timestamp_ms, (int, float)) else None
            identity = {
                "task": task,
                "owner": owner,
                "due": due_text,
            }
            fingerprint = self._fingerprint(identity)
            previous = existing.get(fingerprint)
            cur.execute(
                """
                INSERT INTO action_items(
                    id, summary_id, seq, fingerprint, task, owner, due_text,
                    timestamp_ms, status, completed_at, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    previous["id"] if previous else uuid.uuid4(),
                    summary_id,
                    seq,
                    fingerprint,
                    task,
                    owner,
                    due_text,
                    timestamp_ms,
                    previous["status"] if previous else "OPEN",
                    previous["completed_at"] if previous else None,
                    previous["created_at"] if previous else datetime.now(timezone.utc),
                ),
            )

    def rebuild_summary_projections(self) -> int:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT id, content FROM summaries ORDER BY created_at")
            rows = list(cur.fetchall())
            for row in rows:
                self._sync_summary_items(cur, row["id"], row["content"])
            return len(rows)

    def get_summary(self, plaud_id: str) -> dict | None:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.provider, s.model, s.prompt_version, s.content,
                       s.created_at, s.updated_at
                FROM summaries s
                JOIN transcripts t ON t.id = s.transcript_id
                JOIN recordings r ON r.id = t.recording_id
                WHERE r.plaud_id = %s
                """,
                (plaud_id,),
            )
            return cur.fetchone()

    def list_decisions(self, limit: int = 100) -> list[dict]:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.id, r.plaud_id, r.name, d.seq, d.text, d.timestamp_ms,
                       s.updated_at AS summary_updated_at
                FROM decisions d
                JOIN summaries s ON s.id = d.summary_id
                JOIN transcripts t ON t.id = s.transcript_id
                JOIN recordings r ON r.id = t.recording_id
                ORDER BY COALESCE(r.created_at, r.discovered_at) DESC, d.seq
                LIMIT %s
                """,
                (limit,),
            )
            return list(cur.fetchall())

    def list_action_items(self, status: str = "OPEN", limit: int = 100) -> list[dict]:
        status = status.upper()
        if status not in {"OPEN", "DONE", "DISMISSED", "ALL"}:
            raise ValueError("invalid action status")
        where = "" if status == "ALL" else "WHERE a.status = %s"
        params: tuple = (limit,) if status == "ALL" else (status, limit)
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT a.id, r.plaud_id, r.name, a.seq, a.task, a.owner,
                       a.due_text AS due, a.timestamp_ms, a.status,
                       a.completed_at, a.updated_at
                FROM action_items a
                JOIN summaries s ON s.id = a.summary_id
                JOIN transcripts t ON t.id = s.transcript_id
                JOIN recordings r ON r.id = t.recording_id
                {where}
                ORDER BY a.created_at DESC, a.seq
                LIMIT %s
                """,
                params,
            )
            return list(cur.fetchall())

    def update_action_status(self, action_id: str, status: str) -> dict | None:
        status = status.upper()
        if status not in {"OPEN", "DONE", "DISMISSED"}:
            raise ValueError("invalid action status")
        try:
            parsed_id = uuid.UUID(action_id)
        except ValueError as exc:
            raise ValueError("action id must be a UUID") from exc
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE action_items
                SET status=%s,
                    completed_at=CASE WHEN %s = 'DONE' THEN now() ELSE NULL END,
                    updated_at=now()
                WHERE id=%s
                RETURNING id, task, owner, due_text AS due, timestamp_ms,
                          status, completed_at, updated_at
                """,
                (status, status, parsed_id),
            )
            return cur.fetchone()

    def search_knowledge(self, query: str, limit: int = 20) -> list[dict]:
        query = query.strip()
        if not query:
            raise ValueError("search query must not be empty")
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                WITH needle AS (SELECT lower(%s) AS value), matches AS (
                    SELECT 'transcript'::text AS source_type, r.plaud_id, r.name,
                           ts.start_ms AS timestamp_ms, ts.text
                    FROM transcript_segments ts
                    JOIN transcripts t ON t.id = ts.transcript_id
                    JOIN recordings r ON r.id = t.recording_id, needle n
                    WHERE strpos(lower(ts.text), n.value) > 0
                    UNION ALL
                    SELECT 'summary', r.plaud_id, r.name, NULL,
                           s.content->>'summary'
                    FROM summaries s
                    JOIN transcripts t ON t.id = s.transcript_id
                    JOIN recordings r ON r.id = t.recording_id, needle n
                    WHERE strpos(lower(s.content->>'summary'), n.value) > 0
                    UNION ALL
                    SELECT 'decision', r.plaud_id, r.name, d.timestamp_ms, d.text
                    FROM decisions d
                    JOIN summaries s ON s.id = d.summary_id
                    JOIN transcripts t ON t.id = s.transcript_id
                    JOIN recordings r ON r.id = t.recording_id, needle n
                    WHERE strpos(lower(d.text), n.value) > 0
                    UNION ALL
                    SELECT 'action_item', r.plaud_id, r.name, a.timestamp_ms, a.task
                    FROM action_items a
                    JOIN summaries s ON s.id = a.summary_id
                    JOIN transcripts t ON t.id = s.transcript_id
                    JOIN recordings r ON r.id = t.recording_id, needle n
                    WHERE strpos(lower(a.task), n.value) > 0
                )
                SELECT * FROM matches LIMIT %s
                """,
                (query, limit),
            )
            return list(cur.fetchall())

    def search_transcripts(self, query: str, limit: int = 20) -> list[dict]:
        query = query.strip()
        if not query:
            raise ValueError("search query must not be empty")
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.plaud_id, r.name, r.created_at,
                       s.seq, s.start_ms, s.end_ms, s.text
                FROM transcript_segments s
                JOIN transcripts t ON t.id = s.transcript_id
                JOIN recordings r ON r.id = t.recording_id
                WHERE strpos(lower(s.text), lower(%s)) > 0
                ORDER BY COALESCE(r.created_at, r.discovered_at) DESC, s.seq
                LIMIT %s
                """,
                (query, limit),
            )
            return list(cur.fetchall())

    def list_embedding_sources(self) -> list[dict]:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT 'transcript'::text AS source_type, ts.id::text AS source_id,
                       r.id AS recording_id, r.plaud_id, r.name,
                       ts.start_ms AS timestamp_ms, ts.text
                FROM transcript_segments ts
                JOIN transcripts t ON t.id = ts.transcript_id
                JOIN recordings r ON r.id = t.recording_id
                WHERE btrim(ts.text) <> ''
                UNION ALL
                SELECT 'summary', s.id::text, r.id, r.plaud_id, r.name, NULL,
                       concat_ws(E'\n', s.content->>'summary',
                           array_to_string(ARRAY(
                               SELECT jsonb_array_elements_text(
                                   COALESCE(s.content->'topics', '[]'::jsonb)
                               )
                           ), E'\n'))
                FROM summaries s
                JOIN transcripts t ON t.id = s.transcript_id
                JOIN recordings r ON r.id = t.recording_id
                WHERE btrim(COALESCE(s.content->>'summary', '')) <> ''
                UNION ALL
                SELECT 'decision', d.id::text, r.id, r.plaud_id, r.name,
                       d.timestamp_ms, d.text
                FROM decisions d
                JOIN summaries s ON s.id = d.summary_id
                JOIN transcripts t ON t.id = s.transcript_id
                JOIN recordings r ON r.id = t.recording_id
                WHERE btrim(d.text) <> ''
                UNION ALL
                SELECT 'action_item', a.id::text, r.id, r.plaud_id, r.name,
                       a.timestamp_ms,
                       concat_ws(E'\n', a.task,
                           CASE WHEN a.owner IS NOT NULL THEN 'Owner: ' || a.owner END,
                           CASE WHEN a.due_text IS NOT NULL THEN 'Due: ' || a.due_text END)
                FROM action_items a
                JOIN summaries s ON s.id = a.summary_id
                JOIN transcripts t ON t.id = s.transcript_id
                JOIN recordings r ON r.id = t.recording_id
                WHERE btrim(a.task) <> ''
                ORDER BY source_type, source_id
                """
            )
            return list(cur.fetchall())

    def list_embedding_metadata(self, provider: str, model: str) -> list[dict]:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT source_type, source_id, content_hash, dimensions
                FROM embeddings WHERE provider=%s AND model=%s
                """,
                (provider, model),
            )
            return list(cur.fetchall())

    def upsert_embedding(
        self,
        source: dict,
        provider: str,
        model: str,
        dimensions: int,
        content_hash: str,
        embedding: list[float],
    ) -> None:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO embeddings(
                    id, source_type, source_id, recording_id, provider, model,
                    dimensions, content_hash, text_snapshot, timestamp_ms, embedding
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT(source_type, source_id, provider, model) DO UPDATE SET
                    recording_id=EXCLUDED.recording_id,
                    dimensions=EXCLUDED.dimensions,
                    content_hash=EXCLUDED.content_hash,
                    text_snapshot=EXCLUDED.text_snapshot,
                    timestamp_ms=EXCLUDED.timestamp_ms,
                    embedding=EXCLUDED.embedding,
                    updated_at=now()
                """,
                (
                    uuid.uuid4(),
                    source["source_type"],
                    source["source_id"],
                    source["recording_id"],
                    provider,
                    model,
                    dimensions,
                    content_hash,
                    source["text"],
                    source["timestamp_ms"],
                    json.dumps(embedding),
                ),
            )

    def delete_embeddings_for_model(self, provider: str, model: str) -> int:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM embeddings WHERE provider=%s AND model=%s", (provider, model))
            return cur.rowcount

    def delete_stale_embeddings(
        self,
        provider: str,
        model: str,
        current_keys: set[tuple[str, str]],
    ) -> int:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT source_type, source_id FROM embeddings WHERE provider=%s AND model=%s",
                (provider, model),
            )
            stale = [
                (row["source_type"], row["source_id"], provider, model)
                for row in cur.fetchall()
                if (row["source_type"], row["source_id"]) not in current_keys
            ]
            if stale:
                cur.executemany(
                    """
                    DELETE FROM embeddings
                    WHERE source_type=%s AND source_id=%s AND provider=%s AND model=%s
                    """,
                    stale,
                )
            return len(stale)

    def list_embeddings(self, provider: str, model: str) -> list[dict]:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT e.source_type, e.source_id, r.plaud_id, r.name, r.created_at,
                       e.timestamp_ms, e.text_snapshot AS text, e.embedding
                FROM embeddings e
                JOIN recordings r ON r.id = e.recording_id
                WHERE e.provider=%s AND e.model=%s
                """,
                (provider, model),
            )
            return list(cur.fetchall())

    def embedding_index_signature(self, provider: str, model: str) -> str:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*)::int AS count, max(updated_at) AS latest
                FROM embeddings WHERE provider=%s AND model=%s
                """,
                (provider, model),
            )
            row = cur.fetchone()
            latest = row["latest"].isoformat() if row["latest"] else "none"
            return f"{row['count']}:{latest}"

    def get_rag_answer(self, cache_key: str) -> dict | None:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT response, created_at, updated_at
                FROM rag_answers WHERE cache_key=%s
                """,
                (cache_key,),
            )
            return cur.fetchone()

    def save_rag_answer(
        self,
        *,
        cache_key: str,
        query: str,
        filters: dict,
        provider: str,
        model: str,
        prompt_version: str,
        index_signature: str,
        response: dict,
    ) -> None:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rag_answers(
                    id, cache_key, query, filters, provider, model,
                    prompt_version, index_signature, response
                ) VALUES (%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT(cache_key) DO UPDATE SET
                    response=EXCLUDED.response,
                    updated_at=now()
                """,
                (
                    uuid.uuid4(), cache_key, query, json.dumps(filters), provider,
                    model, prompt_version, index_signature, json.dumps(response, default=str),
                ),
            )
