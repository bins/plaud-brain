import json
import os
from pathlib import Path
from uuid import UUID

import pytest

psycopg = pytest.importorskip("psycopg")

from plaud_brain.store import PostgresStore
from plaud_brain.types import PlaudRecording, TranscriptResult, TranscriptSegment


DB = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB, reason="TEST_DATABASE_URL not set")


@pytest.fixture(scope="module", autouse=True)
def schema():
    sql = Path("infra/migrations/001_init.sql").read_text(encoding="utf-8")
    with psycopg.connect(DB, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS processing_runs, transcript_segments, transcripts, recordings CASCADE")
        for statement in sql.split(";"):
            if statement.strip():
                conn.execute(statement)
    yield


def test_recording_upsert_is_idempotent():
    store = PostgresStore(DB)
    rec = PlaudRecording(plaud_id="integration123456", name="First")
    a = store.upsert_recording(rec)
    b = store.upsert_recording(PlaudRecording(plaud_id="integration123456", name="Renamed"))
    assert a["id"] == b["id"]
    assert b["name"] == "Renamed"
    with store.connect() as conn:
        count = conn.execute("select count(*) AS n from recordings where plaud_id=%s", (rec.plaud_id,)).fetchone()["n"]
    assert count == 1


def test_transcript_replace_does_not_duplicate_segments():
    store = PostgresStore(DB)
    row = store.upsert_recording(PlaudRecording(plaud_id="integration654321", name="Transcript"))
    result1 = TranscriptResult(
        engine="whisper.cpp",
        model="test.bin",
        language="fr",
        duration_ms=2000,
        raw_json={"result": {"language": "fr"}},
        segments=[
            TranscriptSegment(0, 0, 1000, "bonjour"),
            TranscriptSegment(1, 1000, 2000, "salut"),
        ],
        normalized_audio=Path("/tmp/test.wav"),
    )
    tid1 = store.save_transcript(row["id"], result1)
    result2 = TranscriptResult(
        engine="whisper.cpp",
        model="test.bin",
        language="fr",
        duration_ms=1000,
        raw_json={"result": {"language": "fr"}, "revision": 2},
        segments=[TranscriptSegment(0, 0, 1000, "bonjour corrigé")],
        normalized_audio=Path("/tmp/test.wav"),
    )
    tid2 = store.save_transcript(row["id"], result2)
    assert tid1 == tid2
    with store.connect() as conn:
        segs = conn.execute(
            "select seq,text from transcript_segments where transcript_id=%s order by seq", (tid1,)
        ).fetchall()
        status = conn.execute("select status from recordings where id=%s", (row["id"],)).fetchone()["status"]
    assert [(r["seq"], r["text"]) for r in segs] == [(0, "bonjour corrigé")]
    assert status == "TRANSCRIBED"
