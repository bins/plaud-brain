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
    with psycopg.connect(DB, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS rag_answers, embeddings, decisions, action_items, summaries, processing_runs, transcript_segments, transcripts, recordings CASCADE")
        for migration in sorted(Path("infra/migrations").glob("*.sql")):
            sql = migration.read_text(encoding="utf-8")
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


def test_list_get_and_search_transcripts():
    store = PostgresStore(DB)
    row = store.upsert_recording(PlaudRecording(plaud_id="integration-search", name="Searchable"))
    store.save_transcript(
        row["id"],
        TranscriptResult(
            engine="whisper.cpp",
            model="test.bin",
            language="en",
            duration_ms=2000,
            raw_json={"result": {"language": "en"}},
            segments=[
                TranscriptSegment(0, 0, 1000, "Discuss the launch plan"),
                TranscriptSegment(1, 1000, 2000, "Assign the follow-up"),
            ],
            normalized_audio=Path("/tmp/test.wav"),
        ),
    )

    recordings = store.list_recordings(limit=10)
    listed = next(item for item in recordings if item["plaud_id"] == "integration-search")
    assert listed["segment_count"] == 2
    assert listed["language"] == "en"

    transcript = store.get_transcript("integration-search")
    assert transcript is not None
    assert [segment["text"] for segment in transcript["segments"]] == [
        "Discuss the launch plan",
        "Assign the follow-up",
    ]

    matches = store.search_transcripts("LAUNCH", limit=10)
    assert [(match["plaud_id"], match["seq"]) for match in matches] == [
        ("integration-search", 0),
    ]

    assert store.get_transcript("missing") is None
    with pytest.raises(ValueError, match="must not be empty"):
        store.search_transcripts("   ")


def test_summary_upsert_and_pending_selection():
    store = PostgresStore(DB)
    row = store.upsert_recording(PlaudRecording(plaud_id="integration-summary", name="Summary"))
    transcript_id = store.save_transcript(
        row["id"],
        TranscriptResult(
            engine="whisper.cpp",
            model="test.bin",
            language="en",
            duration_ms=1000,
            raw_json={"result": {"language": "en"}},
            segments=[TranscriptSegment(0, 0, 1000, "Summarize this")],
            normalized_audio=Path("/tmp/test.wav"),
        ),
    )
    assert "integration-summary" in {item["plaud_id"] for item in store.pending_summaries(20)}

    first = store.save_summary(
        transcript_id, "deepseek", "deepseek-flash", "v1", {"summary": "First"}
    )
    second = store.save_summary(
        transcript_id, "deepseek", "deepseek-flash", "v1", {"summary": "Updated"}
    )

    assert first == second
    assert store.get_summary("integration-summary")["content"]["summary"] == "Updated"
    assert "integration-summary" not in {item["plaud_id"] for item in store.pending_summaries(20)}


def test_decision_action_projection_search_and_status_survive_refresh():
    store = PostgresStore(DB)
    row = store.upsert_recording(PlaudRecording(plaud_id="integration-knowledge", name="Knowledge"))
    transcript_id = store.save_transcript(
        row["id"],
        TranscriptResult(
            engine="whisper.cpp",
            model="test.bin",
            language="en",
            duration_ms=2000,
            raw_json={"result": {"language": "en"}},
            segments=[TranscriptSegment(0, 0, 2000, "Discuss orbital launch")],
            normalized_audio=Path("/tmp/test.wav"),
        ),
    )
    content = {
        "summary": "An orbital planning meeting",
        "topics": ["Planning"],
        "decisions": [{"text": "Approve orbital launch", "timestamp_ms": 500}],
        "action_items": [
            {"task": "Prepare orbital checklist", "owner": "Alice", "due": None, "timestamp_ms": 1000}
        ],
    }
    store.save_summary(transcript_id, "deepseek", "deepseek-flash", "v1", content)

    decisions = [item for item in store.list_decisions() if item["plaud_id"] == "integration-knowledge"]
    actions = [item for item in store.list_action_items("open") if item["plaud_id"] == "integration-knowledge"]
    assert [item["text"] for item in decisions] == ["Approve orbital launch"]
    decision_id = decisions[0]["id"]
    assert len(actions) == 1
    action_id = actions[0]["id"]
    assert store.update_action_status(str(action_id), "done")["status"] == "DONE"

    refreshed_content = {**content, "action_items": [{**content["action_items"][0], "timestamp_ms": 1500}]}
    store.save_summary(transcript_id, "deepseek", "deepseek-flash", "v1", refreshed_content)
    refreshed = [item for item in store.list_action_items("done") if item["plaud_id"] == "integration-knowledge"]
    refreshed_decisions = [
        item for item in store.list_decisions() if item["plaud_id"] == "integration-knowledge"
    ]
    assert refreshed_decisions[0]["id"] == decision_id
    assert refreshed[0]["id"] == action_id
    assert refreshed[0]["status"] == "DONE"
    assert refreshed[0]["timestamp_ms"] == 1500

    matches = store.search_knowledge("orbital", 20)
    assert {item["source_type"] for item in matches} == {
        "transcript", "summary", "decision", "action_item"
    }


def test_embedding_sources_and_cache_lifecycle():
    store = PostgresStore(DB)
    sources = store.list_embedding_sources()
    source = next(item for item in sources if item["source_type"] == "decision")
    vector = [0.25, 0.75]
    store.upsert_embedding(source, "test", "embed-v1", 2, "hash-1", vector)

    metadata = store.list_embedding_metadata("test", "embed-v1")
    assert any(item["source_id"] == source["source_id"] for item in metadata)
    embedded = store.list_embeddings("test", "embed-v1")
    assert embedded[0]["embedding"] == vector

    source["text"] += " updated"
    store.upsert_embedding(source, "test", "embed-v1", 2, "hash-2", [0.5, 0.5])
    assert store.list_embedding_metadata("test", "embed-v1")[0]["content_hash"] == "hash-2"
    assert store.delete_stale_embeddings("test", "embed-v1", set()) == 1


def test_rag_answer_cache_round_trip():
    store = PostgresStore(DB)
    payload = {
        "query": "What was decided?",
        "answer": "It was approved. [S1]",
        "insufficient_evidence": False,
        "sources": [{"id": "S1"}],
        "provider": "test",
        "model": "qa-v1",
    }
    store.save_rag_answer(
        cache_key="cache-key-1",
        query="What was decided?",
        filters={"top_k": 8},
        provider="test",
        model="qa-v1",
        prompt_version="v1",
        index_signature="2:now",
        response=payload,
    )
    cached = store.get_rag_answer("cache-key-1")
    assert cached["response"] == payload
