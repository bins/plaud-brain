from plaud_brain.presentation import (
    format_timestamp,
    render_knowledge_markdown,
    render_rag_markdown,
    render_transcript_markdown,
    render_transcript_text,
)


def _transcript() -> dict:
    return {
        "plaud_id": "recording-1",
        "name": "Weekly sync",
        "created_at": "2026-09-21T08:00:00+00:00",
        "language": "en",
        "segments": [
            {"start_ms": 0, "speaker": None, "text": "Hello."},
            {"start_ms": 65_000, "speaker": "Alice", "text": "Next item."},
        ],
    }


def test_format_timestamp():
    assert format_timestamp(3_661_999) == "01:01:01"


def test_render_transcript_text():
    assert render_transcript_text(_transcript()) == "Hello.\nNext item."


def test_render_transcript_markdown():
    rendered = render_transcript_markdown(_transcript())
    assert rendered.startswith("# Weekly sync\n")
    assert "`00:01:05` **Alice** Next item." in rendered


def test_render_knowledge_markdown():
    summary = {
        "summary": "A concise result.",
        "topics": ["Planning"],
        "decisions": [{"text": "Ship it"}],
        "action_items": [{"task": "Write tests", "owner": "Alice", "due": None}],
    }
    rendered = render_knowledge_markdown(_transcript(), summary)
    assert rendered.count("# Weekly sync") == 1
    assert "## Summary" in rendered
    assert "Write tests — Alice" in rendered
    assert "## Transcript" in rendered


def test_render_rag_markdown_includes_traceable_sources():
    rendered = render_rag_markdown(
        {
            "answer": "Approved. [S1]",
            "cached": True,
            "provider": "deepseek",
            "model": "flash",
            "sources": [
                {
                    "id": "S1",
                    "plaud_id": "recording-1",
                    "name": "Planning",
                    "source_type": "decision",
                    "timestamp_ms": 42000,
                    "score": 0.8123,
                }
            ],
        }
    )
    assert "Approved. [S1]" in rendered
    assert "Planning — decision — 00:00:42" in rendered
    assert "Cached: yes" in rendered
