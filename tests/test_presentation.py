from plaud_brain.presentation import (
    format_timestamp,
    render_knowledge_markdown,
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
