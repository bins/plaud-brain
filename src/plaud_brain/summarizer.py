from __future__ import annotations

import json

from plaud_brain.adapters.openai_compatible import OpenAICompatibleClient


SYSTEM_PROMPT = """You summarize private meeting and voice-note transcripts.
Return one JSON object and no prose outside JSON. Use the transcript's primary
language. The JSON schema is:
{
  "summary": "concise summary",
  "topics": ["topic"],
  "decisions": [{"text": "decision", "timestamp_ms": 0}],
  "action_items": [
    {"task": "action", "owner": null, "due": null, "timestamp_ms": 0}
  ]
}
Do not invent decisions, owners, dates, or actions. Use empty arrays when none
are supported by the transcript."""


def summarize_transcript(client: OpenAICompatibleClient, transcript: dict) -> dict:
    segments = [
        {
            "start_ms": segment["start_ms"],
            "speaker": segment.get("speaker"),
            "text": segment["text"],
        }
        for segment in transcript["segments"]
    ]
    prompt = json.dumps(
        {
            "title": transcript.get("name"),
            "language": transcript.get("language"),
            "segments": segments,
        },
        ensure_ascii=False,
    )
    result = client.generate_json(SYSTEM_PROMPT, prompt)
    return _normalize_summary(result)


def _normalize_summary(result: dict) -> dict:
    summary = result.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("summary provider returned no summary")
    topics = result.get("topics", [])
    decisions = result.get("decisions", [])
    action_items = result.get("action_items", [])
    if not isinstance(topics, list) or any(not isinstance(topic, str) for topic in topics):
        raise ValueError("summary provider returned invalid topics")
    if not isinstance(decisions, list):
        raise ValueError("summary provider returned invalid decisions")
    if not isinstance(action_items, list):
        raise ValueError("summary provider returned invalid action_items")

    normalized_decisions = []
    for item in decisions:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise ValueError("summary provider returned an invalid decision")
        text = item["text"].strip()
        if not text:
            raise ValueError("summary provider returned an empty decision")
        normalized_decisions.append(
            {"text": text, "timestamp_ms": _optional_timestamp(item.get("timestamp_ms"))}
        )

    normalized_actions = []
    for item in action_items:
        if not isinstance(item, dict) or not isinstance(item.get("task"), str):
            raise ValueError("summary provider returned an invalid action item")
        task = item["task"].strip()
        if not task:
            raise ValueError("summary provider returned an empty action item")
        owner = _optional_text(item.get("owner"), "owner")
        due = _optional_text(item.get("due"), "due")
        normalized_actions.append(
            {
                "task": task,
                "owner": owner,
                "due": due,
                "timestamp_ms": _optional_timestamp(item.get("timestamp_ms")),
            }
        )

    return {
        "summary": summary.strip(),
        "topics": [topic.strip() for topic in topics if topic.strip()],
        "decisions": normalized_decisions,
        "action_items": normalized_actions,
    }


def _optional_timestamp(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError("summary provider returned an invalid timestamp_ms")
    return int(value)


def _optional_text(value, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"summary provider returned an invalid {field}")
    return value.strip() or None


def render_summary_markdown(transcript: dict, summary: dict) -> str:
    title = transcript.get("name") or transcript["plaud_id"]
    lines = [f"# {title}", "", "## Summary", "", summary["summary"], ""]
    lines.extend(["## Topics", ""])
    lines.extend(f"- {topic}" for topic in summary["topics"])
    lines.extend(["", "## Decisions", ""])
    lines.extend(f"- {item['text']}" for item in summary["decisions"])
    lines.extend(["", "## Action items", ""])
    for item in summary["action_items"]:
        detail = item["task"]
        if item.get("owner"):
            detail += f" — {item['owner']}"
        if item.get("due"):
            detail += f" (due: {item['due']})"
        lines.append(f"- {detail}")
    return "\n".join(lines) + "\n"
