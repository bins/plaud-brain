from __future__ import annotations


def format_timestamp(milliseconds: int) -> str:
    total_seconds = max(0, milliseconds // 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def render_transcript_text(transcript: dict) -> str:
    return "\n".join(segment["text"] for segment in transcript["segments"])


def render_transcript_markdown(transcript: dict) -> str:
    title = transcript.get("name") or transcript["plaud_id"]
    created_at = transcript.get("created_at")
    metadata = [
        f"- PLAUD ID: `{transcript['plaud_id']}`",
        f"- Language: `{transcript.get('language') or 'unknown'}`",
    ]
    if created_at is not None:
        metadata.insert(1, f"- Created: `{created_at}`")
    lines = [f"# {title}", "", *metadata, "", "## Transcript", ""]
    for segment in transcript["segments"]:
        timestamp = format_timestamp(segment["start_ms"])
        speaker = f" **{segment['speaker']}**" if segment.get("speaker") else ""
        lines.append(f"- `{timestamp}`{speaker} {segment['text']}")
    return "\n".join(lines) + "\n"


def render_knowledge_markdown(transcript: dict, summary: dict | None) -> str:
    title = transcript.get("name") or transcript["plaud_id"]
    lines = [f"# {title}", "", f"- PLAUD ID: `{transcript['plaud_id']}`", ""]
    if summary is not None:
        lines.extend(["## Summary", "", summary["summary"], "", "## Topics", ""])
        lines.extend(f"- {topic}" for topic in summary.get("topics", []))
        lines.extend(["", "## Decisions", ""])
        lines.extend(f"- {item['text']}" for item in summary.get("decisions", []))
        lines.extend(["", "## Action items", ""])
        for item in summary.get("action_items", []):
            detail = item["task"]
            if item.get("owner"):
                detail += f" — {item['owner']}"
            if item.get("due"):
                detail += f" (due: {item['due']})"
            lines.append(f"- {detail}")
        lines.append("")
    lines.extend(["## Transcript", ""])
    for segment in transcript["segments"]:
        timestamp = format_timestamp(segment["start_ms"])
        speaker = f" **{segment['speaker']}**" if segment.get("speaker") else ""
        lines.append(f"- `{timestamp}`{speaker} {segment['text']}")
    return "\n".join(lines) + "\n"
