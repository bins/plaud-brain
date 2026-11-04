"""Export recordings as Markdown files for the PKB knowledge base.

PKB treats Markdown files as canonical and indexes any change it sees. The
export directory is a projection owned by plaud-brain: PostgreSQL stays the
source of truth, and each file is rewritten whenever its rendering changes.
Rendering is deterministic, so an unchanged recording produces identical bytes
and PKB does not re-index it.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path

from plaud_brain.presentation import format_timestamp

log = logging.getLogger(__name__)

GENERATOR = "plaud-brain-export"
SCHEMA_VERSION = 1
# Marker that identifies files this exporter owns; only those are ever deleted.
OWNERSHIP_MARKER = f"generator: {GENERATOR}"
CLASSIFICATIONS = ("private", "sensitive")
SECTION_MS = 5 * 60 * 1000

_SAFE_ID = re.compile(r"[A-Za-z0-9_-]+")
# Characters YAML forbids or treats as line breaks, plus other controls.
_YAML_ESCAPE = re.compile("[\x00-\x1f\x7f-\x9f  ﻿￾￿]")
_LINE_BREAKS = re.compile(r"[\r\n\u0085  ]+")
_CONTROL = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def export_filename(plaud_id: str) -> str:
    if not _SAFE_ID.fullmatch(plaud_id):
        raise ValueError("recording identifier contains unsupported characters")
    return f"{plaud_id}.md"


def _yaml_str(value: object) -> str:
    """Quote a scalar as a YAML double-quoted string that cannot break out."""
    text = str(value)
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return '"' + _YAML_ESCAPE.sub(lambda m: f"\\u{ord(m.group()):04x}", escaped) + '"'


def _inline(value: object) -> str:
    """One Markdown line: no line breaks, no control characters."""
    text = _CONTROL.sub("", _LINE_BREAKS.sub(" ", str(value)))
    return " ".join(text.split())


def _paragraph_lines(value: object) -> list[str]:
    """Multi-line model text that cannot open headings or fenced blocks."""
    lines = []
    for raw in _LINE_BREAKS.split(_CONTROL.sub("", str(value))):
        line = raw.rstrip()
        stripped = line.lstrip()
        if stripped.startswith(("#", "```", "~~~")):
            line = "\\" + stripped
        lines.append(line)
    while lines and not lines[-1]:
        lines.pop()
    return lines


def _speaker(value: object) -> str:
    return _inline(value).replace("*", "")


def _iso(value) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def render_pkb_document(
    transcript: dict,
    summary: dict | None,
    actions: list[dict],
    classification: str = "private",
) -> str:
    if classification not in CLASSIFICATIONS:
        raise ValueError("PKB export classification must be private or sensitive")
    plaud_id = transcript["plaud_id"]
    export_filename(plaud_id)
    title = _inline(transcript.get("name") or "") or plaud_id
    content = (summary or {}).get("content") or {}

    front = [
        "---",
        f"title: {_yaml_str(title)}",
        "type: plaud-recording",
    ]
    created = _iso(transcript.get("created_at"))
    if created:
        front.append(f"created: {_yaml_str(created)}")
    if summary and _iso(summary.get("updated_at")):
        front.append(f"updated: {_yaml_str(_iso(summary['updated_at']))}")
    front.extend(
        [
            "tags: [plaud, voice]",
            "security:",
            f"  classification: {classification}",
            "  cloud_ai: deny",
            "  local_ai: allow",
            "source:",
            "  type: plaud",
            f"  plaud_id: {_yaml_str(plaud_id)}",
            "metadata:",
            f"  {OWNERSHIP_MARKER}",
            f"  schema_version: {SCHEMA_VERSION}",
            f"  recording_id: {_yaml_str(transcript['recording_id'])}",
        ]
    )
    optional = [
        ("duration_seconds", transcript.get("duration_seconds")),
        ("language", transcript.get("language")),
        ("stt_engine", transcript.get("engine")),
        ("stt_model", transcript.get("model")),
    ]
    if summary:
        optional.extend(
            [
                ("summary_provider", summary.get("provider")),
                ("summary_model", summary.get("model")),
                ("summary_prompt_version", summary.get("prompt_version")),
            ]
        )
    for key, value in optional:
        if value is None or value == "":
            continue
        rendered = str(value) if isinstance(value, int) and not isinstance(value, bool) else _yaml_str(value)
        front.append(f"  {key}: {rendered}")
    front.append("---")

    body = [f"# {title}", ""]
    if summary:
        body.extend(["## Summary", ""])
        body.extend(_paragraph_lines(content.get("summary", "")) or ["(empty)"])
        body.append("")
        topics = [_inline(t) for t in content.get("topics", []) if _inline(t)]
        if topics:
            body.extend(["## Topics", ""])
            body.extend(f"- {topic}" for topic in topics)
            body.append("")
        decisions = [d for d in content.get("decisions", []) if _inline(d.get("text", ""))]
        if decisions:
            body.extend(["## Decisions", ""])
            for item in decisions:
                body.append(f"- {_timestamp_prefix(item.get('timestamp_ms'))}{_inline(item['text'])}")
            body.append("")
    if actions:
        body.extend(["## Action items", ""])
        for item in actions:
            body.append(_render_action(item))
        body.append("")

    body.extend(["## Transcript", ""])
    current_section = None
    for segment in transcript["segments"]:
        text = _inline(segment["text"])
        if not text:
            continue
        section = int(segment["start_ms"]) // SECTION_MS
        if section != current_section:
            if current_section is not None:
                body.append("")
            body.extend([f"### {format_timestamp(section * SECTION_MS)}", ""])
            current_section = section
        speaker = _speaker(segment.get("speaker") or "")
        label = f" **{speaker}**" if speaker else ""
        body.append(f"- `{format_timestamp(int(segment['start_ms']))}`{label} {text}")

    return "\n".join(front + body) + "\n"


def _timestamp_prefix(timestamp_ms) -> str:
    if timestamp_ms is None:
        return ""
    return f"`{format_timestamp(int(timestamp_ms))}` "


def _render_action(item: dict) -> str:
    status = str(item.get("status") or "OPEN").upper()
    box = "[x]" if status == "DONE" else "[ ]"
    detail = _inline(item["task"])
    if item.get("owner"):
        detail += f" — {_inline(item['owner'])}"
    if item.get("due"):
        detail += f" (due: {_inline(item['due'])})"
    if status == "DISMISSED":
        detail += " (dismissed)"
    return f"- {box} {_timestamp_prefix(item.get('timestamp_ms'))}{detail}"


class PkbExporter:
    def __init__(self, store, export_dir: Path, classification: str = "private", file_mode: int = 0o644):
        if classification not in CLASSIFICATIONS:
            raise ValueError("PKB_EXPORT_CLASSIFICATION must be private or sensitive")
        if not 0 <= file_mode <= 0o777:
            raise ValueError("PKB_EXPORT_FILE_MODE must be an octal permission such as 644")
        self.store = store
        self.export_dir = export_dir
        self.classification = classification
        self.file_mode = file_mode

    def sync(self) -> dict[str, int]:
        directory = self._checked_directory()
        counts = {"exported": 0, "export_unchanged": 0, "export_deleted": 0, "export_failed": 0}
        keep: set[str] = set()
        for row in self.store.list_pkb_export_ids():
            plaud_id = row["plaud_id"]
            try:
                name = export_filename(plaud_id)
                keep.add(name)
                rendered = self._render(plaud_id)
                if rendered is None:
                    keep.discard(name)
                    continue
                if self._write_if_changed(directory / name, rendered.encode("utf-8")):
                    counts["exported"] += 1
                else:
                    counts["export_unchanged"] += 1
            except Exception:
                # The ID is safe to log; transcript content is never logged.
                log.exception("PKB export failed for %s", plaud_id)
                counts["export_failed"] += 1
        # Delete only after a complete pass, and never on an empty result: a
        # wrong DATABASE_URL must not empty the export directory.
        if counts["export_failed"] == 0 and keep:
            counts["export_deleted"] = self._delete_stale(directory, keep)
        elif not keep:
            log.warning("PKB export found no recordings; stale files were not deleted")
        return counts

    def _render(self, plaud_id: str) -> str | None:
        transcript = self.store.get_transcript(plaud_id)
        if transcript is None or not transcript["segments"]:
            return None
        summary = self.store.get_summary(plaud_id)
        actions = self.store.list_recording_actions(plaud_id)
        return render_pkb_document(transcript, summary, actions, self.classification)

    def _checked_directory(self) -> Path:
        directory = self.export_dir
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(
                f"PKB export directory must be an existing real directory: {directory}"
            )
        return directory

    def _write_if_changed(self, path: Path, data: bytes) -> bool:
        try:
            if path.read_bytes() == data:
                return False
        except FileNotFoundError:
            pass
        # A hidden .tmp name is ignored by PKB's watcher until the rename.
        fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, self.file_mode)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        self._fsync_directory(path.parent)
        return True

    def _delete_stale(self, directory: Path, keep: set[str]) -> int:
        deleted = 0
        for path in directory.glob("*.md"):
            if path.name in keep or path.is_symlink() or not path.is_file():
                continue
            if not self._is_owned(path):
                continue
            path.unlink()
            deleted += 1
        if deleted:
            self._fsync_directory(directory)
        return deleted

    @staticmethod
    def _is_owned(path: Path) -> bool:
        with path.open("rb") as handle:
            head = handle.read(4096).decode("utf-8", errors="replace")
        end = head.find("\n---\n", 4)
        front = head[: end if end >= 0 else len(head)]
        return head.startswith("---\n") and f"\n  {OWNERSHIP_MARKER}\n" in front

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
