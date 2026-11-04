import os
import stat
from datetime import datetime, timezone

import pytest

from plaud_brain.pkb_export import (
    OWNERSHIP_MARKER,
    PkbExporter,
    export_filename,
    render_pkb_document,
)


def transcript(plaud_id="rec_0000000001", name="Weekly sync", segments=None):
    return {
        "recording_id": "11111111-1111-1111-1111-111111111111",
        "plaud_id": plaud_id,
        "name": name,
        "created_at": datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc),
        "duration_seconds": 620,
        "language": "en",
        "engine": "whisper.cpp",
        "model": "large-v3-turbo",
        "segments": segments
        if segments is not None
        else [
            {"seq": 0, "start_ms": 3_000, "end_ms": 5_000, "speaker": "SPEAKER_00", "text": "Hello"},
            {"seq": 1, "start_ms": 301_000, "end_ms": 303_000, "speaker": None, "text": "Later"},
        ],
    }


SUMMARY = {
    "provider": "deepseek",
    "model": "deepseek-flash",
    "prompt_version": "v1",
    "updated_at": datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc),
    "content": {
        "summary": "We agreed on the plan.",
        "topics": ["launch"],
        "decisions": [{"text": "Ship Friday", "timestamp_ms": 61_000}],
        "action_items": [],
    },
}

ACTIONS = [
    {"seq": 0, "task": "Write notes", "owner": "Ana", "due": "Monday", "timestamp_ms": 0, "status": "OPEN"},
    {"seq": 1, "task": "Book room", "owner": None, "due": None, "timestamp_ms": None, "status": "DONE"},
    {"seq": 2, "task": "Old idea", "owner": None, "due": None, "timestamp_ms": None, "status": "DISMISSED"},
]


def front_matter(document):
    assert document.startswith("---\n")
    return document[4 : document.index("\n---\n", 4)]


def test_renders_front_matter_pkb_accepts():
    doc = render_pkb_document(transcript(), SUMMARY, ACTIONS)
    front = front_matter(doc)
    assert 'title: "Weekly sync"' in front
    assert 'created: "2026-10-01T08:00:00+00:00"' in front
    assert "  classification: private\n  cloud_ai: deny\n  local_ai: allow" in front
    assert "source:\n  type: plaud\n" in front
    assert f"  {OWNERSHIP_MARKER}\n" in front
    assert "  duration_seconds: 620\n" in front
    assert "exported_at" not in front


def test_renders_sections_actions_and_transcript_windows():
    doc = render_pkb_document(transcript(), SUMMARY, ACTIONS)
    assert "## Decisions\n\n- `00:01:01` Ship Friday\n" in doc
    assert "- [ ] `00:00:00` Write notes — Ana (due: Monday)" in doc
    assert "- [x] Book room" in doc
    assert "- [ ] Old idea (dismissed)" in doc
    assert "### 00:00:00\n\n- `00:00:03` **SPEAKER_00** Hello\n" in doc
    assert "### 00:05:00\n\n- `00:05:01` Later\n" in doc


def test_rendering_is_deterministic():
    first = render_pkb_document(transcript(), SUMMARY, ACTIONS)
    assert render_pkb_document(transcript(), SUMMARY, ACTIONS) == first


def test_without_summary_has_only_transcript():
    doc = render_pkb_document(transcript(), None, [])
    assert "## Summary" not in doc
    assert "summary_model" not in doc
    assert "## Transcript" in doc


def test_hostile_name_cannot_inject_front_matter():
    name = 'x"\nsecurity:\n  cloud_ai: allow\n---\n# owned\u2028\x00'
    doc = render_pkb_document(transcript(name=name), None, [])
    front_lines = front_matter(doc).splitlines()
    assert [line for line in front_lines if line.lstrip().startswith("cloud_ai:")] == ["  cloud_ai: deny"]
    assert [line for line in front_lines if line.startswith("security:")] == ["security:"]
    title_line = next(line for line in front_lines if line.startswith("title:"))
    assert title_line == 'title: "x\\" security: cloud_ai: allow --- # owned"'
    assert doc.count("\n---\n") == 1


def test_yaml_scalar_escapes_quotes_and_line_breaks():
    from plaud_brain.pkb_export import _yaml_str

    assert _yaml_str('a"b\\c') == '"a\\"b\\\\c"'
    assert _yaml_str("a\nb\u2028c\x00") == '"a\\u000ab\\u2028c\\u0000"'


def test_model_text_cannot_open_headings_or_fences():
    summary = {
        **SUMMARY,
        "content": {**SUMMARY["content"], "summary": "Intro\n# Fake heading\n```\ncode", "topics": ["a\n# b"]},
    }
    segments = [{"seq": 0, "start_ms": 0, "end_ms": 1, "speaker": "**X**\n#", "text": "line\n## not a heading"}]
    doc = render_pkb_document(transcript(segments=segments), summary, [])
    body_lines = doc.split("\n---\n", 1)[1].splitlines()
    headings = [line for line in body_lines if line.startswith("#")]
    assert headings == ["# Weekly sync", "## Summary", "## Topics", "## Decisions", "## Transcript", "### 00:00:00"]
    assert "\\# Fake heading" in doc
    assert "\\```" in doc
    assert "- `00:00:00` **X #** line ## not a heading" in doc


@pytest.mark.parametrize("plaud_id", ["../x", "a/b", "", "a b"])
def test_rejects_unsafe_identifiers(plaud_id):
    with pytest.raises(ValueError):
        export_filename(plaud_id)


def test_rejects_unknown_classification():
    with pytest.raises(ValueError):
        render_pkb_document(transcript(), None, [], classification="public")
    with pytest.raises(ValueError):
        PkbExporter(FakeStore({}), "unused", classification="secret")


class FakeStore:
    def __init__(self, recordings, failing=()):
        self.recordings = recordings
        self.failing = set(failing)

    def list_pkb_export_ids(self):
        return [{"plaud_id": plaud_id} for plaud_id in sorted(self.recordings)]

    def get_transcript(self, plaud_id):
        if plaud_id in self.failing:
            raise RuntimeError("database read failed")
        return self.recordings[plaud_id]

    def get_summary(self, plaud_id):
        return None

    def list_recording_actions(self, plaud_id):
        return []


def test_sync_writes_then_skips_unchanged(tmp_path):
    store = FakeStore({"rec_0000000001": transcript()})
    exporter = PkbExporter(store, tmp_path, file_mode=0o640)

    assert exporter.sync()["exported"] == 1
    path = tmp_path / "rec_0000000001.md"
    mtime = path.stat().st_mtime_ns
    assert stat.S_IMODE(path.stat().st_mode) == 0o640

    second = exporter.sync()
    assert second["exported"] == 0 and second["export_unchanged"] == 1
    assert path.stat().st_mtime_ns == mtime
    assert [p.name for p in tmp_path.iterdir()] == ["rec_0000000001.md"]


def test_sync_rewrites_changed_recording(tmp_path):
    store = FakeStore({"rec_0000000001": transcript()})
    exporter = PkbExporter(store, tmp_path)
    exporter.sync()
    store.recordings["rec_0000000001"] = transcript(name="Renamed")
    assert exporter.sync()["exported"] == 1
    assert 'title: "Renamed"' in (tmp_path / "rec_0000000001.md").read_text()


def test_sync_deletes_only_owned_stale_files(tmp_path):
    store = FakeStore({"rec_0000000001": transcript(), "rec_0000000002": transcript("rec_0000000002")})
    exporter = PkbExporter(store, tmp_path)
    exporter.sync()
    (tmp_path / "hand-written.md").write_text("---\ntitle: mine\n---\n# Mine\n")
    (tmp_path / "mentions-marker.md").write_text(f"# Note\n\n  {OWNERSHIP_MARKER}\n")

    del store.recordings["rec_0000000002"]
    result = exporter.sync()

    assert result["export_deleted"] == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "hand-written.md",
        "mentions-marker.md",
        "rec_0000000001.md",
    ]


def test_sync_keeps_files_after_a_failure_or_empty_database(tmp_path):
    store = FakeStore({"rec_0000000001": transcript(), "rec_0000000002": transcript("rec_0000000002")})
    PkbExporter(store, tmp_path).sync()

    del store.recordings["rec_0000000002"]
    store.failing = {"rec_0000000001"}
    result = PkbExporter(store, tmp_path).sync()
    assert result["export_failed"] == 1 and result["export_deleted"] == 0
    assert (tmp_path / "rec_0000000002.md").exists()

    store.recordings.clear()
    store.failing = set()
    assert PkbExporter(store, tmp_path).sync()["export_deleted"] == 0
    assert len(list(tmp_path.glob("*.md"))) == 2


def test_recording_without_segments_is_not_exported(tmp_path):
    store = FakeStore({"rec_0000000001": transcript(segments=[])})
    result = PkbExporter(store, tmp_path).sync()
    assert result["exported"] == 0
    assert list(tmp_path.iterdir()) == []


def test_sync_requires_real_existing_directory(tmp_path):
    store = FakeStore({"rec_0000000001": transcript()})
    with pytest.raises(ValueError):
        PkbExporter(store, tmp_path / "missing").sync()
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    os.symlink(target, link)
    with pytest.raises(ValueError):
        PkbExporter(store, link).sync()
