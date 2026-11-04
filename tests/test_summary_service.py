import pytest

from plaud_brain.summary_service import SummaryService


class FakeStore:
    def __init__(self):
        self.cached = None
        self.saved = []
        self.runs = []

    def get_transcript(self, plaud_id):
        if plaud_id == "missing":
            return None
        return {
            "recording_id": "recording-id",
            "transcript_id": "transcript-id",
            "plaud_id": plaud_id,
            "name": "Note",
            "language": "en",
            "segments": [{"start_ms": 0, "speaker": None, "text": "Hello"}],
        }

    def get_summary(self, plaud_id):
        return self.cached

    def start_run(self, recording_id, stage):
        self.runs.append((recording_id, stage, "started"))
        return "run-id"

    def finish_run(self, run_id, success, code=None, message=None):
        self.runs.append((run_id, success, code))

    def save_summary(self, **kwargs):
        self.saved.append(kwargs)

    def pending_summaries(self, limit):
        return [{"plaud_id": "one"}, {"plaud_id": "two"}][:limit]


class FakeClient:
    def __init__(self):
        self.calls = 0

    def generate_json(self, system_prompt, user_prompt):
        self.calls += 1
        return {"summary": "Done", "topics": [], "decisions": [], "action_items": []}


def test_summary_service_reuses_cached_result():
    store = FakeStore()
    store.cached = {"content": {"summary": "Cached", "topics": [], "decisions": [], "action_items": []}}
    client = FakeClient()
    summary, generated = SummaryService(store, client, "deepseek", "model").summarize_one("one")
    assert summary["summary"] == "Cached"
    assert generated is False
    assert client.calls == 0


def test_summary_service_generates_and_persists():
    store = FakeStore()
    client = FakeClient()
    summary, generated = SummaryService(store, client, "deepseek", "model").summarize_one("one")
    assert summary["summary"] == "Done"
    assert generated is True
    assert store.saved[0]["transcript_id"] == "transcript-id"
    assert store.runs[-1] == ("run-id", True, None)


def test_summary_service_reports_missing_transcript():
    with pytest.raises(ValueError, match="Transcript not found"):
        SummaryService(FakeStore(), FakeClient(), "deepseek", "model").summarize_one("missing")


class RefreshStore(FakeStore):
    def __init__(self, pending, reviewed=(), broken=()):
        super().__init__()
        self.cached = {"content": {"summary": "Old", "topics": [], "decisions": [], "action_items": []}}
        self.pending = pending
        self.reviewed = set(reviewed)
        self.broken = set(broken)
        self.queries = []

    def pending_speaker_summary_refreshes(self, max_attempts, limit):
        self.queries.append((max_attempts, limit))
        return [{"plaud_id": plaud_id} for plaud_id in self.pending[:limit]]

    def get_transcript(self, plaud_id):
        if plaud_id in self.broken:
            raise RuntimeError("database read failed")
        return {**super().get_transcript(plaud_id), "transcript_id": f"transcript-{plaud_id}"}

    def save_summary(self, **kwargs):
        from plaud_brain.store import ReviewedActionsError

        if kwargs["require_unreviewed_actions"] and kwargs["transcript_id"] in {
            f"transcript-{plaud_id}" for plaud_id in self.reviewed
        }:
            raise ReviewedActionsError("reviewed")
        super().save_summary(**kwargs)


def test_refresh_regenerates_cached_summaries_with_the_review_guard():
    store = RefreshStore(["one", "two"])
    client = FakeClient()
    counts = SummaryService(store, client, "deepseek", "model").refresh_after_diarization(5, 3)
    assert counts == {"summary_refreshed": 2, "summary_refresh_skipped": 0, "summary_refresh_failed": 0}
    assert client.calls == 2
    assert all(saved["require_unreviewed_actions"] for saved in store.saved)
    assert store.queries == [(3, 5)]


def test_refresh_skips_reviewed_actions_and_continues_after_failures():
    store = RefreshStore(["reviewed", "broken", "fine"], reviewed={"reviewed"}, broken={"broken"})
    counts = SummaryService(store, FakeClient(), "deepseek", "model").refresh_after_diarization(5, 3)
    assert counts == {"summary_refreshed": 1, "summary_refresh_skipped": 1, "summary_refresh_failed": 1}
    assert [saved["transcript_id"] for saved in store.saved] == ["transcript-fine"]
    # The refused save is recorded as an unsuccessful run, not left RUNNING.
    assert ("run-id", False, "ReviewedActionsError") in store.runs


def test_plain_summaries_do_not_use_the_review_guard():
    store = FakeStore()
    SummaryService(store, FakeClient(), "deepseek", "model").summarize_one("one", refresh=True)
    assert store.saved[0]["require_unreviewed_actions"] is False
