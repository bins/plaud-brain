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
