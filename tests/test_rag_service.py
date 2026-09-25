from contextlib import contextmanager
from datetime import date, datetime, timezone

import pytest

from plaud_brain.rag_service import RagService


class FakeStore:
    def __init__(self):
        self.signature = "2:now"
        self.cache = {}
        self.saved = []
        self.lock_entries = 0

    @contextmanager
    def rag_answer_lock(self, cache_key):
        self.lock_entries += 1
        yield

    def embedding_index_signature(self, provider, model):
        return self.signature

    def get_rag_answer(self, cache_key):
        response = self.cache.get(cache_key)
        return {"response": response} if response else None

    def save_rag_answer(self, **kwargs):
        self.saved.append(kwargs)
        self.cache[kwargs["cache_key"]] = kwargs["response"]


class FakeEmbeddings:
    provider = "embedding-provider"
    model = "embedding-model"

    def __init__(self, matches):
        self.matches = matches
        self.calls = []

    def search(self, query, limit, **filters):
        self.calls.append((query, limit, filters))
        return self.matches[:limit]


class FakeClient:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def generate_json(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return self.result


def _match(score=0.8):
    return {
        "source_type": "decision",
        "source_id": "decision-1",
        "plaud_id": "recording-1",
        "name": "Planning",
        "created_at": datetime(2026, 9, 1, tzinfo=timezone.utc),
        "timestamp_ms": 42000,
        "score": score,
        "text": "The team approved the launch.",
    }


def test_ask_generates_grounded_answer_and_reuses_cache():
    store = FakeStore()
    embeddings = FakeEmbeddings([_match()])
    client = FakeClient(
        {
            "answer": "The launch was approved. [S1]",
            "used_sources": ["S1"],
            "insufficient_evidence": False,
        }
    )
    service = RagService(store, embeddings, client, "deepseek", "flash")

    first = service.ask("What was decided?", source_types={"decision"})
    second = service.ask("What was decided?", source_types={"decision"})

    assert first["cached"] is False
    assert first["sources"][0]["id"] == "S1"
    assert "text" not in first["sources"][0]
    assert second["cached"] is True
    assert len(embeddings.calls) == 1
    assert len(client.calls) == 1
    assert store.lock_entries == 1


def test_cache_invalidates_when_index_changes():
    store = FakeStore()
    embeddings = FakeEmbeddings([_match()])
    client = FakeClient(
        {"answer": "Approved. [S1]", "used_sources": ["S1"], "insufficient_evidence": False}
    )
    service = RagService(store, embeddings, client, "deepseek", "flash")
    service.ask("Decision?")
    store.signature = "3:later"
    service.ask("Decision?")
    assert len(client.calls) == 2


def test_low_score_returns_deterministic_insufficient_answer():
    store = FakeStore()
    embeddings = FakeEmbeddings([_match(score=0.2)])
    client = FakeClient({})
    service = RagService(store, embeddings, client, "deepseek", "flash", min_score=0.5)

    result = service.ask("Unknown?")

    assert result["insufficient_evidence"] is True
    assert result["sources"] == []
    assert client.calls == []


@pytest.mark.parametrize(
    "response",
    [
        {"answer": "Claim [S9]", "used_sources": ["S9"], "insufficient_evidence": False},
        {"answer": "Uncited claim", "used_sources": [], "insufficient_evidence": False},
        {"answer": "Claim", "used_sources": ["S1"], "insufficient_evidence": False},
        {"answer": "No evidence [S1]", "used_sources": ["S1"], "insufficient_evidence": True},
    ],
)
def test_rejects_invalid_provider_citations(response):
    service = RagService(FakeStore(), FakeEmbeddings([_match()]), FakeClient(response), "deepseek", "flash")
    with pytest.raises(ValueError):
        service.ask("Question", refresh=True)


def test_passes_retrieval_filters():
    embeddings = FakeEmbeddings([_match()])
    service = RagService(
        FakeStore(),
        embeddings,
        FakeClient({"answer": "Yes [S1]", "used_sources": ["S1"], "insufficient_evidence": False}),
        "deepseek",
        "flash",
    )
    service.ask(
        "Question",
        recording="recording-1",
        source_types={"decision"},
        since=date(2026, 9, 1),
        until=date(2026, 9, 30),
    )
    filters = embeddings.calls[0][2]
    assert filters == {
        "recording": "recording-1",
        "source_types": {"decision"},
        "since": date(2026, 9, 1),
        "until": date(2026, 9, 30),
    }


def test_rejects_inverted_date_range():
    service = RagService(FakeStore(), FakeEmbeddings([_match()]), FakeClient({}), "deepseek", "flash")
    with pytest.raises(ValueError, match="since date"):
        service.ask("Question", since=date(2026, 9, 2), until=date(2026, 9, 1))
