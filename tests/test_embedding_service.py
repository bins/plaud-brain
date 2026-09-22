from contextlib import contextmanager

from plaud_brain.embedding_service import EmbeddingService


class FakeClient:
    def __init__(self):
        self.calls = []

    def embed(self, texts, input_type=None):
        self.calls.append((texts, input_type))
        vectors = {
            "alpha": [1.0, 0.0],
            "beta": [0.0, 1.0],
            "alpha changed": [0.8, 0.2],
            "find alpha": [1.0, 0.0],
        }
        return [vectors[text] for text in texts]


class FakeStore:
    def __init__(self):
        self.sources = [
            {
                "source_type": "transcript",
                "source_id": "1",
                "recording_id": "r1",
                "plaud_id": "p1",
                "name": "One",
                "timestamp_ms": 0,
                "text": "alpha",
            },
            {
                "source_type": "decision",
                "source_id": "2",
                "recording_id": "r1",
                "plaud_id": "p1",
                "name": "One",
                "timestamp_ms": 10,
                "text": "beta",
            },
        ]
        self.rows = {}
        self.lock_entries = 0

    @contextmanager
    def embedding_sync_lock(self):
        self.lock_entries += 1
        yield

    def list_embedding_sources(self):
        return [dict(row) for row in self.sources]

    def list_embedding_metadata(self, provider, model):
        return [
            {
                "source_type": key[0],
                "source_id": key[1],
                "content_hash": row["content_hash"],
                "dimensions": row["dimensions"],
            }
            for key, row in self.rows.items()
        ]

    def upsert_embedding(self, source, provider, model, dimensions, content_hash, embedding):
        self.rows[(source["source_type"], source["source_id"])] = {
            **source,
            "dimensions": dimensions,
            "content_hash": content_hash,
            "embedding": embedding,
        }

    def delete_embeddings_for_model(self, provider, model):
        count = len(self.rows)
        self.rows.clear()
        return count

    def delete_stale_embeddings(self, provider, model, current_keys):
        stale = [key for key in self.rows if key not in current_keys]
        for key in stale:
            del self.rows[key]
        return len(stale)

    def list_embeddings(self, provider, model):
        return [
            {
                "source_type": row["source_type"],
                "source_id": row["source_id"],
                "plaud_id": row["plaud_id"],
                "name": row["name"],
                "timestamp_ms": row["timestamp_ms"],
                "text": row["text"],
                "embedding": row["embedding"],
            }
            for row in self.rows.values()
        ]


def test_sync_is_incremental_and_removes_stale_rows():
    store = FakeStore()
    client = FakeClient()
    service = EmbeddingService(store, client, "test", "model", 2, batch_size=8)

    assert service.sync() == {"sources": 2, "indexed": 2, "reused": 0, "deleted": 0}
    assert service.sync() == {"sources": 2, "indexed": 0, "reused": 2, "deleted": 0}
    assert len(client.calls) == 1
    assert store.lock_entries == 2

    store.sources[0]["text"] = "alpha changed"
    store.sources.pop()
    assert service.sync() == {"sources": 1, "indexed": 1, "reused": 0, "deleted": 1}


def test_search_combines_semantic_score_and_literal_match():
    store = FakeStore()
    client = FakeClient()
    service = EmbeddingService(store, client, "test", "model", 2)
    service.sync()

    matches = service.search("find alpha", limit=2)

    assert matches[0]["text"] == "alpha"
    assert matches[0]["semantic_score"] == 1.0
    assert matches[0]["exact_match"] is False
    assert "embedding" not in matches[0]

    literal_matches = service.search("alpha", limit=2)
    assert literal_matches[0]["exact_match"] is True
    assert literal_matches[0]["score"] == 1.08


def test_service_rejects_invalid_configuration():
    store = FakeStore()
    client = FakeClient()
    try:
        EmbeddingService(store, client, "test", "model", 0)
    except ValueError as exc:
        assert "dimensions" in str(exc)
    else:
        raise AssertionError("zero dimensions must be rejected")

    try:
        EmbeddingService(store, client, "test", "model", 2, batch_size=0)
    except ValueError as exc:
        assert "batch size" in str(exc)
    else:
        raise AssertionError("zero batch size must be rejected")
