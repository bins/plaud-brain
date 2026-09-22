import json

import pytest

from plaud_brain.adapters.embeddings import (
    EmbeddingProviderError,
    OpenAICompatibleEmbeddingClient,
)


class Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(self.payload).encode()


def test_embedding_client_orders_response_and_sends_dimensions(monkeypatch):
    monkeypatch.setenv("TEST_EMBEDDING_KEY", "secret")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data)
        return Response(
            {"data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]}
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleEmbeddingClient(
        "test", "https://example.test/v1", "embed", "TEST_EMBEDDING_KEY", dimensions=2
    )

    assert client.embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert captured["body"] == {"model": "embed", "input": ["a", "b"], "dimensions": 2}


def test_embedding_client_requires_key(monkeypatch):
    monkeypatch.delenv("MISSING_EMBEDDING_KEY", raising=False)
    client = OpenAICompatibleEmbeddingClient(
        "test", "https://example.test/v1", "embed", "MISSING_EMBEDDING_KEY"
    )
    with pytest.raises(EmbeddingProviderError, match="missing API key"):
        client.embed(["a"])


@pytest.mark.parametrize("value", ["not-a-number", True, float("nan"), float("inf")])
def test_embedding_client_rejects_invalid_vector_values(monkeypatch, value):
    monkeypatch.setenv("TEST_EMBEDDING_KEY", "secret")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: Response({"data": [{"index": 0, "embedding": [value]}]}),
    )
    client = OpenAICompatibleEmbeddingClient(
        "test", "https://example.test/v1", "embed", "TEST_EMBEDDING_KEY"
    )
    with pytest.raises(EmbeddingProviderError, match="invalid vector values"):
        client.embed(["a"])
