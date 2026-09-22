from __future__ import annotations

import hashlib
import math


class EmbeddingService:
    def __init__(
        self,
        store,
        client,
        provider: str,
        model: str,
        dimensions: int,
        batch_size: int = 32,
        document_input_type: str | None = None,
        query_input_type: str | None = None,
    ) -> None:
        if dimensions < 1:
            raise ValueError("embedding dimensions must be positive")
        if batch_size < 1:
            raise ValueError("embedding batch size must be positive")
        self.store = store
        self.client = client
        self.provider = provider
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.document_input_type = document_input_type
        self.query_input_type = query_input_type

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def sync(self, rebuild: bool = False) -> dict[str, int]:
        with self.store.embedding_sync_lock():
            return self._sync_locked(rebuild)

    def _sync_locked(self, rebuild: bool = False) -> dict[str, int]:
        sources = self.store.list_embedding_sources()
        if rebuild:
            self.store.delete_embeddings_for_model(self.provider, self.model)
        existing = {
            (row["source_type"], row["source_id"]): row
            for row in self.store.list_embedding_metadata(self.provider, self.model)
        }
        current_keys = {(row["source_type"], row["source_id"]) for row in sources}
        pending = []
        for source in sources:
            source["content_hash"] = self._hash(source["text"])
            cached = existing.get((source["source_type"], source["source_id"]))
            if (
                cached is None
                or cached["content_hash"] != source["content_hash"]
                or cached["dimensions"] != self.dimensions
            ):
                pending.append(source)

        indexed = 0
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            vectors = self.client.embed(
                [row["text"] for row in batch], input_type=self.document_input_type
            )
            if len(vectors) != len(batch):
                raise ValueError("embedding provider returned the wrong number of vectors")
            for source, vector in zip(batch, vectors):
                if len(vector) != self.dimensions:
                    raise ValueError(
                        f"embedding dimensions mismatch: expected {self.dimensions}, got {len(vector)}"
                    )
                self.store.upsert_embedding(
                    source,
                    self.provider,
                    self.model,
                    self.dimensions,
                    source["content_hash"],
                    vector,
                )
                indexed += 1

        deleted = self.store.delete_stale_embeddings(
            self.provider, self.model, current_keys
        )
        return {"sources": len(sources), "indexed": indexed, "reused": len(sources) - indexed, "deleted": deleted}

    def search(self, query: str, limit: int = 20) -> list[dict]:
        query = query.strip()
        if not query:
            raise ValueError("search query must not be empty")
        vector = self.client.embed([query], input_type=self.query_input_type)[0]
        if len(vector) != self.dimensions:
            raise ValueError(
                f"embedding dimensions mismatch: expected {self.dimensions}, got {len(vector)}"
            )
        query_norm = math.sqrt(math.fsum(value * value for value in vector))
        if query_norm == 0:
            raise ValueError("embedding provider returned a zero query vector")
        needle = query.casefold()
        results = []
        for row in self.store.list_embeddings(self.provider, self.model):
            candidate = row.pop("embedding")
            if len(candidate) != len(vector):
                continue
            candidate_norm = math.sqrt(math.fsum(value * value for value in candidate))
            cosine = (
                math.fsum(a * b for a, b in zip(vector, candidate))
                / (query_norm * candidate_norm)
                if candidate_norm
                else 0.0
            )
            exact_match = needle in row["text"].casefold()
            results.append(
                {
                    **row,
                    "semantic_score": round(cosine, 6),
                    "exact_match": exact_match,
                    "score": round(cosine + (0.08 if exact_match else 0.0), 6),
                }
            )
        results.sort(key=lambda item: (item["score"], item["semantic_score"]), reverse=True)
        return results[:limit]
