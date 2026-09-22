from __future__ import annotations

import json
import math
import numbers
import os
import time
import urllib.error
import urllib.request


class EmbeddingProviderError(RuntimeError):
    pass


class OpenAICompatibleEmbeddingClient:
    def __init__(
        self,
        provider: str,
        base_url: str,
        model: str,
        api_key_env: str,
        dimensions: int | None = None,
        timeout_seconds: int = 60,
        retries: int = 3,
    ) -> None:
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.dimensions = dimensions
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    def embed(self, texts: list[str], input_type: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise EmbeddingProviderError(f"missing API key environment variable: {self.api_key_env}")
        body: dict[str, object] = {"model": self.model, "input": texts}
        if self.dimensions is not None:
            body["dimensions"] = self.dimensions
        if input_type:
            body["input_type"] = input_type
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "plaud-brain/0.1",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = json.load(response)
                rows = sorted(payload.get("data", []), key=lambda row: row.get("index", 0))
                vectors = [row.get("embedding") for row in rows]
                if len(vectors) != len(texts) or any(not isinstance(vector, list) for vector in vectors):
                    raise EmbeddingProviderError("embedding provider returned an invalid response")
                dimensions = {len(vector) for vector in vectors}
                if len(dimensions) != 1 or 0 in dimensions:
                    raise EmbeddingProviderError("embedding provider returned inconsistent dimensions")
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, numbers.Real)
                    or not math.isfinite(float(value))
                    for vector in vectors
                    for value in vector
                ):
                    raise EmbeddingProviderError("embedding provider returned invalid vector values")
                return [[float(value) for value in vector] for vector in vectors]
            except urllib.error.HTTPError as exc:
                last_error = EmbeddingProviderError(
                    f"{self.provider} embeddings request failed with HTTP {exc.code}"
                )
                if exc.code < 500 and exc.code != 429:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = EmbeddingProviderError(
                    f"{self.provider} embeddings request failed: {exc.__class__.__name__}"
                )
            if attempt + 1 < self.retries:
                time.sleep(2**attempt)
        raise last_error or EmbeddingProviderError("embedding provider request failed")
