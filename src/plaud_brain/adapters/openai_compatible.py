from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any


class CloudProviderError(RuntimeError):
    pass


class OpenAICompatibleClient:
    def __init__(
        self,
        provider: str,
        base_url: str,
        model: str,
        api_key_env: str,
        timeout_seconds: int = 300,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self._opener = opener or urllib.request.urlopen

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise CloudProviderError(
                f"Missing API key environment variable: {self.api_key_env}"
            )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "plaud-brain/0.1",
            },
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise CloudProviderError(
                f"{self.provider} API returned HTTP {exc.code}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise CloudProviderError(f"{self.provider} API request failed") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CloudProviderError(
                f"{self.provider} API returned an invalid response"
            ) from exc

        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise CloudProviderError(
                f"{self.provider} API response has no message content"
            ) from exc
        return _parse_json_content(content, self.provider)


def _parse_json_content(content: Any, provider: str) -> dict:
    if not isinstance(content, str):
        raise CloudProviderError(f"{provider} API returned non-text content")
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CloudProviderError(
            f"{provider} API message is not valid JSON"
        ) from exc
    if not isinstance(result, dict):
        raise CloudProviderError(f"{provider} API JSON must be an object")
    return result
