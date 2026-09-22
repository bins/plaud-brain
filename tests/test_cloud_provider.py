import json
import urllib.error

import pytest

from plaud_brain.adapters.openai_compatible import CloudProviderError, OpenAICompatibleClient
from plaud_brain.summarizer import _normalize_summary, render_summary_markdown, summarize_transcript


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_openai_compatible_client_uses_configured_endpoint(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "secret-value")
    captured = {}

    def opener(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.headers["Authorization"]
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "summary": "Done",
                                    "topics": [],
                                    "decisions": [],
                                    "action_items": [],
                                }
                            )
                        }
                    }
                ]
            }
        )

    client = OpenAICompatibleClient(
        provider="deepseek",
        base_url="https://api.deepseek.com/",
        model="deepseek-flash",
        api_key_env="TEST_PROVIDER_KEY",
        timeout_seconds=42,
        opener=opener,
    )
    result = client.generate_json("system", "user")

    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["authorization"] == "Bearer secret-value"
    assert captured["payload"]["model"] == "deepseek-flash"
    assert captured["timeout"] == 42
    assert result["summary"] == "Done"


def test_client_requires_configured_key(monkeypatch):
    monkeypatch.delenv("MISSING_PROVIDER_KEY", raising=False)
    client = OpenAICompatibleClient(
        "custom",
        "https://example.invalid",
        "model",
        "MISSING_PROVIDER_KEY",
    )
    with pytest.raises(CloudProviderError, match="MISSING_PROVIDER_KEY"):
        client.generate_json("system", "user")


def test_client_does_not_expose_http_error_body(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "secret-value")

    def opener(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            {},
            None,
        )

    client = OpenAICompatibleClient(
        "deepseek",
        "https://api.deepseek.com",
        "deepseek-flash",
        "TEST_PROVIDER_KEY",
        opener=opener,
    )
    with pytest.raises(CloudProviderError, match="HTTP 401"):
        client.generate_json("system", "private transcript")


def test_summarize_and_render_markdown():
    class FakeClient:
        def generate_json(self, system_prompt, user_prompt):
            assert "private meeting" in system_prompt
            assert "Hello" in user_prompt
            return {
                "summary": "A short summary",
                "topics": ["Planning"],
                "decisions": [{"text": "Ship it", "timestamp_ms": 0}],
                "action_items": [
                    {
                        "task": "Write tests",
                        "owner": "Alice",
                        "due": None,
                        "timestamp_ms": 1000,
                    }
                ],
            }

    transcript = {
        "plaud_id": "recording-1",
        "name": "Meeting",
        "language": "en",
        "segments": [
            {"start_ms": 0, "speaker": None, "text": "Hello"},
        ],
    }
    summary = summarize_transcript(FakeClient(), transcript)
    rendered = render_summary_markdown(transcript, summary)

    assert summary["topics"] == ["Planning"]
    assert "# Meeting" in rendered
    assert "Write tests — Alice" in rendered


@pytest.mark.parametrize(
    "payload",
    [
        {"summary": "ok", "topics": [123]},
        {"summary": "ok", "decisions": [{"timestamp_ms": 0}]},
        {"summary": "ok", "action_items": [{"task": ""}]},
        {"summary": "ok", "action_items": [{"task": "Do it", "owner": 123}]},
        {"summary": "ok", "decisions": [{"text": "Decide", "timestamp_ms": -1}]},
    ],
)
def test_summary_normalization_rejects_invalid_nested_values(payload):
    with pytest.raises(ValueError):
        _normalize_summary(payload)
