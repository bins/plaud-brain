import json
import threading
import urllib.error
import urllib.request

import pytest

from plaud_brain.web import HTML, JS, PlaudBrainServer, RequestHandler, WebApplication


class FakeStore:
    def dashboard(self):
        return {"recordings": 2, "transcripts": 2, "summaries": 2, "open_actions": 1, "decisions": 3, "embeddings": 10}

    def list_recordings(self, limit):
        return [{"plaud_id": "recording-1", "name": "One"}]

    def get_transcript(self, plaud_id):
        return {"plaud_id": plaud_id, "segments": [{"text": "private"}]}

    def get_summary(self, plaud_id):
        return {"content": {"summary": "summary"}}

    def list_action_items(self, status, limit):
        return [{"id": "action-1", "status": status}]

    def update_action_status(self, action_id, status):
        return {"id": action_id, "status": status.upper()}


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    def search(self, query, limit, **filters):
        self.calls.append((query, limit, filters))
        return [{"text": "match", "score": 0.8}]


class FakeRag:
    def __init__(self):
        self.calls = []

    def ask(self, question, **kwargs):
        self.calls.append((question, kwargs))
        return {"answer": "answer", "sources": [], "cached": False}


@pytest.fixture
def app(monkeypatch):
    monkeypatch.delenv("TEST_WEB_TOKEN", raising=False)
    return WebApplication(FakeStore(), FakeEmbeddings(), FakeRag(), "127.0.0.1", "TEST_WEB_TOKEN")


@pytest.fixture
def server(app):
    httpd = PlaudBrainServer(("127.0.0.1", 0), RequestHandler, app)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=2)


def _json_request(url, payload):
    return urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def test_loopback_server_health_and_security_headers(server):
    with urllib.request.urlopen(f"{server}/api/health") as response:
        assert json.load(response) == {"status": "ok"}
        assert response.headers["X-Frame-Options"] == "DENY"
        assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
        assert response.headers["Cache-Control"] == "no-store"


def test_web_ask_and_filters(server, app):
    request = _json_request(
        f"{server}/api/ask",
        {
            "question": "What happened?",
            "source_types": ["decision"],
            "since": "2026-09-01",
            "until": "2026-09-30",
        },
    )
    with urllib.request.urlopen(request) as response:
        assert json.load(response)["answer"] == "answer"
    _, kwargs = app.rag_service.calls[0]
    assert kwargs["source_types"] == {"decision"}
    assert kwargs["since"].isoformat() == "2026-09-01"


def test_non_loopback_requires_token(monkeypatch):
    monkeypatch.delenv("TEST_WEB_TOKEN", raising=False)
    with pytest.raises(ValueError, match="TEST_WEB_TOKEN"):
        WebApplication(FakeStore(), FakeEmbeddings(), FakeRag(), "0.0.0.0", "TEST_WEB_TOKEN")


def test_configured_token_protects_api(monkeypatch):
    monkeypatch.setenv("TEST_WEB_TOKEN", "a-secure-test-token")
    app = WebApplication(FakeStore(), FakeEmbeddings(), FakeRag(), "127.0.0.1", "TEST_WEB_TOKEN")
    httpd = PlaudBrainServer(("127.0.0.1", 0), RequestHandler, app)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{httpd.server_port}/api/health"
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(url)
        assert exc.value.code == 401
        request = urllib.request.Request(url, headers={"Authorization": "Bearer a-secure-test-token"})
        with urllib.request.urlopen(request) as response:
            assert response.status == 200
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_rejects_invalid_source_type(server):
    request = _json_request(
        f"{server}/api/search",
        {"query": "test", "source_types": ["unknown"]},
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(request)
    assert exc.value.code == 400


def test_rejects_dns_rebinding_host(server):
    request = urllib.request.Request(
        f"{server}/api/health",
        headers={"Host": "attacker.example"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(request)
    assert exc.value.code == 421


def test_ui_avoids_html_injection_sinks():
    assert "innerHTML" not in JS
    assert "eval(" not in JS
    assert '<script src="/app.js"></script>' in HTML
