from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import os
import threading
from datetime import date
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse


log = logging.getLogger(__name__)
MAX_BODY_BYTES = 64 * 1024
SOURCE_TYPES = {"transcript", "summary", "decision", "action_item"}


class WebApplication:
    def __init__(self, store, embedding_service, rag_service, host: str, token_env: str) -> None:
        self.store = store
        self.embedding_service = embedding_service
        self.rag_service = rag_service
        self.token = os.getenv(token_env, "").strip()
        self.expensive_requests = threading.BoundedSemaphore(2)
        try:
            is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            is_loopback = host.lower() == "localhost"
        self.loopback = is_loopback
        if not is_loopback and not self.token:
            raise ValueError(f"{token_env} is required when WEB_HOST is not loopback")

    def host_allowed(self, host_header: str | None) -> bool:
        if not self.loopback:
            return True
        if not host_header:
            return False
        try:
            hostname = urlparse(f"//{host_header}").hostname
            if hostname is None:
                return False
            if hostname.lower() == "localhost":
                return True
            return ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            return False

    def authorized(self, authorization: str | None) -> bool:
        if not self.token:
            return True
        prefix = "Bearer "
        if not authorization or not authorization.startswith(prefix):
            return False
        return hmac.compare_digest(authorization[len(prefix) :], self.token)

    def dashboard(self) -> dict:
        return self.store.dashboard()

    def recordings(self) -> list[dict]:
        return self.store.list_recordings(100)

    def recording(self, plaud_id: str) -> dict:
        transcript = self.store.get_transcript(plaud_id)
        if transcript is None:
            raise LookupError("recording not found")
        summary = self.store.get_summary(plaud_id)
        return {"transcript": transcript, "summary": summary["content"] if summary else None}

    def actions(self, status: str) -> list[dict]:
        return self.store.list_action_items(status, 200)

    def update_action(self, action_id: str, status: str) -> dict:
        item = self.store.update_action_status(action_id, status)
        if item is None:
            raise LookupError("action item not found")
        return item

    def search(self, body: dict) -> list[dict]:
        query = _text(body, "query", max_length=2000)
        limit = _limit(body.get("limit", 20), maximum=100)
        filters = _filters(body)
        with self.expensive_requests:
            return self.embedding_service.search(query, limit=limit, **filters)

    def ask(self, body: dict) -> dict:
        question = _text(body, "question", max_length=4000)
        filters = _filters(body)
        refresh = body.get("refresh", False)
        if not isinstance(refresh, bool):
            raise ValueError("refresh must be boolean")
        with self.expensive_requests:
            return self.rag_service.ask(question, refresh=refresh, **filters)


def _text(body: dict, key: str, max_length: int) -> str:
    value = body.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must not be empty")
    value = value.strip()
    if len(value) > max_length:
        raise ValueError(f"{key} is too long")
    return value


def _limit(value, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"limit must be between 1 and {maximum}")
    return value


def _filters(body: dict) -> dict:
    recording = body.get("recording") or None
    if recording is not None and (not isinstance(recording, str) or len(recording) > 256):
        raise ValueError("recording is invalid")
    raw_types = body.get("source_types") or []
    if not isinstance(raw_types, list) or any(item not in SOURCE_TYPES for item in raw_types):
        raise ValueError("source_types is invalid")
    try:
        since = date.fromisoformat(body["since"]) if body.get("since") else None
        until = date.fromisoformat(body["until"]) if body.get("until") else None
    except (TypeError, ValueError) as exc:
        raise ValueError("dates must use YYYY-MM-DD") from exc
    if since and until and since > until:
        raise ValueError("since date must not be after until date")
    return {
        "recording": recording,
        "source_types": set(raw_types),
        "since": since,
        "until": until,
    }


class PlaudBrainServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, app: WebApplication):
        self.app = app
        super().__init__(address, handler)


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "plaud-brain"

    @property
    def app(self) -> WebApplication:
        return self.server.app

    def do_GET(self) -> None:
        if not self.app.host_allowed(self.headers.get("Host")):
            return self._error(HTTPStatus.MISDIRECTED_REQUEST, "host not allowed")
        path = urlparse(self.path).path
        if path == "/":
            return self._static(HTML, "text/html; charset=utf-8")
        if path == "/app.css":
            return self._static(CSS, "text/css; charset=utf-8")
        if path == "/app.js":
            return self._static(JS, "text/javascript; charset=utf-8")
        if not path.startswith("/api/"):
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        if not self._authorized():
            return
        try:
            if path == "/api/health":
                return self._json({"status": "ok"})
            if path == "/api/dashboard":
                return self._json(self.app.dashboard())
            if path == "/api/recordings":
                return self._json(self.app.recordings())
            if path.startswith("/api/recordings/"):
                plaud_id = unquote(path.removeprefix("/api/recordings/"))
                if not plaud_id or len(plaud_id) > 256 or "/" in plaud_id:
                    raise ValueError("recording id is invalid")
                return self._json(self.app.recording(plaud_id))
            if path.startswith("/api/actions/"):
                status = path.removeprefix("/api/actions/").upper()
                return self._json(self.app.actions(status))
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        except LookupError as exc:
            return self._error(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            return self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception:
            log.exception("web GET failed")
            return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "request failed")

    def do_POST(self) -> None:
        if not self.app.host_allowed(self.headers.get("Host")):
            return self._error(HTTPStatus.MISDIRECTED_REQUEST, "host not allowed")
        path = urlparse(self.path).path
        if not path.startswith("/api/"):
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        if not self._authorized():
            return
        try:
            body = self._body()
            if path == "/api/search":
                return self._json(self.app.search(body))
            if path == "/api/ask":
                return self._json(self.app.ask(body))
            if path.startswith("/api/actions/"):
                action_id = unquote(path.removeprefix("/api/actions/"))
                if not action_id or len(action_id) > 64 or "/" in action_id:
                    raise ValueError("action id is invalid")
                status = _text(body, "status", 20)
                return self._json(self.app.update_action(action_id, status))
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        except LookupError as exc:
            return self._error(HTTPStatus.NOT_FOUND, str(exc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._error(HTTPStatus.BAD_REQUEST, "invalid JSON body")
        except ValueError as exc:
            return self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception:
            log.exception("web POST failed")
            return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "request failed")

    def _body(self) -> dict:
        if self.headers.get_content_type() != "application/json":
            raise ValueError("content type must be application/json")
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length or "0")
        except ValueError as exc:
            raise ValueError("invalid content length") from exc
        if length < 1 or length > MAX_BODY_BYTES:
            raise ValueError("request body size is invalid")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def _authorized(self) -> bool:
        if self.app.authorized(self.headers.get("Authorization")):
            return True
        self._error(HTTPStatus.UNAUTHORIZED, "authorization required")
        return False

    def _json(self, value, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self._security_headers("application/json; charset=utf-8", len(body), no_store=True)
        self.end_headers()
        self.wfile.write(body)

    def _static(self, value: str, content_type: str) -> None:
        body = value.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self._security_headers(content_type, len(body), no_store=False)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json({"error": message}, status)

    def _security_headers(self, content_type: str, length: int, no_store: bool) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; connect-src 'self'; img-src 'self' data:; "
            "style-src 'self'; script-src 'self'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.send_header("Cache-Control", "no-store" if no_store else "private, max-age=300")

    def log_message(self, format: str, *args) -> None:
        # Paths may contain recording identifiers; keep them out of journald.
        return


def serve(app: WebApplication, host: str, port: int) -> None:
    server = PlaudBrainServer((host, port), RequestHandler, app)
    log.info("web interface listening on http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PLAUD Brain</title><link rel="stylesheet" href="/app.css"></head>
<body><header><div><span class="eyebrow">PRIVATE KNOWLEDGE</span><h1>PLAUD Brain</h1></div><div id="health" class="pill">Connecting…</div></header>
<main><section class="hero"><div><p class="kicker">Ask your recordings</p><h2>Turn conversations into clear next steps.</h2><p class="muted">Answers stay grounded in timestamped sources.</p></div>
<form id="ask-form"><textarea id="question" rows="3" maxlength="4000" placeholder="What did we decide about…?" required></textarea><div class="form-row"><select id="ask-type"><option value="">All sources</option><option value="decision">Decisions</option><option value="action_item">Action items</option><option value="summary">Summaries</option><option value="transcript">Transcripts</option></select><button type="submit">Ask</button></div></form></section>
<section id="answer" class="answer hidden" aria-live="polite"></section>
<section id="metrics" class="metrics"></section>
<section class="grid"><article><div class="section-head"><h2>Recordings</h2><button id="reload" class="ghost">Refresh</button></div><div id="recordings" class="list"></div></article>
<article><div class="section-head"><h2>Open actions</h2><span id="action-count" class="count"></span></div><div id="actions" class="list"></div></article></section>
<section class="search"><div class="section-head"><h2>Semantic search</h2></div><form id="search-form"><input id="search-query" maxlength="2000" placeholder="Find a topic, decision, or phrase" required><button type="submit">Search</button></form><div id="search-results" class="list"></div></section>
</main><dialog id="detail"><button id="close-detail" class="close" aria-label="Close">×</button><div id="detail-content"></div></dialog><script src="/app.js"></script></body></html>"""

CSS = """:root{color-scheme:dark;--bg:#0b1118;--panel:#121b25;--panel2:#172330;--line:#263647;--text:#edf4f7;--muted:#9eb0bd;--accent:#67e8c5;--accent2:#70a8ff;--danger:#ff8e8e}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% 0,#173044 0,transparent 32rem),var(--bg);color:var(--text);font:15px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif}header{max-width:1180px;margin:auto;padding:28px 28px 14px;display:flex;justify-content:space-between;align-items:center}h1,h2,h3,p{margin-top:0}h1{font-size:20px;letter-spacing:-.02em;margin:2px 0}.eyebrow,.kicker{font-size:11px;letter-spacing:.16em;color:var(--accent);font-weight:700}.pill,.count{border:1px solid var(--line);border-radius:999px;padding:5px 10px;color:var(--muted);font-size:12px}.pill.ok{color:var(--accent);border-color:#2f806d}main{max-width:1180px;margin:auto;padding:14px 28px 60px}.hero{display:grid;grid-template-columns:1fr 1.15fr;gap:40px;align-items:end;padding:42px;border:1px solid var(--line);background:linear-gradient(135deg,rgba(25,43,58,.96),rgba(15,25,35,.96));border-radius:24px;box-shadow:0 24px 80px #0005}.hero h2{font-size:38px;line-height:1.08;letter-spacing:-.04em;max-width:520px}.muted{color:var(--muted)}textarea,input,select{width:100%;background:#091018;color:var(--text);border:1px solid var(--line);border-radius:12px;padding:13px 14px;font:inherit;outline:none}textarea:focus,input:focus,select:focus{border-color:var(--accent2);box-shadow:0 0 0 3px #70a8ff22}.form-row,#search-form{display:flex;gap:10px;margin-top:10px}.form-row select{max-width:180px}button{border:0;border-radius:12px;background:var(--accent);color:#06251d;padding:11px 18px;font-weight:750;cursor:pointer}button:hover{filter:brightness(1.07)}button:disabled{opacity:.5;cursor:wait}.ghost{background:transparent;color:var(--muted);border:1px solid var(--line);padding:7px 11px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}.metric,article,.search,.answer{background:var(--panel);border:1px solid var(--line);border-radius:18px}.metric{padding:18px}.metric strong{font-size:27px;display:block}.metric span{color:var(--muted);font-size:12px}.grid{display:grid;grid-template-columns:1.15fr .85fr;gap:18px}article,.search{padding:22px}.section-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px}.section-head h2{font-size:18px;margin:0}.list{display:grid;gap:8px}.item{padding:13px 14px;background:var(--panel2);border:1px solid transparent;border-radius:12px}.item.clickable{cursor:pointer}.item.clickable:hover{border-color:var(--accent2)}.item h3{font-size:14px;margin-bottom:4px}.meta{color:var(--muted);font-size:12px}.action-row{display:flex;gap:12px;justify-content:space-between;align-items:flex-start}.action-row button{padding:6px 9px;font-size:11px}.search{margin-top:18px}.answer{margin-top:18px;padding:24px}.answer .answer-text{font-size:17px;white-space:pre-wrap}.sources{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px}.source{background:#0d1821;border:1px solid var(--line);border-radius:10px;padding:8px 10px;font-size:12px;color:var(--muted)}.hidden{display:none}.error{color:var(--danger)}dialog{width:min(840px,calc(100% - 30px));max-height:85vh;overflow:auto;background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:18px;padding:28px}dialog::backdrop{background:#020609cc}.close{position:sticky;float:right;top:0;background:var(--panel2);color:var(--text);font-size:22px;padding:2px 10px}.segment{padding:10px 0;border-top:1px solid var(--line)}@media(max-width:760px){header,main{padding-left:16px;padding-right:16px}.hero,.grid{grid-template-columns:1fr}.hero{padding:26px;gap:20px}.hero h2{font-size:30px}.metrics{grid-template-columns:repeat(2,1fr)}.form-row{align-items:stretch}.form-row select{max-width:none}#search-form,.form-row{flex-direction:column}}"""

JS = """let token='';const $=s=>document.querySelector(s);const el=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n};
async function api(path,options={}){options.headers={...(options.headers||{})};if(token)options.headers.Authorization=`Bearer ${token}`;let r=await fetch(path,options);if(r.status===401&&!token){token=prompt('Access token')||'';if(token)return api(path,options)}const data=await r.json();if(!r.ok)throw new Error(data.error||'Request failed');return data}
const fmtTime=ms=>{const s=Math.floor((ms||0)/1000);return [Math.floor(s/3600),Math.floor(s%3600/60),s%60].map(v=>String(v).padStart(2,'0')).join(':')};
function showError(target,error){target.replaceChildren(el('p','error',error.message))}
async function load(){try{await api('/api/health');$('#health').textContent='System healthy';$('#health').classList.add('ok');const [d,recs,acts]=await Promise.all([api('/api/dashboard'),api('/api/recordings'),api('/api/actions/OPEN')]);renderMetrics(d);renderRecordings(recs);renderActions(acts)}catch(e){$('#health').textContent='Unavailable';showError($('#recordings'),e)}}
function renderMetrics(d){const values=[['Recordings',d.recordings],['Transcripts',d.transcripts],['Open actions',d.open_actions],['Knowledge items',d.embeddings]];$('#metrics').replaceChildren(...values.map(([k,v])=>{const n=el('div','metric');n.append(el('strong','',String(v)),el('span','',k));return n}))}
function renderRecordings(rows){$('#recordings').replaceChildren(...rows.map(r=>{const n=el('div','item clickable');n.tabIndex=0;n.append(el('h3','',r.name||r.plaud_id),el('div','meta',`${r.language||'unknown'} · ${r.segment_count} segments · ${r.summarized?'summarized':'no summary'}`));n.onclick=()=>openRecording(r.plaud_id);n.onkeydown=e=>{if(e.key==='Enter')n.click()};return n}))}
function renderActions(rows){$('#action-count').textContent=String(rows.length);$('#actions').replaceChildren(...rows.map(a=>{const n=el('div','item action-row');const text=el('div');text.append(el('h3','',a.task),el('div','meta',[a.owner,a.due].filter(Boolean).join(' · ')||'No owner or due date'));const b=el('button','ghost','Done');b.onclick=async()=>{b.disabled=true;try{await api(`/api/actions/${encodeURIComponent(a.id)}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({status:'done'})});load()}catch(e){b.disabled=false;alert(e.message)}};n.append(text,b);return n}))}
async function openRecording(id){const dlg=$('#detail');$('#detail-content').replaceChildren(el('p','muted','Loading…'));dlg.showModal();try{const d=await api(`/api/recordings/${encodeURIComponent(id)}`);const box=$('#detail-content');box.replaceChildren();box.append(el('h2','',d.transcript.name||d.transcript.plaud_id));if(d.summary){box.append(el('h3','','Summary'),el('p','',d.summary.summary))}box.append(el('h3','','Transcript'));for(const s of d.transcript.segments){const n=el('div','segment');n.append(el('div','meta',fmtTime(s.start_ms)),el('div','',s.text));box.append(n)}}catch(e){showError($('#detail-content'),e)}}
$('#close-detail').onclick=()=>$('#detail').close();$('#reload').onclick=load;
$('#ask-form').onsubmit=async e=>{e.preventDefault();const out=$('#answer'),button=e.submitter;out.classList.remove('hidden');out.replaceChildren(el('p','muted','Searching your recordings…'));button.disabled=true;try{const type=$('#ask-type').value;const body={question:$('#question').value,source_types:type?[type]:[]};const r=await api('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const text=el('div','answer-text',r.answer);const sources=el('div','sources');for(const s of r.sources)sources.append(el('div','source',`[${s.id}] ${s.name||s.plaud_id} · ${s.source_type}${s.timestamp_ms!==null?' · '+fmtTime(s.timestamp_ms):''}`));out.replaceChildren(text,sources,el('div','meta',r.cached?'Cached answer':'Fresh answer'))}catch(err){showError(out,err)}finally{button.disabled=false}};
$('#search-form').onsubmit=async e=>{e.preventDefault();const out=$('#search-results'),button=e.submitter;out.replaceChildren(el('p','muted','Searching…'));button.disabled=true;try{const rows=await api('/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:$('#search-query').value,limit:20})});out.replaceChildren(...rows.map(r=>{const n=el('div','item');n.append(el('div','meta',`${r.source_type} · ${(r.score||0).toFixed(3)} · ${r.timestamp_ms!==null?fmtTime(r.timestamp_ms):'no timestamp'}`),el('div','',r.text));return n}))}catch(err){showError(out,err)}finally{button.disabled=false}};load();"""
