# Architecture decisions

## ADR-001 — PostgreSQL is source of truth
Accepted. Notion will be a later projection only.

## ADR-002 — M1 STT backend selection
Superseded on the current AMD Phoenix/RADV host. A real `large-v3-turbo`
transcription caused a compute-ring timeout, GPU reset, and
`vk::DeviceLostError`. M1 therefore supports an environment-controlled CPU
fallback and this host uses `WHISPER_NO_GPU=true`. Vulkan remains available for
explicit retesting; benchmark before changing the host default.

## ADR-003 — Pin whisper.cpp v1.9.4
Accepted for reproducibility. Upgrade only through an explicit tested change.

## ADR-004 — Normalize all PLAUD audio through ffmpeg
Accepted. This avoids relying on whisper.cpp support for every source container/codec.

## ADR-005 — Do not enable VAD in M1
Accepted. Keep the first timestamp contract to segment-level offsets in the original normalized audio timeline.

## ADR-006 — No cloud transcription in M0/M1
Accepted. Cloud fallback requires an explicit privacy policy in a later milestone.

## ADR-007 — M2a starts with local transcript retrieval
Accepted. Listing, rendering, exporting, and literal case-insensitive search
read only from PostgreSQL. No transcript or audio content leaves the host.
Summarization and external projections require a later explicit privacy and
provider decision.

## ADR-008 — Cloud summaries use a provider-neutral compatible adapter
Accepted with explicit user approval. DeepSeek `deepseek-flash` is the initial
default. Provider name, base URL, model, and API-key environment variable are
configuration, so another OpenAI-compatible cloud can be selected without
changing application code. Only transcript text and timestamps are sent; audio
stays local. API keys remain in the ignored `.env` file.

## ADR-009 — Phoenix Vulkan disables Flash Attention
Accepted. On Radeon 780M/Phoenix with RADV, the default Flash Attention path
reproducibly caused an amdgpu compute-ring timeout and GPU reset. The same model
and audio completed successfully with `-nfa`. GPU selection and Flash Attention
are therefore independent settings; this host should use
`WHISPER_NO_GPU=false` with `WHISPER_FLASH_ATTN=false` after validation.

## ADR-010 — Cloud STT is opt-in and locally recoverable
Accepted with explicit user approval as a post-M1 extension. OpenRouter Whisper
Large V3 Turbo is the initial cloud backend. Provider URL, model, and API-key
environment variable remain configuration. Long recordings are split into
bounded WAV chunks; completed chunk responses and the merged result are cached
for idempotent retries. `STT_FALLBACK_PROVIDER=local` keeps whisper.cpp as the
recovery path. Enabling cloud STT uploads recording audio to the selected
provider, while API keys stay only in the ignored `.env` file.

## ADR-011 — Summaries are a replaceable PostgreSQL projection
Accepted. A transcript remains the source record and has at most one current
structured summary. Provider/model/prompt version are stored with the JSON.
Automatic runs only fill missing summaries; replacing one requires an explicit
refresh. Summary failures are audited without changing `TRANSCRIBED` status.

## ADR-012 — M2 normalizes actionable knowledge without vector search
Accepted. Decisions and action items are relational projections of the current
summary. Action state is user-owned and survives refresh when the deterministic
content fingerprint still matches. M2 search is literal and local across all
knowledge text; embeddings and semantic retrieval are deferred until corpus
size and measured recall justify the operational complexity.

## ADR-013 — M3 uses provider-neutral incremental embeddings
Accepted. OpenRouter with `qwen/qwen3-embedding-8b` and 1024 dimensions is the
initial configuration, but provider URL, model, dimensions, and key environment
variable are configurable. Source content is hashed, so repeated indexing does
not resend unchanged private text or incur duplicate embedding cost.

## ADR-014 — Start with JSONB vectors and application cosine search
Accepted. The installed PostgreSQL 16 image does not expose the pgvector
extension, and the personal corpus is currently small. JSONB persistence plus
an in-process scan avoids an infrastructure migration. Move to pgvector when
measured index size or query latency warrants approximate/vector indexing.

## ADR-015 — M4 answers require validated source citations
Accepted. The QA model receives only the question and top retrieved sources.
It must return structured JSON with inline source IDs. Deterministic code rejects
unknown sources, uncited answers, mismatched inline citations, and citations on
an insufficient-evidence response. Source text is explicitly treated as
untrusted data to reduce prompt-injection risk.

## ADR-016 — Cache answers against the embedding index signature
Accepted. A cached answer is reusable only when the question, filters,
provider/model settings, prompt version, and embedding row count/latest update
are unchanged. Per-key PostgreSQL advisory locking prevents concurrent cache
misses from creating duplicate paid requests.

## ADR-017 — M5 is a loopback-first, dependency-free web layer
Accepted. The interface reuses the Python services and PostgreSQL authority,
binds to `127.0.0.1` by default, and is reached remotely through an SSH tunnel.
A non-loopback bind requires a configured Bearer token. Static assets are
served with a restrictive CSP, dynamic text is inserted with DOM text nodes,
request bodies and filters are bounded, and only two semantic/QA requests run
concurrently. Loopback mode rejects non-loopback Host headers to prevent DNS
rebinding. No transcript, token, or recording identifier is written to the HTTP
access log.

## ADR-018 — M6 stores diarization separately from transcription
Accepted. Speaker detection must not rewrite or rerun a valid transcript.
Provider output is stored as time-ranged anonymous speaker turns and projected
onto existing transcript segments by greatest temporal overlap. Results are
cached against the source audio checksum, provider, and model. The interface is
provider-neutral because local pyannote and cloud diarized transcription have
different privacy, cost, and hardware tradeoffs.

## ADR-019 — Enable local pyannote Community-1 for manual diarization
Proposed. Pending the M6a acceptance results recorded in
`docs/M6_RUNBOOK.md` ("M6a acceptance results").

Context: ADR-018 made diarization a separate, provider-neutral projection.
The local adapter for `pyannote/speaker-diarization-community-1` (CPU PyTorch
2.8.0, pyannote.audio 4.0.7) ran at about 0.47–0.54× realtime, peaking at
about 2.3 GiB RSS on 60 s and 300 s samples on bmax-ai, and cache reuse
avoided any repeated model run.

Decision (to confirm): set `DIARIZATION_PROVIDER=pyannote` on bmax-ai for
operator-invoked `plaud-brain diarize <plaud-id>` only. Speaker labels stay
anonymous (`SPEAKER_00`, ...) and are not identities. Audio stays on the host;
the Hugging Face token is used only to download the model and lives only in
`.env`. The cloud alternative (`gpt-4o-transcribe-diarize`) is not adopted,
because it uploads audio and transcribes again.

Out of scope for this decision: automatic diarization in `run-once`,
backfill of existing recordings, regenerating summaries after diarization, and
speaker naming. Each needs its own decision after this one is accepted.

Acceptance evidence required before changing the status to Accepted:
- every row of the M6a acceptance table is `Pass`;
- the long-recording row shows memory and wall time acceptable to run
  alongside the 15-minute ingestion timer;
- the manual speaker-count check on a known multi-speaker sample is
  plausible;
- normalized audio under `data/work/<plaud-id>/diarization/` is removed
  after each run, or a retention rule is written here.

Rollback: set `DIARIZATION_PROVIDER=disabled`. Stored transcripts are never
modified by diarization except for the nullable `speaker` projection, which
can be cleared without touching STT text or timestamps.

## ADR-020 — Opt-in automatic diarization in run-once
Proposed. Depends on ADR-019.

Decision (to confirm): `DIARIZATION_AUTO=true` diarizes recordings without a
speaker projection during `run-once`, after STT and before automatic
summaries. Each run is bounded by `DIARIZATION_AUTO_LIMIT` (default 2) and
selects newest first so a backfill never delays fresh recordings. A recording is
skipped after `DIARIZATION_MAX_ATTEMPTS` unsuccessful runs, counting stale
`RUNNING` rows so an OOM-killed process cannot loop. A PostgreSQL advisory lock
per recording serializes manual and scheduled runs, and the normalized audio
copy is always removed. Diarization failures never block ingestion, summaries,
or indexing. Existing summaries are not regenerated here; see ADR-022.

Acceptance evidence: ADR-019 accepted; the long-recording row of the M6a table
shows that `DIARIZATION_AUTO_LIMIT` recordings fit inside `TimeoutStartSec=3h`
together with STT.

## ADR-021 — Feed PKB through a deterministic Markdown export
Accepted. PKB indexes canonical Markdown files, and its capture API keeps every
file immutable, which conflicts with summaries, speaker labels, and action
status that change after transcription. plaud-brain therefore owns a dedicated
export folder inside the PKB root and rewrites one file per recording whenever
its rendering changes. PostgreSQL remains authoritative; the files are a
regenerable projection. Exports never grant cloud AI access, write no
timestamps of their own, and delete only files carrying the exporter's
ownership marker after a complete, non-empty pass.

## ADR-022 — Refresh pre-diarization summaries only when no action was reviewed
Accepted. A summary older than its recording's diarization was generated
without speaker labels. `summarize-stale`, and opt-in
`SUMMARY_REFRESH_AFTER_DIARIZATION` in `run-once`, regenerate such summaries,
bounded by `SUMMARY_REFRESH_LIMIT` and by three unsuccessful attempts since the
latest diarization. Action identity includes the owner, and speaker labels make
owners likely to change, so a refresh could reset `DONE` or `DISMISSED` items
to `OPEN`. Automatic refresh therefore skips recordings with any reviewed
action, and the save re-checks this under row locks in the same transaction,
so a status change made during generation is never overwritten. Explicit
`summarize --refresh` keeps its existing behaviour.
