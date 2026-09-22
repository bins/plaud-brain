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
