# Data model — M5

M5 adds no tables. The local web interface reads and updates the existing
PostgreSQL model through the same store methods used by the CLI.

## recordings
One row per PLAUD recording. `plaud_id` is the external idempotency key.

Important fields: name, created_at, duration_seconds, status, audio_path, audio_sha256, error fields.

## transcripts
One row per recording in M1. Stores STT engine/model/language and the raw whisper JSON for future re-parsing.

## transcript_segments
Ordered, timestamped segments. Speaker is nullable because diarization is M6 scope.

## summaries
One current structured summary per transcript. Stores provider, model, prompt
version, and JSON content. The unique transcript key makes automatic summary
generation idempotent; an explicit refresh replaces the current content.

## decisions
Ordered decisions projected from a structured summary, linked back to its
recording through the summary and transcript. Optional timestamps reference the
original audio timeline.

## action_items
Ordered actionable work projected from a summary. Each item has a deterministic
fingerprint and a mutable `OPEN`, `DONE`, or `DISMISSED` status. Matching items
retain identity and status when the summary is refreshed.

## processing_runs
Audit trail of each download, STT, and summary attempt, including failures.

## embeddings
Incremental semantic-search projection for transcript segments, summaries,
decisions, and action items. Rows store the source identity, recording link,
provider/model/dimensions, content hash, text snapshot, and JSONB vector.
The content hash prevents paid recomputation when source text is unchanged.

## rag_answers
Cached grounded answers keyed by a SHA-256 digest of the normalized question,
filters, QA/embedding models, prompt version, and embedding-index signature.
The stored JSON contains only validated public citation metadata; retrieved
evidence text is sent to the provider but is not duplicated into this table.
