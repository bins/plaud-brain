# Architecture — M3

M0/M1 deliberately implements only ingestion and local transcription.

## Components

1. **PlaudCliAdapter** — executes the official `plaud` CLI and converts terminal output into internal records.
2. **AudioDownloader** — obtains the temporary audio URL from the adapter and stores content atomically.
3. **WhisperCppTranscriber** — normalizes source audio through ffmpeg and calls `whisper-cli` with JSON output.
4. **PostgresStore** — owns persistence, state transitions, transcripts, segments, and processing-run audit rows.
5. **Pipeline** — orchestration only. It contains no PLAUD-specific parsing and no STT-specific JSON details.

## State machine

```text
DISCOVERED
   | download
   v
DOWNLOADED
   | local STT
   v
TRANSCRIBED

Download failure -> FAILED_DOWNLOAD -> retry download
STT failure      -> FAILED_STT      -> retry STT
```

The next milestone will extend `TRANSCRIBED` rather than changing its meaning.

M2b adds a summary projection after `TRANSCRIBED`. Summary presence is derived
from the `summaries` table rather than overloading the recording state. Missing
summaries are retryable, and a successful stored summary is reused unless an
operator explicitly requests `--refresh`.

M2 completes the local knowledge layer by projecting decisions and action items
from summary JSON. PostgreSQL remains authoritative for action status. Literal
knowledge search spans transcript segments, summary text, decisions, and action
items.

M3 adds an incremental embedding projection and hybrid retrieval. Every source
item is keyed by its database identity and content hash. Changed and new items
are embedded in batches; unchanged vectors are reused and deleted sources are
pruned. Query ranking is cosine similarity plus a small exact-substring bonus.
The embedding provider is replaceable through environment configuration.

## Idempotency

- `recordings.plaud_id` is unique.
- `transcripts.recording_id` is unique.
- `transcript_segments(transcript_id, seq)` is unique.
- `embeddings(source_type, source_id, provider, model)` is unique.
- Downloads are written to a temporary file, checksummed, then atomically renamed.
- Existing valid audio is reused.
- A recording already in `TRANSCRIBED` is skipped unless a future explicit reprocess command is added.

## External assumptions

The official PLAUD CLI provides browse commands, recording IDs, `plaud file <id>`, and `plaud audio <id>` returning a temporary URL. CLI parsing is intentionally isolated because terminal formatting is not an API contract.

whisper.cpp is pinned to a known release and built with Vulkan on the target host. Source audio is normalized to PCM WAV before inference to avoid container/codec decode inconsistencies.
