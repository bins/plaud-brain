# M2 runbook

## Acceptance checks

1. `plaud-brain recordings` reports `summarized: true` for processed recordings.
2. `plaud-brain show <plaud-id> --format markdown` renders summary and transcript.
3. `plaud-brain decisions` lists structured decisions with source recording IDs.
4. `plaud-brain actions --status open` lists actionable work.
5. `plaud-brain action <uuid> done` persists completion.
6. `plaud-brain knowledge-search <query>` searches transcript, summary, decision,
   and action-item text without calling a cloud service.
7. Re-running `summarize <plaud-id>` reuses the stored result; only `--refresh`
   calls the provider again.

## Recovery

- A failed cloud summary remains absent and is retried by the next automatic run.
- A failed cloud STT request falls back to local whisper.cpp when configured.
- Successful STT chunks and summaries are reused on retry.
- Database migrations are additive and recordings/transcripts remain the source
  records for rebuilding all M2 projections.
