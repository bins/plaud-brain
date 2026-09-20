# Agent Engineering Contract

## Mission
Implement a reliable, privacy-first pipeline: PLAUD -> local audio -> local STT -> PostgreSQL.
M0/M1 must not introduce Hermes, OpenClaw, Notion, or cloud STT.

## Ownership
- Codex: primary implementation, integration, and merge owner.
- Claude Code: architecture/security/review; avoid rewriting working implementation without an explicit review finding.
- DeepSeek: tests, benchmark analysis, docs, log analysis, repetitive implementation.

## Rules
1. PostgreSQL is the source of truth.
2. Deterministic operations stay in deterministic code.
3. Keep PLAUD and STT behind adapters.
4. Every pipeline stage is idempotent and retryable.
5. Never commit secrets, PLAUD tokens, recordings, transcripts, or model binaries.
6. Do not enable cloud upload in M0/M1.
7. A transcription is successful only if a parseable JSON output file exists; do not trust whisper-cli exit code alone.
8. Do not enable whisper.cpp VAD in M1. Segment timestamps are sufficient; token-level VAD timestamp behavior is intentionally out of scope.
9. Any schema change must update infra/migrations and docs/DATA_MODEL.md.
10. Scope expansion must be recorded in docs/DECISIONS.md.

## Required checks
- `python -m compileall src`
- `pytest -q`
- shell scripts pass `bash -n`
- M1 manual smoke test on Ubuntu 24.04 target host
