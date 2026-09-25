# M4 runbook — cited question answering

## Configure

QA defaults to DeepSeek and can reuse `DEEPSEEK_API_KEY`. Provider URL, model,
key environment variable, timeout, evidence count, and minimum retrieval score
are configurable with the `QA_*` settings documented in `.env.example`.

## Migrate

```bash
make db-migrate
```

## Ask

```bash
plaud-brain ask "What did we decide?"
plaud-brain ask "What are the next actions?" --source-type action_item
plaud-brain ask "Summarize this recording" --recording <plaud-id>
plaud-brain ask "What changed?" --since 2026-09-01 --until 2026-09-30
```

Use `--format json` for automation and `--output answer.md` for a file. A second
identical request should report `Cached: yes`. `--refresh` deliberately calls
the providers again.

## Trust boundary

The query embedding provider receives the question. The QA provider receives
the question and selected evidence text. Audio is never uploaded by M4. Answers
with invalid or missing citations fail closed instead of being cached.
