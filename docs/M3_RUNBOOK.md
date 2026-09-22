# M3 runbook — semantic retrieval

## Configure

Keep the provider key only in the ignored `.env` file. The default settings are
documented in `.env.example`; `OPENROUTER_API_KEY` is shared with cloud STT.

## Migrate and backfill

```bash
docker compose exec -T postgres psql -U plaud -d plaud_brain \
  < infra/migrations/004_embeddings.sql
plaud-brain index
```

A second `plaud-brain index` should report zero indexed items and reuse every
source. Use `--rebuild` only when intentionally replacing all vectors for the
configured provider/model.

## Search

```bash
plaud-brain semantic-search "what did we decide about the launch?" --limit 10
```

Results include source type, recording identity, timestamp, semantic score,
literal-match flag, and hybrid score. Higher scores rank first.

## Scheduled indexing

Set `EMBEDDING_AUTO=true`. `run-once` then updates embeddings after ingestion
and optional summaries. An embedding outage is reported as `index_failed: 1`
without discarding successful ingestion or summary work.
