CREATE TABLE IF NOT EXISTS rag_answers (
    id UUID PRIMARY KEY,
    cache_key TEXT NOT NULL UNIQUE,
    query TEXT NOT NULL,
    filters JSONB NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    index_signature TEXT NOT NULL,
    response JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS rag_answers_updated_idx
    ON rag_answers(updated_at DESC);
