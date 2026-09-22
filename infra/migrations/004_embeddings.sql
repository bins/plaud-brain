CREATE TABLE IF NOT EXISTS embeddings (
    id UUID PRIMARY KEY,
    source_type TEXT NOT NULL CHECK (source_type IN (
        'transcript', 'summary', 'decision', 'action_item'
    )),
    source_id TEXT NOT NULL,
    recording_id UUID NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    content_hash TEXT NOT NULL,
    text_snapshot TEXT NOT NULL,
    timestamp_ms BIGINT,
    embedding JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(source_type, source_id, provider, model)
);

CREATE INDEX IF NOT EXISTS embeddings_model_idx
    ON embeddings(provider, model);

CREATE INDEX IF NOT EXISTS embeddings_recording_idx
    ON embeddings(recording_id, source_type);
