CREATE TABLE IF NOT EXISTS decisions (
    id UUID PRIMARY KEY,
    summary_id UUID NOT NULL REFERENCES summaries(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    fingerprint TEXT NOT NULL,
    text TEXT NOT NULL,
    timestamp_ms BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(summary_id, seq),
    UNIQUE(summary_id, fingerprint)
);

CREATE INDEX IF NOT EXISTS decisions_timestamp_idx
    ON decisions(summary_id, timestamp_ms);

CREATE TABLE IF NOT EXISTS action_items (
    id UUID PRIMARY KEY,
    summary_id UUID NOT NULL REFERENCES summaries(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    fingerprint TEXT NOT NULL,
    task TEXT NOT NULL,
    owner TEXT,
    due_text TEXT,
    timestamp_ms BIGINT,
    status TEXT NOT NULL DEFAULT 'OPEN'
        CHECK (status IN ('OPEN', 'DONE', 'DISMISSED')),
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(summary_id, seq),
    UNIQUE(summary_id, fingerprint)
);

CREATE INDEX IF NOT EXISTS action_items_status_idx
    ON action_items(status, created_at DESC);
