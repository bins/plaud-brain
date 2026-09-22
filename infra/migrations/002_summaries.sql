ALTER TABLE processing_runs
    DROP CONSTRAINT IF EXISTS processing_runs_stage_check;

ALTER TABLE processing_runs
    ADD CONSTRAINT processing_runs_stage_check
    CHECK (stage IN ('DOWNLOAD', 'STT', 'SUMMARY'));

CREATE TABLE IF NOT EXISTS summaries (
    id UUID PRIMARY KEY,
    transcript_id UUID NOT NULL UNIQUE REFERENCES transcripts(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    content JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
