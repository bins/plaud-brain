ALTER TABLE processing_runs
    DROP CONSTRAINT IF EXISTS processing_runs_stage_check;

ALTER TABLE processing_runs
    ADD CONSTRAINT processing_runs_stage_check
    CHECK (stage IN ('DOWNLOAD', 'STT', 'SUMMARY', 'DIARIZATION'));

CREATE TABLE IF NOT EXISTS diarizations (
    id UUID PRIMARY KEY,
    transcript_id UUID NOT NULL UNIQUE REFERENCES transcripts(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    audio_sha256 TEXT NOT NULL,
    raw_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS speaker_turns (
    id BIGSERIAL PRIMARY KEY,
    diarization_id UUID NOT NULL REFERENCES diarizations(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    start_ms BIGINT NOT NULL CHECK (start_ms >= 0),
    end_ms BIGINT NOT NULL CHECK (end_ms > start_ms),
    speaker TEXT NOT NULL CHECK (btrim(speaker) <> ''),
    confidence DOUBLE PRECISION CHECK (
        confidence IS NULL OR (confidence >= 0 AND confidence <= 1)
    ),
    UNIQUE(diarization_id, seq)
);

CREATE INDEX IF NOT EXISTS speaker_turns_diarization_idx
    ON speaker_turns(diarization_id, seq);
