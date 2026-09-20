CREATE TABLE IF NOT EXISTS recordings (
    id UUID PRIMARY KEY,
    plaud_id TEXT NOT NULL UNIQUE,
    name TEXT,
    created_at TIMESTAMPTZ,
    duration_seconds INTEGER,
    status TEXT NOT NULL CHECK (status IN (
        'DISCOVERED','DOWNLOADED','TRANSCRIBED','FAILED_DOWNLOAD','FAILED_STT'
    )),
    audio_path TEXT,
    audio_sha256 TEXT,
    error_code TEXT,
    error_message TEXT,
    discovered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS recordings_status_idx ON recordings(status);
CREATE INDEX IF NOT EXISTS recordings_created_at_idx ON recordings(created_at DESC);

CREATE TABLE IF NOT EXISTS transcripts (
    id UUID PRIMARY KEY,
    recording_id UUID NOT NULL UNIQUE REFERENCES recordings(id) ON DELETE CASCADE,
    engine TEXT NOT NULL,
    model TEXT NOT NULL,
    language TEXT,
    duration_ms BIGINT,
    raw_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS transcript_segments (
    id BIGSERIAL PRIMARY KEY,
    transcript_id UUID NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    start_ms BIGINT NOT NULL,
    end_ms BIGINT NOT NULL,
    speaker TEXT,
    text TEXT NOT NULL,
    UNIQUE(transcript_id, seq)
);

CREATE INDEX IF NOT EXISTS transcript_segments_transcript_idx ON transcript_segments(transcript_id, seq);

CREATE TABLE IF NOT EXISTS processing_runs (
    id UUID PRIMARY KEY,
    recording_id UUID NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    stage TEXT NOT NULL CHECK (stage IN ('DOWNLOAD','STT')),
    status TEXT NOT NULL CHECK (status IN ('RUNNING','SUCCEEDED','FAILED')),
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at TIMESTAMPTZ,
    error_code TEXT,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS processing_runs_recording_idx ON processing_runs(recording_id, started_at DESC);
