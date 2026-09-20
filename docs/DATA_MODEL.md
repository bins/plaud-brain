# Data model — M1

## recordings
One row per PLAUD recording. `plaud_id` is the external idempotency key.

Important fields: name, created_at, duration_seconds, status, audio_path, audio_sha256, error fields.

## transcripts
One row per recording in M1. Stores STT engine/model/language and the raw whisper JSON for future re-parsing.

## transcript_segments
Ordered, timestamped segments. Speaker is nullable because diarization is M6 scope.

## processing_runs
Audit trail of each download/STT attempt, including success/failure and error detail.
