# M6a runbook — speaker diarization foundation

## Current scope

The first M6a slice provides:

- provider-neutral diarization result types;
- deterministic speaker-to-transcript alignment;
- PostgreSQL persistence for diarization and speaker turns;
- provider/model/audio-checksum cache reuse;
- processing-run success/failure audit;
- no repeated STT and no change to original transcript language.

It intentionally does not install or enable a model yet.

The adapter and CLI are ready but disabled by default:

```dotenv
DIARIZATION_PROVIDER=disabled
DIARIZATION_MODEL=pyannote/speaker-diarization-community-1
DIARIZATION_TOKEN_ENV=HF_TOKEN
```

After accepting the model conditions and placing `HF_TOKEN` only in `.env`:

```bash
.venv/bin/pip install -e '.[diarization]'
# change DIARIZATION_PROVIDER=pyannote
.venv/bin/plaud-brain diarize <plaud-id>
```

Use `--refresh` only for a deliberate rerun. A matching audio checksum,
provider, and model reuses the stored result by default.

## Provider benchmark gate

Preferred first benchmark: local
`pyannote/speaker-diarization-community-1`.

- It runs locally and exposes exclusive diarization designed to reconcile with
  transcription timestamps.
- The model accepts mono 16 kHz audio, matching the existing normalized format.
- Access requires accepting the Hugging Face model conditions and providing a
  token; never commit the token.
- bmax-ai currently has no torch or pyannote installation and no Hugging Face
  token, so installation is deferred until explicit credential setup.

Cloud alternative: OpenAI `gpt-4o-transcribe-diarize`.

- It returns `diarized_json` and requires automatic chunking for audio longer
  than 30 seconds.
- It uploads audio and performs transcription again, so it is not the default
  incremental path.
- It requires a direct OpenAI API key; the existing OpenRouter key is not
  treated as equivalent.

Official references:

- https://huggingface.co/pyannote/speaker-diarization-community-1
- https://developers.openai.com/api/reference/resources/audio/transcriptions

## Benchmark corpus

Use copied excerpts, never modify the authoritative source audio:

1. 60 seconds, two speakers, low overlap;
2. 5 minutes, two or three speakers;
3. one difficult excerpt with overlap or background noise.

Record wall time, peak memory, detected speaker count, manual speaker-confusion
notes, and whether transcript-segment alignment is usable. Do not commit audio
or speaker content.

## Apply migration

```bash
make db-migrate
```

Migration 006 only adds `diarizations`, `speaker_turns`, and the
`DIARIZATION` processing stage. Existing transcripts remain unchanged until
a provider is deliberately run.

## Acceptance for enabling a provider

- A 60-second sample completes without exhausting host memory.
- Anonymous speaker count is plausible.
- Speaker changes align acceptably with existing transcript timestamps.
- Repeating the same provider/model/audio combination is a cache hit.
- Provider failure leaves the existing transcript intact and records a failed
  processing run.
- Any cloud provider has an explicit audio-upload privacy decision.
