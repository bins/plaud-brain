# M6a runbook — speaker diarization foundation

## Current scope

The first M6a slice provides:

- provider-neutral diarization result types;
- deterministic speaker-to-transcript alignment;
- PostgreSQL persistence for diarization and speaker turns;
- provider/model/audio-checksum cache reuse;
- processing-run success/failure audit;
- no repeated STT and no change to original transcript language.

The adapter and CLI remain disabled by default:

```dotenv
DIARIZATION_PROVIDER=disabled
DIARIZATION_MODEL=pyannote/speaker-diarization-community-1
DIARIZATION_TOKEN_ENV=HF_TOKEN
```

After accepting the model conditions and placing `HF_TOKEN` only in `.env`:

```bash
.venv/bin/pip install torch==2.8.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cpu
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
- The model accepts mono 16 kHz audio. The adapter preloads PCM WAV and uses
  the configured ffmpeg binary to normalize compressed source audio without
  rerunning STT.
- Access requires accepting the Hugging Face model conditions and providing a
  token; never commit the token.
- bmax-ai uses CPU-only PyTorch 2.8.0 and pyannote.audio 4.0.7. Install the CPU
  wheels before the optional extra so pip does not download unused NVIDIA CUDA
  packages.

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

Validated on bmax-ai on 2026-09-26:

- model warm load: 3.43 seconds and about 551 MiB RSS;
- 60 seconds: 28.09 seconds (0.468 realtime), 2.21 GiB peak RSS;
- 300 seconds: 161.71 seconds (0.539 realtime), 2.31 GiB peak RSS;
- full database path: compressed-audio normalization, 260 stored turns and 368
  labeled transcript segments;
- immediate repeat: cache hit in 0.021 seconds with no repeated model run.

Speaker counts are anonymous and must still be judged manually on a known
multi-speaker sample before treating them as identity labels.

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
