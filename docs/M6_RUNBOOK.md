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

## M6a acceptance results

Fill in on bmax-ai. Use copied excerpts only; record no transcript text,
speaker content, recording names, or PLAUD IDs here. `Pass` / `Fail` /
`Pending` per row. ADR-019 is accepted only when every row is `Pass`.

Host / versions: bmax-ai, CPU-only PyTorch 2.8.0, pyannote.audio 4.0.7,
model `pyannote/speaker-diarization-community-1`, date: ____

| # | Check | Sample | Wall time | Realtime factor | Peak RSS | Speakers detected / expected | Result | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | 60 s, two speakers, low overlap | copied excerpt | 28.09 s | 0.468 | 2.21 GiB | __ / 2 | Pending | timing recorded 2026-09-26; speaker count not yet judged |
| 2 | 5 min, two or three speakers | copied excerpt | 161.71 s | 0.539 | 2.31 GiB | __ / __ | Pending | timing recorded 2026-09-26; speaker count not yet judged |
| 3 | Difficult: overlap or background noise | copied excerpt | | | | __ / __ | Pending | |
| 4 | Long recording (≥ 45 min) | full copy | | | | __ / __ | Pending | must fit beside the ingestion timer |
| 5 | Speaker changes align with transcript timestamps | rows 1–3 | — | — | — | — | Pending | describe misaligned segments, if any |
| 6 | Repeat run is a cache hit | row 1 | | — | — | — | Pending | 0.021 s observed 2026-09-26 |
| 7 | Provider failure keeps transcript intact, run recorded `FAILED` | forced failure (e.g. bad token) | — | — | — | — | Pending | check `processing_runs` |
| 8 | Normalized audio under `data/work/<id>/diarization/` removed after run | row 1 | — | — | — | — | Pending | or record a retention rule in ADR-019 |
| 9 | Cloud provider privacy decision | — | — | — | — | — | N/A | no cloud provider proposed |

Speaker-count check: judge rows 1–4 by listening to the copied excerpt and
comparing it with `plaud-brain show <plaud-id>`; note systematic confusions
(e.g. two similar voices merged) without quoting content.

## Automatic diarization and backfill

Enable only after ADR-019 is accepted and ADR-020's evidence is recorded.

```dotenv
DIARIZATION_PROVIDER=pyannote
DIARIZATION_AUTO=true
DIARIZATION_AUTO_LIMIT=2
DIARIZATION_MAX_ATTEMPTS=3
```

- `run-once` order: discover → download/STT → diarization → summaries →
  embedding index. Output adds `diarized`, `diarization_cached`, and
  `diarization_failed`.
- Selection: transcribed recordings with audio, at least one segment, and no
  diarization row; newest first. A recording that already has a diarization from
  another provider or model is not picked up; rerun it with
  `diarize <plaud-id> --refresh`.
- Attempt cap: `FAILED` and stale `RUNNING` rows both count, so a run killed by
  the OOM killer does not retry forever. A successful manual `diarize` clears it
  from the queue.
- Timing: at about 0.5× realtime, two 1-hour recordings add about an hour to a
  run. `plaud-brain.service` allows 3 hours (`TimeoutStartSec=3h`), and the
  timer starts the next run only 15 minutes after the previous one finishes.
- Backfill: `plaud-brain diarize-pending --limit N`, ideally while the timer is
  stopped or at night. The per-recording advisory lock makes an overlapping
  timer run wait and then reuse the stored result.
- Summaries generated before diarization keep no speaker labels; refresh them
  deliberately with `plaud-brain summarize <plaud-id> --refresh`.
- Work files under `data/work/<plaud-id>/diarization/` are deleted after every
  run, successful or not.

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
