# plaud-brain

M0/M1 implementation of a self-hosted PLAUD ingestion and local speech-to-text pipeline.

## M1 flow

```text
PLAUD device -> PLAUD Cloud -> PLAUD CLI -> audio download
                                      |
                                      v
                              local data/audio
                                      |
                                      v
                         ffmpeg -> 16 kHz mono WAV
                                      |
                                      v
                       whisper.cpp + Vulkan (local)
                                      |
                                      v
                         PostgreSQL transcript store
```

No PLAUD transcription minutes are consumed by this pipeline: it fetches the original audio and performs STT locally.

## Target host

- Ubuntu 24.04
- x86_64
- AMD Ryzen 7 8745HS / Radeon 780M-class iGPU
- Node.js >= 20 for PLAUD CLI
- Python 3.12
- Docker / Docker Compose for PostgreSQL
- ffmpeg / ffprobe
- Vulkan-capable Mesa driver

## Bootstrap

```bash
cp .env.example .env
make check
make install
make db-up

npm install -g @plaud-ai/cli
plaud login
plaud me

make whisper-build
make whisper-model

make run-once
```

`plaud login` is interactive and intentionally not automated. PLAUD stores its own OAuth tokens under `~/.plaud/`.

## Commands

```bash
plaud-brain discover                 # discover recent recordings, upsert into DB
plaud-brain process                  # process pending downloads/transcriptions
plaud-brain run-once                 # discover + process
plaud-brain status                   # status counts
plaud-brain benchmark path/to.wav    # benchmark configured whisper model
```

## Security

- PostgreSQL binds to localhost only.
- Audio/transcripts are ignored by git.
- No cloud STT is enabled.
- No PLAUD credential is copied into this repository.
- Work recordings should remain local unless a later explicit policy enables cloud fallback.

See `docs/M1_RUNBOOK.md` for the host smoke test.
