# plaud-brain

M5 implementation of a self-hosted PLAUD ingestion, configurable speech-to-text,
structured-summary, hybrid semantic retrieval, cited question-answering, and
local web interface.

## Current flow

```text
PLAUD device -> PLAUD Cloud -> PLAUD CLI -> audio download
                                      |
                                      v
                              local data/audio
                                      |
                                      v
                         ffmpeg -> bounded WAV chunks
                                      |
                          +-----------+-----------+
                          |                       |
                          v                       v
                  OpenRouter Whisper      whisper.cpp CPU
                       (primary)          (local fallback)
                                      |
                                      v
                         PostgreSQL transcript store
```

No PLAUD transcription minutes are consumed by this pipeline: it fetches the
original audio and uses the configured local or independent cloud STT backend.

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

`make db-up` waits for PostgreSQL and applies every migration in order. The
Compose database is exposed only on `127.0.0.1:5433`.

On the current AMD Phoenix/RADV host, set `WHISPER_NO_GPU=true` in `.env`
to use the stable CPU path. Set it to `false` only after validating the Vulkan
driver and workload; the initial Vulkan smoke test caused a GPU reset.
`WHISPER_THREADS=8` uses the eight physical CPU cores on this host; a 60-second
sample improved from 51 seconds at four threads to 33 seconds at eight threads.

Cloud STT is opt-in. Set `STT_PROVIDER=openrouter` and keep the OpenRouter key
only in the ignored `.env` file. Audio is converted to 16 kHz mono WAV chunks,
each successful chunk response is cached, and timestamps are merged back onto
the original timeline. `STT_FALLBACK_PROVIDER=local` preserves local CPU STT
when the cloud endpoint is unavailable. Set `STT_BASE_URL`, `STT_MODEL`, and
`STT_API_KEY_ENV` to switch to another OpenAI-compatible transcription API.

`plaud login` is interactive and intentionally not automated. PLAUD stores its own OAuth tokens under `~/.plaud/`.

## Commands

```bash
plaud-brain discover                 # discover recent recordings, upsert into DB
plaud-brain process                  # process pending downloads/transcriptions
plaud-brain run-once                 # discover + process
plaud-brain status                   # status counts
plaud-brain recordings --limit 20    # list local recordings and transcript status
plaud-brain transcript <plaud-id>    # print a transcript as plain text
plaud-brain transcript <plaud-id> --format markdown --output transcript.md
plaud-brain search "launch plan"     # search timestamped transcript segments
plaud-brain knowledge-search "launch" # search transcripts, summaries, decisions, actions
plaud-brain index                    # incrementally embed changed knowledge
plaud-brain semantic-search "launch risks" # hybrid semantic + literal ranking
plaud-brain ask "what did we decide?" # grounded answer with recording citations
plaud-brain serve                     # local web interface on 127.0.0.1:8787
plaud-brain diarize <plaud-id>         # add cached anonymous speaker labels
plaud-brain show <plaud-id>           # combined summary and timestamped transcript
plaud-brain decisions                 # list extracted decisions
plaud-brain actions                   # list open action items
plaud-brain action <uuid> done        # update one action item
plaud-brain summarize <plaud-id>      # cloud summary using the selected provider
plaud-brain summary <plaud-id>        # read an existing summary without a cloud call
plaud-brain benchmark path/to.wav    # benchmark configured whisper model
```

Literal retrieval reads only from local PostgreSQL. `index` sends knowledge
text to the configured embedding provider, and `semantic-search` sends the
query. Audio is never sent by the embedding path.

## Cloud summaries

Cloud summaries use an OpenAI-compatible Chat Completions endpoint. DeepSeek is
the default provider; only transcript text and timestamps are sent, never the
source audio. Configure the untracked `.env` file:

```dotenv
SUMMARY_PROVIDER=deepseek
SUMMARY_BASE_URL=https://api.deepseek.com
SUMMARY_MODEL=deepseek-flash
SUMMARY_API_KEY_ENV=DEEPSEEK_API_KEY
DEEPSEEK_API_KEY=your-secret-key
```

Do not commit the API key. To switch providers, change `SUMMARY_PROVIDER`,
`SUMMARY_BASE_URL`, `SUMMARY_MODEL`, and `SUMMARY_API_KEY_ENV` to another
OpenAI-compatible service. Generate a summary with:

```bash
plaud-brain summarize <plaud-id> --format markdown --output summary.md
```

Summaries are stored in PostgreSQL and reused by default. Pass `--refresh` to
replace one deliberately. Set `SUMMARY_AUTO=true` to let `run-once` generate
summaries for transcribed recordings that do not have one yet.

Decisions and action items are projected into relational tables when a summary
is stored. Action status can be `open`, `done`, or `dismissed`; matching items
keep their ID and status across an explicit summary refresh.

## Semantic retrieval

M3 uses an OpenAI-compatible embeddings endpoint. The default is OpenRouter's
`qwen/qwen3-embedding-8b` at 1024 dimensions. Content hashes make indexing
incremental: unchanged transcript segments, summaries, decisions, and actions
reuse stored vectors without another API call. Configure `EMBEDDING_AUTO=true`
to update the index at the end of `run-once`.

The current corpus is small, so vectors are stored as JSONB in PostgreSQL and
cosine similarity is calculated in the application. This avoids replacing the
current database image solely to add pgvector. Revisit pgvector or a dedicated
vector store when scan latency becomes material.

## Cited question answering

`ask` retrieves the strongest local evidence and sends only the question plus
those selected text sources to the configured QA provider. Answers must cite
the supplied source IDs inline; unknown, missing, or inconsistent citations are
rejected. Returned sources include recording identity, source type, timestamp,
and relevance score.

```bash
plaud-brain ask "What are my next actions?"
plaud-brain ask "What was decided?" --source-type decision --since 2026-09-01
plaud-brain ask "What happened?" --recording <plaud-id> --format json
```

Answers are cached against the question, filters, provider configuration, and
current embedding-index signature. Repeating a question does not make another
embedding or QA request while the knowledge index is unchanged. Use `--refresh`
for an intentional regeneration.

## Speaker diarization

M6a stores speaker turns separately from transcription and aligns anonymous
speaker labels onto the existing timestamped segments. It never reruns or
rewrites valid STT. The first adapter targets local pyannote Community-1 and is
disabled until its optional dependency is installed and the model conditions
are accepted:

```bash
.venv/bin/pip install torch==2.8.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e '.[diarization]'
# set DIARIZATION_PROVIDER=pyannote and HF_TOKEN only in .env
plaud-brain diarize <plaud-id>
```

The adapter preloads PCM WAV audio and normalizes compressed source audio with
the configured ffmpeg binary, so TorchCodec's optional system-FFmpeg decoder is
not required. Install the CPU PyTorch wheels first on non-NVIDIA hosts to avoid
pulling unused CUDA packages.

Matching audio checksum, provider, and model reuse the stored result. See
`docs/M6_RUNBOOK.md` before enabling a model or uploading audio to any cloud.

## Scheduled ingestion

The user-level systemd units in `deploy/systemd/` run ingestion 15 minutes
after the previous run finishes, preventing overlapping CPU transcriptions.
They also provide the non-interactive `PATH` required by the locally installed
PLAUD CLI, Node.js, ffmpeg, and ffprobe.

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/plaud-brain.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now plaud-brain.timer
systemctl --user status plaud-brain.timer
```

Inspect recent runs with:

```bash
journalctl --user -u plaud-brain.service -n 100 --no-pager
```

## Database backups

The backup timer creates a verified PostgreSQL custom-format dump every day and
keeps 30 days by default:

```bash
chmod +x scripts/backup_postgres.sh
cp deploy/systemd/plaud-brain-backup.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now plaud-brain-backup.timer
systemctl --user start plaud-brain-backup.service
```

Backups are written with mode `600` under `data/backups/`, which contains
private transcript data and must not be committed. See
`docs/BACKUP_RUNBOOK.md` for the isolated restore-test procedure.

## Local web interface

The M5 interface shows health counts, recordings and transcripts, open actions,
semantic search, and cited Q&A. It binds only to `127.0.0.1:8787` by default.
Reach it from another computer through an SSH tunnel:

```bash
ssh -L 8787:127.0.0.1:8787 bins@bmax-ai
```

Then open `http://127.0.0.1:8787`. Install the separate user service with:

```bash
cp deploy/systemd/plaud-brain-web.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now plaud-brain-web.service
```

Keep `WEB_HOST=127.0.0.1` for private use. Any non-loopback bind is rejected
unless the environment variable named by `WEB_API_TOKEN_ENV` contains a
Bearer token. The browser keeps that token in memory only.

## Security

- PostgreSQL binds to localhost only.
- Audio/transcripts are ignored by git.
- Cloud STT and cloud summaries are explicitly configured in the untracked `.env`.
- OpenRouter receives bounded audio chunks; DeepSeek receives transcript text.
- QA sends the question and selected evidence text to the configured provider.
- The web service is loopback-only by default and writes no transcript content to its request log.
- No PLAUD credential is copied into this repository.
- Provider failures fall back to local STT, and successful cloud results are cached locally.

See `docs/M1_RUNBOOK.md` for the host smoke test.
