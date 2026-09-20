# M1 runbook

## 1. Host prerequisites

```bash
sudo apt update
sudo apt install -y git cmake build-essential ffmpeg libvulkan-dev vulkan-tools glslc
vulkaninfo --summary
```

Ensure `/dev/dri` exists and the AMD GPU appears in `vulkaninfo`.

## 2. PLAUD CLI

```bash
node --version     # must be >=20
npm install -g @plaud-ai/cli
plaud login
plaud me
plaud recent --days 1
```

Do not copy `~/.plaud/tokens.json` into the repository.

## 3. App + database

```bash
cp .env.example .env
make install
make db-up
```

## 4. whisper.cpp

```bash
make whisper-build
make whisper-model
```

The build is pinned to v1.9.4 and uses `-DGGML_VULKAN=1`.

## 5. Smoke test

Create/sync a short PLAUD recording, then:

```bash
make run-once
.venv/bin/plaud-brain status
```

Expected final state: `TRANSCRIBED`.

Inspect:

```bash
docker compose exec postgres psql -U plaud -d plaud_brain -c \
  "select plaud_id,name,status,audio_path from recordings order by discovered_at desc limit 5;"

docker compose exec postgres psql -U plaud -d plaud_brain -c \
  "select t.language, count(s.*) segments from transcripts t join transcript_segments s on s.transcript_id=t.id group by t.id;"
```

## 6. Benchmark

Run against representative French, English, Chinese, and mixed-language recordings:

```bash
.venv/bin/plaud-brain benchmark ./sample.wav --label french-meeting
```

Results append to `benchmarks/results/stt.csv`.
Record transcription quality manually in `benchmarks/results/notes.md` until a labeled corpus exists.
