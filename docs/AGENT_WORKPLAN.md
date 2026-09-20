# M0/M1 agent work plan

## Codex — implementation owner

1. Clone/open this repository on the target Ubuntu 24.04 host.
2. Run all static/unit checks.
3. Validate the PostgreSQL integration tests.
4. Install/login to the official PLAUD CLI and capture the real `plaud recent --days 1`, `plaud file <id>`, and `plaud audio <id>` behavior without committing private output.
5. Adjust only `PlaudCliAdapter` if real terminal formatting differs from fixtures.
6. Build pinned whisper.cpp v1.9.4 with Vulkan.
7. Download `large-v3-turbo` and run the benchmark corpus.
8. Execute the M1 smoke test end-to-end.
9. Record host/version/RTF results in a new non-private benchmark summary. Never commit transcript text.

Acceptance: one newly synced PLAUD recording reaches `TRANSCRIBED` without consuming PLAUD transcription minutes.

## Claude Code — independent review

Review after Codex has a passing M1 smoke test. Focus on:
- retry/state-machine correctness;
- failure windows between file and DB writes;
- command argument safety;
- PLAUD credential/privacy handling;
- PostgreSQL invariants;
- whether Whisper JSON parsing is stable for v1.9.4;
- operational recovery after reboot or partial failure.

Return findings only. Do not add M2 features.

## DeepSeek — benchmark/test owner

After the first successful M1 run:
- expand parser fixtures using sanitized shapes from actual CLI output;
- add failure tests for expired audio URLs, network interruption, ffmpeg failure, missing model, and missing whisper JSON;
- analyze STT benchmark CSV for French/English/Mandarin/mixed speech;
- propose a benchmark conclusion, but do not change the default model automatically.

## Merge gate

Codex integrates review fixes, runs the full suite, repeats one real PLAUD smoke test, and updates `docs/DECISIONS.md` only when evidence warrants a decision change.
