# Architecture decisions

## ADR-001 — PostgreSQL is source of truth
Accepted. Notion will be a later projection only.

## ADR-002 — M1 default STT is whisper.cpp via Vulkan
Accepted provisionally. Benchmark before final long-term model decision.

## ADR-003 — Pin whisper.cpp v1.9.4
Accepted for reproducibility. Upgrade only through an explicit tested change.

## ADR-004 — Normalize all PLAUD audio through ffmpeg
Accepted. This avoids relying on whisper.cpp support for every source container/codec.

## ADR-005 — Do not enable VAD in M1
Accepted. Keep the first timestamp contract to segment-level offsets in the original normalized audio timeline.

## ADR-006 — No cloud transcription in M0/M1
Accepted. Cloud fallback requires an explicit privacy policy in a later milestone.
