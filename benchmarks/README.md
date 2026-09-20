# STT benchmarks

Performance is measured as realtime factor (RTF): `elapsed_seconds / audio_seconds`. Lower is better. `x_realtime = 1 / RTF`.

M1 starts with `large-v3-turbo`. After the baseline is stable, compare quantized turbo, large-v3, and alternative engines in separate ADRs rather than changing production defaults silently.
