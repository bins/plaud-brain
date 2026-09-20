from __future__ import annotations

import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from plaud_brain.adapters.whisper_cpp import WhisperCppTranscriber


def probe_duration_ms(ffprobe_bin: str, audio: Path) -> int:
    cmd = [ffprobe_bin, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(audio)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    seconds = float(json.loads(proc.stdout)["format"]["duration"])
    return int(seconds * 1000)


def run_benchmark(transcriber: WhisperCppTranscriber, ffprobe_bin: str, audio: Path, out_csv: Path, label: str) -> dict:
    audio_ms = probe_duration_ms(ffprobe_bin, audio)
    work = out_csv.parent / "work" / f"bench-{int(time.time())}"
    started = time.perf_counter()
    result = transcriber.transcribe(audio, work)
    elapsed = time.perf_counter() - started
    rtf = elapsed / (audio_ms / 1000.0)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "audio": str(audio),
        "audio_seconds": round(audio_ms / 1000.0, 3),
        "elapsed_seconds": round(elapsed, 3),
        "realtime_factor": round(rtf, 5),
        "x_realtime": round(1 / rtf, 2) if rtf else None,
        "engine": result.engine,
        "model": result.model,
        "language": result.language,
        "segments": len(result.segments),
    }
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return row
