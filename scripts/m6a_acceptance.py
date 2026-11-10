#!/usr/bin/env python3
"""Collect the numbers for the M6a acceptance table in docs/M6_RUNBOOK.md.

Run on bmax-ai from the repository root with the diarization extra installed
and DIARIZATION_PROVIDER=pyannote in .env:

    .venv/bin/python scripts/m6a_acceptance.py bench --label "60 s" excerpt-60s.wav
    .venv/bin/python scripts/m6a_acceptance.py check <plaud-id>

`bench` diarizes copied audio files without touching the database (rows 1-4:
wall time, realtime factor, peak memory, speakers detected).

`check` uses one transcribed recording in the database (rows 5-8 and its own
timing). It reruns diarization with --refresh, repeats it to confirm a cache
hit, forces a provider failure by pointing the token at an unset variable, and
verifies that the transcript text and timestamps never change. The refresh
replaces that recording's stored speaker labels with a new run of the same
model, which is what `diarize --refresh` always does.

Output contains no transcript text, recording names, or PLAUD IDs, so the
printed rows can be pasted into the runbook. Speaker counts and alignment
still need a human: listen to the excerpt and compare.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UNSET_TOKEN_ENV = "PLAUD_BRAIN_M6A_UNSET_TOKEN"


@dataclass
class ChildRun:
    returncode: int
    seconds: float
    peak_rss_bytes: int
    stdout: str
    stderr: str


def run_measured(command: list[str], env: dict[str, str] | None = None) -> ChildRun:
    """Run a command and report its own wall time and peak resident memory."""
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        start = time.monotonic()
        process = subprocess.Popen(command, stdout=out, stderr=err, env=env, cwd=REPO)
        _, status, usage = os.wait4(process.pid, 0)
        seconds = time.monotonic() - start
        process.returncode = os.waitstatus_to_exitcode(status)
        out.seek(0)
        err.seek(0)
        return ChildRun(
            returncode=process.returncode,
            seconds=seconds,
            # Linux reports ru_maxrss in KiB.
            peak_rss_bytes=usage.ru_maxrss * 1024,
            stdout=out.read().decode("utf-8", "replace"),
            stderr=err.read().decode("utf-8", "replace"),
        )


def gib(value: int) -> str:
    return f"{value / 1024**3:.2f} GiB"


def realtime(seconds: float, audio_seconds: float | None) -> str:
    if not audio_seconds:
        return "?"
    return f"{seconds / audio_seconds:.3f}"


def segments_digest(segments: list[dict]) -> str:
    """Hash STT text and timestamps; speaker labels are deliberately excluded."""
    payload = [
        [int(s["seq"]), int(s["start_ms"]), int(s["end_ms"]), s["text"]] for s in segments
    ]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()


def alignment_stats(segments: list[dict], turns: list[dict]) -> dict:
    """How well the assigned speaker's turns cover each transcript segment."""
    labeled = 0
    weak = 0
    changes: list[int] = []
    previous = None
    for segment in segments:
        speaker = segment.get("speaker")
        if not speaker:
            continue
        labeled += 1
        start, end = int(segment["start_ms"]), int(segment["end_ms"])
        covered = sum(
            max(0, min(end, int(t["end_ms"])) - max(start, int(t["start_ms"])))
            for t in turns
            if t["speaker"] == speaker
        )
        if end > start and covered / (end - start) < 0.5:
            weak += 1
        if previous is not None and speaker != previous:
            changes.append(start)
        previous = speaker
    return {
        "segments": len(segments),
        "labeled": labeled,
        "weak": weak,
        "changes": changes,
    }


def format_ms(value: int) -> str:
    seconds = value // 1000
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def audio_seconds(path: Path) -> float | None:
    ffprobe = os.getenv("FFPROBE_BIN", "ffprobe")
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        return float(result.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def bench_child(path: Path) -> int:
    """Diarize one file in a fresh process so its memory is measured alone."""
    from plaud_brain.adapters.pyannote_diarization import PyannoteDiarizer
    from plaud_brain.config import Settings

    settings = Settings.from_env()
    diarizer = PyannoteDiarizer(
        model=settings.diarization_model,
        token_env=settings.diarization_token_env,
        min_speakers=settings.diarization_min_speakers,
        max_speakers=settings.diarization_max_speakers,
        ffmpeg_bin=settings.ffmpeg_bin,
    )
    with tempfile.TemporaryDirectory(prefix="m6a-bench-") as work:
        result = diarizer.diarize(path, Path(work))
    print(json.dumps({"turns": len(result.turns), "speakers": len({t.speaker for t in result.turns})}))
    return 0


def bench(args) -> int:
    print("| Sample | Wall time | Realtime factor | Peak RSS | Speakers detected | Exit |")
    print("|---|---|---|---|---|---|")
    failed = False
    for index, raw in enumerate(args.audio):
        path = Path(raw).resolve()
        label = args.label[index] if index < len(args.label) else f"sample {index + 1}"
        run = run_measured([sys.executable, str(Path(__file__).resolve()), "_bench-child", str(path)])
        speakers = "?"
        if run.returncode == 0:
            speakers = str(json.loads(run.stdout.strip().splitlines()[-1])["speakers"])
        else:
            failed = True
            print(f"<!-- {label} failed: {last_line(run.stderr)} -->", file=sys.stderr)
        print(
            f"| {label} | {run.seconds:.2f} s | {realtime(run.seconds, audio_seconds(path))} "
            f"| {gib(run.peak_rss_bytes)} | {speakers} | {run.returncode} |"
        )
    return 1 if failed else 0


def last_line(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1][:300] if lines else "(no output)"


def latest_diarization_run(store, recording_id) -> dict | None:
    with store.connect() as conn:
        return conn.execute(
            """
            SELECT status, error_code, started_at FROM processing_runs
            WHERE recording_id = %s AND stage = 'DIARIZATION'
            ORDER BY started_at DESC LIMIT 1
            """,
            (recording_id,),
        ).fetchone()


def check(args) -> int:
    from plaud_brain.config import Settings
    from plaud_brain.store import PostgresStore

    settings = Settings.from_env()
    if settings.diarization_provider != "pyannote":
        print("DIARIZATION_PROVIDER must be pyannote in .env", file=sys.stderr)
        return 2
    store = PostgresStore(settings.database_url)
    plaud_id = args.plaud_id
    transcript = store.get_transcript(plaud_id)
    if transcript is None or not transcript["segments"]:
        print("recording has no transcript segments", file=sys.stderr)
        return 2
    before = segments_digest(transcript["segments"])
    duration_s = (transcript.get("duration_ms") or 0) / 1000 or transcript.get("duration_seconds")
    work_dir = settings.data_dir / "work" / plaud_id / "diarization"
    cli = [str(Path(sys.executable).parent / "plaud-brain")]
    results: dict[str, tuple[str, str]] = {}

    # Fresh run (its timing also fills row 4 when this is the long recording).
    fresh = run_measured(cli + ["diarize", plaud_id, "--refresh"])
    fresh_ok = fresh.returncode == 0
    stored = store.get_diarization(plaud_id) if fresh_ok else None
    transcript = store.get_transcript(plaud_id)
    stats = alignment_stats(transcript["segments"], stored["turns"] if stored else [])
    speakers = len({t["speaker"] for t in stored["turns"]}) if stored else 0

    # Row 8: no normalized audio left behind.
    if not fresh_ok:
        results["8"] = ("Not checked", "fresh run failed")
    elif work_dir.exists():
        results["8"] = ("Fail", f"left: {sorted(p.name for p in work_dir.iterdir())}")
    else:
        results["8"] = ("Pass", "work dir removed")

    # Row 6: an immediate repeat is a cache hit.
    repeat = run_measured(cli + ["diarize", plaud_id])
    cached = False
    if repeat.returncode == 0:
        try:
            cached = json.loads(repeat.stdout).get("cached") is True
        except json.JSONDecodeError:
            cached = False
    if fresh_ok:
        results["6"] = ("Pass" if cached else "Fail", f"{repeat.seconds:.3f} s, cached={cached}")
    else:
        results["6"] = ("Not checked", "fresh run failed")

    # Row 7: a provider failure keeps transcript and stored speakers intact.
    previous = store.get_diarization(plaud_id)
    env = {k: v for k, v in os.environ.items() if k != UNSET_TOKEN_ENV}
    env["DIARIZATION_TOKEN_ENV"] = UNSET_TOKEN_ENV
    failure = run_measured(cli + ["diarize", plaud_id, "--refresh"], env=env)
    run = latest_diarization_run(store, transcript["recording_id"])
    after_failure = store.get_diarization(plaud_id)
    intact = (
        previous is not None
        and after_failure is not None
        and after_failure["updated_at"] == previous["updated_at"]
        and len(after_failure["turns"]) == len(previous["turns"])
    )
    failed_recorded = run is not None and run["status"] == "FAILED"
    results["7"] = (
        "Pass" if failure.returncode != 0 and failed_recorded and intact and not work_dir.exists() else "Fail",
        ("no stored diarization to protect; " if previous is None else "")
        + f"exit {failure.returncode}, last run {run['status'] if run else 'none'}"
        f" ({run['error_code'] if run else '-'}), stored speakers {'unchanged' if intact else 'CHANGED'}",
    )

    after = segments_digest(store.get_transcript(plaud_id)["segments"])
    text_ok = after == before

    # Row 5 needs a human; these numbers point at where to listen.
    if fresh_ok:
        results["5"] = (
            "Pending",
            f"{stats['labeled']}/{stats['segments']} segments labeled, {stats['weak']} with <50% "
            f"turn coverage, {len(stats['changes'])} speaker changes",
        )
    else:
        results["5"] = ("Not checked", "fresh run failed")

    print("## This recording (paste timing into row 4 if it is the long recording)\n")
    print("| Wall time | Realtime factor | Peak RSS | Speakers detected | Exit |")
    print("|---|---|---|---|---|")
    print(
        f"| {fresh.seconds:.2f} s | {realtime(fresh.seconds, duration_s)} | {gib(fresh.peak_rss_bytes)} "
        f"| {speakers if fresh_ok else '?'} | {fresh.returncode} |"
    )
    if not fresh_ok:
        print(f"\nFresh run failed: {last_line(fresh.stderr)}")
    print("\n## Rows 5-8\n")
    print("| # | Result | Notes |")
    print("|---|---|---|")
    for row in ("5", "6", "7", "8"):
        print(f"| {row} | {results[row][0]} | {results[row][1]} |")
    print(f"\nTranscript text and timestamps unchanged: {'yes' if text_ok else 'NO'}")
    if fresh_ok and stats["changes"]:
        shown = ", ".join(format_ms(ms) for ms in stats["changes"][: args.show_changes])
        print(f"First speaker changes to listen for: {shown}")

    automated = [results[r][0] for r in ("6", "7", "8")]
    return 0 if fresh_ok and text_ok and all(r == "Pass" for r in automated) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bench", help="time copied audio files without the database (rows 1-4)")
    b.add_argument("audio", nargs="+")
    b.add_argument("--label", action="append", default=[], help="table label per file, in order")
    c = sub.add_parser("check", help="verify rows 5-8 on one transcribed recording")
    c.add_argument("plaud_id")
    c.add_argument("--show-changes", type=int, default=10)
    child = sub.add_parser("_bench-child")
    child.add_argument("audio")
    args = parser.parse_args(argv)
    if args.command == "bench":
        return bench(args)
    if args.command == "check":
        return check(args)
    return bench_child(Path(args.audio))


if __name__ == "__main__":
    sys.exit(main())
