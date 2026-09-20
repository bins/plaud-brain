from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from plaud_brain.adapters.downloader import AudioDownloader
from plaud_brain.adapters.plaud_cli import PlaudCliAdapter
from plaud_brain.adapters.whisper_cpp import WhisperCppTranscriber
from plaud_brain.benchmark import run_benchmark
from plaud_brain.config import Settings
from plaud_brain.pipeline import Pipeline
from plaud_brain.store import PostgresStore


def build(settings: Settings) -> tuple[PostgresStore, Pipeline, WhisperCppTranscriber]:
    store = PostgresStore(settings.database_url)
    plaud = PlaudCliAdapter(settings.plaud_bin)
    transcriber = WhisperCppTranscriber(
        whisper_bin=settings.whisper_bin,
        model_path=settings.whisper_model,
        ffmpeg_bin=settings.ffmpeg_bin,
        language=settings.whisper_language,
    )
    pipeline = Pipeline(store, plaud, AudioDownloader(), transcriber, settings.data_dir)
    return store, pipeline, transcriber


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="plaud-brain")
    p.add_argument("--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("discover")
    proc = sub.add_parser("process")
    proc.add_argument("--limit", type=int, default=20)
    once = sub.add_parser("run-once")
    once.add_argument("--limit", type=int, default=20)
    sub.add_parser("status")
    bench = sub.add_parser("benchmark")
    bench.add_argument("audio", type=Path)
    bench.add_argument("--label", default="manual")
    bench.add_argument("--out", type=Path, default=Path("benchmarks/results/stt.csv"))
    return p


def main() -> None:
    args = parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = Settings.from_env()
    store, pipeline, transcriber = build(settings)
    if args.command == "discover":
        print(json.dumps({"discovered": pipeline.discover(settings.plaud_lookback_days, settings.plaud_page_size)}))
    elif args.command == "process":
        print(json.dumps(pipeline.process(args.limit)))
    elif args.command == "run-once":
        discovered = pipeline.discover(settings.plaud_lookback_days, settings.plaud_page_size)
        result = pipeline.process(args.limit)
        print(json.dumps({"discovered": discovered, **result}))
    elif args.command == "status":
        print(json.dumps(store.status_counts(), indent=2, default=str))
    elif args.command == "benchmark":
        row = run_benchmark(transcriber, settings.ffprobe_bin, args.audio.resolve(), args.out.resolve(), args.label)
        print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
