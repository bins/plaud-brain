from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

from plaud_brain.adapters.downloader import AudioDownloader
from plaud_brain.adapters.cloud_stt import FallbackTranscriber, OpenAICompatibleSTTTranscriber
from plaud_brain.adapters.embeddings import EmbeddingProviderError, OpenAICompatibleEmbeddingClient
from plaud_brain.adapters.openai_compatible import CloudProviderError, OpenAICompatibleClient
from plaud_brain.adapters.plaud_cli import PlaudCliAdapter
from plaud_brain.adapters.whisper_cpp import WhisperCppTranscriber
from plaud_brain.benchmark import run_benchmark
from plaud_brain.config import Settings
from plaud_brain.embedding_service import EmbeddingService
from plaud_brain.pipeline import Pipeline
from plaud_brain.presentation import (
    render_knowledge_markdown,
    render_rag_markdown,
    render_transcript_markdown,
    render_transcript_text,
)
from plaud_brain.rag_service import RagService
from plaud_brain.store import PostgresStore
from plaud_brain.summarizer import render_summary_markdown
from plaud_brain.summary_service import SummaryService


def build(settings: Settings) -> tuple[PostgresStore, Pipeline, WhisperCppTranscriber]:
    store = PostgresStore(settings.database_url)
    plaud = PlaudCliAdapter(settings.plaud_bin)
    local_transcriber = WhisperCppTranscriber(
        whisper_bin=settings.whisper_bin,
        model_path=settings.whisper_model,
        ffmpeg_bin=settings.ffmpeg_bin,
        language=settings.whisper_language,
        threads=settings.whisper_threads,
        no_gpu=settings.whisper_no_gpu,
        flash_attn=settings.whisper_flash_attn,
    )
    if settings.stt_provider == "local":
        transcriber = local_transcriber
    else:
        cloud_transcriber = OpenAICompatibleSTTTranscriber(
            provider=settings.stt_provider,
            base_url=settings.stt_base_url,
            model=settings.stt_model,
            api_key_env=settings.stt_api_key_env,
            ffmpeg_bin=settings.ffmpeg_bin,
            language=settings.whisper_language,
            chunk_seconds=settings.stt_chunk_seconds,
            timeout_seconds=settings.stt_timeout_seconds,
        )
        if settings.stt_fallback_provider == "local":
            transcriber = FallbackTranscriber(cloud_transcriber, local_transcriber)
        elif settings.stt_fallback_provider in {"", "none"}:
            transcriber = cloud_transcriber
        else:
            raise ValueError("STT_FALLBACK_PROVIDER must be local or none")
    pipeline = Pipeline(store, plaud, AudioDownloader(), transcriber, settings.data_dir)
    return store, pipeline, local_transcriber


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
    recordings = sub.add_parser("recordings")
    recordings.add_argument("--limit", type=int, default=20)
    transcript = sub.add_parser("transcript")
    transcript.add_argument("plaud_id")
    transcript.add_argument("--format", choices=("text", "markdown", "json"), default="text")
    transcript.add_argument("--output", type=Path)
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=20)
    knowledge_search = sub.add_parser("knowledge-search")
    knowledge_search.add_argument("query")
    knowledge_search.add_argument("--limit", type=int, default=20)
    semantic_search = sub.add_parser("semantic-search")
    semantic_search.add_argument("query")
    semantic_search.add_argument("--limit", type=int, default=20)
    _add_retrieval_filters(semantic_search)
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--format", choices=("json", "markdown"), default="markdown")
    ask.add_argument("--output", type=Path)
    ask.add_argument("--refresh", action="store_true")
    _add_retrieval_filters(ask)
    index = sub.add_parser("index")
    index.add_argument("--rebuild", action="store_true")
    actions = sub.add_parser("actions")
    actions.add_argument("--status", choices=("open", "done", "dismissed", "all"), default="open")
    actions.add_argument("--limit", type=int, default=100)
    action = sub.add_parser("action")
    action.add_argument("action_id")
    action.add_argument("status", choices=("open", "done", "dismissed"))
    decisions = sub.add_parser("decisions")
    decisions.add_argument("--limit", type=int, default=100)
    show = sub.add_parser("show")
    show.add_argument("plaud_id")
    show.add_argument("--format", choices=("json", "markdown"), default="markdown")
    show.add_argument("--output", type=Path)
    summarize = sub.add_parser("summarize")
    summarize.add_argument("plaud_id")
    summarize.add_argument("--format", choices=("json", "markdown"), default="markdown")
    summarize.add_argument("--output", type=Path)
    summarize.add_argument("--refresh", action="store_true")
    summary = sub.add_parser("summary")
    summary.add_argument("plaud_id")
    summary.add_argument("--format", choices=("json", "markdown"), default="markdown")
    summary.add_argument("--output", type=Path)
    bench = sub.add_parser("benchmark")
    bench.add_argument("audio", type=Path)
    bench.add_argument("--label", default="manual")
    bench.add_argument("--out", type=Path, default=Path("benchmarks/results/stt.csv"))
    return p


def _add_retrieval_filters(command: argparse.ArgumentParser) -> None:
    command.add_argument("--recording")
    command.add_argument(
        "--source-type",
        action="append",
        choices=("transcript", "summary", "decision", "action_item"),
        dest="source_types",
    )
    command.add_argument("--since", type=date.fromisoformat)
    command.add_argument("--until", type=date.fromisoformat)


def main() -> None:
    args = parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = Settings.from_env()
    store, pipeline, transcriber = build(settings)
    summary_client = OpenAICompatibleClient(
        provider=settings.summary_provider,
        base_url=settings.summary_base_url,
        model=settings.summary_model,
        api_key_env=settings.summary_api_key_env,
        timeout_seconds=settings.summary_timeout_seconds,
    )
    summary_service = SummaryService(
        store, summary_client, settings.summary_provider, settings.summary_model
    )
    embedding_client = OpenAICompatibleEmbeddingClient(
        provider=settings.embedding_provider,
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        api_key_env=settings.embedding_api_key_env,
        dimensions=settings.embedding_dimensions,
        timeout_seconds=settings.embedding_timeout_seconds,
    )
    embedding_service = EmbeddingService(
        store,
        embedding_client,
        settings.embedding_provider,
        settings.embedding_model,
        settings.embedding_dimensions,
        batch_size=settings.embedding_batch_size,
        document_input_type=settings.embedding_document_input_type or None,
        query_input_type=settings.embedding_query_input_type or None,
    )
    qa_client = OpenAICompatibleClient(
        provider=settings.qa_provider,
        base_url=settings.qa_base_url,
        model=settings.qa_model,
        api_key_env=settings.qa_api_key_env,
        timeout_seconds=settings.qa_timeout_seconds,
    )
    rag_service = RagService(
        store,
        embedding_service,
        qa_client,
        settings.qa_provider,
        settings.qa_model,
        top_k=settings.qa_top_k,
        min_score=settings.qa_min_score,
    )
    if args.command == "discover":
        print(json.dumps({"discovered": pipeline.discover(settings.plaud_lookback_days, settings.plaud_page_size)}))
    elif args.command == "process":
        print(json.dumps(pipeline.process(args.limit)))
    elif args.command == "run-once":
        discovered = pipeline.discover(settings.plaud_lookback_days, settings.plaud_page_size)
        result = pipeline.process(args.limit)
        summary_result = (
            summary_service.summarize_pending(args.limit)
            if settings.summary_auto
            else {"summarized": 0, "summary_failed": 0}
        )
        index_result: dict[str, int] = {
            "sources": 0,
            "indexed": 0,
            "reused": 0,
            "deleted": 0,
            "index_failed": 0,
        }
        if settings.embedding_auto:
            try:
                index_result.update(embedding_service.sync())
            except Exception:
                logging.getLogger(__name__).exception("embedding index failed")
                index_result["index_failed"] = 1
        print(json.dumps({"discovered": discovered, **result, **summary_result, **index_result}))
    elif args.command == "status":
        print(json.dumps(store.status_counts(), indent=2, default=str))
    elif args.command == "recordings":
        print(json.dumps(store.list_recordings(args.limit), indent=2, default=str, ensure_ascii=False))
    elif args.command == "transcript":
        transcript = store.get_transcript(args.plaud_id)
        if transcript is None:
            raise SystemExit(f"Transcript not found: {args.plaud_id}")
        if args.format == "json":
            rendered = json.dumps(transcript, indent=2, default=str, ensure_ascii=False) + "\n"
        elif args.format == "markdown":
            rendered = render_transcript_markdown(transcript)
        else:
            rendered = render_transcript_text(transcript) + "\n"
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
    elif args.command == "search":
        print(json.dumps(store.search_transcripts(args.query, args.limit), indent=2, default=str, ensure_ascii=False))
    elif args.command == "knowledge-search":
        print(json.dumps(store.search_knowledge(args.query, args.limit), indent=2, default=str, ensure_ascii=False))
    elif args.command == "semantic-search":
        try:
            matches = embedding_service.search(
                args.query,
                args.limit,
                recording=args.recording,
                source_types=set(args.source_types or []),
                since=args.since,
                until=args.until,
            )
        except (EmbeddingProviderError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        print(json.dumps(matches, indent=2, default=str, ensure_ascii=False))
    elif args.command == "ask":
        try:
            result = rag_service.ask(
                args.question,
                recording=args.recording,
                source_types=set(args.source_types or []),
                since=args.since,
                until=args.until,
                refresh=args.refresh,
            )
        except (CloudProviderError, EmbeddingProviderError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        rendered = (
            json.dumps(result, indent=2, default=str, ensure_ascii=False) + "\n"
            if args.format == "json"
            else render_rag_markdown(result)
        )
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
    elif args.command == "index":
        try:
            result = embedding_service.sync(rebuild=args.rebuild)
        except (EmbeddingProviderError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        print(json.dumps(result, indent=2))
    elif args.command == "actions":
        print(json.dumps(store.list_action_items(args.status, args.limit), indent=2, default=str, ensure_ascii=False))
    elif args.command == "action":
        item = store.update_action_status(args.action_id, args.status)
        if item is None:
            raise SystemExit(f"Action item not found: {args.action_id}")
        print(json.dumps(item, indent=2, default=str, ensure_ascii=False))
    elif args.command == "decisions":
        print(json.dumps(store.list_decisions(args.limit), indent=2, default=str, ensure_ascii=False))
    elif args.command == "show":
        transcript = store.get_transcript(args.plaud_id)
        if transcript is None:
            raise SystemExit(f"Transcript not found: {args.plaud_id}")
        cached = store.get_summary(args.plaud_id)
        summary = cached["content"] if cached else None
        if args.format == "json":
            rendered = json.dumps(
                {"transcript": transcript, "summary": summary},
                indent=2,
                default=str,
                ensure_ascii=False,
            ) + "\n"
        else:
            rendered = render_knowledge_markdown(transcript, summary)
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
    elif args.command == "summarize":
        transcript = store.get_transcript(args.plaud_id)
        if transcript is None:
            raise SystemExit(f"Transcript not found: {args.plaud_id}")
        try:
            summary, _ = summary_service.summarize_one(args.plaud_id, refresh=args.refresh)
        except (CloudProviderError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        if args.format == "json":
            rendered = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        else:
            rendered = render_summary_markdown(transcript, summary)
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
    elif args.command == "summary":
        transcript = store.get_transcript(args.plaud_id)
        cached = store.get_summary(args.plaud_id)
        if transcript is None:
            raise SystemExit(f"Transcript not found: {args.plaud_id}")
        if cached is None:
            raise SystemExit(f"Summary not found: {args.plaud_id}")
        summary = cached["content"]
        if args.format == "json":
            rendered = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        else:
            rendered = render_summary_markdown(transcript, summary)
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
    elif args.command == "benchmark":
        row = run_benchmark(transcriber, settings.ffprobe_bin, args.audio.resolve(), args.out.resolve(), args.label)
        print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
