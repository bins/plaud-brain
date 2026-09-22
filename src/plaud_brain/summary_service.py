from __future__ import annotations

import logging

from plaud_brain.summarizer import summarize_transcript

log = logging.getLogger(__name__)

PROMPT_VERSION = "v1"


class SummaryService:
    def __init__(self, store, client, provider: str, model: str) -> None:
        self.store = store
        self.client = client
        self.provider = provider
        self.model = model

    def summarize_one(self, plaud_id: str, refresh: bool = False) -> tuple[dict, bool]:
        transcript = self.store.get_transcript(plaud_id)
        if transcript is None:
            raise ValueError(f"Transcript not found: {plaud_id}")
        if not refresh:
            cached = self.store.get_summary(plaud_id)
            if cached is not None:
                return cached["content"], False

        run = self.store.start_run(transcript["recording_id"], "SUMMARY")
        try:
            summary = summarize_transcript(self.client, transcript)
            self.store.save_summary(
                transcript_id=transcript["transcript_id"],
                provider=self.provider,
                model=self.model,
                prompt_version=PROMPT_VERSION,
                content=summary,
            )
            self.store.finish_run(run, True)
            return summary, True
        except Exception as exc:
            self.store.finish_run(run, False, exc.__class__.__name__, str(exc))
            raise

    def summarize_pending(self, limit: int = 20) -> dict[str, int]:
        counts = {"summarized": 0, "summary_failed": 0}
        for item in self.store.pending_summaries(limit):
            try:
                _, generated = self.summarize_one(item["plaud_id"])
                if generated:
                    counts["summarized"] += 1
            except Exception:
                log.exception("summary failed for %s", item["plaud_id"])
                counts["summary_failed"] += 1
        return counts
