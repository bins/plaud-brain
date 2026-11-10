import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "m6a_acceptance.py"
spec = importlib.util.spec_from_file_location("m6a_acceptance", SCRIPT)
m6a = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m6a  # dataclasses look the module up while loading
spec.loader.exec_module(m6a)


def test_digest_ignores_speaker_labels_but_not_text_or_timing():
    base = [{"seq": 0, "start_ms": 0, "end_ms": 1000, "text": "hi", "speaker": None}]
    labeled = [{**base[0], "speaker": "SPEAKER_00"}]
    assert m6a.segments_digest(base) == m6a.segments_digest(labeled)
    assert m6a.segments_digest(base) != m6a.segments_digest([{**base[0], "text": "hey"}])
    assert m6a.segments_digest(base) != m6a.segments_digest([{**base[0], "end_ms": 999}])


def test_alignment_stats_counts_weak_coverage_and_changes():
    segments = [
        {"start_ms": 0, "end_ms": 1000, "speaker": "A"},
        {"start_ms": 1000, "end_ms": 2000, "speaker": "B"},
        {"start_ms": 2000, "end_ms": 3000, "speaker": "B"},
        {"start_ms": 3000, "end_ms": 4000, "speaker": None},
    ]
    turns = [
        {"start_ms": 0, "end_ms": 1000, "speaker": "A"},
        {"start_ms": 1000, "end_ms": 2000, "speaker": "B"},
        {"start_ms": 2000, "end_ms": 2300, "speaker": "B"},
    ]
    stats = m6a.alignment_stats(segments, turns)
    assert stats == {"segments": 4, "labeled": 3, "weak": 1, "changes": [1000]}


def test_run_measured_reports_exit_code_and_memory():
    run = m6a.run_measured([sys.executable, "-c", "import sys; print('ok'); sys.exit(3)"])
    assert run.returncode == 3
    assert run.stdout.strip() == "ok"
    assert run.peak_rss_bytes > 0
    assert run.seconds > 0


def test_formatting_helpers():
    assert m6a.format_ms(3_725_000) == "01:02:05"
    assert m6a.realtime(30.0, 60.0) == "0.500"
    assert m6a.realtime(30.0, None) == "?"
    assert m6a.gib(2 * 1024**3) == "2.00 GiB"
