from plaud_brain.adapters.whisper_cpp import parse_whisper_segments


def test_parse_segments():
    raw = {
        "result": {"language": "fr"},
        "transcription": [
            {"offsets": {"from": 0, "to": 1200}, "text": " Bonjour."},
            {"offsets": {"from": 1200, "to": 3100}, "text": " On commence."},
        ],
    }
    segs = parse_whisper_segments(raw)
    assert len(segs) == 2
    assert segs[0].start_ms == 0
    assert segs[1].end_ms == 3100
    assert segs[0].text == "Bonjour."
