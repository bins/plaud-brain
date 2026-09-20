from plaud_brain.adapters.plaud_cli import parse_key_values, parse_recording_table


def test_parse_unicode_table():
    text = """
┌──────────────────────┬──────────────────┬───────────────────────────┬──────────┐
│ ID                   │ Name             │ Created At                │ Duration │
├──────────────────────┼──────────────────┼───────────────────────────┼──────────┤
│ abcdef1234567890     │ Team sync        │ 2026-09-20T08:00:00+02:00 │ 42:10    │
│ fedcba0987654321     │ Voice memo       │ 2026-09-20T09:00:00+02:00 │ 3m 5s    │
└──────────────────────┴──────────────────┴───────────────────────────┴──────────┘
"""
    rows = parse_recording_table(text)
    assert len(rows) == 2
    assert rows[0].plaud_id == "abcdef1234567890"
    assert rows[0].name == "Team sync"
    assert rows[0].duration_seconds == 2530
    assert rows[1].duration_seconds == 185


def test_parse_key_value_output():
    text = """
id          abcdef1234567890
name        Example
created_at  2026-09-20T08:00:00Z
"""
    data = parse_key_values(text)
    assert data["id"] == "abcdef1234567890"
    assert data["name"] == "Example"
