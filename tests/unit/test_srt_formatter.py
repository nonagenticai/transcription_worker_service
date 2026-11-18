"""SRT output formatting."""

from __future__ import annotations

from src.services.backends.base import TranscriptSegment
from src.services.srt_formatter import _format_timestamp, format_srt


def test_format_timestamp_zero() -> None:
    assert _format_timestamp(0) == "00:00:00,000"


def test_format_timestamp_hours_minutes_seconds() -> None:
    assert _format_timestamp(3661.25) == "01:01:01,250"


def test_format_timestamp_negative_clamped() -> None:
    assert _format_timestamp(-1.5) == "00:00:00,000"


def test_format_srt_multi_segment() -> None:
    segs = [
        TranscriptSegment(id=0, start=0.0, end=1.2, text="Hello world"),
        TranscriptSegment(id=1, start=1.5, end=3.0, text="Second line."),
    ]
    srt = format_srt(segs)
    assert "1\n00:00:00,000 --> 00:00:01,200\nHello world" in srt
    assert "2\n00:00:01,500 --> 00:00:03,000\nSecond line." in srt
    assert srt.endswith("\n")


def test_format_srt_empty() -> None:
    assert format_srt([]) == "\n"


def test_format_srt_rounding_overflow() -> None:
    # 0.9999 → 999 ms (rounded), verify no 1000 leak
    assert _format_timestamp(0.9999) == "00:00:01,000"


def test_format_timestamp_carries_seconds_overflow_into_minutes() -> None:
    # Regression: used to emit "00:01:60,000", which is not valid SRT.
    assert _format_timestamp(119.9996) == "00:02:00,000"


def test_format_timestamp_carries_minutes_overflow_into_hours() -> None:
    # Regression: used to emit "00:59:60,000".
    assert _format_timestamp(3599.9999) == "01:00:00,000"


def test_format_timestamp_never_emits_out_of_range_fields() -> None:
    """No input may produce a 60-second or 60-minute field."""
    probes = [0.9999, 59.9999, 119.9996, 599.9999, 3599.9999, 7199.99999, 86399.9999]
    for value in probes:
        stamp = _format_timestamp(value)
        hours, minutes, rest = stamp.split(":")
        secs, millis = rest.split(",")
        assert int(minutes) < 60, stamp
        assert int(secs) < 60, stamp
        assert int(millis) < 1000, stamp
        assert int(hours) >= 0, stamp
