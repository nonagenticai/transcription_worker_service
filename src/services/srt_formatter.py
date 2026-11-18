"""SRT subtitle formatter for TranscriptSegment lists."""

from __future__ import annotations

from collections.abc import Iterable

from src.services.backends.base import TranscriptSegment


def _format_timestamp(seconds: float) -> str:
    """Render seconds as an SRT ``HH:MM:SS,mmm`` timestamp.

    Rounding is done ONCE, on total milliseconds, then decomposed. The previous
    implementation split the value first and carried the millisecond rounding
    overflow into seconds only, so ``119.9996`` produced ``00:01:60,000`` and
    ``3599.9999`` produced ``00:59:60,000`` — neither is a valid SRT timestamp,
    and players reject or mis-seek the whole cue.
    """
    if seconds < 0:
        seconds = 0.0
    total_millis = round(seconds * 1000)
    millis = total_millis % 1000
    total_secs = total_millis // 1000
    secs = total_secs % 60
    minutes = (total_secs // 60) % 60
    hours = total_secs // 3600
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_srt(segments: Iterable[TranscriptSegment]) -> str:
    """Render a list of TranscriptSegment into SRT-formatted text."""
    lines: list[str] = []
    for index, seg in enumerate(segments, start=1):
        lines.append(str(index))
        lines.append(f"{_format_timestamp(seg.start)} --> {_format_timestamp(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
