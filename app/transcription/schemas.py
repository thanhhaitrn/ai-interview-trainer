"""Result schemas for video/audio transcription."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class TranscriptWord:
    """A single timestamped word from the transcript."""

    start: float
    end: float
    word: str


@dataclass(slots=True)
class TranscriptSegment:
    """A single timestamped chunk of transcribed speech."""

    start: float
    end: float
    text: str
    words: list[TranscriptWord] = field(default_factory=list)


@dataclass(slots=True)
class TranscriptResult:
    """JSON-friendly transcript returned by ``VideoTranscriber``."""

    source_path: str
    language: str
    duration_sec: float
    model: str
    text: str
    segments: list[TranscriptSegment] = field(default_factory=list)

    def all_words(self) -> list[TranscriptWord]:
        """Flatten word-level timestamps across all segments in order."""
        return [word for segment in self.segments for word in segment.words]

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to a plain dictionary for JSON output."""
        return asdict(self)
