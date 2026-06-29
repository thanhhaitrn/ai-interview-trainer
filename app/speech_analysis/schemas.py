"""Result schemas for speech fluency and voice-quality analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class FluencyMetrics:
    """Delivery/fluency metrics derived from transcript + word timestamps."""

    total_words: int
    span_sec: float
    speech_rate_wpm: float
    articulation_rate_wpm: float
    pause_count: int
    pause_total_sec: float
    mean_pause_sec: float
    long_pause_count: int
    filler_count: int
    filler_per_min: float
    repetition_count: int
    mean_length_of_run: float
    run_count: int
    max_run_length: int
    filler_breakdown: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VoiceMetrics:
    """Acoustic voice-quality metrics; high jitter/shimmer/pitch variability
    indicate a shaky or trembling voice ("giọng run")."""

    sample_rate: int
    voiced_fraction: float
    f0_mean_hz: float | None
    f0_std_hz: float | None
    f0_min_hz: float | None
    f0_max_hz: float | None
    f0_cv: float | None
    jitter_local_pct: float | None
    shimmer_local_pct: float | None
    intensity_mean_db: float | None
    intensity_std_db: float | None
    tremor_label: str
    tremor_indicators: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
