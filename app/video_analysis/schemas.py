"""Result schemas for lightweight video analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class HeadMovementAmount:
    """Average movement of the main face center across sampled frames."""

    score: float
    label: str


@dataclass(slots=True)
class VideoPresentationMetrics:
    """Observable presentation metrics from sampled video frames."""

    face_visible_ratio: float
    camera_facing_ratio: float
    candidate_centered_ratio: float
    head_movement_amount: HeadMovementAmount
    smile_frequency: float | None


@dataclass(slots=True)
class VideoQualityMetrics:
    """Basic video quality metrics from sampled video frames."""

    brightness_mean: float
    blur_score_mean: float
    resolution: str
    fps: float
    multiple_faces_detected: bool


@dataclass(slots=True)
class VideoAnalysisResult:
    """JSON-friendly summary returned by ``VideoAnalyzer``."""

    video_path: str
    frame_count: int
    fps: float
    width: int
    height: int
    resolution: str
    duration_sec: float
    sampled_frames: int
    video_presentation: VideoPresentationMetrics
    video_quality: VideoQualityMetrics
    face_detected: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to a plain dictionary for JSON output."""
        return asdict(self)
