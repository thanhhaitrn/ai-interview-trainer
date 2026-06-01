"""Result schema for lightweight video analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class VideoAnalysisResult:
    """Summary metrics returned by ``VideoAnalyzer``."""

    video_path: str
    frame_count: int
    fps: float
    width: int
    height: int
    duration_sec: float
    sampled_frames: int
    brightness_mean: float
    blur_score_mean: float
    face_detected: bool
    face_visible_ratio: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Convert dataclass to a plain dictionary for JSON output."""
        return asdict(self)

