"""Simple video analyzer for local OpenCV experiments."""

from __future__ import annotations

from pathlib import Path

import cv2

from app.video_analysis.features import (
    calculate_blur_score,
    calculate_brightness,
    detect_faces,
)
from app.video_analysis.schemas import VideoAnalysisResult


class VideoAnalyzer:
    """Analyze basic quality and face visibility signals from a video file."""

    def __init__(self) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade at: {cascade_path}")

    def analyze(self, video_path: str, sample_every_n: int = 10) -> VideoAnalysisResult:
        """Analyze a video and return a JSON-friendly summary object."""
        if sample_every_n <= 0:
            raise ValueError("sample_every_n must be greater than 0.")

        path = Path(video_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video file: {path}")

        warnings: list[str] = []

        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration_sec = float(frame_count / fps) if fps > 0 else 0.0

            sampled_frames = 0
            frames_with_face = 0
            brightness_total = 0.0
            blur_total = 0.0

            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                if frame_index % sample_every_n == 0:
                    sampled_frames += 1

                    brightness = calculate_brightness(frame)
                    blur_score = calculate_blur_score(frame)
                    faces = detect_faces(frame, self.face_cascade)

                    brightness_total += brightness
                    blur_total += blur_score
                    if faces:
                        frames_with_face += 1

                frame_index += 1

            if sampled_frames == 0:
                brightness_mean = 0.0
                blur_score_mean = 0.0
                face_visible_ratio = 0.0
                warnings.append("No frames could be sampled")
            else:
                brightness_mean = brightness_total / sampled_frames
                blur_score_mean = blur_total / sampled_frames
                face_visible_ratio = frames_with_face / sampled_frames

                if frames_with_face == 0:
                    warnings.append("No face detected in sampled frames")
                if brightness_mean < 60:
                    warnings.append("Low lighting detected")
                if blur_score_mean < 80:
                    warnings.append("Video appears blurry")

            return VideoAnalysisResult(
                video_path=str(path),
                frame_count=frame_count,
                fps=fps,
                width=width,
                height=height,
                duration_sec=duration_sec,
                sampled_frames=sampled_frames,
                brightness_mean=brightness_mean,
                blur_score_mean=blur_score_mean,
                face_detected=frames_with_face > 0,
                face_visible_ratio=face_visible_ratio,
                warnings=warnings,
            )
        finally:
            capture.release()

