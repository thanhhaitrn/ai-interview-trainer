"""Simple video analyzer for local OpenCV experiments."""

from __future__ import annotations

import math
from pathlib import Path

import cv2

from app.video_analysis.features import (
    box_center,
    calculate_blur_score,
    calculate_brightness,
    crop_box_region,
    count_smile_candidates_in_face,
    detect_faces,
    filter_face_boxes,
    filter_significant_face_boxes,
    is_box_centered,
    load_haar_cascade,
    pick_largest_box,
)
from app.video_analysis.schemas import (
    HeadMovementAmount,
    VideoAnalysisResult,
    VideoPresentationMetrics,
    VideoQualityMetrics,
)


LOW_LIGHTING_THRESHOLD = 60
# Laplacian blur scores vary a lot by camera/video codec, so keep this conservative.
BLURRY_THRESHOLD = 15
FACE_DETECTION_SCALE_FACTOR = 1.08
FACE_DETECTION_MIN_NEIGHBORS = 8
FACE_DETECTION_MIN_SIZE = 90
MIN_RAW_SMILE_CANDIDATES = 15


class VideoAnalyzer:
    """Analyze basic quality and face visibility signals from a video file."""

    def __init__(self) -> None:
        self.face_cascade = load_haar_cascade("haarcascade_frontalface_default.xml")
        if self.face_cascade is None:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            raise RuntimeError(f"Failed to load Haar cascade at: {cascade_path}")

        self.smile_cascade = load_haar_cascade("haarcascade_smile.xml")

    def _head_movement_label(self, score: float) -> str:
        if score < 0.04:
            return "low"
        if score < 0.12:
            return "moderate"
        return "high"

    def _calculate_smile_frequency(self, candidate_counts: list[int]) -> float:
        """
        Estimate smile frequency from raw Haar smile candidate counts.

        This follows OpenCV's smile demo idea: raw nested detections are more
        useful as an intensity signal than a single yes/no cascade box.
        """
        if not candidate_counts:
            return 0.0

        max_count = max(candidate_counts)
        min_count = min(candidate_counts)

        if max_count < MIN_RAW_SMILE_CANDIDATES:
            return 0.0

        if max_count == min_count:
            return 1.0 if max_count >= MIN_RAW_SMILE_CANDIDATES * 2 else 0.0

        dynamic_threshold = min_count + max(3, int((max_count - min_count) * 0.45))
        smiling_frames = sum(
            1
            for count in candidate_counts
            if count >= MIN_RAW_SMILE_CANDIDATES and count >= dynamic_threshold
        )
        return smiling_frames / len(candidate_counts)

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
            resolution = f"{width}x{height}"

            sampled_frames = 0
            frames_with_face = 0
            camera_facing_frames = 0
            centered_frames = 0
            smile_candidate_counts: list[int] = []
            multiple_faces_detected = False
            brightness_total = 0.0
            blur_total = 0.0
            movement_total = 0.0
            movement_steps = 0
            previous_face_center: tuple[float, float] | None = None
            frame_diagonal = math.hypot(width, height) if width and height else 1.0

            if self.smile_cascade is None:
                warnings.append("Smile detection unavailable")

            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                if frame_index % sample_every_n == 0:
                    sampled_frames += 1

                    brightness = calculate_brightness(frame)
                    raw_faces = detect_faces(
                        frame,
                        self.face_cascade,
                        scale_factor=FACE_DETECTION_SCALE_FACTOR,
                        min_neighbors=FACE_DETECTION_MIN_NEIGHBORS,
                        min_face_size=FACE_DETECTION_MIN_SIZE,
                    )
                    valid_faces = filter_face_boxes(
                        raw_faces,
                        frame_height=height,
                        frame_width=width,
                        min_face_size=FACE_DETECTION_MIN_SIZE,
                    )
                    main_face = pick_largest_box(valid_faces)
                    significant_faces = filter_significant_face_boxes(
                        valid_faces,
                        main_face,
                    )
                    blur_frame = (
                        crop_box_region(frame, main_face)
                        if main_face is not None
                        else frame
                    )
                    blur_score = calculate_blur_score(blur_frame)

                    brightness_total += brightness
                    blur_total += blur_score
                    if len(significant_faces) > 1:
                        multiple_faces_detected = True

                    if main_face is not None:
                        frames_with_face += 1
                        face_center = box_center(main_face)

                        if is_box_centered(main_face, width, height):
                            centered_frames += 1
                            # This accepts screen-facing or camera-facing posture; it is not gaze tracking.
                            camera_facing_frames += 1

                        if previous_face_center is not None:
                            movement_total += (
                                math.dist(previous_face_center, face_center)
                                / frame_diagonal
                            )
                            movement_steps += 1

                        previous_face_center = face_center

                        if self.smile_cascade is not None:
                            smile_candidate_counts.append(
                                count_smile_candidates_in_face(
                                    frame,
                                    main_face,
                                    self.smile_cascade,
                                )
                            )

                frame_index += 1

            if sampled_frames == 0:
                brightness_mean = 0.0
                blur_score_mean = 0.0
                face_visible_ratio = 0.0
                camera_facing_ratio = 0.0
                candidate_centered_ratio = 0.0
                smile_frequency = None if self.smile_cascade is None else 0.0
                warnings.append("No frames could be sampled")
            else:
                brightness_mean = brightness_total / sampled_frames
                blur_score_mean = blur_total / sampled_frames
                face_visible_ratio = frames_with_face / sampled_frames
                camera_facing_ratio = camera_facing_frames / sampled_frames
                candidate_centered_ratio = centered_frames / sampled_frames
                smile_frequency = (
                    None
                    if self.smile_cascade is None
                    else self._calculate_smile_frequency(smile_candidate_counts)
                )

                if frames_with_face == 0:
                    warnings.append("No face detected in sampled frames")
                if brightness_mean < LOW_LIGHTING_THRESHOLD:
                    warnings.append("Low lighting detected")
                if blur_score_mean < BLURRY_THRESHOLD:
                    warnings.append("Video appears blurry")
                if multiple_faces_detected:
                    warnings.append("Multiple faces detected")

            warnings.append(
                "Camera-facing ratio is approximate; it accepts screen-facing or camera-facing posture and does not measure real eye contact"
            )

            movement_score = movement_total / movement_steps if movement_steps else 0.0

            return VideoAnalysisResult(
                video_path=str(path),
                frame_count=frame_count,
                fps=fps,
                width=width,
                height=height,
                resolution=resolution,
                duration_sec=duration_sec,
                sampled_frames=sampled_frames,
                video_presentation=VideoPresentationMetrics(
                    face_visible_ratio=face_visible_ratio,
                    camera_facing_ratio=camera_facing_ratio,
                    candidate_centered_ratio=candidate_centered_ratio,
                    head_movement_amount=HeadMovementAmount(
                        score=movement_score,
                        label=self._head_movement_label(movement_score),
                    ),
                    smile_frequency=smile_frequency,
                ),
                video_quality=VideoQualityMetrics(
                    brightness_mean=brightness_mean,
                    blur_score_mean=blur_score_mean,
                    resolution=resolution,
                    fps=fps,
                    multiple_faces_detected=multiple_faces_detected,
                ),
                face_detected=frames_with_face > 0,
                warnings=warnings,
            )
        finally:
            capture.release()
