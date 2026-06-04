"""Frame-level OpenCV feature helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import cv2

Box = tuple[int, int, int, int]


def calculate_brightness(frame: Any) -> float:
    """Return mean grayscale intensity for a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def calculate_blur_score(frame: Any) -> float:
    """Return blur score using variance of Laplacian."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def crop_box_region(frame: Any, box: Box, margin_ratio: float = 0.15) -> Any:
    """Crop a box with a small margin, staying inside frame bounds."""
    x, y, w, h = box
    frame_height, frame_width = frame.shape[:2]
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)

    x1 = max(x - margin_x, 0)
    y1 = max(y - margin_y, 0)
    x2 = min(x + w + margin_x, frame_width)
    y2 = min(y + h + margin_y, frame_height)

    return frame[y1:y2, x1:x2]


def load_haar_cascade(file_name: str) -> cv2.CascadeClassifier | None:
    """Load one OpenCV Haar cascade by file name."""
    cascade_path = Path(cv2.data.haarcascades) / file_name
    cascade = cv2.CascadeClassifier(str(cascade_path))
    return None if cascade.empty() else cascade


def to_box_list(raw_boxes: Iterable[Iterable[int]]) -> list[Box]:
    """Convert OpenCV boxes into plain integer tuples."""
    return [
        (int(x), int(y), int(w), int(h))
        for x, y, w, h in raw_boxes
    ]


def detect_faces(
    frame: Any,
    face_cascade: Any,
    *,
    scale_factor: float = 1.08,
    min_neighbors: int = 8,
    min_face_size: int = 90,
) -> list[Box]:
    """Detect faces in one frame using Haar cascades."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_face_size, min_face_size),
    )
    return to_box_list(faces)


def filter_face_boxes(
    boxes: list[Box],
    frame_height: int,
    frame_width: int | None = None,
    min_face_size: int = 90,
) -> list[Box]:
    """Remove common false-positive face boxes."""
    valid: list[Box] = []
    dynamic_min_size = min_face_size
    if frame_width is not None:
        dynamic_min_size = max(min_face_size, int(min(frame_width, frame_height) * 0.12))

    for x, y, w, h in boxes:
        if w < dynamic_min_size or h < dynamic_min_size:
            continue

        aspect_ratio = float(w) / float(h) if h > 0 else 0.0
        if aspect_ratio < 0.75 or aspect_ratio > 1.35:
            continue

        center_y = y + (h / 2.0)
        if center_y > frame_height * 0.85:
            continue

        valid.append((x, y, w, h))

    return valid


def filter_significant_face_boxes(
    boxes: list[Box],
    main_box: Box | None,
    min_area_ratio: float = 0.45,
) -> list[Box]:
    """Keep boxes large enough to plausibly be another nearby face."""
    if main_box is None:
        return []

    main_area = main_box[2] * main_box[3]
    if main_area <= 0:
        return []

    return [
        box
        for box in boxes
        if (box[2] * box[3]) >= main_area * min_area_ratio
    ]


def pick_largest_box(boxes: list[Box]) -> Box | None:
    """Return the largest face box by area."""
    if not boxes:
        return None
    return max(boxes, key=lambda box: box[2] * box[3])


def box_center(box: Box) -> tuple[float, float]:
    """Return center x/y for a bounding box."""
    x, y, w, h = box
    return x + (w / 2.0), y + (h / 2.0)


def is_box_centered(
    box: Box,
    frame_width: int,
    frame_height: int,
) -> bool:
    """Check whether the main face center is in the central frame region."""
    center_x, center_y = box_center(box)
    return (
        frame_width * 0.30 <= center_x <= frame_width * 0.70
        and frame_height * 0.20 <= center_y <= frame_height * 0.75
    )


def detect_smile_in_face(
    frame: Any,
    face_box: Box,
    smile_cascade: Any,
) -> bool:
    """Detect a smile inside the lower half of the main face."""
    return count_smile_candidates_in_face(frame, face_box, smile_cascade) >= 8


def count_smile_candidates_in_face(
    frame: Any,
    face_box: Box,
    smile_cascade: Any,
) -> int:
    """Count raw smile candidates in the lower half of the main face."""
    x, y, w, h = face_box

    mouth_x1 = x
    mouth_x2 = x + w
    mouth_y1 = y + int(h * 0.50)
    mouth_y2 = y + h
    mouth_region = frame[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
    if mouth_region.size == 0:
        return 0

    gray_mouth = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
    gray_mouth = cv2.equalizeHist(gray_mouth)

    min_smile_size = (
        max(24, int(w * 0.12)),
        max(12, int(h * 0.06)),
    )
    max_smile_size = (
        max(min_smile_size[0] + 1, int(w * 0.90)),
        max(min_smile_size[1] + 1, int(h * 0.45)),
    )

    smiles = smile_cascade.detectMultiScale(
        gray_mouth,
        scaleFactor=1.1,
        minNeighbors=0,
        minSize=min_smile_size,
        maxSize=max_smile_size,
    )

    valid_count = 0
    for smile_box in smiles:
        _, _, smile_width, smile_height = smile_box
        if smile_height <= 0:
            continue

        aspect_ratio = float(smile_width) / float(smile_height)
        width_ratio = float(smile_width) / float(w)
        height_ratio = float(smile_height) / float(h)

        if (
            1.2 <= aspect_ratio <= 9.0
            and 0.12 <= width_ratio <= 0.90
            and 0.04 <= height_ratio <= 0.45
        ):
            valid_count += 1

    return valid_count
