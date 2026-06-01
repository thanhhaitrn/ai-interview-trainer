"""Frame-level OpenCV feature helpers."""

from __future__ import annotations

from typing import Any

import cv2


def calculate_brightness(frame: Any) -> float:
    """Return mean grayscale intensity for a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def calculate_blur_score(frame: Any) -> float:
    """Return blur score using variance of Laplacian."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def detect_faces(frame: Any, face_cascade: Any) -> list[tuple[int, int, int, int]]:
    """Detect faces in one frame using Haar cascades."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    return [
        (int(x), int(y), int(w), int(h))
        for x, y, w, h in faces
    ]

