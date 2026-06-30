"""Simple webcam face-detection test using OpenCV Haar cascade."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2

Box = tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run webcam face detection.")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index for cv2.VideoCapture (default: 0).",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.08,
        help="Haar cascade scale factor (default: 1.08).",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=8,
        help="Haar cascade min neighbors (default: 8).",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=90,
        help="Minimum face width/height in pixels (default: 90).",
    )
    parser.add_argument(
        "--allow-multiple-faces",
        action="store_true",
        help="Draw all valid faces instead of only the largest one.",
    )
    parser.add_argument(
        "--max-missed-frames",
        type=int,
        default=5,
        help="Keep last valid box for this many missed frames (default: 5).",
    )
    parser.add_argument(
        "--debug-raw",
        action="store_true",
        help="Draw rejected raw detections in red for debugging.",
    )
    return parser.parse_args()


def load_face_cascade() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade: {cascade_path}")
    return face_cascade


def to_box_list(raw_faces: Iterable[Iterable[int]]) -> list[Box]:
    return [
        (int(x), int(y), int(w), int(h))
        for x, y, w, h in raw_faces
    ]


def filter_face_boxes(
    boxes: list[Box],
    frame_height: int,
    min_face_size: int,
) -> tuple[list[Box], list[Box]]:
    valid: list[Box] = []
    rejected: list[Box] = []

    for x, y, w, h in boxes:
        if w < min_face_size or h < min_face_size:
            rejected.append((x, y, w, h))
            continue

        aspect_ratio = float(w) / float(h) if h > 0 else 0.0
        if aspect_ratio < 0.75 or aspect_ratio > 1.35:
            rejected.append((x, y, w, h))
            continue

        center_y = y + (h / 2.0)
        if center_y > frame_height * 0.85:
            rejected.append((x, y, w, h))
            continue

        valid.append((x, y, w, h))

    return valid, rejected


def pick_largest_box(boxes: list[Box]) -> Box | None:
    if not boxes:
        return None
    return max(boxes, key=lambda box: box[2] * box[3])


def draw_box(frame, box: Box, color: tuple[int, int, int], thickness: int = 2) -> None:
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)


def draw_overlay(
    frame,
    raw_count: int,
    valid_count: int,
    displayed_count: int,
) -> None:
    cv2.putText(
        frame,
        f"Raw detections: {raw_count}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Valid detections: {valid_count}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Displayed faces: {displayed_count}",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "Press q to quit",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )


def main() -> None:
    args = parse_args()

    if args.scale_factor <= 1.0:
        raise ValueError("--scale-factor must be greater than 1.0.")
    if args.min_neighbors < 0:
        raise ValueError("--min-neighbors must be >= 0.")
    if args.min_face_size <= 0:
        raise ValueError("--min-face-size must be > 0.")
    if args.max_missed_frames < 0:
        raise ValueError("--max-missed-frames must be >= 0.")

    face_cascade = load_face_cascade()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam at camera index {args.camera_index}."
        )

    window_name = "Face Camera Test (press q to quit)"
    last_valid_box: Box | None = None
    missed_frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            raw_faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=args.scale_factor,
                minNeighbors=args.min_neighbors,
                minSize=(args.min_face_size, args.min_face_size),
            )
            raw_boxes = to_box_list(raw_faces)

            frame_height = frame.shape[0]
            valid_boxes, rejected_boxes = filter_face_boxes(
                boxes=raw_boxes,
                frame_height=frame_height,
                min_face_size=args.min_face_size,
            )

            displayed_boxes: list[Box] = []
            largest_valid = pick_largest_box(valid_boxes)

            if valid_boxes:
                missed_frames = 0
                if args.allow_multiple_faces:
                    displayed_boxes = list(valid_boxes)
                    if largest_valid is not None:
                        last_valid_box = largest_valid
                elif largest_valid is not None:
                    displayed_boxes = [largest_valid]
                    last_valid_box = largest_valid
            else:
                missed_frames += 1
                if last_valid_box is not None and missed_frames <= args.max_missed_frames:
                    displayed_boxes = [last_valid_box]
                else:
                    last_valid_box = None

            if args.debug_raw:
                for box in rejected_boxes:
                    draw_box(frame, box, color=(0, 0, 255), thickness=2)

            for box in displayed_boxes:
                draw_box(frame, box, color=(0, 255, 0), thickness=2)

            draw_overlay(
                frame=frame,
                raw_count=len(raw_boxes),
                valid_count=len(valid_boxes),
                displayed_count=len(displayed_boxes),
            )

            cv2.imshow(window_name, frame)

            # Exit when user presses the "q" key.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
