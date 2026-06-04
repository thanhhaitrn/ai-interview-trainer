"""Terminal-only video interview recording and analysis experiment."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2

from app.video_analysis import VideoAnalyzer


CAMERA_WARMUP_SECONDS = 5.0
FRAME_RETRY_SECONDS = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record or analyze a video interview answer from the terminal."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index for cv2.VideoCapture. Default: 0",
    )
    parser.add_argument(
        "--record-seconds",
        type=int,
        default=20,
        help="Maximum recording length in seconds. Default: 20",
    )
    parser.add_argument(
        "--sample-every-n",
        type=int,
        default=10,
        help="Analyze every Nth frame. Default: 10",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary webcam recording after analysis.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runtime_uploads/temp_recordings"),
        help="Temporary recording folder. Default: runtime_uploads/temp_recordings",
    )
    return parser.parse_args()


def open_camera(camera_index: int) -> cv2.VideoCapture:
    """Open webcam, preferring the native macOS backend when available."""
    backend_attempts: list[tuple[str, int]] = []
    if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        backend_attempts.append(("AVFoundation", cv2.CAP_AVFOUNDATION))
    backend_attempts.append(("default backend", cv2.CAP_ANY))

    for backend_name, backend_id in backend_attempts:
        capture = cv2.VideoCapture(camera_index, backend_id)
        if capture.isOpened():
            return capture
        capture.release()

    raise RuntimeError(f"Could not open webcam at camera index {camera_index}.")


def read_camera_frame(
    capture: cv2.VideoCapture,
    *,
    timeout_seconds: float,
) -> tuple[bool, object | None]:
    """Wait briefly for a usable camera frame."""
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        ok, frame = capture.read()
        if ok and frame is not None and getattr(frame, "size", 0) > 0:
            return True, frame
        time.sleep(0.05)

    return False, None


def make_temp_video_base(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    return output_dir / f"video_interview_{timestamp}_{unique_id}"


def open_video_writer(
    base_path: Path,
    fps: float,
    width: int,
    height: int,
) -> tuple[cv2.VideoWriter, Path]:
    attempts = [
        (base_path.with_suffix(".mp4"), "mp4v"),
        (base_path.with_suffix(".avi"), "XVID"),
    ]

    for output_path, codec in attempts:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer, output_path
        writer.release()

    raise RuntimeError("Could not create an .mp4 or .avi video writer.")


def draw_recording_overlay(frame, elapsed_sec: float) -> None:
    cv2.putText(
        frame,
        "Recording... Press q to stop early",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Elapsed: {elapsed_sec:.1f}s",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )


def record_webcam_answer(
    *,
    camera_index: int,
    record_seconds: int,
    output_dir: Path,
) -> Path:
    if record_seconds <= 0:
        raise ValueError("--record-seconds must be greater than 0.")

    capture = open_camera(camera_index)

    writer: cv2.VideoWriter | None = None
    output_path: Path | None = None
    frames_written = 0
    window_name = "Video Interview Recording"

    try:
        print("Starting camera. Waiting for the first frame...")
        ok, frame = read_camera_frame(
            capture,
            timeout_seconds=CAMERA_WARMUP_SECONDS,
        )
        if not ok:
            raise RuntimeError(
                "Could not read from webcam. Check macOS camera permission, "
                "close other apps using the camera, or try --camera-index 1."
            )

        height, width = frame.shape[:2]
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 20.0)
        if fps <= 0:
            fps = 20.0

        writer, output_path = open_video_writer(
            make_temp_video_base(output_dir),
            fps=fps,
            width=width,
            height=height,
        )

        start_time = time.monotonic()

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed >= record_seconds:
                break

            preview_frame = frame.copy()
            draw_recording_overlay(preview_frame, elapsed)
            cv2.imshow(window_name, preview_frame)

            writer.write(frame)
            frames_written += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            ok, frame = read_camera_frame(
                capture,
                timeout_seconds=FRAME_RETRY_SECONDS,
            )
            if not ok:
                print("Stopped early because webcam frame could not be read.")
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if output_path is None or frames_written == 0:
        raise RuntimeError("No video frames were recorded.")

    return output_path


def ask_yes_no(prompt: str) -> bool:
    while True:
        answer = input(prompt).strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please enter y or n.")


def ask_existing_video_path() -> Path:
    raw_path = input("Enter video path to analyze: ").strip()
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    return path


def analyze_and_print(video_path: Path, sample_every_n: int) -> None:
    analyzer = VideoAnalyzer()
    result = analyzer.analyze(str(video_path), sample_every_n=sample_every_n)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    temp_video_path: Path | None = None

    try:
        should_record = ask_yes_no("Try video interview recording? (y/n): ")

        if should_record:
            temp_video_path = record_webcam_answer(
                camera_index=args.camera_index,
                record_seconds=args.record_seconds,
                output_dir=args.output_dir,
            )
            analyze_and_print(temp_video_path, args.sample_every_n)
        else:
            existing_video_path = ask_existing_video_path()
            analyze_and_print(existing_video_path, args.sample_every_n)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    finally:
        if temp_video_path is not None and not args.keep_temp:
            try:
                temp_video_path.unlink(missing_ok=True)
            except OSError as error:
                print(f"Warning: could not delete temp video: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
