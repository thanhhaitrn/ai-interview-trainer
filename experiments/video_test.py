"""Run local video analysis from the command line."""

from __future__ import annotations

import argparse
import json
import sys

from app.video_analysis import VideoAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze basic video quality and face visibility.")
    parser.add_argument("video_path", help="Path to a local video file.")
    parser.add_argument(
        "--sample-every-n",
        type=int,
        default=10,
        help="Analyze every Nth frame. Default: 10",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        analyzer = VideoAnalyzer()
        result = analyzer.analyze(
            video_path=args.video_path,
            sample_every_n=args.sample_every_n,
        )
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()

