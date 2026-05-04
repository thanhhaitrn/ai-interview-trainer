from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUESTS_DIR = Path("data/requests")
EVALUATIONS_DIR = Path("data/evaluations")


def load_request_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_answer_from_request_file(path: Path) -> dict[str, Any]:
    from app.service import evaluate_answer

    request_payload = load_request_json(path)
    return evaluate_answer(request_payload)


def save_evaluated_answer(
    request_json_path: Path,
    output_dir: Path = EVALUATIONS_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_result = evaluate_answer_from_request_file(request_json_path)
    output_path = output_dir / f"{request_json_path.stem}_evaluation.json"

    output_path.write_text(
        json.dumps(evaluation_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an interview answer from a JSON request file."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=REQUESTS_DIR / "evaluation_request.json",
        help="Path to the evaluation request JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EVALUATIONS_DIR,
        help="Folder where the evaluation JSON file will be saved.",
    )
    args = parser.parse_args()

    if not args.input_path.exists():
        parser.error(f"Input file does not exist: {args.input_path}")

    output_path = save_evaluated_answer(args.input_path, args.output_dir)
    print(f"Saved answer evaluation JSON to {output_path}")


if __name__ == "__main__":
    main()
