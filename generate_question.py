from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUESTS_DIR = Path("data/requests")
QUESTIONS_DIR = Path("data/questions")


def load_request_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def generate_question_from_request_file(path: Path) -> dict[str, Any]:
    from app.service import generate_question

    request_payload = load_request_json(path)
    return generate_question(request_payload)


def save_generated_question(
    request_json_path: Path,
    output_dir: Path = QUESTIONS_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_question = generate_question_from_request_file(request_json_path)
    output_path = output_dir / f"{request_json_path.stem}_question.json"

    output_path.write_text(
        json.dumps(generated_question, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an interview question from a JSON request file."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=REQUESTS_DIR / "question_request.json",
        help="Path to the question request JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=QUESTIONS_DIR,
        help="Folder where the generated question JSON file will be saved.",
    )
    args = parser.parse_args()

    if not args.input_path.exists():
        parser.error(f"Input file does not exist: {args.input_path}")

    output_path = save_generated_question(args.input_path, args.output_dir)
    print(f"Saved generated question JSON to {output_path}")


if __name__ == "__main__":
    main()
