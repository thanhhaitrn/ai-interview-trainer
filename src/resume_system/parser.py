import argparse
import json
from pathlib import Path


RAW_DIR = Path("data/raw")
PARSED_DIR = Path("data/parsed")


def parse_pdf_to_json(pdf_path: Path, output_dir: Path = PARSED_DIR) -> Path:
    from docling.document_converter import DocumentConverter

    output_dir.mkdir(parents=True, exist_ok=True)

    converter = DocumentConverter()
    result = converter.convert(pdf_path)

    output_path = output_dir / f"{pdf_path.stem}_parsed.json"

    output_path.write_text(
        json.dumps(result.document.export_to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a resume PDF into Docling JSON.")
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=RAW_DIR / "resume1.pdf",
        help="Path to the resume PDF file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PARSED_DIR,
        help="Folder where the parsed JSON file will be saved.",
    )
    args = parser.parse_args()

    if not args.input_path.exists():
        parser.error(f"Input file does not exist: {args.input_path}")

    output_path = parse_pdf_to_json(args.input_path, args.output_dir)
    print(f"Saved parsed JSON to {output_path}")


if __name__ == "__main__":
    main()
