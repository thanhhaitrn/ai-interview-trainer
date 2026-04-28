from pathlib import Path
import json 

from docling.document_converter import DocumentConverter

RAW_DIR = Path("data/raw")
PARSED_DIR = Path("data/parsed")

def parse_pdf_to_json(pdf_path: Path) -> Path:
    PARSED_DIR.mkdir(parents=True, exist_ok=True)

    converter = DocumentConverter()
    result = converter.convert(pdf_path)

    output_path = PARSED_DIR / f"{pdf_path.stem}_parsed.json"

    output_path.write_text(
        json.dumps(result.document.export_to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return output_path

if __name__ == "__main__":
    pdf_path = RAW_DIR / "resume1.pdf"
    output_path = parse_pdf_to_json(pdf_path)
    print(f"Saved parsed JSON to {output_path}")