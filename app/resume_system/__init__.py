"""Resume parsing and normalization entrypoints for the app package."""

from app.resume_system.parser import PARSED_DIR, RAW_DIR, parse_pdf_to_json
from app.resume_system.resume_normalizer import (
    LLM_DIR,
    normalize_resume,
    save_llm_resume,
)

__all__ = [
    "LLM_DIR",
    "PARSED_DIR",
    "RAW_DIR",
    "normalize_resume",
    "parse_pdf_to_json",
    "save_llm_resume",
]
