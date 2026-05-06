"""Normalize Docling resume output into a compact LLM-ready JSON schema."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PARSED_DIR = Path("data/parsed")
LLM_DIR = Path("data/llm")
COMMON_COURSES_PATH = Path(__file__).with_name("common_university_courses.txt")

SECTION_HEADERS_KEYWORDS = {
    "EDUCATION": ["EDUCATION", "ACADEMIC BACKGROUND"],
    "COURSES": ["COURSES", "RELEVANT COURSES"],
    "WORK EXPERIENCE": ["EXPERIENCE", "WORK EXPERIENCE", "EMPLOYMENT HISTORY"],
    "SKILLS": ["SKILLS", "TECHNICAL SKILLS", "CORE SKILLS"],
    "PROJECTS": ["PROJECTS"]
}

# Month names are converted to YYYY-MM strings throughout the normalized output.
MONTHS = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
}


def load_docling_json(path: Path) -> dict[str, Any]:
    """Read a Docling JSON export from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def extract_text_blocks(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect normalized text blocks with their Docling labels and page numbers."""
    blocks = []

    for index, item in enumerate(doc.get("texts", [])):
        text = clean_text(item.get("text", ""))
        if not text:
            continue

        page_no = None
        if item.get("prov"):
            page_no = item["prov"][0].get("page_no")

        blocks.append(
            {
                "index": index,
                "label": item.get("label"),
                "text": text,
                "page_no": page_no,
            }
        )

    return blocks


def clean_text(value: str) -> str:
    """Trim text and remove spaces that Docling may leave before punctuation."""
    text = value.strip()
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text


def page_count(blocks: list[dict[str, Any]]) -> int:
    """Return the highest page number found in the extracted text blocks."""
    pages = [block["page_no"] for block in blocks if block.get("page_no") is not None]
    return max(pages, default=0)


def title_case(value: str | None) -> str | None:
    """Convert a candidate field to title case while preserving missing values."""
    if value is None:
        return None
    return value.title()


def detect_section_header(text: str) -> str | None:
    """Map a short heading-like line to one of the canonical resume sections."""
    upper_text = text.upper()

    for section, keywords in SECTION_HEADERS_KEYWORDS.items():
        if (
            any(keyword in upper_text for keyword in keywords)
            and len(upper_text.split()) <= 3
        ):
            return section

    return None


def is_boundary_block(block: dict[str, Any]) -> bool:
    """Identify blocks that likely end the current section."""
    text = block["text"].strip()
    words = text.split()

    return (
        block.get("label") == "section_header"
        or (text.isupper() and 0 < len(words) <= 4)
    )


def matches_current_section(block: dict[str, Any], current_section: str | None) -> bool:
    """Decide whether a text block still belongs to the active section."""
    if current_section is None:
        return False

    text = block["text"].strip()
    upper_text = text.upper()
    word_count = len(text.split())
    label = block.get("label")

    if current_section == "EDUCATION":
        degree_tokens = {"B.S.", "B.A.", "M.S.", "M.A.", "PH.D."}

        # Uppercase short blocks usually signal a new heading, except degree names.
        if text.isupper() and word_count <= 4 and upper_text not in degree_tokens:
            return False

        return bool(
            upper_text in degree_tokens
            or
            re.search(r"[A-Za-z]+\s+\d{4}\s*-\s*[A-Za-z]+\s+\d{4}", text)
            or re.search(r"\b(?:GPA|UNIVERSITY|COLLEGE|INSTITUTE|SCHOOL)\b", upper_text)
            or any(token in upper_text for token in degree_tokens)
            or (word_count <= 12 and not text.endswith("."))
        )

    if current_section == "COURSES":
        return bool(
            label == "list_item"
            or not text.isupper()
        )

    if current_section == "SKILLS":
        return bool(
            ":" in text
            or "," in text
            or "(" in text
            or ")" in text
            or label == "list_item"
        )

    if current_section == "WORK EXPERIENCE":
        # Job content can include short titles, but all-caps short text is a boundary.
        if text.isupper() and word_count <= 4:
            return False

        return bool(
            label == "list_item"
            or re.search(r"[A-Za-z]+\s+\d{4}\s*-\s*([A-Za-z]+\s+\d{4}|Present)", text)
            or re.search(r"/\s*[^/]+,\s*[A-Z]{2}", text)
            or (word_count <= 6 and not text.endswith("."))
        )

    if current_section == "PROJECTS":
        # Project entries are often title lines followed by sentence-style bullets.
        if text.isupper() and word_count <= 4:
            return False

        return bool(
            label == "list_item"
            or text.endswith(".")
            or (label == "section_header" and word_count <= 6 and not text.isupper())
        )

    return False


def split_sections(blocks: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Group text blocks into canonical resume sections using heading heuristics."""
    sections: dict[str, list[str]] = {}
    current_section: str | None = None

    for block in blocks:
        text = block["text"].strip()
        matched_section = detect_section_header(text)

        if matched_section:
            current_section = matched_section
            sections.setdefault(current_section, [])
            continue

        if current_section and matches_current_section(block, current_section):
            sections[current_section].append(text)
            continue

        if is_boundary_block(block):
            current_section = None

        sections.setdefault("Others", []).append(text)

    return sections


def month_year_to_iso(value: str | None) -> str | None:
    """Convert dates like 'January 2024' into a YYYY-MM string."""
    if not value:
        return None

    match = re.search(r"([A-Za-z]+)\s+(\d{4})", value)
    if not match:
        return None

    month = MONTHS.get(match.group(1).lower())
    year = match.group(2)
    if not month:
        return None

    return f"{year}-{month}"


def parse_location(value: str | None) -> dict[str, str | None]:
    """Split a comma-separated location into city, state, and country fields."""
    if not value:
        return {"city": None, "state": None, "country": None}

    parts = [part.strip() for part in value.split(",")]
    return {
        "city": parts[0] if parts else None,
        "state": parts[1] if len(parts) > 1 else None,
        "country": parts[2] if len(parts) > 2 else None,
    }


def normalize_list_item(value: str) -> str:
    """Clean bullet markers, trailing punctuation, and repeated whitespace."""
    return re.sub(r"\s+", " ", value.strip(" \t\r\n-*\u2022\u00b7."))


def split_list_items(value: str) -> list[str]:
    """Split a delimited list while preserving commas inside parenthetical text."""
    # Split on explicit separators, but keep separators inside parentheses.
    items = []
    current = []
    paren_depth = 0

    for char in value:
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth = max(paren_depth - 1, 0)

        if char in {",", ";", "|"} and paren_depth == 0:
            item = normalize_list_item("".join(current))
            if item:
                items.append(item)
            current = []
            continue

        current.append(char)

    last_item = normalize_list_item("".join(current))
    if last_item:
        items.append(last_item)

    return items


def load_common_courses(path: Path = COMMON_COURSES_PATH) -> list[str]:
    """Load the curated course names used to split compact course lists."""
    if not path.exists():
        return []

    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def course_pattern(course_name: str) -> re.Pattern[str] | None:
    """Build a regex that matches a course name across punctuation variants."""
    tokens = re.findall(r"[A-Za-z0-9]+", course_name.replace("&", " and "))
    if not tokens:
        return None

    token_patterns = [
        r"(?:and|&)" if token.lower() == "and" else re.escape(token)
        for token in tokens
    ]
    pattern = r"(?<![A-Za-z0-9])" + r"[\W_]+".join(token_patterns) + r"(?![A-Za-z0-9])"
    return re.compile(pattern, flags=re.IGNORECASE)


def select_course_matches(text: str, common_courses: list[str]) -> list[tuple[int, int, str]]:
    """Find non-overlapping known course names in their original text order."""
    matches = []
    for course_name in common_courses:
        pattern = course_pattern(course_name)
        if pattern is None:
            continue

        for match in pattern.finditer(text):
            matches.append((match.start(), match.end(), course_name))

    selected = []
    for start, end, course_name in sorted(matches, key=lambda item: (item[0], -(item[1] - item[0]))):
        overlaps_existing_match = any(
            start < selected_end and end > selected_start
            for selected_start, selected_end, _ in selected
        )
        if overlaps_existing_match:
            continue

        selected.append((start, end, course_name))

    return sorted(selected, key=lambda item: item[0])


def course_matches_cover_text(text: str, matches: list[tuple[int, int, str]]) -> bool:
    """Check whether detected courses account for every word in the source text."""
    word_spans = [(match.start(), match.end()) for match in re.finditer(r"[A-Za-z0-9]+", text)]
    if not word_spans:
        return False

    return all(
        any(match_start <= word_start and word_end <= match_end for match_start, match_end, _ in matches)
        for word_start, word_end in word_spans
    )


def split_known_courses(value: str, common_courses: list[str]) -> list[str]:
    """Return known course names only when they fully cover the input item."""
    item = normalize_list_item(value)
    if not item:
        return []

    matches = select_course_matches(item, common_courses)
    if not matches or not course_matches_cover_text(item, matches):
        return [item]

    return [course_name for _, _, course_name in matches]



def parse_skills(lines: list[str]) -> list[dict[str, list[str]]]:
    """Normalize skill lines into one de-duplicated bullet list."""
    bullets = []
    seen = set()

    for line in lines:
        normalized_line = normalize_list_item(line)
        if not normalized_line:
            continue

        for skill in split_list_items(normalized_line):
            dedupe_key = skill.lower()
            if dedupe_key in seen:
                continue

            seen.add(dedupe_key)
            bullets.append(skill)

    return [{"bullets": bullets}] if bullets else []


def parse_courses(lines: list[str]) -> list[str]:
    """Normalize and de-duplicate relevant course names."""
    courses = []
    seen = set()
    common_courses = load_common_courses()

    for line in lines:
        for item in split_list_items(line):
            for course in split_known_courses(item, common_courses):
                dedupe_key = course.lower()
                if dedupe_key in seen:
                    continue

                seen.add(dedupe_key)
                courses.append(course)

    return courses


def parse_date_location(value: str) -> dict[str, Any]:
    """Extract start date, end date, current status, and location from job metadata."""
    date_match = re.search(
        r"([A-Za-z]+\s+\d{4})\s*-\s*([A-Za-z]+\s+\d{4}|Present)",
        value,
        flags=re.IGNORECASE,
    )
    location_match = re.search(r"/\s*([^/]+,\s*[A-Z]{2})", value)

    end_value = date_match.group(2) if date_match else None

    return {
        "start_date": month_year_to_iso(date_match.group(1)) if date_match else None,
        "end_date": None if end_value and end_value.lower() == "present" else month_year_to_iso(end_value),
        "is_current": bool(end_value and end_value.lower() == "present"),
        "location": location_match.group(1).strip() if location_match else None,
    }


def parse_work_experience(lines: list[str]) -> list[dict[str, Any]]:
    """Convert work-experience section lines into structured job entries."""
    items = []
    index = 0

    while index < len(lines):
        if index + 2 >= len(lines):
            break

        job_title = lines[index]
        company = lines[index + 1]
        date_location = lines[index + 2]
        consumed = 3

        # Some Docling outputs merge the company and date/location onto one line.
        if not re.search(r"[A-Za-z]+\s+\d{4}\s*-", date_location):
            combined_line = company
            company = re.sub(r"\s+[A-Za-z]+\s+\d{4}\s*-.*$", "", combined_line).strip()
            date_location = combined_line[len(company) :].strip()
            consumed = 2

        metadata = parse_date_location(date_location)

        bullets = []
        index += consumed
        while index < len(lines):
            next_line = lines[index]
            next_two_lines = " ".join(lines[index : index + 3])
            # A short non-sentence line followed by dates marks the next job entry.
            looks_like_next_job = (
                index + 1 < len(lines)
                and not next_line.endswith(".")
                and re.search(r"[A-Za-z]+\s+\d{4}\s*-", next_two_lines)
            )
            if looks_like_next_job:
                break

            bullets.append(next_line.rstrip(" .") + ".")
            index += 1

        items.append(
            {
                "company": company,
                "job_title": job_title,
                "location": metadata["location"],
                "start_date": metadata["start_date"],
                "end_date": metadata["end_date"],
                "is_current": metadata["is_current"],
                "bullets": bullets,
            }
        )

    return items


def parse_projects(lines: list[str]) -> list[dict[str, Any]]:
    """Convert project section lines into project names with bullet details."""
    items = []
    current_project: dict[str, Any] | None = None

    for line in lines:
        is_project_name = len(line) < 70 and not line.endswith(".")

        if is_project_name:
            if current_project:
                items.append(current_project)
            current_project = {"project_name": line, "bullets": []}
            continue

        if current_project is None:
            current_project = {"project_name": None, "bullets": []}

        current_project["bullets"].append(line.rstrip(" .") + ".")

    if current_project:
        items.append(current_project)

    return items


def parse_education(lines: list[str], course_lines: list[str] | None = None) -> list[dict[str, Any]]:
    """Convert education lines plus course lines into one structured school entry."""
    if not lines:
        return []

    degree = lines[0]
    detail = lines[1] if len(lines) > 1 else ""

    date_match = re.search(r"([A-Za-z]+\s+\d{4})\s*-\s*([A-Za-z]+\s+\d{4})", detail)
    gpa_match = re.search(r"GPA:\s*([\d.]+)", detail, flags=re.IGNORECASE)
    location_match = re.search(r"([A-Za-z ]+,\s*[A-Z]{2})", detail)

    before_dates = detail[: date_match.start()].strip() if date_match else detail
    field_of_study, institution = split_field_and_institution(before_dates)

    return [
        {
            "degree": degree,
            "field_of_study": field_of_study,
            "institution": institution,
            "location": location_match.group(1).strip() if location_match else None,
            "start_date": month_year_to_iso(date_match.group(1)) if date_match else None,
            "end_date": month_year_to_iso(date_match.group(2)) if date_match else None,
            "gpa": gpa_match.group(1) if gpa_match else None,
            "relevant_courses": parse_courses(course_lines or []),
        }
    ]


def split_field_and_institution(value: str) -> tuple[str | None, str | None]:
    """Separate field of study from the institution name when both are present."""
    words = value.split()

    if len(words) >= 2 and words[-1] in {"University", "College", "Institute", "School"}:
        institution = " ".join(words[-2:])
        field_of_study = " ".join(words[:-2])
        return field_of_study or None, institution

    school_match = re.search(r"(.+?)\s+(.+(?:University|College|Institute|School))$", value)
    if school_match:
        return school_match.group(1).strip(), school_match.group(2).strip()

    return value or None, None


def normalize_resume(docling_json_path: Path) -> dict[str, Any]:
    """Normalize one Docling resume JSON file into the target LLM schema."""
    doc = load_docling_json(docling_json_path)
    blocks = extract_text_blocks(doc)
    sections = split_sections(blocks)

    origin = doc.get("origin", {})

    return {
        "document_type": "resume",
        "source": {
            "file_name": origin.get("filename") or docling_json_path.name,
            "page_count": page_count(blocks),
        },
        "candidate": {
            "full_name": title_case(blocks[0]["text"]) if len(blocks) > 0 else None,
            "headline": title_case(blocks[1]["text"]) if len(blocks) > 1 else None,
        },
        "sections": [
            {
                "section_name": "Work Experience",
                "items": parse_work_experience(sections.get("WORK EXPERIENCE", [])),
            },
            {
                "section_name": "Projects",
                "items": parse_projects(sections.get("PROJECTS", [])),
            },
            {
                "section_name": "Skills",
                "items": parse_skills(sections.get("SKILLS", [])),
            },
            {
                "section_name": "Education",
                "items": parse_education(
                    sections.get("EDUCATION", []),
                    sections.get("COURSES", []),
                ),
            },
        ],
    }


def save_llm_resume(docling_json_path: Path, output_dir: Path = LLM_DIR) -> Path:
    """Normalize a Docling JSON file and write the LLM-ready JSON output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = normalize_resume(docling_json_path)
    new_name = f"{docling_json_path.stem}_llm{docling_json_path.suffix}"
    output_path = output_dir / new_name

    output_path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return output_path


def main() -> None:
    """Run the command-line normalization workflow."""
    parser = argparse.ArgumentParser(description="Convert Docling resume JSON into LLM-ready resume JSON.")
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=PARSED_DIR / "resume1_parsed.json",
        help="Path to a raw Docling JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=LLM_DIR,
        help="Folder where the normalized JSON file will be saved.",
    )
    args = parser.parse_args()

    output_path = save_llm_resume(args.input_path, args.output_dir)
    print(f"Saved LLM-ready resume JSON to {output_path}")


if __name__ == "__main__":
    main()
