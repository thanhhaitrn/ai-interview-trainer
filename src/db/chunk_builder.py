from __future__ import annotations

import re
import sqlite3
from typing import Any


Chunk = dict[str, Any]


def _use_row_factory(conn: sqlite3.Connection) -> None:
    """Return SQLite rows that can be accessed by column name."""

    conn.row_factory = sqlite3.Row


def _clean_text(value: Any) -> str:
    """Convert nullable DB values into clean one-line text."""

    if value is None:
        return ""

    return str(value).strip()


def _join_parts(parts: list[str], separator: str = " ") -> str:
    """Join only non-empty parts so chunk text does not contain noisy blanks."""

    return separator.join(part for part in parts if part).strip()


def _format_date_range(start_date: Any, end_date: Any, is_current: Any = 0) -> str:
    """Build a readable date range for semantic search text."""

    start = _clean_text(start_date) or "unknown start date"
    end = "present" if is_current else _clean_text(end_date)

    if not end:
        end = "unknown end date"

    if start == "unknown start date" and end == "unknown end date":
        return ""

    return f"from {start} to {end}"


def _fetch_bullets(
    cur: sqlite3.Cursor,
    table_name: str,
    foreign_key: str,
    parent_id: int,
) -> list[str]:
    """Fetch bullet text for child tables such as work_bullets and project_bullets."""

    rows = cur.execute(
        f"""
        SELECT bullet_text
        FROM {table_name}
        WHERE {foreign_key} = ?
        ORDER BY id
        """,
        (parent_id,),
    ).fetchall()

    return [_clean_text(row["bullet_text"]) for row in rows if _clean_text(row["bullet_text"])]


def split_paragraphs(text: str) -> list[str]:
    """Split raw job description text into paragraphs."""

    # Normalize Windows/Mac line endings to Unix line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove extra spaces around each line.
    lines = [line.strip() for line in text.split("\n")]

    # Join lines back so blank lines remain as paragraph separators.
    normalized_text = "\n".join(lines)

    # Split on one or more blank lines.
    paragraphs = re.split(r"\n\s*\n+", normalized_text)

    # Remove empty paragraphs.
    return [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]


def is_standalone_heading(paragraph: str) -> bool:
    """Detect a short job description heading that stands on its own line."""

    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if len(lines) != 1:
        return False

    line = lines[0]

    # Treat "line.length <= 3" as word count <= 3.
    # This catches headings like "Job Summary", "Experience", and "Nice To Have".
    if len(line.split()) > 3:
        return False

    # Short sentence-style paragraphs should stay as content, not headings.
    return not line.endswith((".", "!", "?"))


def _heading_to_section_type(heading: str) -> str:
    """Convert a heading into a stable section_type value."""

    slug = re.sub(r"[^a-z0-9]+", "_", heading.lower()).strip("_")
    if slug.startswith("job_"):
        return slug

    return f"job_{slug}" if slug else "job_section"


def split_job_sections(text: str) -> list[tuple[str, str]]:
    """
    Split a job description by detected headings.

    If no standalone heading is detected, return an empty list so callers can
    fall back to paragraph chunking.
    """

    paragraphs = split_paragraphs(text)
    sections: list[tuple[str, str]] = []
    current_heading = "Overview"
    current_parts: list[str] = []
    found_heading = False

    for paragraph in paragraphs:
        if is_standalone_heading(paragraph):
            found_heading = True

            # Save content collected under the previous heading before moving on.
            if current_parts:
                sections.append((current_heading, "\n\n".join(current_parts)))

            current_heading = paragraph.strip().rstrip(":")
            current_parts = []
            continue

        current_parts.append(paragraph)

    if current_parts:
        sections.append((current_heading, "\n\n".join(current_parts)))

    return sections if found_heading else []


def fetch_resume_core(conn: sqlite3.Connection, resume_id: int) -> dict[str, Any]:
    """Fetch the parent resume row used to verify that the resume exists."""

    _use_row_factory(conn)
    cur = conn.cursor()

    row = cur.execute(
        """
        SELECT id, file_name, name, headline
        FROM resumes_doc
        WHERE id = ?
        """,
        (resume_id,),
    ).fetchone()

    if row is None:
        raise ValueError(f"Resume id {resume_id} does not exist.")

    return dict(row)


def build_resume_core_chunk(conn: sqlite3.Connection, resume_id: int) -> list[Chunk]:
    """Build a high-level chunk from the parent resume row."""

    resume = fetch_resume_core(conn, resume_id)

    # This chunk keeps candidate-level fields searchable even if no section mentions them.
    text = _join_parts(
        [
            "Resume profile:",
            _clean_text(resume["name"]) or "unknown candidate",
            f"Headline: {_clean_text(resume['headline'])}." if _clean_text(resume["headline"]) else "",
            f"File name: {_clean_text(resume['file_name'])}." if _clean_text(resume["file_name"]) else "",
        ]
    )

    return [
        {
            "resume_id": resume_id,
            "section_type": "resume_core",
            "source_table": "resumes_doc",
            "source_id": int(resume["id"]),
            "chunk_text": text,
        }
    ]


def build_work_chunks(conn: sqlite3.Connection, resume_id: int) -> list[Chunk]:
    """Build one semantic chunk for each work experience row."""

    _use_row_factory(conn)
    cur = conn.cursor()

    jobs = cur.execute(
        """
        SELECT id, company, job_title, location, start_date, end_date, is_current
        FROM work_experiences
        WHERE resume_id = ?
        ORDER BY id
        """,
        (resume_id,),
    ).fetchall()

    chunks: list[Chunk] = []

    for job in jobs:
        bullets = _fetch_bullets(cur, "work_bullets", "work_experience_id", int(job["id"]))
        bullet_text = _join_parts(bullets)
        date_range = _format_date_range(job["start_date"], job["end_date"], job["is_current"])

        # Keep the chunk as a complete sentence so embedding search has enough context.
        text = _join_parts(
            [
                "Work experience:",
                f"Candidate worked as {_clean_text(job['job_title']) or 'unknown role'}",
                f"at {_clean_text(job['company']) or 'unknown company'}.",
                f"Period: {date_range}." if date_range else "",
                f"Location: {_clean_text(job['location']) or 'unknown location'}.",
                f"Responsibilities and achievements: {bullet_text}" if bullet_text else "",
            ]
        )

        chunks.append(
            {
                "resume_id": resume_id,
                "section_type": "work_experience",
                "source_table": "work_experiences",
                "source_id": int(job["id"]),
                "chunk_text": text,
            }
        )

    return chunks


def build_worker_chunks(conn: sqlite3.Connection, resume_id: int) -> list[Chunk]:
    """Backward-compatible wrapper for the old function name."""

    return build_work_chunks(conn, resume_id)


def build_project_chunks(conn: sqlite3.Connection, resume_id: int) -> list[Chunk]:
    """Build one semantic chunk for each project."""

    _use_row_factory(conn)
    cur = conn.cursor()

    projects = cur.execute(
        """
        SELECT id, project_name
        FROM projects
        WHERE resume_id = ?
        ORDER BY id
        """,
        (resume_id,),
    ).fetchall()

    chunks: list[Chunk] = []

    for project in projects:
        bullets = _fetch_bullets(cur, "project_bullets", "project_id", int(project["id"]))
        bullet_text = _join_parts(bullets)

        text = _join_parts(
            [
                "Project:",
                _clean_text(project["project_name"]) or "unknown project",
                f"Details and outcomes: {bullet_text}" if bullet_text else "",
            ]
        )

        chunks.append(
            {
                "resume_id": resume_id,
                "section_type": "project",
                "source_table": "projects",
                "source_id": int(project["id"]),
                "chunk_text": text,
            }
        )

    return chunks


def build_skill_chunks(conn: sqlite3.Connection, resume_id: int) -> list[Chunk]:
    """Build one combined skills chunk for the resume."""

    _use_row_factory(conn)
    cur = conn.cursor()

    rows = cur.execute(
        """
        SELECT skill_text
        FROM skills
        WHERE resume_id = ?
        ORDER BY id
        """,
        (resume_id,),
    ).fetchall()

    skills = [_clean_text(row["skill_text"]) for row in rows if _clean_text(row["skill_text"])]
    if not skills:
        return []

    # Skills are usually short, so grouping them creates a stronger searchable chunk.
    return [
        {
            "resume_id": resume_id,
            "section_type": "skills",
            "source_table": "skills",
            "source_id": None,
            "chunk_text": f"Skills: {_join_parts(skills, ', ')}",
        }
    ]


def build_education_chunks(conn: sqlite3.Connection, resume_id: int) -> list[Chunk]:
    """Build one semantic chunk for each education row."""

    _use_row_factory(conn)
    cur = conn.cursor()

    education_rows = cur.execute(
        """
        SELECT id, degree, field_of_study, institution, location, start_date, end_date, gpa
        FROM education
        WHERE resume_id = ?
        ORDER BY id
        """,
        (resume_id,),
    ).fetchall()

    chunks: list[Chunk] = []

    for education in education_rows:
        course_rows = cur.execute(
            """
            SELECT course_name
            FROM education_courses
            WHERE education_id = ?
            ORDER BY id
            """,
            (education["id"],),
        ).fetchall()

        courses = [
            _clean_text(row["course_name"])
            for row in course_rows
            if _clean_text(row["course_name"])
        ]
        date_range = _format_date_range(education["start_date"], education["end_date"])

        text = _join_parts(
            [
                "Education:",
                _clean_text(education["degree"]),
                _clean_text(education["field_of_study"]),
                f"at {_clean_text(education['institution'])}." if _clean_text(education["institution"]) else "",
                f"Period: {date_range}." if date_range else "",
                f"Location: {_clean_text(education['location'])}." if _clean_text(education["location"]) else "",
                f"GPA: {_clean_text(education['gpa'])}." if _clean_text(education["gpa"]) else "",
                f"Relevant courses: {_join_parts(courses, ', ')}." if courses else "",
            ]
        )

        chunks.append(
            {
                "resume_id": resume_id,
                "section_type": "education",
                "source_table": "education",
                "source_id": int(education["id"]),
                "chunk_text": text,
            }
        )

    return chunks


def build_resume_chunks(conn: sqlite3.Connection, resume_id: int) -> list[Chunk]:
    """Build all semantic chunks for one resume."""

    chunks = build_resume_core_chunk(conn, resume_id)
    chunks.extend(build_work_chunks(conn, resume_id))
    chunks.extend(build_project_chunks(conn, resume_id))
    chunks.extend(build_skill_chunks(conn, resume_id))
    chunks.extend(build_education_chunks(conn, resume_id))

    return chunks


def insert_resume_chunks(
    conn: sqlite3.Connection,
    resume_id: int,
    *,
    replace_existing: bool = True,
) -> int:
    """
    Build chunks and insert them into resume_chunks.

    replace_existing=True makes the function safe to rerun for the same resume
    after parsing logic or source data changes.
    """

    chunks = build_resume_chunks(conn, resume_id)
    cur = conn.cursor()

    if replace_existing:
        cur.execute(
            """
            DELETE FROM resume_chunks
            WHERE resume_id = ?
            """,
            (resume_id,),
        )

    cur.executemany(
        """
        INSERT INTO resume_chunks (
            resume_id,
            section_type,
            source_table,
            source_id,
            chunk_text
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                chunk["resume_id"],
                chunk["section_type"],
                chunk["source_table"],
                chunk["source_id"],
                chunk["chunk_text"],
            )
            for chunk in chunks
        ],
    )

    return len(chunks)


def build_jd_chunks(conn: sqlite3.Connection, job_id: int) -> list[Chunk]:
    """Build heading-based chunks for one job description, with paragraph fallback."""

    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    row = cur.execute(
        """
        SELECT id, file_name, job_title, company, raw_text
        FROM job_descriptions
        WHERE id = ?
        """,
        (job_id,),
    ).fetchone()

    if row is None:
        raise ValueError(f"Job description id {job_id} does not exist.")

    chunks: list[Chunk] = []
    sections = split_job_sections(row["raw_text"])


    if sections:
        for index, (heading, section_text) in enumerate(sections, start=1):
            chunk_text = (
                f"Job description section {index} - {heading}: "
                f"{section_text}"
            )

            chunks.append(
                {
                    "job_id": job_id,
                    "section_type": _heading_to_section_type(heading),
                    "source_table": "job_descriptions",
                    "source_id": row["id"],
                    "chunk_text": chunk_text,
                }
            )

        return chunks

    paragraphs = split_paragraphs(row["raw_text"])

    for index, paragraph in enumerate(paragraphs, start=1):
        chunk_text = (
            f"Job description paragraph {index}: "
            f"{paragraph}"
        )

        chunks.append(
            {
                "job_id": job_id,
                "section_type": "job_paragraph",
                "source_table": "job_descriptions",
                "source_id": row["id"],
                "chunk_text": chunk_text,
            }
        )

    return chunks


def insert_job_chunks(
    conn: sqlite3.Connection,
    job_id: int,
    *,
    replace_existing: bool = True,
) -> int:
    """Build and insert job chunks into job_chunks."""

    chunks = build_jd_chunks(conn, job_id)
    cur = conn.cursor()

    if replace_existing:
        cur.execute(
            """
            DELETE FROM job_chunks
            WHERE job_id = ?
            """,
            (job_id,),
        )

    cur.executemany(
        """
        INSERT INTO job_chunks (
            job_id,
            section_type,
            source_table,
            source_id,
            chunk_text
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                chunk["job_id"],
                chunk["section_type"],
                chunk["source_table"],
                chunk["source_id"],
                chunk["chunk_text"],
            )
            for chunk in chunks
        ],
    )

    return len(chunks)
