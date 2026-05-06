from __future__ import annotations

import sqlite3
from pathlib import Path

def normalize_job_text(text: str) -> str:
    """Normalize raw job descriptions text before storing and chunking"""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip spaces on each line, keep blank lines for paragraph splitting
    lines = [line.strip() for line in text.split("\n")]

    # Remove blank lines while preserving paragraph boundaries
    normalized_lines = []
    previous_blank = False

    for line in lines:
        is_blank = line == ""

        if is_blank and previous_blank:
            continue

        normalized_lines.append(line)
        previous_blank = is_blank

    return "\n".join(normalized_lines).strip()

def insert_jd(
        conn: sqlite3.Connection,
        *,
        file_name: str,
        raw_text: str,
        job_title: str | None = None,
        company: str | None = None,
) -> int:
    """Insert one raw job description into job_descriptions"""

    normalize_text = normalize_job_text(raw_text)

    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO job_descriptions (
            file_name,
            job_title,
            company,
            raw_text
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            file_name,
            job_title,
            company,
            normalize_text
        ),
    )

    return int(cur.lastrowid)

def insert_jd_file(
        conn: sqlite3.Connection,
        path: str | Path,
        *,
        job_title: str | None = None,
        company: str | None = None,
) -> int:
    """Read a .txt job description file and insert it into the database"""
    
    path = Path(path)
    raw_text = path.read_text(encoding="utf-8")

    return insert_jd(
        conn,
        file_name=path.name,
        raw_text=raw_text,
        job_title=job_title,
        company=company,
    )