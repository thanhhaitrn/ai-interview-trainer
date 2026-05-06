import json
import sqlite3
from pathlib import Path

from src.db.chunk_builder import build_jd_chunks, insert_job_chunks, insert_resume_chunks
from src.db.insert_jd import insert_jd, insert_jd_file
from src.db.insert_resume import insert_resume_json
from src.db.schema import init_db


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESUME_JSON_PATH = PROJECT_ROOT / "data" / "resumes" / "llm" / "resume1_parsed_llm.json"
JD_TEXT_PATH = PROJECT_ROOT / "data" / "jobs" / "sample.txt"

REQUIRED_TABLES = {
    "resumes_doc",
    "work_experiences",
    "work_bullets",
    "projects",
    "project_bullets",
    "skills",
    "education",
    "education_courses",
    "resume_chunks",
    "job_descriptions",
    "job_chunks",
}


def _load_json(path: Path) -> dict:
    """Load a normalized resume JSON file."""

    return json.loads(path.read_text(encoding="utf-8"))


def _create_test_db(tmp_path: Path, db_name: str = "test_chunks.db") -> Path:
    """Create a fresh SQLite database for chunk builder tests."""

    db_path = tmp_path / db_name
    init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            """
        ).fetchall()

    table_names = {row[0] for row in rows}
    missing_tables = REQUIRED_TABLES - table_names
    assert not missing_tables, f"Missing DB tables: {sorted(missing_tables)}"

    return db_path


def _create_db_from_resume_json(
    tmp_path: Path,
    db_name: str = "resume_from_json.db",
) -> tuple[Path, int]:
    """Create a test DB and seed it from data/resumes/llm/resume1_parsed_llm.json."""

    assert RESUME_JSON_PATH.exists(), f"Missing normalized resume JSON: {RESUME_JSON_PATH}"

    db_path = _create_test_db(tmp_path, db_name)
    resume_data = _load_json(RESUME_JSON_PATH)

    with sqlite3.connect(db_path) as conn:
        resume_id = insert_resume_json(conn, resume_data)
        conn.commit()

        row = conn.execute(
            """
            SELECT file_name, name, headline
            FROM resumes_doc
            WHERE id = ?
            """,
            (resume_id,),
        ).fetchone()

    assert row == ("resume1.pdf", "Trish Mathers", "Entry-Level Data Scientist")

    return db_path, resume_id


def _print_chunk_rows(title: str, rows: list[sqlite3.Row]) -> None:
    """Print chunks in a readable format for pytest -s."""

    print(f"\n\n{title}")
    print("=" * 100)

    for index, row in enumerate(rows, start=1):
        print(f"\nChunk {index}")
        print("-" * 100)
        print(f"id: {row['id']}")
        print(f"section_type: {row['section_type']}")
        print(f"source_table: {row['source_table']}")
        print(f"source_id: {row['source_id']}")
        print("chunk_text:")
        print(row["chunk_text"])


def test_full_flow_create_db_insert_resume_and_jd_then_view_chunks(tmp_path):
    """
    Full local flow:
    1. Create a fresh SQLite database.
    2. Insert resume1 data.
    3. Insert sample job description text.
    4. Build and store resume/JD chunks.
    5. Print chunks so they can be viewed with: python -m pytest -s tests/test_chunk_builder.py

    Note:
    This test creates the DB from data/llm/resume1_parsed_llm.json, then adds
    the sample job description text.
    """

    assert JD_TEXT_PATH.exists(), f"Missing job description text file: {JD_TEXT_PATH}"

    db_path, resume_id = _create_db_from_resume_json(tmp_path, "resume_job_chunks.db")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        job_id = insert_jd_file(
            conn,
            JD_TEXT_PATH,
            job_title="Data Analyst Intern",
            company="Impiricus",
        )

        resume_chunk_count = insert_resume_chunks(conn, resume_id)
        job_chunk_count = insert_job_chunks(conn, job_id)
        conn.commit()

        resume_chunks = conn.execute(
            """
            SELECT id, section_type, source_table, source_id, chunk_text
            FROM resume_chunks
            WHERE resume_id = ?
            ORDER BY id
            """,
            (resume_id,),
        ).fetchall()

        job_chunks = conn.execute(
            """
            SELECT id, section_type, source_table, source_id, chunk_text
            FROM job_chunks
            WHERE job_id = ?
            ORDER BY id
            """,
            (job_id,),
        ).fetchall()

    assert resume_chunk_count == len(resume_chunks)
    assert job_chunk_count == len(job_chunks)
    assert resume_chunk_count > 0
    assert job_chunk_count > 0
    assert [row["section_type"] for row in job_chunks] == [
        "job_who_we_are",
        "job_summary",
        "job_duties_responsibilities",
        "job_experience",
        "job_nice_to_have",
        "job_compensation",
        "job_important_notice",
    ]
    assert all("Job description paragraph" not in row["chunk_text"] for row in job_chunks)
    assert "Impiricus is seeking a Data Analyst Intern" in job_chunks[1]["chunk_text"]

    _print_chunk_rows("RESUME CHUNKS", resume_chunks)
    _print_chunk_rows("JOB DESCRIPTION CHUNKS", job_chunks)


def test_jd_chunk_builder_falls_back_to_paragraphs_when_no_headings(tmp_path):
    """JD text without standalone short headings should use paragraph chunks."""

    db_path = _create_test_db(tmp_path, "fallback_chunks.db")

    with sqlite3.connect(db_path) as conn:
        job_id = insert_jd(
            conn,
            file_name="no_heading.txt",
            raw_text=(
                "Python and SQL are required.\n\n"
                "Excel is helpful.\n\n"
                "Candidates should communicate clearly."
            ),
        )

        chunks = build_jd_chunks(conn, job_id)

    assert [chunk["section_type"] for chunk in chunks] == [
        "job_paragraph",
        "job_paragraph",
        "job_paragraph",
    ]
    assert chunks[0]["chunk_text"].startswith("Job description paragraph 1:")
