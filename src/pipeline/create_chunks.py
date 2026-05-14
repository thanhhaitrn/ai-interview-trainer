import argparse
import sqlite3
from pathlib import Path

from src.db.chunk_builder import insert_job_chunks, insert_resume_chunks
from src.db.schema import init_db


DB_PATH = Path("db/resume_kb.db")


def _unique_ids(ids: list[int]) -> list[int]:
    """Keep ids unique while preserving command-line order."""

    seen = set()
    unique = []

    for item_id in ids:
        if item_id in seen:
            continue

        seen.add(item_id)
        unique.append(item_id)

    return unique


def fetch_resume_ids(conn: sqlite3.Connection) -> list[int]:
    """Return all resume ids that exist in the database."""

    rows = conn.execute(
        """
        SELECT id
        FROM resumes_doc
        ORDER BY id
        """
    ).fetchall()

    return [int(row["id"]) for row in rows]


def fetch_job_ids(conn: sqlite3.Connection) -> list[int]:
    """Return all job description ids that exist in the database."""

    rows = conn.execute(
        """
        SELECT id
        FROM job_descriptions
        ORDER BY id
        """
    ).fetchall()

    return [int(row["id"]) for row in rows]


def create_resume_chunks(
    conn: sqlite3.Connection,
    resume_ids: list[int],
    *,
    replace_existing: bool,
) -> dict[int, int]:
    """Create resume chunks and return chunk counts by resume id."""

    counts = {}

    for resume_id in _unique_ids(resume_ids):
        counts[resume_id] = insert_resume_chunks(
            conn,
            resume_id,
            replace_existing=replace_existing,
        )

    return counts


def create_job_chunks(
    conn: sqlite3.Connection,
    job_ids: list[int],
    *,
    replace_existing: bool,
) -> dict[int, int]:
    """Create job description chunks and return chunk counts by job id."""

    counts = {}

    for job_id in _unique_ids(job_ids):
        counts[job_id] = insert_job_chunks(
            conn,
            job_id,
            replace_existing=replace_existing,
        )

    return counts


def print_resume_chunks(conn: sqlite3.Connection, resume_ids: list[int]) -> None:
    """Print stored resume chunks for manual inspection."""

    for resume_id in _unique_ids(resume_ids):
        rows = conn.execute(
            """
            SELECT id, section_type, source_table, source_id, chunk_text
            FROM resume_chunks
            WHERE resume_id = ?
            ORDER BY id
            """,
            (resume_id,),
        ).fetchall()

        print(f"\nResume chunks for resume_id={resume_id}")
        print("=" * 80)

        for row in rows:
            print(f"\nid={row['id']}")
            print(f"section_type={row['section_type']}")
            print(f"source_table={row['source_table']}")
            print(f"source_id={row['source_id']}")
            print(row["chunk_text"])


def print_job_chunks(conn: sqlite3.Connection, job_ids: list[int]) -> None:
    """Print stored job description chunks for manual inspection."""

    for job_id in _unique_ids(job_ids):
        rows = conn.execute(
            """
            SELECT id, section_type, source_table, source_id, chunk_text
            FROM job_chunks
            WHERE job_id = ?
            ORDER BY id
            """,
            (job_id,),
        ).fetchall()

        print(f"\nJob description chunks for job_id={job_id}")
        print("=" * 80)

        for row in rows:
            print(f"\nid={row['id']}")
            print(f"section_type={row['section_type']}")
            print(f"source_table={row['source_table']}")
            print(f"source_id={row['source_id']}")
            print(row["chunk_text"])


def parse_args() -> argparse.Namespace:
    """Read command-line arguments for chunk creation."""

    parser = argparse.ArgumentParser(
        description="Create semantic chunks for inserted resumes and job descriptions."
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help="Path to SQLite database. Default: db/resume_kb.db",
    )
    parser.add_argument(
        "--resume-id",
        type=int,
        action="append",
        default=[],
        help="Resume id to chunk. Can be passed multiple times.",
    )
    parser.add_argument(
        "--job-id",
        type=int,
        action="append",
        default=[],
        help="Job description id to chunk. Can be passed multiple times.",
    )
    parser.add_argument(
        "--all-resumes",
        action="store_true",
        help="Create chunks for every resume in the database.",
    )
    parser.add_argument(
        "--all-jobs",
        action="store_true",
        help="Create chunks for every job description in the database.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append chunks instead of replacing existing chunks for the same document.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print chunks after inserting them.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = args.db

    init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        resume_ids = args.resume_id
        job_ids = args.job_id

        # If no explicit target is passed, process all known documents.
        if args.all_resumes or not (resume_ids or job_ids or args.all_jobs):
            resume_ids = fetch_resume_ids(conn)

        if args.all_jobs or not (args.resume_id or args.job_id or args.all_resumes):
            job_ids = fetch_job_ids(conn)

        replace_existing = not args.append

        resume_counts = create_resume_chunks(
            conn,
            resume_ids,
            replace_existing=replace_existing,
        )
        job_counts = create_job_chunks(
            conn,
            job_ids,
            replace_existing=replace_existing,
        )
        conn.commit()

        for resume_id, count in resume_counts.items():
            print(f"Created {count} resume chunks for resume_id={resume_id}")

        for job_id, count in job_counts.items():
            print(f"Created {count} job chunks for job_id={job_id}")

        if not resume_counts and not job_counts:
            print("No resumes or job descriptions found to chunk.")

        if args.show:
            print_resume_chunks(conn, resume_ids)
            print_job_chunks(conn, job_ids)

    print(f"db_path={db_path}")


if __name__ == "__main__":
    main()
