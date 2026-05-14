import argparse
import sqlite3
from pathlib import Path

from src.db.chunk_builder import insert_job_chunks
from src.db.insert_jd import insert_jd_file
from src.db.schema import init_db


DB_PATH = Path("db/resume_kb.db")
JD_PATH = Path("data/jobs/sample.txt")


def reset_db(db_path: str | Path) -> None:
    """
    Delete the existing SQLite database file.
    Use this only when you want a completely fresh database.
    """

    db_path = Path(db_path)

    if db_path.exists():
        db_path.unlink()
        print(f"Deleted old database: {db_path}")


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Insert one job description .txt file into the SQLite database."
    )

    parser.add_argument(
        "txt_path",
        nargs="?",
        type=Path,
        default=JD_PATH,
        help="Path to the job description .txt file. Default: data/jobs/sample.txt",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help="Path to SQLite database. Default: db/resume_kb.db",
    )
    parser.add_argument(
        "--job-title",
        type=str,
        default=None,
        help="Optional job title metadata.",
    )
    parser.add_argument(
        "--company",
        type=str,
        default=None,
        help="Optional company metadata.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing database before inserting data.",
    )
    parser.add_argument(
        "--create-chunks",
        action="store_true",
        help="Create job chunks immediately after inserting the job description.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print generated job chunks. Implies --create-chunks.",
    )

    return parser.parse_args()


def print_job_chunks(conn: sqlite3.Connection, job_id: int) -> None:
    """Print stored chunks for one job description."""

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


def main() -> None:
    args = parse_args()

    txt_path = args.txt_path
    db_path = args.db

    if args.reset:
        reset_db(db_path)

    init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        job_id = insert_jd_file(
            conn,
            txt_path,
            job_title=args.job_title,
            company=args.company,
        )

        chunk_count = None
        if args.create_chunks or args.show:
            chunk_count = insert_job_chunks(conn, job_id)

        conn.commit()

        if args.show:
            print_job_chunks(conn, job_id)

    print("Inserted job description successfully.")
    print(f"job_id={job_id}")

    if chunk_count is not None:
        print(f"job_chunk_count={chunk_count}")

    print(f"txt_path={txt_path}")
    print(f"db_path={db_path}")


if __name__ == "__main__":
    main()
