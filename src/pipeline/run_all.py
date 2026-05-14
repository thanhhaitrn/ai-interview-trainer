import argparse
import json
import sqlite3
from pathlib import Path

from src.db.chunk_builder import insert_job_chunks, insert_resume_chunks
from src.db.insert_jd import insert_jd_file
from src.db.insert_resume import insert_resume_json
from src.db.qdrant_store import get_collection_name, get_qdrant_client
from src.db.schema import init_db
from src.pipeline.create_chunks import print_job_chunks, print_resume_chunks
from src.pipeline.embed_chunks import embed_table


DB_PATH = Path("db/resume_kb.db")
RESUME_JSON_PATH = Path("data/resumes/llm/resume1_parsed_llm.json")
JD_TEXT_PATH = Path("data/jobs/sample.txt")


def load_json(path: str | Path) -> dict:
    """Load a normalized resume JSON file."""

    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def reset_db(db_path: str | Path) -> None:
    """Delete the existing SQLite database file."""

    db_path = Path(db_path)

    if db_path.exists():
        db_path.unlink()
        print(f"Deleted old database: {db_path}")


def parse_args() -> argparse.Namespace:
    """Read command-line arguments for the full ingestion and chunking flow."""

    parser = argparse.ArgumentParser(
        description="Run the full local pipeline: ingest data, create chunks, and embed chunks into Qdrant."
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help="Path to SQLite database. Default: db/resume_kb.db",
    )
    parser.add_argument(
        "--resume-json",
        type=Path,
        default=RESUME_JSON_PATH,
        help="Path to normalized resume JSON. Default: data/resumes/llm/resume1_parsed_llm.json",
    )
    parser.add_argument(
        "--jd-txt",
        type=Path,
        default=JD_TEXT_PATH,
        help="Path to job description .txt file. Default: data/jobs/sample.txt",
    )
    parser.add_argument(
        "--job-title",
        type=str,
        default=None,
        help="Optional job title metadata for the JD.",
    )
    parser.add_argument(
        "--company",
        type=str,
        default=None,
        help="Optional company metadata for the JD.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing database before running the pipeline.",
    )
    parser.add_argument(
        "--skip-resume",
        action="store_true",
        help="Skip resume ingest and resume chunk creation.",
    )
    parser.add_argument(
        "--skip-jd",
        action="store_true",
        help="Skip JD ingest and JD chunk creation.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print created chunks after the pipeline finishes.",
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Skip embedding chunks into Qdrant.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding batch size. Default: 16.",
    )
    parser.add_argument(
        "--collection",
        default=get_collection_name(),
        help="Qdrant collection name. Default: QDRANT_COLLECTION or resume_kb_chunks.",
    )
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Delete and recreate the Qdrant collection before uploading vectors.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    db_path = args.db

    if args.reset:
        reset_db(db_path)

    init_db(db_path)

    resume_id = None
    job_id = None
    resume_chunk_count = None
    job_chunk_count = None
    resume_embed_count = None
    job_embed_count = None

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        if not args.skip_resume:
            resume_data = load_json(args.resume_json)
            resume_id = insert_resume_json(conn, resume_data)
            resume_chunk_count = insert_resume_chunks(conn, resume_id)

        if not args.skip_jd:
            job_id = insert_jd_file(
                conn,
                args.jd_txt,
                job_title=args.job_title,
                company=args.company,
            )
            job_chunk_count = insert_job_chunks(conn, job_id)

        conn.commit()

        if args.show:
            if resume_id is not None:
                print_resume_chunks(conn, [resume_id])

            if job_id is not None:
                print_job_chunks(conn, [job_id])

    if not args.skip_embed:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            client = get_qdrant_client()

            resume_embed_count, collection_ready = embed_table(
                conn,
                client,
                "resume_chunks",
                batch_size=args.batch_size,
                collection_name=args.collection,
                recreate_collection=args.recreate_collection,
                collection_ready=False,
            )
            job_embed_count, _ = embed_table(
                conn,
                client,
                "job_chunks",
                batch_size=args.batch_size,
                collection_name=args.collection,
                recreate_collection=args.recreate_collection and not collection_ready,
                collection_ready=collection_ready,
            )

    print("Pipeline finished successfully.")

    if resume_id is not None:
        print(f"resume_id={resume_id}")
        print(f"resume_chunk_count={resume_chunk_count}")
        print(f"resume_json={args.resume_json}")

    if job_id is not None:
        print(f"job_id={job_id}")
        print(f"job_chunk_count={job_chunk_count}")
        print(f"jd_txt={args.jd_txt}")

    if resume_embed_count is not None:
        print(f"resume_embed_count={resume_embed_count}")
        print(f"job_embed_count={job_embed_count}")
        print(f"qdrant_collection={args.collection}")

    print(f"db_path={db_path}")


if __name__ == "__main__":
    main()
