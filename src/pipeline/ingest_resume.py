import argparse
import json
import sqlite3
from pathlib import Path

from src.db.schema import init_db
from src.db.insert_resume import insert_resume_json


DB_PATH = Path("db/resume_kb.db")
JSON_PATH = Path("data/resumes/llm/resume1_parsed_llm.json")


def load_json(path: str | Path) -> dict:
    """Load parsed resume JSON from a file."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def reset_db(db_path: str | Path) -> None:
    """
    Delete the existing SQLite database file.
    Use this when you want a completely fresh database.
    Be careful: this removes all old inserted resumes.
    """
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()
        print(f"Deleted old database: {db_path}")

def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Insert one parsed resume JSON file into the SQLite resume database."
    )

    parser.add_argument(
        "json_path",
        nargs="?",
        type=Path,
        default=JSON_PATH,
        help="Path to the parsed resume JSON file. Example: data/resume1.json",
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help="Path to SQLite database. Default: db/resume_kb.db",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing database before inserting data.",
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    json_path = args.json_path
    db_path = args.db

    if args.reset:
        reset_db(db_path)

    init_db(db_path)

    data = load_json(json_path)

    with sqlite3.connect(db_path) as conn:
        resume_id = insert_resume_json(conn, data)
        conn.commit()

    print("Inserted resume successfully.")
    print(f"resume_id={resume_id}")
    print(f"json_path={json_path}")
    print(f"db_path={db_path}")

if __name__ == "__main__":
    main()
