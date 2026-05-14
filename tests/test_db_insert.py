import sqlite3

from src.db.schema import init_db
from src.db.insert_resume import insert_resume_json


def test_insert_resume_json(tmp_path):
    db_path = tmp_path / "test_resume_kb.db"

    data = {
        "document_type": "resume",
        "source": {
            "file_name": "resume1.pdf",
            "page_count": 1
        },
        "candidate": {
            "full_name": "Trish Mathers",
            "headline": "Entry-Level Data Scientist"
        },
        "sections": [
            {
                "section_name": "Skills",
                "items": [
                    {
                        "bullets": ["SQL", "Python", "Excel"]
                    }
                ]
            }
        ]
    }

    init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        resume_id = insert_resume_json(conn, data)
        conn.commit()

        rows = conn.execute("""
            SELECT skill_text
            FROM skills
            WHERE resume_id = ?;
        """, (resume_id,)).fetchall()

    assert len(rows) == 3
    assert rows[0][0] == "SQL"


def test_insert_resume_json_inserts_work_experience(tmp_path):
    db_path = tmp_path / "test_resume_kb.db"

    data = {
        "document_type": "resume",
        "source": {
            "file_name": "resume1.pdf",
            "page_count": 1
        },
        "candidate": {
            "full_name": "Trish Mathers",
            "headline": "Entry-Level Data Scientist"
        },
        "sections": [
            {
                "section_name": "Work Experience",
                "items": [
                    {
                        "company": "Niantic",
                        "job_title": "Data Scientist Intern",
                        "location": "Seattle, WA",
                        "start_date": "2020-04",
                        "end_date": "2021-04",
                        "is_current": False,
                        "bullets": ["Built regression automation."]
                    }
                ]
            }
        ]
    }

    init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        resume_id = insert_resume_json(conn, data)
        conn.commit()

        resume_row = conn.execute("""
            SELECT file_name, page_count, name
            FROM resumes_doc
            WHERE id = ?;
        """, (resume_id,)).fetchone()

        work_row = conn.execute("""
            SELECT id, company, job_title
            FROM work_experiences
            WHERE resume_id = ?;
        """, (resume_id,)).fetchone()

        bullet_rows = conn.execute("""
            SELECT bullet_text
            FROM work_bullets
            WHERE work_experience_id = ?;
        """, (work_row[0],)).fetchall()

    assert resume_row == ("resume1.pdf", 1, "Trish Mathers")
    assert work_row[1:] == ("Niantic", "Data Scientist Intern")
    assert bullet_rows == [("Built regression automation.",)]


def test_init_db_removes_legacy_sql_vector_storage(tmp_path):
    db_path = tmp_path / "legacy_vector_store.db"

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE resume_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_id INTEGER NOT NULL,
                section_type TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                embedding_id TEXT
            );
        """)
        conn.execute("""
            CREATE TABLE job_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL,
                section_type TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                embedding_id TEXT
            );
        """)
        conn.execute("""
            CREATE TABLE chunk_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id INTEGER NOT NULL,
                embedding_json TEXT NOT NULL
            );
        """)
        conn.commit()

    init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        table_row = conn.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'chunk_embeddings'
        """).fetchone()
        resume_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(resume_chunks)").fetchall()
        }
        job_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(job_chunks)").fetchall()
        }

    assert table_row is None
    assert "embedding_id" not in resume_columns
    assert "embedding_id" not in job_columns
