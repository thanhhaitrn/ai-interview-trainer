from pathlib import Path
import sqlite3


def _column_exists(cur: sqlite3.Cursor, table_name: str, column_name: str) -> bool:
    rows = cur.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(row[1] == column_name for row in rows)


def _drop_column_if_exists(
    cur: sqlite3.Cursor,
    table_name: str,
    column_name: str,
) -> None:
    if _column_exists(cur, table_name, column_name):
        cur.execute(f"ALTER TABLE {table_name} DROP COLUMN {column_name}")


def _drop_legacy_vector_storage(cur: sqlite3.Cursor) -> None:
    """Remove vector artifacts from the old SQLite-backed vector store."""

    cur.execute("DROP TABLE IF EXISTS chunk_embeddings")
    _drop_column_if_exists(cur, "resume_chunks", "embedding_id")
    _drop_column_if_exists(cur, "job_chunks", "embedding_id")


def init_db(db_path: str | Path) -> None:
    """Create SQLite tables for resume documents and extracted facts."""

    # Convert db_path into a Path object so we can use Path methods.
    # This allows the function to accept both:
    # - string path: "db/resume_kb.db"
    # - Path object: Path("db/resume_kb.db")
    db_path = Path(db_path)

    # Make sure the parent folder exist and creates the "db" folder if it does not exist.
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

         # Table 1: resumes. This is the parent table.
        cur.execute("""
        CREATE TABLE IF NOT EXISTS resumes_doc (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            page_count INTEGER,
            name TEXT,
            headline TEXT,
            raw_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ); 
            """)

         # Table 2: work resume_id connects this work back to the resume.
        cur.execute("""
        CREATE TABLE IF NOT EXISTS work_experiences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_id INTEGER NOT NULL,
            company TEXT,
            job_title TEXT,
            location TEXT,
            start_date TEXT,
            end_date TEXT,
            is_current INTEGER,
            FOREIGN KEY (resume_id) REFERENCES resumes_doc(id)
        );
        """)

        # Table 3: work_bullets. work_experience_id connects each bullet to its parent job.
        cur.execute("""
        CREATE TABLE IF NOT EXISTS work_bullets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            work_experience_id INTEGER NOT NULL,
            bullet_text TEXT NOT NULL,
            FOREIGN KEY (work_experience_id) REFERENCES work_experiences(id)
            );
            """)

        # Table 4: projects. One row = one project.
        cur.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_id INTEGER NOT NULL,
            project_name TEXT,
            FOREIGN KEY (resume_id) REFERENCES resumes_doc(id)
            );
            """)

        # Table 5: project_bullets. One row = one bullet point under one project. project_id connects each bullet to its parent project.
        cur.execute("""
        CREATE TABLE IF NOT EXISTS project_bullets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            bullet_text TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id)
        );
        """)

        # Table 6: skills. One row = one skill or one skill line.
        cur.execute("""
        CREATE TABLE IF NOT EXISTS skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_id INTEGER NOT NULL,
            skill_text TEXT NOT NULL,
            category TEXT,
            FOREIGN KEY (resume_id) REFERENCES resumes_doc(id)
        );
        """)

        #Table 7: education. Courses are separated into education_courses because one education record can have many relevant courses.
        cur.execute("""
        CREATE TABLE IF NOT EXISTS education (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_id INTEGER NOT NULL,
            degree TEXT,
            field_of_study TEXT,
            institution TEXT,
            location TEXT,
            start_date TEXT,
            end_date TEXT,
            gpa TEXT,
            FOREIGN KEY (resume_id) REFERENCES resumes_doc(id)
        );
        """)

        # Table 8: education_courses. education_id connects each course to its education record.
        cur.execute("""
        CREATE TABLE IF NOT EXISTS education_courses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            education_id INTEGER NOT NULL,
            course_name TEXT NOT NULL,
            FOREIGN KEY (education_id) REFERENCES education(id)
        );
        """)

        # Table 9: table to create semantic chunks for embeddings
        cur.execute("""
        CREATE TABLE IF NOT EXISTS resume_chunks (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_id INTEGER NOT NULL,
            section_type TEXT NOT NULL,
            source_table TEXT,
            source_id INTEGER,
            chunk_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (resume_id) REFERENCES resumes_doc(id) ON DELETE CASCADE 
        );
        """)

        # Table 10: raw job description documents
        cur.execute("""
        CREATE TABLE IF NOT EXISTS job_descriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            job_title TEXT,
            company TEXT,
            raw_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Table 11: semantic chunks created from job descriptions
        cur.execute("""
        CREATE TABLE IF NOT EXISTS job_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            section_type TEXT NOT NULL,
            source_table TEXT,
            source_id INTEGER,
            chunk_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_id) REFERENCES job_descriptions(id) ON DELETE CASCADE
        );
        """)

        _drop_legacy_vector_storage(cur)
        conn.commit()
