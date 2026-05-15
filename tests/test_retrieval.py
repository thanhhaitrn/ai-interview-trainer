"""Tests for automatic SQLite/Qdrant context retrieval during question generation."""

from __future__ import annotations

import os
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.retrieval import retrieve_table_context_for_question
from app.schemas import QuestionRequest
from app.service import build_question_state, enrich_question_state_with_retrieval


def create_test_db(db_path: Path) -> None:
    """Create minimal tables/rows needed by retrieval tests."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE resumes_doc (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE resume_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_id INTEGER NOT NULL,
                chunk_text TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE job_descriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_title TEXT,
                company TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE job_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL,
                chunk_text TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            INSERT INTO resumes_doc (id, file_name)
            VALUES (1, 'friend1_parsed_llm.json')
            """
        )
        conn.executemany(
            """
            INSERT INTO resume_chunks (resume_id, chunk_text)
            VALUES (?, ?)
            """,
            [
                (
                    1,
                    "Work experience: implemented Python REST API endpoints and handled debugging.",
                ),
                (
                    1,
                    "Skills: SQL schema design, query optimization, logging, testing.",
                ),
            ],
        )

        conn.execute(
            """
            INSERT INTO job_descriptions (id, job_title, company)
            VALUES (1, 'Junior Backend Developer', 'Nordic Cloud Labs')
            """
        )
        conn.executemany(
            """
            INSERT INTO job_chunks (job_id, chunk_text)
            VALUES (?, ?)
            """,
            [
                (
                    1,
                    "Job description: design and implement REST APIs in Python.",
                ),
                (
                    1,
                    "Job description: debug production incidents and communicate root causes.",
                ),
            ],
        )
        conn.commit()


class RetrievalTestCase(unittest.TestCase):
    """Validate automatic retrieval and service wiring behavior."""

    def test_retrieval_uses_sqlite_fallback_when_qdrant_returns_empty(self) -> None:
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "resume_kb.db"
            create_test_db(db_path)

            env_overrides = {
                "INTERVIEW_AUTO_RETRIEVAL": "1",
                "INTERVIEW_DB_PATH": str(db_path),
                "INTERVIEW_RETRIEVAL_TOP_K": "2",
            }

            with patch.dict(os.environ, env_overrides, clear=False):
                with patch(
                    "app.retrieval.retrieve_from_qdrant",
                    return_value=([], [], {"source": "qdrant", "used": False}),
                ):
                    result = retrieve_table_context_for_question(
                        cv_context=["Candidate has project experience."],
                        job_description_context=["Role needs REST APIs and SQL skills."],
                        interview_type="technical",
                        difficulty="medium",
                        resume_file_name="friend1_parsed_llm.json",
                        job_role_title="Junior Backend Developer",
                        job_company="Nordic Cloud Labs",
                    )

            cv_context = result["cv_context"]
            job_context = result["job_description_context"]

            self.assertTrue(
                any("Retrieved resume evidence:" in line for line in cv_context),
            )
            self.assertTrue(
                any("Retrieved job evidence:" in line for line in job_context),
            )
            self.assertGreaterEqual(result["metadata"]["resume_chunks_added"], 1)
            self.assertGreaterEqual(result["metadata"]["job_chunks_added"], 1)

    def test_retrieval_can_be_disabled_by_env(self) -> None:
        env_overrides = {
            "INTERVIEW_AUTO_RETRIEVAL": "0",
        }

        with patch.dict(os.environ, env_overrides, clear=False):
            result = retrieve_table_context_for_question(
                cv_context=["legacy cv"],
                job_description_context=["legacy job"],
                interview_type="technical",
                difficulty="easy",
            )

        self.assertEqual(result["cv_context"], ["legacy cv"])
        self.assertEqual(result["job_description_context"], ["legacy job"])
        self.assertEqual(result["metadata"]["enabled"], False)

    def test_service_enrichment_uses_retrieval_output(self) -> None:
        payload = {
            "cv_context": ["legacy cv"],
            "job_description_context": ["legacy job"],
            "interview_type": "technical",
            "difficulty": "medium",
        }
        request = QuestionRequest.model_validate(payload)
        state = build_question_state(request)

        with patch(
            "app.service.retrieve_table_context_for_question",
            return_value={
                "cv_context": ["legacy cv", "Retrieved resume evidence: chunk 1"],
                "job_description_context": ["legacy job", "Retrieved job evidence: chunk 1"],
                "metadata": {"enabled": True},
            },
        ):
            enriched = enrich_question_state_with_retrieval(state, request)

        self.assertIn("Retrieved resume evidence: chunk 1", enriched["cv_context"])
        self.assertIn("Retrieved job evidence: chunk 1", enriched["job_description_context"])


if __name__ == "__main__":
    unittest.main()

