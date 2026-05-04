"""Regression tests for the simple top-level script entrypoints."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from evaluate_answer import save_evaluated_answer
from generate_question import save_generated_question


class FakeOllamaResponse:
    """Minimal response object that behaves like `requests.Response`."""

    def __init__(self, payload: dict[str, object]):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, str]:
        return {"response": json.dumps(self._payload)}


@contextmanager
def patched_ollama_post(payload: dict[str, object]):
    """Temporarily replace `requests.post` inside the Ollama client module."""
    import app.llm_api_client as llm_api_client

    original_post = llm_api_client.requests.post

    def fake_post(*args, **kwargs):
        return FakeOllamaResponse(payload)

    llm_api_client.requests.post = fake_post
    try:
        yield
    finally:
        llm_api_client.requests.post = original_post


class ScriptEntrypointTestCase(unittest.TestCase):
    """Exercise the new script-style entrypoints."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_show_workflow_script_prints_summary(self):
        result = subprocess.run(
            [
                sys.executable,
                str(self.project_root / "show_workflow.py"),
            ],
            cwd=self.project_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("%% question_graph", result.stdout)
        self.assertIn("%% evaluation_graph", result.stdout)
        self.assertIn("graph TD;", result.stdout)
        self.assertIn("__start__", result.stdout)

    def test_generate_question_script_saves_output_json(self):
        os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434"
        os.environ["OLLAMA_MODEL"] = "test-model"
        os.environ["OLLAMA_API_KEY"] = ""

        with tempfile.TemporaryDirectory() as output_dir:
            ollama_response = {
                "question": "Describe a React project where you integrated an external API.",
                "type": "technical",
                "difficulty": "medium",
                "focus_area": "React and API integration",
                "why_this_question": "It checks practical frontend experience.",
                "expected_good_answer_points": [
                    "Project context",
                    "Integration approach",
                    "Personal contribution",
                    "Outcome"
                ]
            }

            with patched_ollama_post(ollama_response):
                output_path = save_generated_question(
                    self.project_root / "data/requests/question_request.json",
                    Path(output_dir)
                )

            self.assertTrue(output_path.exists())

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIn("question", payload)

    def test_evaluate_answer_script_saves_output_json(self):
        os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434"
        os.environ["OLLAMA_MODEL"] = "test-model"
        os.environ["OLLAMA_API_KEY"] = ""

        with tempfile.TemporaryDirectory() as output_dir:
            ollama_response = {
                "overall_score": 7,
                "category_scores": {
                    "relevance": 8,
                    "clarity": 7,
                    "specificity": 6,
                    "job_alignment": 7,
                    "cv_alignment": 7,
                    "structure": 7
                },
                "strengths": ["Relevant answer with clear project context."],
                "weaknesses": ["Could include more measurable impact."],
                "missing_details": ["Quantified results"],
                "improved_answer": "I built a React dashboard with API integration and improved load time by 20%.",
                "next_advice": "Add concrete metrics and decision tradeoffs."
            }

            with patched_ollama_post(ollama_response):
                output_path = save_evaluated_answer(
                    self.project_root / "data/requests/evaluation_request.json",
                    Path(output_dir)
                )

            self.assertTrue(output_path.exists())

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIn("overall_score", payload)


if __name__ == "__main__":
    unittest.main()
