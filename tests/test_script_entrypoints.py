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


class FakeOllamaRawResponse:
    """Minimal response object that returns raw model text."""

    def __init__(self, raw_text: str):
        self._raw_text = raw_text

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, str]:
        return {"response": self._raw_text}


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


@contextmanager
def patched_ollama_raw_text(raw_text: str):
    """Temporarily replace `requests.post` to return raw LLM text."""
    import app.llm_api_client as llm_api_client

    original_post = llm_api_client.requests.post

    def fake_post(*args, **kwargs):
        return FakeOllamaRawResponse(raw_text)

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
        self.assertIn("flowchart TD", result.stdout)
        self.assertIn("InterviewAgent profile", result.stdout)
        self.assertIn("Call Ollama /api/generate once", result.stdout)

    def test_generate_question_script_saves_output_json(self):
        os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434"
        os.environ["OLLAMA_MODEL"] = "test-model"
        os.environ["OLLAMA_API_KEY"] = ""

        with tempfile.TemporaryDirectory() as output_dir:
            ollama_response = {
                "interview_stage": "technical_screen",
                "seniority_level": "junior",
                "difficulty_level": "medium",
                "question_count": 2,
                "questions": [
                    {
                        "id": "q1",
                        "question": "Describe a backend API you implemented and why you designed it that way.",
                        "competency": "Backend API Development",
                        "technique": "project_deep_dive",
                        "difficulty": "medium",
                        "reason_for_asking": "Assess implementation depth.",
                        "resume_grounding": "Candidate worked on SQL and modeling projects.",
                        "job_alignment": "Role requires backend API implementation.",
                        "expected_strong_answer_signals": ["Concrete endpoint design decisions."],
                        "red_flags": ["No concrete implementation details."],
                        "follow_up_questions": ["How did you validate error handling?"],
                        "scoring_guidance": {
                            "strong_answer": "Clear design, tradeoffs, and validation.",
                            "average_answer": "Basic implementation with limited rationale.",
                            "weak_answer": "Vague or theoretical answer only."
                        }
                    }
                ],
                "coverage_summary": {
                    "competencies_covered": ["Backend API Development"],
                    "techniques_used": ["project_deep_dive"],
                    "notes": "Focused on core API competency first."
                }
            }

            with patched_ollama_post(ollama_response):
                output_path = save_generated_question(
                    self.project_root / "data/requests/question_request.json",
                    Path(output_dir)
                )

            self.assertTrue(output_path.exists())

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIn("questions", payload)
            self.assertIn("question", payload)

    def test_evaluate_answer_script_saves_output_json(self):
        os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434"
        os.environ["OLLAMA_MODEL"] = "test-model"
        os.environ["OLLAMA_API_KEY"] = ""

        with tempfile.TemporaryDirectory() as output_dir:
            ollama_response = {
                "overall_score": 4.0,
                "overall_rating": "strong",
                "hiring_signal": "positive",
                "confidence": "medium",
                "summary": "Good answer with concrete debugging evidence.",
                "criteria_scores": [
                    {
                        "criterion": "Technical Accuracy",
                        "weight": 30,
                        "score": 4,
                        "weighted_score": 1.2,
                        "reason": "Correct use of query optimization.",
                        "evidence_from_answer": ["Added index and query changes."],
                        "missing_evidence": ["Did not discuss tradeoffs."],
                        "improvement_advice": "Explain tradeoffs and alternatives."
                    }
                ],
                "strengths": ["Clear debugging steps."],
                "weaknesses": ["Limited tradeoff discussion."],
                "red_flags": [],
                "follow_up_questions": ["How would you monitor regression risk after deployment?"],
                "candidate_coaching": {
                    "better_answer_strategy": "Include context, decision, validation, and tradeoffs.",
                    "example_improvement": "I profiled the query plan, added an index, and validated p95 latency reduction."
                },
                "fairness_check": {
                    "used_only_job_relevant_evidence": True,
                    "ignored_protected_characteristics": True,
                    "notes": "Scored only technical evidence."
                }
            }

            with patched_ollama_post(ollama_response):
                output_path = save_evaluated_answer(
                    self.project_root / "data/requests/evaluation_request.json",
                    Path(output_dir)
                )

            self.assertTrue(output_path.exists())

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIn("overall_score", payload)
            self.assertIn("category_scores", payload)
            self.assertIn("next_advice", payload)

    def test_generate_question_legacy_request_saves_output_json(self):
        os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434"
        os.environ["OLLAMA_MODEL"] = "test-model"
        os.environ["OLLAMA_API_KEY"] = ""

        with tempfile.TemporaryDirectory() as output_dir:
            ollama_response = {
                "question": "Tell me about a backend bug you fixed.",
                "type": "technical",
                "difficulty": "medium",
                "focus_area": "Debugging",
                "why_this_question": "Checks practical troubleshooting ability.",
                "expected_good_answer_points": [
                    "Specific bug",
                    "Diagnosis process",
                    "Fix and verification"
                ]
            }

            with patched_ollama_post(ollama_response):
                output_path = save_generated_question(
                    self.project_root / "data/requests/question_request_legacy.json",
                    Path(output_dir)
                )

            self.assertTrue(output_path.exists())

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIn("question", payload)

    def test_evaluate_answer_legacy_request_supports_code_fence_json(self):
        os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434"
        os.environ["OLLAMA_MODEL"] = "test-model"
        os.environ["OLLAMA_API_KEY"] = ""

        with tempfile.TemporaryDirectory() as output_dir:
            raw_json = """```json
{
  "overall_score": 3.5,
  "category_scores": {
    "technical_accuracy": 4
  },
  "strengths": ["Clear diagnosis flow."],
  "weaknesses": ["Could provide stronger metrics."],
  "missing_details": ["Long-term prevention strategy"],
  "improved_answer": "I profiled the endpoint and validated the fix with load tests.",
  "next_advice": "Add tradeoff rationale and monitoring details."
}
```"""

            with patched_ollama_raw_text(raw_json):
                output_path = save_evaluated_answer(
                    self.project_root / "data/requests/evaluation_request_legacy.json",
                    Path(output_dir)
                )

            self.assertTrue(output_path.exists())

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["overall_score"], 3.5)
            self.assertIn("next_advice", payload)


if __name__ == "__main__":
    unittest.main()
