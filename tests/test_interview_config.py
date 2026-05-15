"""Unit tests for interview config wiring and prompt constraints."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from app.interview_agent import InterviewAgent, InterviewAgentProfile
from app.schemas import EvaluationRequest, QuestionRequest
from app.service import build_evaluation_profile, build_question_profile


class InterviewConfigTestCase(unittest.TestCase):
    """Verify config mapping and structured prompt design requirements."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.agent = InterviewAgent(profile=InterviewAgentProfile())

    def test_question_profile_loads_rich_config(self):
        payload = json.loads(
            (self.project_root / "data/requests/question_request.json").read_text(encoding="utf-8")
        )
        request = QuestionRequest.model_validate(payload)
        profile = build_question_profile(request)

        self.assertEqual(profile.interview_stage, "technical_screen")
        self.assertEqual(profile.seniority_level, "junior")
        self.assertEqual(profile.question_count, 6)
        self.assertIn("situational", profile.question_techniques)
        self.assertTrue(profile.question_constraints["require_followups"])

    def test_evaluation_profile_loads_rich_config(self):
        payload = json.loads(
            (self.project_root / "data/requests/evaluation_request.json").read_text(encoding="utf-8")
        )
        request = EvaluationRequest.model_validate(payload)
        profile = build_evaluation_profile(request)

        self.assertEqual(profile.evaluation_mode, "coaching")
        self.assertEqual(profile.scale, "1-5")
        self.assertTrue(profile.evidence_required)
        self.assertIn("1", profile.rating_anchors)
        self.assertGreaterEqual(len(profile.rubric), 5)

    def test_question_prompt_contains_required_sections(self):
        prompt = self.agent._build_question_prompt(
            cv_context=["Built REST APIs", "Used SQL and debugging workflows"],
            job_description_context=["Junior backend role", "Requires API and SQL fundamentals"],
            interview_type="technical",
            difficulty="medium",
            profile=InterviewAgentProfile(),
        )

        self.assertIn("Structured Employment Interview", prompt)
        self.assertIn("Critical Incident Technique", prompt)
        self.assertIn("Situational Interview", prompt)
        self.assertIn("Behavior Description Interview", prompt)
        self.assertIn("Output JSON schema", prompt)
        self.assertIn('"expected_strong_answer_signals"', prompt)
        self.assertIn('"follow_up_questions"', prompt)

    def test_evaluation_prompt_contains_rubric_and_evidence_rules(self):
        prompt = self.agent._build_evaluation_prompt(
            cv_context=["Implemented backend APIs in Python"],
            job_description_context=["Role requires API debugging and communication"],
            question="Tell me about a backend incident you resolved.",
            expected_good_answer_points=["Incident", "Diagnosis", "Fix", "Outcome"],
            student_answer="I investigated logs, identified a query bottleneck, and improved latency.",
            profile=InterviewAgentProfile(),
        )

        self.assertIn("Behaviorally Anchored Rating Scales", prompt)
        self.assertIn("Rubric criteria", prompt)
        self.assertIn("Rating anchors", prompt)
        self.assertIn("Score only what is present in the answer.", prompt)
        self.assertIn('"hiring_signal"', prompt)
        self.assertIn('"evidence_from_answer"', prompt)


if __name__ == "__main__":
    unittest.main()
