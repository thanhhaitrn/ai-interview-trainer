"""Tests for graph state and the agent compatibility facade."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.agent import InterviewAgent, get_agent_profile
from app.graph import GraphState, merge_dicts


class InterviewGraphStateTestCase(unittest.TestCase):
    """Verify graph state and direct agent call behavior."""

    def test_graph_state_accepts_workflow_node_keys(self):
        state: GraphState = {
            "request_payload": {"interview_type": "technical"},
            "status": "pending",
            "turn_summaries": [],
            "turns": [],
            "errors": [],
        }

        self.assertEqual(state["status"], "pending")
        self.assertEqual(state["request_payload"], {"interview_type": "technical"})
        self.assertEqual(state["turn_summaries"], [])
        self.assertEqual(state["turns"], [])
        self.assertEqual(state["errors"], [])

    def test_merge_dicts_overwrites_right_hand_values(self):
        self.assertEqual(
            merge_dicts({"score": 1, "status": "old"}, {"status": "new"}),
            {"score": 1, "status": "new"},
        )

    def test_agent_facade_generates_structured_question(self):
        agent = InterviewAgent(profile=get_agent_profile())

        def fake_structured_call(prompt, schema, **kwargs):
            return schema.model_validate(
                {
                    "interview_stage": "technical_screen",
                    "seniority_level": "junior",
                    "difficulty_level": "medium",
                    "question_count": 0,
                }
            )

        with patch(
            "app.agent.llm_client.call_llm_with_structured_output",
            side_effect=fake_structured_call,
        ):
            result = agent.generate_question_structured(
                cv_context=["Built REST APIs"],
                job_description_context=["Backend role"],
                interview_type="technical",
                difficulty="medium",
            )

        self.assertEqual(result.interview_stage, "technical_screen")


if __name__ == "__main__":
    unittest.main()
