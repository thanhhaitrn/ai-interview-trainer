"""Interview agent orchestration around prompt builders and the LLM API."""

from __future__ import annotations

from typing import Any

from app.agent.profile import AgentProfile, get_agent_profile
from app.agent.outputs import EvaluatedAnswerOutput, GeneratedQuestionOutput
from app.agent.prompts import (
    build_evaluation_chat_prompt,
    build_question_chat_prompt,
)
from app.agent import llm_client


class InterviewAgent:
    """Unified agent that runs both question generation and answer evaluation."""

    def __init__(
        self,
        profile: AgentProfile | None = None,
    ):
        self.profile = profile or get_agent_profile()

    def generate_question(
        self,
        cv_context: Any,
        job_description_context: Any,
        interview_type: str,
        difficulty: str | None,
        profile: AgentProfile | None = None,
    ) -> str:
        return self.generate_question_structured(
            cv_context=cv_context,
            job_description_context=job_description_context,
            interview_type=interview_type,
            difficulty=difficulty,
            profile=profile,
        ).model_dump_json(exclude_none=True)

    def generate_question_structured(
        self,
        cv_context: Any,
        job_description_context: Any,
        interview_type: str,
        difficulty: str | None,
        profile: AgentProfile | None = None,
    ) -> GeneratedQuestionOutput:
        runtime_profile = profile or self.profile
        prompt = build_question_chat_prompt(
            cv_context=cv_context,
            job_description_context=job_description_context,
            interview_type=interview_type,
            difficulty=difficulty,
            profile=runtime_profile,
        )
        result = llm_client.call_llm_with_structured_output(
            prompt,
            GeneratedQuestionOutput,
        )
        return result

    def evaluate_answer(
        self,
        cv_context: Any,
        job_description_context: Any,
        question: str | dict[str, Any],
        expected_good_answer_points: list[str],
        student_answer: str,
        profile: AgentProfile | None = None,
        delivery_metrics: dict[str, Any] | None = None,
    ) -> str:
        return self.evaluate_answer_structured(
            cv_context=cv_context,
            job_description_context=job_description_context,
            question=question,
            expected_good_answer_points=expected_good_answer_points,
            student_answer=student_answer,
            profile=profile,
            delivery_metrics=delivery_metrics,
        ).model_dump_json(exclude_none=True)

    def evaluate_answer_structured(
        self,
        cv_context: Any,
        job_description_context: Any,
        question: str | dict[str, Any],
        expected_good_answer_points: list[str],
        student_answer: str,
        profile: AgentProfile | None = None,
        delivery_metrics: dict[str, Any] | None = None,
    ) -> EvaluatedAnswerOutput:
        runtime_profile = profile or self.profile
        prompt = build_evaluation_chat_prompt(
            cv_context=cv_context,
            job_description_context=job_description_context,
            question=question,
            expected_good_answer_points=expected_good_answer_points,
            student_answer=student_answer,
            profile=runtime_profile,
            delivery_metrics=delivery_metrics,
        )
        result = llm_client.call_llm_with_structured_output(
            prompt,
            EvaluatedAnswerOutput,
            temperature=0.0,
        )
        return result

    def workflow_steps(self) -> list[str]:
        """Return runtime workflow steps used by this single agent."""
        from app.graph.workflow import workflow_steps

        return workflow_steps()


interview_agent = InterviewAgent()
