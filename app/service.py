"""Reusable service functions for the script-style entrypoints."""

import json
from typing import Any

from app.agent_graph import question_graph, evaluation_graph
from app.schemas import QuestionRequest, EvaluationRequest


def parse_json_response(response_text: str) -> Any:
    """Turn the model's raw text response into a Python object."""
    # The model is instructed to return JSON text, so decode it here.
    try:
        return json.loads(response_text)
    # Raise a plain ValueError here so both CLI and API callers can decide
    # how they want to surface the failure.
    except json.JSONDecodeError as exc:
        raise ValueError("LLM returned invalid JSON") from exc


def format_context_value(value: Any) -> str:
    """Convert nested values into a single human-readable line."""
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (int, float, bool)):
        return str(value)

    if isinstance(value, list):
        rendered_items = [
            format_context_value(item)
            for item in value
        ]
        clean_items = [item for item in rendered_items if item]
        return ", ".join(clean_items)

    if isinstance(value, dict):
        parts = []
        for key, item_value in value.items():
            rendered_value = format_context_value(item_value)
            if not rendered_value:
                continue
            key_label = key.replace("_", " ")
            parts.append(f"{key_label}: {rendered_value}")
        return "; ".join(parts)

    return str(value)


def build_resume_context(request: QuestionRequest | EvaluationRequest) -> list[str]:
    """Create cv_context lines from structured resume data when provided."""
    if request.resume is None:
        return list(request.cv_context)

    context_lines: list[str] = []
    candidate = request.resume.candidate

    if candidate:
        if candidate.full_name:
            context_lines.append(f"Candidate name: {candidate.full_name}")
        if candidate.headline:
            context_lines.append(f"Candidate headline: {candidate.headline}")

    for section in request.resume.sections:
        for item in section.items:
            rendered_item = format_context_value(item)
            if rendered_item:
                context_lines.append(f"{section.section_name}: {rendered_item}")

    if context_lines:
        return context_lines

    return list(request.cv_context)


def build_job_description_context(request: QuestionRequest | EvaluationRequest) -> list[str]:
    """Create job_description_context lines from structured job data when provided."""
    if request.job_description is None:
        return list(request.job_description_context)

    context_lines: list[str] = []
    job = request.job_description

    scalar_fields = [
        ("Role title", job.role_title),
        ("Company", job.company),
        ("Location", job.location),
        ("Seniority", job.seniority),
        ("Employment type", job.employment_type),
        ("Summary", job.summary),
    ]

    for label, value in scalar_fields:
        rendered_value = format_context_value(value)
        if rendered_value:
            context_lines.append(f"{label}: {rendered_value}")

    list_fields = [
        ("Responsibility", job.responsibilities),
        ("Requirement", job.requirements),
        ("Preferred skill", job.preferred_skills),
        ("Tech stack", job.tech_stack),
    ]

    for label, values in list_fields:
        for value in values:
            rendered_value = format_context_value(value)
            if rendered_value:
                context_lines.append(f"{label}: {rendered_value}")

    if context_lines:
        return context_lines

    return list(request.job_description_context)


def build_question_state(request: QuestionRequest) -> dict[str, Any]:
    """Map validated question request payload into graph state keys."""
    return {
        "cv_context": build_resume_context(request),
        "job_description_context": build_job_description_context(request),
        "interview_type": request.interview_type,
        "difficulty": request.difficulty,
    }


def build_evaluation_state(request: EvaluationRequest) -> dict[str, Any]:
    """Map validated evaluation request payload into graph state keys."""
    return {
        "cv_context": build_resume_context(request),
        "job_description_context": build_job_description_context(request),
        "question": request.question,
        "expected_good_answer_points": request.expected_good_answer_points,
        "student_answer": request.student_answer,
    }


def generate_question(request: QuestionRequest | dict[str, Any]) -> Any:
    """Run the question-generation graph and return decoded JSON output."""
    # Validate raw dictionaries before they enter the graph.
    validated_request = QuestionRequest.model_validate(request)
    # Convert the validated request into the graph state format.
    state = build_question_state(validated_request)
    # Run the dedicated question-generation graph from start to finish.
    result = question_graph.invoke(state)
    # Decode the graph's raw model output before returning it.
    return parse_json_response(result["raw_llm_output"])


def evaluate_answer(request: EvaluationRequest | dict[str, Any]) -> Any:
    """Run the answer-evaluation graph and return decoded JSON output."""
    # Validate raw dictionaries before they enter the graph.
    validated_request = EvaluationRequest.model_validate(request)
    # Convert the validated request into the graph state format.
    state = build_evaluation_state(validated_request)
    # Run the dedicated evaluation graph from start to finish.
    result = evaluation_graph.invoke(state)
    # Decode the graph's raw model output before returning it.
    return parse_json_response(result["raw_llm_output"])
