"""Reusable service functions for the script-style entrypoints."""

from __future__ import annotations

import json
import re
from typing import Any

from app.interview_agent import InterviewAgentProfile, interview_agent
from app.retrieval import retrieve_table_context_for_question
from app.schemas import EvaluationRequest, QuestionRequest


def strip_markdown_code_fences(text: str) -> str:
    """Remove a surrounding markdown code block if one exists."""
    stripped = text.strip()

    fenced_block = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.S | re.I)
    if fenced_block:
        return fenced_block.group(1).strip()

    fenced_search = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.S | re.I)
    if fenced_search:
        return fenced_search.group(1).strip()

    return stripped


def parse_json_response(response_text: str) -> Any:
    """Turn model text response into Python JSON with lightweight repair steps."""
    cleaned = strip_markdown_code_fences(response_text)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Some models prepend or append commentary. Try extracting JSON object span.
        object_start = cleaned.find("{")
        object_end = cleaned.rfind("}")

        if object_start != -1 and object_end != -1 and object_end > object_start:
            candidate_object = cleaned[object_start : object_end + 1]
            try:
                return json.loads(candidate_object)
            except json.JSONDecodeError:
                pass

        # Also support top-level arrays when the model emits a list response.
        array_start = cleaned.find("[")
        array_end = cleaned.rfind("]")

        if array_start != -1 and array_end != -1 and array_end > array_start:
            candidate_array = cleaned[array_start : array_end + 1]
            try:
                return json.loads(candidate_array)
            except json.JSONDecodeError:
                pass

        preview = cleaned[:200].replace("\n", "\\n")
        raise ValueError(f"LLM returned invalid JSON. Preview: {preview}")


def model_to_dict(value: Any) -> dict[str, Any]:
    """Convert pydantic models or dictionaries into plain dictionaries."""
    if value is None:
        return {}

    if hasattr(value, "model_dump"):
        return value.model_dump()

    if isinstance(value, dict):
        return dict(value)

    return {}


def model_to_list_of_dicts(values: Any) -> list[dict[str, Any]]:
    """Convert a list of pydantic models/dicts into plain dictionaries."""
    if not values:
        return []

    items = []
    for value in values:
        if hasattr(value, "model_dump"):
            items.append(value.model_dump())
        elif isinstance(value, dict):
            items.append(dict(value))
    return items


def merge_bool_flags(
    base_flags: dict[str, bool],
    override_flags: dict[str, Any] | None,
) -> dict[str, bool]:
    """Merge override boolean flags into defaults while keeping booleans only."""
    merged = dict(base_flags)
    if not override_flags:
        return merged

    for key, value in override_flags.items():
        if isinstance(value, bool):
            merged[key] = value

    return merged


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
    """Map validated question request payload into runtime state keys."""
    return {
        "cv_context": build_resume_context(request),
        "job_description_context": build_job_description_context(request),
        "interview_type": request.interview_type,
        "difficulty": request.difficulty,
    }


def enrich_question_state_with_retrieval(
    state: dict[str, Any],
    request: QuestionRequest,
) -> dict[str, Any]:
    """Automatically augment question context with SQLite/Qdrant retrieval."""
    resume_file_name = None
    if request.resume and request.resume.source:
        resume_file_name = request.resume.source.file_name

    job_role_title = None
    job_company = None
    if request.job_description:
        job_role_title = request.job_description.role_title
        job_company = request.job_description.company

    retrieval_result = retrieve_table_context_for_question(
        cv_context=list(state.get("cv_context", [])),
        job_description_context=list(state.get("job_description_context", [])),
        interview_type=str(state.get("interview_type", request.interview_type)),
        difficulty=str(state.get("difficulty", request.difficulty)),
        resume_file_name=resume_file_name,
        job_role_title=job_role_title,
        job_company=job_company,
    )

    enriched_state = dict(state)
    cv_context = retrieval_result.get("cv_context")
    job_context = retrieval_result.get("job_description_context")

    if isinstance(cv_context, list):
        enriched_state["cv_context"] = cv_context
    if isinstance(job_context, list):
        enriched_state["job_description_context"] = job_context

    return enriched_state


def build_evaluation_state(request: EvaluationRequest) -> dict[str, Any]:
    """Map validated evaluation request payload into runtime state keys."""
    return {
        "cv_context": build_resume_context(request),
        "job_description_context": build_job_description_context(request),
        "question": request.question,
        "expected_good_answer_points": request.expected_good_answer_points,
        "student_answer": request.student_answer,
    }


def build_question_profile(request: QuestionRequest) -> InterviewAgentProfile:
    """Build interview profile from question request configs with safe defaults."""
    base = InterviewAgentProfile()
    config = request.interview_config

    if config is None:
        # Keep legacy requests simple by defaulting to a single generated question.
        return InterviewAgentProfile(
            interview_stage=base.interview_stage,
            seniority_level=base.seniority_level,
            difficulty_level=request.difficulty or base.difficulty_level,
            question_count=1,
            question_techniques=list(base.question_techniques),
            competencies=list(base.competencies),
            question_constraints=dict(base.question_constraints),
            fairness_rules=dict(base.fairness_rules),
        )

    question_constraints = merge_bool_flags(
        base.question_constraints,
        model_to_dict(config.question_constraints),
    )
    fairness_rules = merge_bool_flags(
        base.fairness_rules,
        model_to_dict(config.fairness_rules),
    )

    question_count = config.question_count if config.question_count is not None else base.question_count
    techniques = list(config.question_techniques) if config.question_techniques else list(base.question_techniques)
    competencies = model_to_list_of_dicts(config.competencies)

    return InterviewAgentProfile(
        interview_stage=config.interview_stage or base.interview_stage,
        seniority_level=config.seniority_level or base.seniority_level,
        difficulty_level=config.difficulty_level or request.difficulty or base.difficulty_level,
        question_count=question_count,
        question_techniques=techniques,
        competencies=competencies,
        question_constraints=question_constraints,
        fairness_rules=fairness_rules,
    )


def build_evaluation_profile(request: EvaluationRequest) -> InterviewAgentProfile:
    """Build interview profile from evaluation request configs with safe defaults."""
    base = InterviewAgentProfile()
    config = request.evaluation_config

    if config is None:
        return InterviewAgentProfile(
            evaluation_mode=base.evaluation_mode,
            scale=base.scale,
            evidence_required=base.evidence_required,
            rubric=list(base.rubric),
            rating_anchors=dict(base.rating_anchors),
            fairness_rules=dict(base.fairness_rules),
        )

    fairness_rules = merge_bool_flags(
        base.fairness_rules,
        model_to_dict(config.fairness_rules),
    )

    rubric_criteria = list(base.rubric)
    rating_anchors = dict(base.rating_anchors)

    if config.rubric is not None:
        rubric_data = model_to_dict(config.rubric)
        criteria = rubric_data.get("criteria") or []
        if criteria:
            rubric_criteria = model_to_list_of_dicts(criteria)

        custom_anchors = rubric_data.get("rating_anchors")
        if isinstance(custom_anchors, dict) and custom_anchors:
            rating_anchors = {
                str(key): str(value)
                for key, value in custom_anchors.items()
            }

    return InterviewAgentProfile(
        evaluation_mode=config.evaluation_mode or base.evaluation_mode,
        scale=config.scale or base.scale,
        evidence_required=config.evidence_required,
        rubric=rubric_criteria,
        rating_anchors=rating_anchors,
        fairness_rules=fairness_rules,
    )


def add_legacy_question_fields(result: Any) -> Any:
    """Add legacy top-level question fields when rich output is returned."""
    if not isinstance(result, dict):
        return result

    questions = result.get("questions")
    if not isinstance(questions, list) or not questions:
        return result

    first_question = questions[0]
    if not isinstance(first_question, dict):
        return result

    result.setdefault("question", first_question.get("question", ""))
    result.setdefault("type", first_question.get("technique", "structured"))
    result.setdefault("difficulty", first_question.get("difficulty", ""))
    result.setdefault("focus_area", first_question.get("competency", ""))
    result.setdefault("why_this_question", first_question.get("reason_for_asking", ""))
    result.setdefault(
        "expected_good_answer_points",
        list(first_question.get("expected_strong_answer_signals", [])),
    )
    return result


def add_legacy_evaluation_fields(result: Any) -> Any:
    """Add legacy evaluation fields when rich output is returned."""
    if not isinstance(result, dict):
        return result

    criteria_scores = result.get("criteria_scores")
    if not isinstance(criteria_scores, list):
        return result

    legacy_category_scores: dict[str, float | int] = {}
    missing_details: list[str] = []

    for item in criteria_scores:
        if not isinstance(item, dict):
            continue

        criterion = str(item.get("criterion", "")).strip()
        score = item.get("score")
        if criterion and isinstance(score, (int, float)):
            legacy_key = criterion.lower().replace(" ", "_")
            legacy_category_scores[legacy_key] = score

        for missing_item in item.get("missing_evidence", []):
            if isinstance(missing_item, str):
                missing_details.append(missing_item)

    result.setdefault("category_scores", legacy_category_scores)
    result.setdefault("missing_details", missing_details)

    coaching = result.get("candidate_coaching", {})
    if isinstance(coaching, dict):
        result.setdefault("improved_answer", coaching.get("example_improvement", ""))
        result.setdefault("next_advice", coaching.get("better_answer_strategy", ""))

    return result


def generate_question(request: QuestionRequest | dict[str, Any]) -> Any:
    """Run single-agent question generation and return decoded JSON output."""
    validated_request = QuestionRequest.model_validate(request)
    state = build_question_state(validated_request)
    state = enrich_question_state_with_retrieval(state, validated_request)
    profile = build_question_profile(validated_request)

    result_text = interview_agent.generate_question(
        cv_context=state["cv_context"],
        job_description_context=state["job_description_context"],
        interview_type=state["interview_type"],
        difficulty=state["difficulty"],
        profile=profile,
    )
    result_json = parse_json_response(result_text)
    return add_legacy_question_fields(result_json)


def evaluate_answer(request: EvaluationRequest | dict[str, Any]) -> Any:
    """Run single-agent answer evaluation and return decoded JSON output."""
    validated_request = EvaluationRequest.model_validate(request)
    state = build_evaluation_state(validated_request)
    profile = build_evaluation_profile(validated_request)

    result_text = interview_agent.evaluate_answer(
        cv_context=state["cv_context"],
        job_description_context=state["job_description_context"],
        question=state["question"],
        expected_good_answer_points=state["expected_good_answer_points"],
        student_answer=state["student_answer"],
        profile=profile,
    )
    result_json = parse_json_response(result_text)
    return add_legacy_evaluation_fields(result_json)
