"""Single prompt profile for the interview agent."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


AgentProfile = dict[str, Any]


def default_rubric() -> list[dict[str, Any]]:
    return [
        {
            "name": "Technical Accuracy",
            "weight": 30,
            "description": "Correctness, relevance, and depth of technical explanation",
        },
        {
            "name": "Specific Evidence",
            "weight": 25,
            "description": "Concrete examples, implementation details, decisions, metrics, or outcomes",
        },
        {
            "name": "Problem-Solving Process",
            "weight": 20,
            "description": "Reasoning, decomposition, debugging, and tradeoff analysis",
        },
        {
            "name": "Communication Clarity",
            "weight": 15,
            "description": "Answer is structured, understandable, and concise",
        },
        {
            "name": "Role Relevance",
            "weight": 10,
            "description": "Answer maps clearly to role requirements and target competency",
        },
    ]


AGENT_PROFILE: AgentProfile = {
    "name": "interview_agent",
    "role": "Structured interview designer and evaluator for software engineering roles.",
    "system_instruction": (
        "Use only the provided resume, job description, interview data, rubric, "
        "and candidate answers. Return only the bound structured output. Do not "
        "include markdown, code fences, or commentary outside the schema."
    ),
    "interview_stage": "technical_screen",
    "seniority_level": None,
    "difficulty_level": None,
    "question_count": 5,
    "max_followups_per_question": 1,
    "question_techniques": [
        "project_deep_dive",
        "technical_probe",
        "situational",
        "behavioral_star",
    ],
    "competencies": [],
    "question_constraints": {
        "avoid_yes_no_questions": True,
        "avoid_trivia": True,
        "avoid_leading_questions": True,
        "avoid_multi_part_questions": False,
        "require_followups": False,
        "allow_dynamic_followups": True,
        "require_expected_signals": True,
        "require_red_flags": True,
        "require_reason_for_asking": True,
        "require_resume_grounding": True,
        "require_job_alignment": True,
    },
    "fairness_rules": {
        "job_related_only": True,
        "ignore_protected_characteristics": True,
        "do_not_penalize_non_native_english": True,
        "score_only_observed_evidence": True,
        "avoid_school_prestige_bias": True,
        "do_not_infer_missing_information": True,
    },
    "evaluation_mode": "coaching",
    "scale": "1-5",
    "evidence_required": True,
    "rubric": default_rubric(),
    "rating_anchors": {
        "1": "Very weak: no relevant evidence, incorrect, vague, or unrelated answer",
        "2": "Weak: partially relevant but shallow, incomplete, or contains major gaps",
        "3": "Acceptable: mostly relevant answer with some concrete evidence but limited depth",
        "4": "Strong: clear, relevant, evidence-based answer with good reasoning and details",
        "5": "Excellent: deep, specific, well-structured answer with tradeoffs, outcomes, and strong job alignment",
    },
}


def get_agent_profile(**overrides: Any) -> AgentProfile:
    """Return a fresh copy of the single agent profile with optional overrides."""
    profile = deepcopy(AGENT_PROFILE)
    profile.update({key: value for key, value in overrides.items() if value is not None})
    return profile


def merge_profile(base: AgentProfile | None = None, **overrides: Any) -> AgentProfile:
    """Copy a profile dict and apply optional runtime overrides."""
    profile = deepcopy(base or AGENT_PROFILE)
    profile.update({key: value for key, value in overrides.items() if value is not None})
    return profile


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


def build_question_profile(request: Any) -> AgentProfile:
    """Build runtime profile data for question generation."""
    base = get_agent_profile()
    config = getattr(request, "interview_config", None)

    if config is None:
        return merge_profile(
            base,
            difficulty_level=getattr(request, "difficulty", None),
        )

    question_constraints = merge_bool_flags(
        base["question_constraints"],
        model_to_dict(config.question_constraints),
    )
    question_constraints["require_followups"] = False
    question_constraints["allow_dynamic_followups"] = True
    fairness_rules = merge_bool_flags(
        base["fairness_rules"],
        model_to_dict(config.fairness_rules),
    )

    question_count = (
        config.question_count
        if config.question_count is not None
        else base["question_count"]
    )
    techniques = (
        list(config.question_techniques)
        if config.question_techniques
        else list(base["question_techniques"])
    )

    return merge_profile(
        base,
        interview_stage=config.interview_stage or base["interview_stage"],
        seniority_level=config.seniority_level,
        difficulty_level=(
            config.difficulty_level
            or getattr(request, "difficulty", None)
        ),
        question_count=question_count,
        max_followups_per_question=base["max_followups_per_question"],
        question_techniques=techniques,
        competencies=model_to_list_of_dicts(config.competencies),
        question_constraints=question_constraints,
        fairness_rules=fairness_rules,
    )


def build_evaluation_profile(request: Any) -> AgentProfile:
    """Build runtime profile data for answer evaluation."""
    base = get_agent_profile()
    config = getattr(request, "evaluation_config", None)

    if config is None:
        return base

    fairness_rules = merge_bool_flags(
        base["fairness_rules"],
        model_to_dict(config.fairness_rules),
    )

    rubric_criteria = list(base["rubric"])
    rating_anchors = dict(base["rating_anchors"])

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

    return merge_profile(
        base,
        evaluation_mode=config.evaluation_mode or base["evaluation_mode"],
        scale=config.scale or base["scale"],
        evidence_required=config.evidence_required,
        rubric=rubric_criteria,
        rating_anchors=rating_anchors,
        fairness_rules=fairness_rules,
    )
