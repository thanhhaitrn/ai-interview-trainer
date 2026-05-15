"""Single configurable interview agent for question generation and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from app.llm_api_client import call_llm_api


DEFAULT_QUESTION_TECHNIQUES = [
    "project_deep_dive",
    "technical_probe",
    "situational",
    "behavioral_star",
]

DEFAULT_QUESTION_CONSTRAINTS = {
    "avoid_yes_no_questions": True,
    "avoid_trivia": True,
    "avoid_leading_questions": True,
    "avoid_multi_part_questions": False,
    "require_followups": True,
    "require_expected_signals": True,
    "require_red_flags": True,
    "require_reason_for_asking": True,
    "require_resume_grounding": True,
    "require_job_alignment": True,
}

DEFAULT_FAIRNESS_RULES = {
    "job_related_only": True,
    "ignore_protected_characteristics": True,
    "do_not_penalize_non_native_english": True,
    "score_only_observed_evidence": True,
    "avoid_school_prestige_bias": True,
    "do_not_infer_missing_information": True,
}

DEFAULT_RATING_ANCHORS = {
    "1": "Very weak: no relevant evidence, incorrect, vague, or unrelated answer",
    "2": "Weak: partially relevant but shallow, incomplete, or contains major gaps",
    "3": "Acceptable: mostly relevant answer with some concrete evidence but limited depth",
    "4": "Strong: clear, relevant, evidence-based answer with good reasoning and details",
    "5": "Excellent: deep, specific, well-structured answer with tradeoffs, outcomes, and strong job alignment",
}


def _default_rubric() -> list[dict[str, Any]]:
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


def format_list(items: List[str]) -> str:
    """Render a list as prompt bullets, or a fallback message when empty."""
    if not items:
        return "No context provided."

    return "\n".join(f"- {item}" for item in items)


@dataclass(frozen=True)
class InterviewAgentProfile:
    """Configurable interview profile used to build structured prompts."""

    name: str = "InterviewCoach Agent"
    role: str = (
        "Structured interview designer and evaluator for software engineering roles."
    )
    rules: List[str] = field(
        default_factory=lambda: [
            "Use only provided resume, job, question, and answer evidence.",
            "Return strict JSON only.",
            "Do not include markdown.",
            "Do not add commentary outside JSON.",
        ]
    )
    interview_stage: str = "technical_screen"
    seniority_level: str = "junior"
    difficulty_level: str = "medium"
    question_count: int = 5
    question_techniques: List[str] = field(
        default_factory=lambda: list(DEFAULT_QUESTION_TECHNIQUES)
    )
    competencies: List[dict[str, Any]] = field(default_factory=list)
    question_constraints: dict[str, bool] = field(
        default_factory=lambda: dict(DEFAULT_QUESTION_CONSTRAINTS)
    )
    evaluation_mode: str = "coaching"
    scale: str = "1-5"
    evidence_required: bool = True
    rubric: List[dict[str, Any]] = field(default_factory=_default_rubric)
    rating_anchors: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_RATING_ANCHORS)
    )
    fairness_rules: dict[str, bool] = field(
        default_factory=lambda: dict(DEFAULT_FAIRNESS_RULES)
    )


class InterviewAgent:
    """Unified agent that runs both question generation and answer evaluation."""

    def __init__(self, profile: InterviewAgentProfile):
        self.profile = profile

    def _default_rubric(self) -> list[dict[str, Any]]:
        return _default_rubric()

    def _normalize_weights(
        self,
        items: list[dict[str, Any]],
        weight_key: str = "weight",
    ) -> list[dict[str, Any]]:
        if not items:
            return []

        copied_items = [dict(item) for item in items]
        parsed_weights: list[float | None] = []

        for item in copied_items:
            raw_weight = item.get(weight_key)
            if isinstance(raw_weight, (int, float)):
                parsed_weights.append(max(float(raw_weight), 0.0))
            else:
                parsed_weights.append(None)

        if all(weight is None for weight in parsed_weights):
            equal_weight = round(100.0 / len(copied_items), 2)
            for item in copied_items:
                item[weight_key] = equal_weight
            return copied_items

        numeric_weights = [weight or 0.0 for weight in parsed_weights]
        total_weight = sum(numeric_weights)

        if total_weight <= 0:
            equal_weight = round(100.0 / len(copied_items), 2)
            for item in copied_items:
                item[weight_key] = equal_weight
            return copied_items

        for index, item in enumerate(copied_items):
            normalized = round((numeric_weights[index] / total_weight) * 100.0, 2)
            item[weight_key] = normalized

        return copied_items

    def _format_competencies(self, competencies: list[dict[str, Any]]) -> str:
        if not competencies:
            return "- No explicit competencies provided. Infer from role requirements."

        normalized = self._normalize_weights(competencies, weight_key="weight")
        lines = []

        for index, competency in enumerate(normalized, start=1):
            name = str(competency.get("name", f"Competency {index}")).strip()
            description = str(competency.get("description") or "").strip()
            weight = competency.get("weight", 0)

            if description:
                lines.append(
                    f"- {index}. {name} (weight={weight}%): {description}"
                )
            else:
                lines.append(f"- {index}. {name} (weight={weight}%)")

        return "\n".join(lines)

    def _format_rubric(self, rubric: list[dict[str, Any]]) -> str:
        if not rubric:
            rubric = self._default_rubric()

        normalized = self._normalize_weights(rubric, weight_key="weight")
        lines = []

        for index, criterion in enumerate(normalized, start=1):
            name = str(criterion.get("name", f"Criterion {index}")).strip()
            description = str(criterion.get("description") or "").strip()
            weight = criterion.get("weight", 0)

            if description:
                lines.append(f"- {index}. {name} (weight={weight}%): {description}")
            else:
                lines.append(f"- {index}. {name} (weight={weight}%)")

        return "\n".join(lines)

    def _format_rating_anchors(self, rating_anchors: dict[str, str]) -> str:
        if not rating_anchors:
            rating_anchors = dict(DEFAULT_RATING_ANCHORS)

        def sort_key(raw_key: str) -> tuple[int, str]:
            if raw_key.isdigit():
                return (0, f"{int(raw_key):04d}")
            return (1, raw_key)

        ordered_keys = sorted(rating_anchors.keys(), key=sort_key)
        return "\n".join(
            f"- {key}: {rating_anchors[key]}"
            for key in ordered_keys
        )

    def _format_question_constraints(self, constraints: dict[str, bool]) -> str:
        if not constraints:
            constraints = dict(DEFAULT_QUESTION_CONSTRAINTS)

        lines = []
        for key, value in constraints.items():
            label = key.replace("_", " ")
            lines.append(f"- {label}: {value}")
        return "\n".join(lines)

    def _format_fairness_rules(self, fairness_rules: dict[str, bool]) -> str:
        if not fairness_rules:
            fairness_rules = dict(DEFAULT_FAIRNESS_RULES)

        lines = []
        for key, value in fairness_rules.items():
            label = key.replace("_", " ")
            lines.append(f"- {label}: {value}")
        return "\n".join(lines)

    def _safe_json_instruction(self, schema: str) -> str:
        return (
            "Return only valid JSON matching this schema.\n"
            "Do not wrap JSON in markdown.\n"
            "Do not include commentary outside JSON.\n"
            f"{schema}"
        )

    def _profile_block(self, profile: InterviewAgentProfile) -> str:
        rules_text = "\n".join(f"- {rule}" for rule in profile.rules)
        return (
            f"You are {profile.name}.\n"
            f"Role: {profile.role}\n"
            "Global Rules:\n"
            f"{rules_text}\n"
        )

    def _build_question_prompt(
        self,
        cv_context: List[str],
        job_description_context: List[str],
        interview_type: str,
        difficulty: str,
        profile: InterviewAgentProfile,
    ) -> str:
        question_schema = """
{
  "interview_stage": "",
  "seniority_level": "",
  "difficulty_level": "",
  "question_count": 0,
  "questions": [
    {
      "id": "q1",
      "question": "",
      "competency": "",
      "technique": "",
      "difficulty": "",
      "reason_for_asking": "",
      "resume_grounding": "",
      "job_alignment": "",
      "expected_strong_answer_signals": [""],
      "red_flags": [""],
      "follow_up_questions": [""],
      "scoring_guidance": {
        "strong_answer": "",
        "average_answer": "",
        "weak_answer": ""
      }
    }
  ],
  "coverage_summary": {
    "competencies_covered": [""],
    "techniques_used": [""],
    "notes": ""
  },
  "legacy_compatibility": {
    "question": "",
    "type": "",
    "difficulty": "",
    "focus_area": "",
    "why_this_question": "",
    "expected_good_answer_points": [""]
  }
}
""".strip()

        return f"""
{self._profile_block(profile)}

Role:
You are a structured interview designer for software engineering roles.

Task:
Generate job-relevant, fair, evidence-seeking interview questions grounded in the provided resume and job description.

Scientific Interview Methods To Apply:
- Structured Employment Interview (Campion, Palmer, & Campion, 1997; Levashina et al., 2014):
  standardized format, job-related competencies, predefined criteria, consistent scoring guidance.
- Critical Incident Technique (Flanagan, 1954):
  use realistic incidents where effective/ineffective performance can be observed.
- Situational Interview (Latham, Saari, Pursell, & Campion, 1980):
  include future-oriented hypothetical dilemmas asking what the candidate would do.
- Behavior Description Interview / Patterned Behavior Description Interview (Janz, 1982):
  include past-behavior questions asking what the candidate actually did.
- STAR (Situation, Task, Action, Result) is only a completeness checklist for behavioral answers.

Inputs:
CV context:
{format_list(cv_context)}

Job description context:
{format_list(job_description_context)}

Interview configuration:
- interview stage: {profile.interview_stage}
- seniority level: {profile.seniority_level}
- difficulty level: {profile.difficulty_level or difficulty}
- legacy interview type: {interview_type}
- legacy difficulty: {difficulty}
- desired question count: {profile.question_count}

Competencies:
{self._format_competencies(profile.competencies)}

Question techniques:
{format_list(profile.question_techniques)}

Question constraints:
{self._format_question_constraints(profile.question_constraints)}

Fairness rules:
{self._format_fairness_rules(profile.fairness_rules)}

Question design rules:
- Do not generate generic questions.
- Do not ask yes/no questions unless explicitly required by job constraints.
- Do not ask illegal/discriminatory/personal/family/health/religion/nationality/age/gender/marital/protected-characteristic questions.
- Do not invent resume details.
- Do not invent job requirements.
- If information is missing, ask fair job-related questions to explore missing evidence.
- Prefer open-ended questions.
- Avoid trivia unless the job explicitly requires factual recall.
- Avoid memorization-only prompts.
- Prefer questions that reveal reasoning, tradeoffs, debugging, implementation decisions, communication, and learning.
- Junior focus: fundamentals, practical projects, debugging, learning ability, clear explanations.
- Mid-level focus: ownership, design tradeoffs, cross-team collaboration, decision quality.
- Senior focus: architecture, ambiguity, mentoring, risk, scalability, cross-functional decisions.
- Each question must map to exactly one primary competency.
- Use configured techniques where possible.
- If require_followups is true, include at least one follow_up_questions item per question.
- If require_expected_signals is true, include expected_strong_answer_signals per question.
- If require_red_flags is true, include red_flags per question.
- If require_reason_for_asking is true, include reason_for_asking per question.
- If require_resume_grounding is true, include resume_grounding per question.
- If require_job_alignment is true, include job_alignment per question.

Output JSON schema:
{self._safe_json_instruction(question_schema)}
"""

    def _build_evaluation_prompt(
        self,
        cv_context: List[str],
        job_description_context: List[str],
        question: str,
        expected_good_answer_points: List[str],
        student_answer: str,
        profile: InterviewAgentProfile,
    ) -> str:
        evaluation_schema = """
{
  "overall_score": 0,
  "overall_rating": "",
  "hiring_signal": "mixed",
  "confidence": "medium",
  "summary": "",
  "criteria_scores": [
    {
      "criterion": "",
      "weight": 0,
      "score": 0,
      "weighted_score": 0,
      "reason": "",
      "evidence_from_answer": [""],
      "missing_evidence": [""],
      "improvement_advice": ""
    }
  ],
  "strengths": [""],
  "weaknesses": [""],
  "red_flags": [""],
  "follow_up_questions": [""],
  "candidate_coaching": {
    "better_answer_strategy": "",
    "example_improvement": ""
  },
  "fairness_check": {
    "used_only_job_relevant_evidence": true,
    "ignored_protected_characteristics": true,
    "notes": ""
  },
  "legacy_compatibility": {
    "overall_score": 0,
    "category_scores": {},
    "strengths": [""],
    "weaknesses": [""],
    "missing_details": [""],
    "improved_answer": "",
    "next_advice": ""
  }
}
""".strip()

        return f"""
{self._profile_block(profile)}

Role:
You are a structured interview evaluator for software engineering interviews.

Task:
Evaluate one candidate answer using only job-relevant evidence from the answer.

Scientific Evaluation Methods To Apply:
- Structured Employment Interview principles: standardized criteria and constrained judgment.
- Behaviorally Anchored Rating Scales (Smith & Kendall, 1963): use anchors for 1-5 ratings.
- For past-behavior responses, treat STAR as a completeness checklist only.

Inputs:
CV context:
{format_list(cv_context)}

Job description context:
{format_list(job_description_context)}

Interview question:
{question}

Expected strong-answer points:
{format_list(expected_good_answer_points)}

Candidate answer:
{student_answer}

Evaluation mode:
- mode: {profile.evaluation_mode}
- scale: {profile.scale}
- evidence required: {profile.evidence_required}

Rubric criteria:
{self._format_rubric(profile.rubric)}

Rating anchors:
{self._format_rating_anchors(profile.rating_anchors)}

Fairness rules:
{self._format_fairness_rules(profile.fairness_rules)}

Evidence rules:
- Score only what is present in the answer.
- Do not infer missing skills from the resume alone.
- Do not reward resume claims unless the answer supports them.
- Do not penalize non-native English grammar/accent unless meaning is unclear.
- Do not evaluate protected characteristics.
- If evidence is missing, list it in missing_evidence.
- Every criterion must cite evidence_from_answer or state insufficient evidence.

Scoring rules:
- Use rating anchors consistently.
- Compute weighted overall score from rubric weights and criterion scores.
- Keep weighted math internally consistent.
- overall_score must align with configured scale ({profile.scale}).
- hiring_signal must be one of: strong_positive, positive, mixed, weak, negative.
- confidence must be one of: low, medium, high.
- In coaching mode: practical, actionable advice.
- In strict_hiring mode: conservative, evidence-only judgments.
- In mock_interview mode: supportive but honest.
- In technical_accuracy mode: emphasize correctness and depth.

Output JSON schema:
{self._safe_json_instruction(evaluation_schema)}
"""

    def generate_question(
        self,
        cv_context: List[str],
        job_description_context: List[str],
        interview_type: str,
        difficulty: str,
        profile: InterviewAgentProfile | None = None,
    ) -> str:
        runtime_profile = profile or self.profile
        prompt = self._build_question_prompt(
            cv_context=cv_context,
            job_description_context=job_description_context,
            interview_type=interview_type,
            difficulty=difficulty,
            profile=runtime_profile,
        )
        return call_llm_api(prompt=prompt)

    def evaluate_answer(
        self,
        cv_context: List[str],
        job_description_context: List[str],
        question: str,
        expected_good_answer_points: List[str],
        student_answer: str,
        profile: InterviewAgentProfile | None = None,
    ) -> str:
        runtime_profile = profile or self.profile
        prompt = self._build_evaluation_prompt(
            cv_context=cv_context,
            job_description_context=job_description_context,
            question=question,
            expected_good_answer_points=expected_good_answer_points,
            student_answer=student_answer,
            profile=runtime_profile,
        )
        return call_llm_api(prompt=prompt)

    def workflow_steps(self) -> list[str]:
        """Return runtime workflow steps used by this single agent."""
        return [
            "Load request JSON",
            "Validate request schema",
            "Build context from structured resume and job description",
            "Load interview/evaluation config and construct one InterviewAgent profile",
            "Generate one constrained prompt for the selected task",
            "Call Ollama /api/generate once",
            "Parse returned JSON text",
            "Write output JSON file",
        ]


interview_agent = InterviewAgent(profile=InterviewAgentProfile())
