"""Pydantic request models that validate interview workflow payloads."""

from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel, Field, model_validator


class ResumeSource(BaseModel):
    """Optional metadata describing where the resume content came from."""

    file_name: str | None = None
    page_count: int | None = None


class ResumeCandidate(BaseModel):
    """High-level candidate profile fields from the normalized resume."""

    full_name: str | None = None
    headline: str | None = None


class ResumeSection(BaseModel):
    """A normalized section from the resume output."""

    section_name: str
    items: List[dict[str, Any]] = Field(default_factory=list)


class ResumeDocument(BaseModel):
    """Structured resume payload compatible with `resume_normalizer.py` output."""

    document_type: str = "resume"
    source: ResumeSource | None = None
    candidate: ResumeCandidate | None = None
    sections: List[ResumeSection] = Field(default_factory=list)


class JobDescriptionDocument(BaseModel):
    """Structured job description payload for interview generation/evaluation."""

    role_title: str | None = None
    company: str | None = None
    location: str | None = None
    seniority: str | None = None
    employment_type: str | None = None
    summary: str | None = None
    responsibilities: List[str] = Field(default_factory=list)
    requirements: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    tech_stack: List[str] = Field(default_factory=list)


class Competency(BaseModel):
    """Target competency used for structured interview coverage and scoring."""

    name: str
    description: str | None = None
    weight: float | int | None = None


class QuestionConstraints(BaseModel):
    """Generation rules that enforce structured-interview quality controls."""

    avoid_yes_no_questions: bool = True
    avoid_trivia: bool = True
    avoid_leading_questions: bool = True
    avoid_multi_part_questions: bool = False
    require_followups: bool = True
    require_expected_signals: bool = True
    require_red_flags: bool = True
    require_reason_for_asking: bool = True
    require_resume_grounding: bool = True
    require_job_alignment: bool = True


class FairnessRules(BaseModel):
    """Bias-reduction and fairness controls for generation/evaluation."""

    job_related_only: bool = True
    ignore_protected_characteristics: bool = True
    do_not_penalize_non_native_english: bool = True
    score_only_observed_evidence: bool = True
    avoid_school_prestige_bias: bool = True
    do_not_infer_missing_information: bool = True


class InterviewConfig(BaseModel):
    """Configurable structured-interview design settings."""

    interview_stage: str = "technical_screen"
    seniority_level: str = "junior"
    difficulty_level: str = "medium"
    question_count: int | None = None
    question_techniques: List[str] = Field(default_factory=list)
    competencies: List[Competency] = Field(default_factory=list)
    question_constraints: QuestionConstraints = Field(default_factory=QuestionConstraints)
    fairness_rules: FairnessRules = Field(default_factory=FairnessRules)

    @model_validator(mode="after")
    def validate_question_count(self) -> "InterviewConfig":
        if self.question_count is not None and self.question_count <= 0:
            raise ValueError("interview_config.question_count must be greater than 0.")
        return self


class RubricCriterion(BaseModel):
    """Weighted evaluation criterion used for evidence-based interview scoring."""

    name: str
    weight: float | int
    description: str | None = None


class Rubric(BaseModel):
    """Rubric definition with weighted criteria and optional rating anchors."""

    criteria: List[RubricCriterion] = Field(default_factory=list)
    rating_anchors: dict[str, str] | None = None


class EvaluationConfig(BaseModel):
    """Configurable answer-evaluation settings and rubric controls."""

    evaluation_mode: str = "coaching"
    scale: str = "1-5"
    evidence_required: bool = True
    rubric: Rubric | None = None
    fairness_rules: FairnessRules = Field(default_factory=FairnessRules)


class QuestionRequest(BaseModel):
    """Input payload for question generation (structured-first, legacy-compatible)."""

    # Structured inputs (preferred).
    resume: ResumeDocument | None = None
    job_description: JobDescriptionDocument | None = None

    # Legacy flat fields (kept for backward compatibility).
    cv_context: List[str] = Field(default_factory=list)
    job_description_context: List[str] = Field(default_factory=list)

    # Legacy high-level fields.
    interview_type: str = "technical"
    difficulty: str = "medium"

    # New optional config object.
    interview_config: InterviewConfig | None = None

    @model_validator(mode="after")
    def validate_context_sources(self) -> "QuestionRequest":
        if not self.resume and not self.cv_context:
            raise ValueError(
                "Provide either `resume` (structured) or `cv_context` (legacy)."
            )
        if not self.job_description and not self.job_description_context:
            raise ValueError(
                "Provide either `job_description` (structured) or "
                "`job_description_context` (legacy)."
            )
        return self


class EvaluationRequest(BaseModel):
    """Input payload for answer evaluation (structured-first, legacy-compatible)."""

    # Structured inputs (preferred).
    resume: ResumeDocument | None = None
    job_description: JobDescriptionDocument | None = None

    # Legacy flat fields (kept for backward compatibility).
    cv_context: List[str] = Field(default_factory=list)
    job_description_context: List[str] = Field(default_factory=list)

    question: str
    expected_good_answer_points: List[str] = Field(default_factory=list)
    student_answer: str | None = None
    candidate_answer: str | None = None

    # New optional config object.
    evaluation_config: EvaluationConfig | None = None

    @model_validator(mode="after")
    def validate_context_sources_and_answer(self) -> "EvaluationRequest":
        if not self.resume and not self.cv_context:
            raise ValueError(
                "Provide either `resume` (structured) or `cv_context` (legacy)."
            )
        if not self.job_description and not self.job_description_context:
            raise ValueError(
                "Provide either `job_description` (structured) or "
                "`job_description_context` (legacy)."
            )

        if not self.student_answer and self.candidate_answer:
            self.student_answer = self.candidate_answer

        if not self.student_answer:
            raise ValueError("Provide `student_answer` or `candidate_answer`.")

        return self
