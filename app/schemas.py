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


class QuestionRequest(BaseModel):
    """Input payload for question generation (structured-first, legacy-compatible)."""

    # Structured inputs (preferred).
    resume: ResumeDocument | None = None
    job_description: JobDescriptionDocument | None = None

    # Legacy flat fields (kept for backward compatibility).
    cv_context: List[str] = Field(default_factory=list)
    job_description_context: List[str] = Field(default_factory=list)

    interview_type: str
    difficulty: str

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
    expected_good_answer_points: List[str]
    student_answer: str

    @model_validator(mode="after")
    def validate_context_sources(self) -> "EvaluationRequest":
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
