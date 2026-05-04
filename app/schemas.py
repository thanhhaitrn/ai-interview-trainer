"""Pydantic request models that validate incoming API payloads."""

from pydantic import BaseModel
from typing import List


class QuestionRequest(BaseModel):
    """Input payload for the question-generation workflow."""
    # Bullet points or short phrases extracted from the candidate's CV.
    cv_context: List[str]
    # Bullet points or short phrases extracted from the target job description.
    job_description_context: List[str]
    # High-level interview style, such as technical or behavioral.
    interview_type: str
    # Difficulty target, such as easy, medium, or hard.
    difficulty: str


class EvaluationRequest(BaseModel):
    """Input payload for the answer-evaluation workflow."""
    # Bullet points or short phrases extracted from the candidate's CV.
    cv_context: List[str]
    # Bullet points or short phrases extracted from the target job description.
    job_description_context: List[str]
    # The interview question the candidate was asked.
    question: str
    # The rubric points the model should look for in a strong answer.
    expected_good_answer_points: List[str]
    # The candidate's actual written answer that needs feedback.
    student_answer: str
