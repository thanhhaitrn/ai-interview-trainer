"""Prompt-building helpers for the two LLM tasks in this service."""

def format_list(items):
    """Render a Python list as bullet points inside a prompt string."""
    # When no context exists, make that explicit so the model knows it.
    if not items:
        return "No context provided."

    # Prefix each item with `-` so the prompt reads like a clean bullet list.
    return "\n".join([f"- {item}" for item in items])


def build_question_prompt(
    cv_context,
    job_description_context,
    interview_type,
    difficulty
):
    """Build the exact instruction block for question generation."""
    # Keep the full prompt inline so it is easy to inspect and edit later.
    return f"""
You are an AI interview question generator.

Your job is to generate ONE interview question for a student.

Use the given CV context and job description context.

CV context:
{format_list(cv_context)}

Job description context:
{format_list(job_description_context)}

Interview type:
{interview_type}

Difficulty:
{difficulty}

Requirements:
- The question must match the job description.
- The question should be answerable based on the candidate's CV.
- The question should fit the interview type.
- The question should fit the difficulty level.
- Do not ask multiple questions.
- Return only valid JSON.
- Do not include markdown.
- Do not include explanation outside the JSON.

Return JSON in this exact structure:

{{
  "question": "",
  "type": "",
  "difficulty": "",
  "focus_area": "",
  "why_this_question": "",
  "expected_good_answer_points": []
}}
"""


def build_evaluation_prompt(
    cv_context,
    job_description_context,
    question,
    expected_good_answer_points,
    student_answer
):
    """Build the exact instruction block for answer evaluation."""
    # Keep the full prompt inline so it is easy to inspect and edit later.
    return f"""
You are an AI interview answer evaluator.

Your job is to evaluate a student's written interview answer.

Use:
1. The interview question
2. The student's answer
3. The candidate's CV context
4. The job description context
5. The expected good answer points

CV context:
{format_list(cv_context)}

Job description context:
{format_list(job_description_context)}

Interview question:
{question}

Expected good answer points:
{format_list(expected_good_answer_points)}

Student answer:
{student_answer}

Evaluate the answer based on:
- relevance
- clarity
- specificity
- job alignment
- CV alignment
- structure

Scoring rule:
- Each category score should be from 0 to 10.
- The overall score should be from 0 to 10.
- Be strict but helpful.
- Give practical feedback.
- Return only valid JSON.
- Do not include markdown.
- Do not include explanation outside the JSON.

Return JSON in this exact structure:

{{
  "overall_score": 0,
  "category_scores": {{
    "relevance": 0,
    "clarity": 0,
    "specificity": 0,
    "job_alignment": 0,
    "cv_alignment": 0,
    "structure": 0
  }},
  "strengths": [],
  "weaknesses": [],
  "missing_details": [],
  "improved_answer": "",
  "next_advice": ""
}}
"""
