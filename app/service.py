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


def to_dict(request: Any) -> dict[str, Any]:
    """Convert a Pydantic model into a plain dictionary."""
    # Pydantic v2 exposes `model_dump`, so prefer it when available.
    if hasattr(request, "model_dump"):
        return request.model_dump()

    # Fall back to the older Pydantic v1 API if needed.
    return request.dict()


def generate_question(request: QuestionRequest | dict[str, Any]) -> Any:
    """Run the question-generation graph and return decoded JSON output."""
    # Validate raw dictionaries before they enter the graph.
    validated_request = QuestionRequest.model_validate(request)
    # Convert the validated request into the graph state format.
    state = to_dict(validated_request)
    # Run the dedicated question-generation graph from start to finish.
    result = question_graph.invoke(state)
    # Decode the graph's raw model output before returning it.
    return parse_json_response(result["raw_llm_output"])


def evaluate_answer(request: EvaluationRequest | dict[str, Any]) -> Any:
    """Run the answer-evaluation graph and return decoded JSON output."""
    # Validate raw dictionaries before they enter the graph.
    validated_request = EvaluationRequest.model_validate(request)
    # Convert the validated request into the graph state format.
    state = to_dict(validated_request)
    # Run the dedicated evaluation graph from start to finish.
    result = evaluation_graph.invoke(state)
    # Decode the graph's raw model output before returning it.
    return parse_json_response(result["raw_llm_output"])
