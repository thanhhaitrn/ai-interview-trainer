"""LangGraph workflows that power question generation and answer evaluation."""

from typing import List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from app.prompts import build_question_prompt, build_evaluation_prompt
from app.llm_api_client import call_llm_api


class InterviewState(TypedDict, total=False):
    """Shared state object that flows through every graph node."""
    # common inputs for both workflows
    cv_context: List[str]
    job_description_context: List[str]

    # matter only when generating a new interview question
    interview_type: Optional[str]
    difficulty: Optional[str]

    # matter only when evaluating an answer
    question: Optional[str]
    expected_good_answer_points: Optional[List[str]]
    student_answer: Optional[str]

    # Every agent writes its final raw model output into this field.
    raw_llm_output: str


def question_agent_node(state: InterviewState) -> InterviewState:
    """Build the question prompt, call the model, and store the raw output."""
    # Build the prompt from the incoming graph state.
    prompt = build_question_prompt(
        cv_context=state["cv_context"],
        job_description_context=state["job_description_context"],
        interview_type=state.get("interview_type", "general"),
        difficulty=state.get("difficulty", "medium")
    )

    # Ask the configured provider to generate a question payload.
    response_text = call_llm_api(
        prompt=prompt,
        task="question"
    )

    # Return only the values this node is responsible for updating.
    return {
        "raw_llm_output": response_text
    }


def evaluation_agent_node(state: InterviewState) -> InterviewState:
    """Build the evaluation prompt, call the model, and store the raw output."""
    # Build the prompt from the incoming graph state.
    prompt = build_evaluation_prompt(
        cv_context=state["cv_context"],
        job_description_context=state["job_description_context"],
        question=state["question"],
        expected_good_answer_points=state["expected_good_answer_points"],
        student_answer=state["student_answer"]
    )

    # Ask the configured provider to evaluate the candidate answer.
    response_text = call_llm_api(
        prompt=prompt,
        task="evaluation"
    )

    # Return only the values this node is responsible for updating.
    return {
        "raw_llm_output": response_text
    }

def build_question_graph():
    """Create the simplest graph: start, question node, end."""
    # Tell LangGraph what keys may exist in the shared workflow state.
    graph = StateGraph(InterviewState)

    # Register the node function that performs question generation.
    graph.add_node("question_agent", question_agent_node)

    # Wire the graph so it always executes this one node.
    graph.add_edge(START, "question_agent")
    graph.add_edge("question_agent", END)

    return graph.compile()


def build_evaluation_graph():
    """Create the simplest graph: start, evaluation node, end."""
    graph = StateGraph(InterviewState)

    graph.add_node("evaluation_agent", evaluation_agent_node)

    graph.add_edge(START, "evaluation_agent")
    graph.add_edge("evaluation_agent", END)

    return graph.compile()


# Build these graphs once at import time so every request can reuse them.
question_graph = build_question_graph()
evaluation_graph = build_evaluation_graph()
