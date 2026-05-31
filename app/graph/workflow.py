"""LangGraph workflow assembly for interactive interviews."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.graph.nodes import (
    ask_question_node,
    decide_next_node,
    evaluate_answer_node,
    final_report_node,
    generate_plan_node,
    load_documents_node,
    route_after_decision,
    route_after_plan,
    start_interview_node,
)
from app.graph.state import GraphState
from app.graph.schemas import QuestionRequest


WORKFLOW_STEPS = [
    "Start interview",
    "Load full JD + full resume",
    "Generate interview plan",
    "Ask question",
    "Candidate answers",
    "Evaluate answer with compact question context + rubric",
    "Decide follow-up or next question",
    "Repeat until interview is complete",
    "Final report",
]

DEFAULT_CHECKPOINTER = InMemorySaver()


def workflow_steps() -> list[str]:
    return list(WORKFLOW_STEPS)


def create_initial_state(
    request_payload: dict[str, Any] | QuestionRequest,
    *,
    debug_trace: bool = False,
) -> GraphState:
    if hasattr(request_payload, "model_dump"):
        payload = request_payload.model_dump()
    else:
        payload = dict(request_payload)

    max_followups_value = payload.get("max_followups_per_question")
    max_followups = int(max_followups_value if max_followups_value is not None else 1)

    return {
        "request_payload": payload,
        "debug_trace": debug_trace,
        "status": "pending",
        "current_question_index": 0,
        "current_followup_count": 0,
        "max_followups_per_question": max(0, max_followups),
        "pending_followup_question": None,
        "turn_summaries": [],
        "turns": [],
        "evaluations": [],
        "trace": [],
        "errors": [],
        "node_outputs": {},
    }


def build_interview_workflow(checkpointer: Any | None = None):
    graph = StateGraph(GraphState)
    graph.add_node("start_interview", start_interview_node)
    graph.add_node("load_documents", load_documents_node)
    graph.add_node("generate_plan", generate_plan_node)
    graph.add_node("ask_question", ask_question_node)
    graph.add_node("evaluate_answer", evaluate_answer_node)
    graph.add_node("decide_next", decide_next_node)
    graph.add_node("final_report", final_report_node)

    graph.add_edge(START, "start_interview")
    graph.add_edge("start_interview", "load_documents")
    graph.add_edge("load_documents", "generate_plan")
    graph.add_conditional_edges(
        "generate_plan",
        route_after_plan,
        {
            "ask_question": "ask_question",
            "final_report": "final_report",
        },
    )
    graph.add_edge("ask_question", "evaluate_answer")
    graph.add_edge("evaluate_answer", "decide_next")
    graph.add_conditional_edges(
        "decide_next",
        route_after_decision,
        {
            "ask_question": "ask_question",
            "final_report": "final_report",
        },
    )
    graph.add_edge("final_report", END)
    return graph.compile(checkpointer=checkpointer or DEFAULT_CHECKPOINTER)


def start_interview(
    request_payload: dict[str, Any] | QuestionRequest,
    *,
    thread_id: str | None = None,
    debug_trace: bool = False,
    checkpointer: Any | None = None,
) -> dict[str, Any]:
    resolved_thread_id = thread_id or str(uuid4())
    workflow = build_interview_workflow(checkpointer=checkpointer)
    result = workflow.invoke(
        create_initial_state(
            request_payload,
            debug_trace=debug_trace,
        ),
        config={"configurable": {"thread_id": resolved_thread_id}},
    )

    if isinstance(result, dict):
        result.setdefault("thread_id", resolved_thread_id)

    return result


def resume_interview(
    *,
    thread_id: str,
    answer: str | dict[str, Any],
    checkpointer: Any | None = None,
) -> dict[str, Any]:
    workflow = build_interview_workflow(checkpointer=checkpointer)
    resume_payload = answer if isinstance(answer, dict) else {"answer": answer}
    result = workflow.invoke(
        Command(resume=resume_payload),
        config={"configurable": {"thread_id": thread_id}},
    )

    if isinstance(result, dict):
        result.setdefault("thread_id", thread_id)

    return result

if __name__ == "__main__":
    agentic_graph = build_interview_workflow()
    print(agentic_graph.get_graph().draw_mermaid())
