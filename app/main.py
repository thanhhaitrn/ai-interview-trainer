"""App CLI for resume preparation, workflow inspection, and interviews."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from app.agent import llm_client
from app.agent.outputs import clean_empty_fields
from app.graph.workflow import workflow_steps
from app.graph.workflow import resume_interview, start_interview
from app.resume_system.parser import PARSED_DIR, RAW_DIR, parse_pdf_to_json
from app.resume_system.resume_normalizer import LLM_DIR, save_llm_resume


DATA_DIR = Path("data")
JOBS_DIR = DATA_DIR / "jobs"
INTERVIEW_RUNS_DIR = DATA_DIR / "interview_runs"
VIDEO_DIR = DATA_DIR / "video"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
ANSWER_EVALS_DIR = DATA_DIR / "answer_evaluations"


def build_mermaid_workflow() -> str:
    """Generate Mermaid text from the interview graph workflow steps."""
    lines = ["flowchart TD"]

    for index, step in enumerate(workflow_steps()):
        lines.append(f'  N{index}["{step}"]')
        if index > 0:
            lines.append(f"  N{index - 1} --> N{index}")

    return "\n".join(lines)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    try:
        json.dumps(value)
    except TypeError:
        return str(value)

    return value


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    payload = clean_empty_fields(_jsonable(payload))
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _list_files(folder: Path, suffixes: set[str]) -> list[Path]:
    if not folder.exists():
        return []

    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _select_path(
    *,
    label: str,
    provided_path: Path | None,
    folder: Path,
    suffixes: set[str],
) -> Path:
    if provided_path is not None:
        if not provided_path.exists():
            raise SystemExit(f"{label} file does not exist: {provided_path}")
        if provided_path.suffix.lower() not in suffixes:
            supported = ", ".join(sorted(suffixes))
            raise SystemExit(f"{label} file must use one of: {supported}")
        return provided_path

    files = _list_files(folder, suffixes)
    if not files:
        supported = ", ".join(sorted(suffixes))
        raise SystemExit(f"No {label} files found in {folder} ({supported}).")

    print(f"\nSelect {label}:")
    for index, path in enumerate(files, start=1):
        print(f"  {index}. {path}")

    if not sys.stdin.isatty():
        print(f"Using default {label}: {files[0]}")
        return files[0]

    while True:
        raw_choice = input(f"Choose number [1-{len(files)}] (default 1): ").strip()
        if not raw_choice:
            return files[0]

        if raw_choice.isdigit():
            choice = int(raw_choice)
            if 1 <= choice <= len(files):
                return files[choice - 1]

        print("Invalid choice. Please enter one of the listed numbers.")


def _load_job_payload(path: Path) -> dict[str, Any]:
    return {"job_description_context": [path.read_text(encoding="utf-8").strip()]}


def _build_interview_request(
    *,
    resume_path: Path,
    job_path: Path,
    interview_type: str,
    difficulty: str | None,
    question_count: int | None,
    max_followups_per_question: int,
) -> dict[str, Any]:
    interview_config: dict[str, Any] = {}
    if difficulty is not None:
        interview_config["difficulty_level"] = difficulty

    if question_count is not None:
        interview_config["question_count"] = question_count

    payload: dict[str, Any] = {
        "resume": _read_json(resume_path),
        "interview_type": interview_type,
        "interview_config": interview_config,
        "max_followups_per_question": max(0, max_followups_per_question),
    }
    if difficulty is not None:
        payload["difficulty"] = difficulty

    payload.update(_load_job_payload(job_path))
    return payload


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return clean_empty_fields(value.model_dump(exclude_none=True))

    if isinstance(value, dict):
        return clean_empty_fields(dict(value))

    return {}


def _extract_interrupt_payload(state: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = state.get("__interrupt__")
    if not interrupts:
        return None

    if isinstance(interrupts, (list, tuple)):
        interrupt = interrupts[0]
    else:
        interrupt = interrupts

    value = getattr(interrupt, "value", interrupt)
    if isinstance(value, dict):
        return value

    return {"value": value}


def _question_text(question: dict[str, Any]) -> str:
    return str(question.get("question") or "").strip()


def _print_interview_plan(state: dict[str, Any]) -> None:
    questions = state.get("planned_questions") or []
    print(f"\n[PLAN] Generated {len(questions)} planned question(s).")

    for question in questions:
        question_id = question.get("id") or "q?"
        competency = question.get("competency") or "general"
        text = _question_text(question)
        print(f"- {question_id} [{competency}]: {text}")


def _format_score(score: Any) -> str:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return str(score or "n/a")

    if value.is_integer():
        return str(int(value))

    return f"{value:.1f}"


def _feedback_text(evaluation: dict[str, Any]) -> str:
    if evaluation.get("summary"):
        return str(evaluation["summary"])

    coaching = evaluation.get("candidate_coaching")
    if isinstance(coaching, dict) and coaching.get("better_answer_strategy"):
        return str(coaching["better_answer_strategy"])

    weaknesses = evaluation.get("weaknesses") or []
    strengths = evaluation.get("strengths") or []
    parts = []
    if strengths:
        parts.append(f"Strengths: {', '.join(map(str, strengths[:2]))}")
    if weaknesses:
        parts.append(f"Needs work: {', '.join(map(str, weaknesses[:2]))}")

    return "; ".join(parts) if parts else "No detailed feedback returned."


def _decision_action(state: dict[str, Any], decision: dict[str, Any]) -> str:
    routed_action = state.get("next_node")
    if routed_action:
        return str(routed_action)

    turn = (state.get("turns") or [{}])[-1]
    if isinstance(turn, dict) and turn.get("routed_action"):
        return str(turn["routed_action"])

    return str(decision.get("action") or "final_report")


def _decision_label(action: str) -> str:
    labels = {
        "follow_up": "Ask follow-up",
        "next_question": "Ask next question",
        "final_report": "Create final report",
    }
    return labels.get(action, action)


def _latest_decision_reason(state: dict[str, Any], decision: dict[str, Any]) -> str:
    if decision.get("reason"):
        return str(decision["reason"])

    for key in ("turn_summaries", "turns"):
        turns = state.get(key) or []
        if not turns:
            continue

        latest_turn = turns[-1]
        if not isinstance(latest_turn, dict):
            continue

        if latest_turn.get("decision_reason"):
            return str(latest_turn["decision_reason"])

        turn_decision = latest_turn.get("decision")
        if isinstance(turn_decision, dict) and turn_decision.get("reason"):
            return str(turn_decision["reason"])

    return "No reason returned."


def _print_latest_turn_result(state: dict[str, Any]) -> None:
    evaluation = _model_dump(state.get("latest_evaluation"))
    if evaluation:
        print(
            "\n[EVALUATION] "
            f"Score: {_format_score(evaluation.get('overall_score'))}/5"
        )
        print(f"[FEEDBACK] {_feedback_text(evaluation)}")

    decision = _model_dump(state.get("latest_decision"))
    if decision or state.get("next_node"):
        action = _decision_action(state, decision)
        print("\n[STATE] Current stage: decide_next_step")
        print(f"[DECISION] {_decision_label(action)}")
        print(f"[REASON] {_latest_decision_reason(state, decision)}")


def _token_summary(llm_calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "prompt_tokens": sum(
            int(
                call.get("prompt_tokens")
                or call.get("prompt_tokens_estimate")
                or 0
            )
            for call in llm_calls
        ),
        "completion_tokens": sum(
            int(
                call.get("completion_tokens")
                or call.get("completion_tokens_estimate")
                or 0
            )
            for call in llm_calls
        ),
        "total_tokens": sum(
            int(call.get("total_tokens") or call.get("total_tokens_estimate") or 0)
            for call in llm_calls
        ),
        "token_source": (
            "provider_usage_metadata"
            if any(
                call.get("token_source") == "provider_usage_metadata"
                for call in llm_calls
            )
            else "estimated_chars_div_4"
        ),
    }


def _render_report_markdown(report_payload: dict[str, Any]) -> str:
    metadata = report_payload["metadata"]
    report = report_payload.get("final_report") or {}
    turns = report_payload.get("turn_summaries") or report_payload.get("turns") or []
    token_summary = metadata["tokens"]

    lines = [
        "# Interview Report",
        "",
        f"- Status: {metadata['status']}",
        f"- Resume: {metadata['resume_path']}",
        f"- Job description: {metadata['job_path']}",
        f"- Runtime seconds: {metadata['runtime_seconds']}",
        f"- Tokens: {token_summary['total_tokens']}",
        f"- Token source: {token_summary['token_source']}",
        "",
    ]

    if report:
        lines.extend(
            [
                "## Final Summary",
                "",
                f"- Recommendation: {report.get('overall_recommendation', '')}",
                f"- Summary: {report.get('summary', '')}",
                "",
            ]
        )

        for title, key in (
            ("Strengths", "strengths"),
            ("Risks", "risks"),
            ("Evidence Highlights", "evidence_highlights"),
            ("Suggested Next Steps", "suggested_next_steps"),
        ):
            values = report.get(key) or []
            if values:
                lines.extend([f"## {title}", ""])
                lines.extend(f"- {value}" for value in values)
                lines.append("")

    if turns:
        lines.extend(["## Turns", ""])
        for index, turn in enumerate(turns, start=1):
            question = turn.get("question") or {}
            evaluation = turn.get("evaluation") or {}
            decision = turn.get("decision") or {}
            question_text = (
                turn.get("question_text")
                or _question_text(question)
            )
            score = (
                turn.get("overall_score")
                if turn.get("overall_score") is not None
                else evaluation.get("overall_score")
            )
            feedback = (
                turn.get("evaluation_summary")
                or _feedback_text(evaluation)
            )
            action = (
                turn.get("decision_action")
                or turn.get("routed_action")
                or decision.get("action")
                or ""
            )
            reason = turn.get("decision_reason") or decision.get("reason") or ""
            lines.extend(
                [
                    f"### Turn {index}",
                    "",
                    f"- Question: {question_text}",
                    f"- Answer: {turn.get('answer', '')}",
                    f"- Score: {_format_score(score)}/5",
                    f"- Feedback: {feedback}",
                    f"- Decision: {_decision_label(action)}",
                    f"- Reason: {reason}",
                    "",
                ]
            )

    return "\n".join(lines).strip() + "\n"


def _save_interview_outputs(
    *,
    output_dir: Path,
    thread_id: str,
    started_at: datetime,
    runtime_seconds: float,
    resume_path: Path,
    job_path: Path,
    state: dict[str, Any],
    status: str,
    error: str | None = None,
) -> tuple[Path, Path, Path]:
    llm_calls = llm_client.get_call_trace()
    run_name = f"{started_at.strftime('%Y%m%d_%H%M%S')}_{thread_id[:8]}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "thread_id": thread_id,
        "started_at": started_at.isoformat(),
        "ended_at": datetime.now().astimezone().isoformat(),
        "runtime_seconds": round(runtime_seconds, 4),
        "status": status,
        "resume_path": str(resume_path),
        "job_path": str(job_path),
        "tokens": _token_summary(llm_calls),
    }
    if error:
        metadata["error"] = error

    trace_payload = {
        "metadata": metadata,
        "workflow_trace": state.get("trace", []),
        "llm_calls": llm_calls,
        "planned_questions": state.get("planned_questions", []),
        "turn_summaries": state.get("turn_summaries", []),
        "turns": state.get("turns", []),
        "final_report": _model_dump(state.get("final_report")),
        "state_status": state.get("status"),
    }
    report_payload = {
        "metadata": metadata,
        "final_report": _model_dump(state.get("final_report")),
        "turn_summaries": state.get("turn_summaries", []),
    }

    trace_path = run_dir / "trace.json"
    report_json_path = run_dir / "report.json"
    report_md_path = run_dir / "report.md"
    _write_json(trace_path, trace_payload)
    _write_json(report_json_path, report_payload)
    report_md_path.write_text(
        _render_report_markdown(_jsonable(report_payload)),
        encoding="utf-8",
    )
    return trace_path, report_json_path, report_md_path


def run_interview_cli(args: argparse.Namespace) -> None:
    resume_path = _select_path(
        label="resume",
        provided_path=args.resume_path,
        folder=LLM_DIR,
        suffixes={".json"},
    )
    job_path = _select_path(
        label="job description",
        provided_path=args.job_path,
        folder=JOBS_DIR,
        suffixes={".txt"},
    )

    request_payload = _build_interview_request(
        resume_path=resume_path,
        job_path=job_path,
        interview_type=args.interview_type,
        difficulty=args.difficulty,
        question_count=args.question_count,
        max_followups_per_question=args.max_followups,
    )

    thread_id = args.thread_id or str(uuid4())
    started_at = datetime.now().astimezone()
    started = time.perf_counter()
    state: dict[str, Any] = {}
    status = "completed"
    error_message: str | None = None

    llm_client.reset_call_trace()

    try:
        print("\n[STATE] Current stage: generate_plan")
        state = start_interview(
            request_payload,
            thread_id=thread_id,
            debug_trace=True,
        )
        _print_interview_plan(state)

        while True:
            interrupt_payload = _extract_interrupt_payload(state)
            if state.get("status") == "completed" or state.get("final_report"):
                break

            if not interrupt_payload:
                raise RuntimeError(
                    "Workflow stopped without a question or final report."
                )

            question = (
                interrupt_payload.get("question")
                or state.get("current_question")
                or {}
            )
            print("\n[STATE] Current stage: ask_question")
            print(f"\n[QUESTION] {_question_text(question)}")

            answer = ""
            while not answer:
                answer = input("\n[USER ANSWER] ").strip()
                if answer.lower() in {"exit", "quit", ":q"}:
                    status = "stopped_by_user"
                    break
                if not answer:
                    print("Please enter an answer, or type 'exit' to stop.")

            if status == "stopped_by_user":
                break

            print("\n[STATE] Current stage: evaluate_answer")
            state = resume_interview(thread_id=thread_id, answer=answer)
            _print_latest_turn_result(state)

        if state.get("final_report"):
            final_report = _model_dump(state["final_report"])
            print("\n[STATE] Current stage: final_report")
            print(
                "[REPORT] Recommendation: "
                f"{final_report.get('overall_recommendation', '')}"
            )
            print(f"[REPORT] Summary: {final_report.get('summary', '')}")
    except KeyboardInterrupt:
        status = "stopped_by_user"
        print("\n[STATE] Interview stopped by user.")
    except Exception as exc:
        status = "failed"
        error_message = str(exc)
        print(f"\n[ERROR] {error_message}")
    finally:
        runtime_seconds = time.perf_counter() - started
        trace_path, report_json_path, report_md_path = _save_interview_outputs(
            output_dir=args.output_dir,
            thread_id=thread_id,
            started_at=started_at,
            runtime_seconds=runtime_seconds,
            resume_path=resume_path,
            job_path=job_path,
            state=state,
            status=status,
            error=error_message,
        )
        print(f"\n[TRACE] Saved trace log to {trace_path}")
        print(f"[REPORT] Saved JSON report to {report_json_path}")
        print(f"[REPORT] Saved Markdown report to {report_md_path}")

    if status == "failed":
        raise SystemExit(1)


def _analyze_speech(result: Any, video_path: Path) -> dict[str, Any]:
    """Run fluency (from transcript) and voice (from audio) analysis.

    Fluency is pure-Python and always runs; voice analysis decodes the audio
    and uses Praat, so it is best-effort and never fails the transcription.
    """
    from app.speech_analysis import analyze_fluency, analyze_voice, decode_audio_mono

    analysis: dict[str, Any] = {}

    fluency = analyze_fluency(result.all_words())
    analysis["fluency"] = fluency.to_dict()
    print(
        "[FLUENCY] "
        f"rate {fluency.speech_rate_wpm} wpm | "
        f"pauses {fluency.pause_count} ({fluency.long_pause_count} long) | "
        f"fillers {fluency.filler_count} | "
        f"repeats {fluency.repetition_count} | "
        f"MLR {fluency.mean_length_of_run} words"
    )
    for warning in fluency.warnings:
        print(f"[FLUENCY] note: {warning}")

    try:
        samples, sample_rate = decode_audio_mono(str(video_path))
        voice = analyze_voice(samples, sample_rate)
        analysis["voice"] = voice.to_dict()
        print(
            "[VOICE] "
            f"steadiness {voice.tremor_label} | "
            f"F0 {voice.f0_mean_hz} Hz | "
            f"jitter {voice.jitter_local_pct}% | "
            f"shimmer {voice.shimmer_local_pct}%"
        )
        for indicator in voice.tremor_indicators:
            print(f"[VOICE] note: {indicator}")
    except Exception as exc:  # noqa: BLE001 - voice analysis is best-effort
        analysis["voice"] = {"error": str(exc)}
        print(f"[VOICE] Skipped voice analysis: {exc}")

    return analysis


def run_transcribe_cli(args: argparse.Namespace) -> None:
    from app.speech_analysis import has_audio_stream
    from app.transcription import VideoTranscriber

    video_path = _select_path(
        label="video",
        provided_path=args.input_path,
        folder=VIDEO_DIR,
        suffixes={".mp4", ".mov", ".avi", ".mkv"},
    )

    if not has_audio_stream(str(video_path)):
        raise SystemExit(
            f"No audio track found in {video_path}; nothing to transcribe."
        )

    print(f"\n[TRANSCRIBE] Loading model '{args.model}' and transcribing {video_path} ...")
    transcriber = VideoTranscriber(model_size=args.model)
    result = transcriber.transcribe(str(video_path), language=args.language)

    payload = result.to_dict()
    if not args.skip_analysis:
        payload.update(_analyze_speech(result, video_path))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    text_path = args.output_dir / f"{stem}.txt"
    json_path = args.output_dir / f"{stem}.json"

    text_path.write_text(result.text + "\n", encoding="utf-8")
    _write_json(json_path, payload)

    print(f"[TRANSCRIBE] Language: {result.language} | Duration: {result.duration_sec}s")
    print(f"[TRANSCRIBE] Transcript length: {len(result.text)} characters")
    print(f"[TRANSCRIBE] Saved text transcript to {text_path}")
    print(f"[TRANSCRIBE] Saved JSON transcript to {json_path}")

    if not result.text:
        print(
            "[TRANSCRIBE] Warning: empty transcript "
            "(the video may have no speech audio)."
        )


def _video_presentation_metrics(source_path: str | None) -> dict[str, Any] | None:
    """Run face/presentation analysis on the source video, best-effort."""
    if not source_path or not Path(source_path).exists():
        print("[EVALUATE] --with-video set but source video was not found; skipping face metrics.")
        return None

    try:
        from app.video_analysis import VideoAnalyzer

        result = VideoAnalyzer().analyze(video_path=source_path)
        return result.to_dict().get("video_presentation")
    except Exception as exc:  # noqa: BLE001 - face metrics are optional
        print(f"[EVALUATE] Skipped face metrics: {exc}")
        return None


def run_evaluate_answer_cli(args: argparse.Namespace) -> None:
    from app.agent import llm_client
    from app.agent.agent import InterviewAgent
    from app.agent.profile import build_evaluation_profile
    from app.graph.schemas import EvaluationRequest
    from app.speech_analysis import build_delivery_metrics

    if not args.transcript.exists():
        raise SystemExit(f"Transcript file does not exist: {args.transcript}")

    transcript = _read_json(args.transcript)
    student_answer = str(transcript.get("text") or "").strip()
    if not student_answer:
        raise SystemExit(
            f"Transcript {args.transcript} has no text to evaluate."
        )

    if args.question_file is not None:
        if not args.question_file.exists():
            raise SystemExit(f"Question file does not exist: {args.question_file}")
        question = args.question_file.read_text(encoding="utf-8").strip()
    else:
        question = (args.question or "").strip()
    if not question:
        raise SystemExit("Provide --question or --question-file.")

    resume_path = _select_path(
        label="resume",
        provided_path=args.resume_path,
        folder=LLM_DIR,
        suffixes={".json"},
    )
    job_path = _select_path(
        label="job description",
        provided_path=args.job_path,
        folder=JOBS_DIR,
        suffixes={".txt"},
    )

    video_metrics = (
        _video_presentation_metrics(transcript.get("source_path"))
        if args.with_video
        else None
    )
    delivery_metrics = build_delivery_metrics(transcript, video_metrics)

    payload: dict[str, Any] = {
        "resume": _read_json(resume_path),
        "question": question,
        "student_answer": student_answer,
        "expected_good_answer_points": list(args.expected or []),
        "delivery_metrics": delivery_metrics or None,
    }
    payload.update(_load_job_payload(job_path))
    request = EvaluationRequest.model_validate(payload)

    profile = build_evaluation_profile(request)
    cv_context = request.resume if request.resume is not None else request.cv_context
    job_context = (
        request.job_description
        if request.job_description is not None
        else request.job_description_context
    )

    llm_client.reset_call_trace()
    print(f"\n[EVALUATE] Scoring answer from {args.transcript} ...")
    result = InterviewAgent(profile).evaluate_answer_structured(
        cv_context=cv_context,
        job_description_context=job_context,
        question=request.question,
        expected_good_answer_points=request.expected_good_answer_points,
        student_answer=request.student_answer,
        delivery_metrics=request.delivery_metrics,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.transcript.stem}_evaluation.json"
    _write_json(
        out_path,
        {
            "question": question,
            "transcript_path": str(args.transcript),
            "delivery_metrics": delivery_metrics,
            "evaluation": result,
            "llm_calls": llm_client.get_call_trace(),
        },
    )

    evaluation = _model_dump(result)
    print(
        "[EVALUATE] Overall score: "
        f"{_format_score(evaluation.get('overall_score'))}/5 "
        f"({evaluation.get('overall_rating', 'n/a')}) | "
        f"signal: {evaluation.get('hiring_signal', 'n/a')}"
    )
    delivery = evaluation.get("delivery_assessment") or {}
    if delivery:
        print(
            "[EVALUATE] Delivery: "
            f"fluency {delivery.get('fluency_rating', 'n/a')} | "
            f"voice {delivery.get('voice_steadiness', 'n/a')}"
        )
        if delivery.get("impact_on_communication"):
            print(f"[EVALUATE] {delivery['impact_on_communication']}")
    print(f"[EVALUATE] Saved evaluation to {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run app utilities.")
    subparsers = parser.add_subparsers(dest="command")

    interview_parser = subparsers.add_parser(
        "interview",
        help="Run an interactive interview from an LLM-ready resume and a JD.",
    )
    interview_parser.add_argument(
        "--resume-path",
        type=Path,
        default=None,
        help="Path to an LLM-ready resume JSON file.",
    )
    interview_parser.add_argument(
        "--job-path",
        type=Path,
        default=None,
        help="Path to a job description .txt file.",
    )
    interview_parser.add_argument(
        "--interview-type",
        default="technical",
        help="Interview type passed into the question-generation prompt.",
    )
    interview_parser.add_argument(
        "--difficulty",
        default=None,
        help="Optional difficulty level. If omitted, the LLM infers it.",
    )
    interview_parser.add_argument(
        "--question-count",
        type=int,
        default=None,
        help="Optional number of planned questions. Defaults to profile config.",
    )
    interview_parser.add_argument(
        "--max-followups",
        type=int,
        default=1,
        help="Maximum follow-up questions per planned question, capped by profile.",
    )
    interview_parser.add_argument(
        "--thread-id",
        default=None,
        help="Optional stable LangGraph thread id for this interview run.",
    )
    interview_parser.add_argument(
        "--output-dir",
        type=Path,
        default=INTERVIEW_RUNS_DIR,
        help="Folder where trace logs and reports will be saved.",
    )

    parse_parser = subparsers.add_parser(
        "parse-resume",
        help="Parse a resume PDF into Docling JSON.",
    )
    parse_parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=RAW_DIR / "resume1.pdf",
        help="Path to the resume PDF file.",
    )
    parse_parser.add_argument(
        "--output-dir",
        type=Path,
        default=PARSED_DIR,
        help="Folder where the parsed JSON file will be saved.",
    )

    normalize_parser = subparsers.add_parser(
        "normalize-resume",
        help="Normalize Docling resume JSON into the app resume schema.",
    )
    normalize_parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=PARSED_DIR / "resume1_parsed.json",
        help="Path to a raw Docling JSON file.",
    )
    normalize_parser.add_argument(
        "--output-dir",
        type=Path,
        default=LLM_DIR,
        help="Folder where the normalized JSON file will be saved.",
    )

    prepare_parser = subparsers.add_parser(
        "prepare-resume",
        help="Parse a resume PDF, then normalize the parsed JSON.",
    )
    prepare_parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=RAW_DIR / "resume1.pdf",
        help="Path to the resume PDF file.",
    )
    prepare_parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=PARSED_DIR,
        help="Folder where the parsed JSON file will be saved.",
    )
    prepare_parser.add_argument(
        "--output-dir",
        type=Path,
        default=LLM_DIR,
        help="Folder where the normalized JSON file will be saved.",
    )

    workflow_parser = subparsers.add_parser(
        "show-workflow",
        help="Print the interview workflow as Mermaid text.",
    )
    workflow_parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path where Mermaid text should be written.",
    )

    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Transcribe a video file into text using local faster-whisper.",
    )
    transcribe_parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=None,
        help="Path to a video file. If omitted, choose one from data/video/.",
    )
    transcribe_parser.add_argument(
        "--language",
        default="en",
        help="Spoken language for transcription. Default: en.",
    )
    transcribe_parser.add_argument(
        "--model",
        default="small.en",
        help="faster-whisper model size (e.g. tiny.en, base.en, small.en, medium.en).",
    )
    transcribe_parser.add_argument(
        "--output-dir",
        type=Path,
        default=TRANSCRIPTS_DIR,
        help="Folder where the .txt and .json transcripts will be saved.",
    )
    transcribe_parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Only transcribe; skip fluency and voice-quality analysis.",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate-answer",
        help="Evaluate one answer from a transcript JSON, delivery-aware.",
    )
    evaluate_parser.add_argument(
        "--transcript",
        type=Path,
        required=True,
        help="Path to a transcript JSON produced by `transcribe`.",
    )
    evaluate_parser.add_argument(
        "--question",
        default=None,
        help="The interview question this answer responds to.",
    )
    evaluate_parser.add_argument(
        "--question-file",
        type=Path,
        default=None,
        help="Path to a text file holding the interview question.",
    )
    evaluate_parser.add_argument(
        "--resume-path",
        type=Path,
        default=None,
        help="LLM-ready resume JSON for grounding. Defaults to a file in data/resumes/llm/.",
    )
    evaluate_parser.add_argument(
        "--job-path",
        type=Path,
        default=None,
        help="Job description .txt for grounding. Defaults to a file in data/jobs/.",
    )
    evaluate_parser.add_argument(
        "--expected",
        action="append",
        default=None,
        help="Expected strong-answer signal (repeatable).",
    )
    evaluate_parser.add_argument(
        "--with-video",
        action="store_true",
        help="Also run face/presentation analysis on the source video.",
    )
    evaluate_parser.add_argument(
        "--output-dir",
        type=Path,
        default=ANSWER_EVALS_DIR,
        help="Folder where the evaluation JSON will be saved.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if not argv_list:
        argv_list = ["interview"]

    args = parser.parse_args(argv_list)

    if args.command == "interview":
        run_interview_cli(args)
        return

    if args.command == "parse-resume":
        if not args.input_path.exists():
            raise SystemExit(f"Input file does not exist: {args.input_path}")

        output_path = parse_pdf_to_json(args.input_path, args.output_dir)
        print(f"Saved parsed JSON to {output_path}")
        return

    if args.command == "normalize-resume":
        if not args.input_path.exists():
            raise SystemExit(f"Input file does not exist: {args.input_path}")

        output_path = save_llm_resume(args.input_path, args.output_dir)
        print(f"Saved LLM-ready resume JSON to {output_path}")
        return

    if args.command == "prepare-resume":
        if not args.input_path.exists():
            raise SystemExit(f"Input file does not exist: {args.input_path}")

        parsed_path = parse_pdf_to_json(args.input_path, args.parsed_dir)
        output_path = save_llm_resume(parsed_path, args.output_dir)
        print(f"Saved parsed JSON to {parsed_path}")
        print(f"Saved LLM-ready resume JSON to {output_path}")
        return

    if args.command == "transcribe":
        run_transcribe_cli(args)
        return

    if args.command == "evaluate-answer":
        run_evaluate_answer_cli(args)
        return

    if args.command == "show-workflow":
        mermaid_output = build_mermaid_workflow()
        if args.output_path is not None:
            args.output_path.write_text(mermaid_output, encoding="utf-8")
            print(f"Saved Mermaid workflow to {args.output_path}")
            return

        print(mermaid_output)
        return

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
