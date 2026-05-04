from __future__ import annotations

import argparse
from pathlib import Path

from langchain_core.runnables.graph import MermaidDrawMethod

from app.agent_graph import evaluation_graph, question_graph


def workflow_graphs():
    return {
        "question_graph": question_graph,
        "evaluation_graph": evaluation_graph,
    }


def build_generated_mermaid() -> str:
    sections = []

    for graph_name, compiled_graph in workflow_graphs().items():
        sections.append(f"%% {graph_name}")
        sections.append(compiled_graph.get_graph().draw_mermaid())

    return "\n\n".join(sections)


def draw_graph_png(compiled_graph, draw_method: str) -> bytes:
    method = MermaidDrawMethod.API
    if draw_method == "pyppeteer":
        method = MermaidDrawMethod.PYPPETEER

    return compiled_graph.get_graph().draw_mermaid_png(draw_method=method)


def save_workflow_pngs(output_dir: Path, draw_method: str = "api") -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    for graph_name, compiled_graph in workflow_graphs().items():
        output_path = output_dir / f"{graph_name}.png"
        output_path.write_bytes(draw_graph_png(compiled_graph, draw_method=draw_method))
        output_paths.append(output_path)

    return output_paths


def display_workflow_pngs_in_notebook(draw_method: str = "api") -> None:
    from IPython.display import Image, display

    for compiled_graph in workflow_graphs().values():
        display(Image(draw_graph_png(compiled_graph, draw_method=draw_method)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Mermaid workflows directly from LangGraph code."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path where Mermaid text should be written.",
    )
    parser.add_argument(
        "--png-output-dir",
        type=Path,
        default=None,
        help="Optional folder where workflow PNG files should be written.",
    )
    parser.add_argument(
        "--png-draw-method",
        choices=["api", "pyppeteer"],
        default="api",
        help=(
            "Rendering backend for PNG output. Use `pyppeteer` for local rendering "
            "if mermaid.ink is blocked."
        ),
    )
    args = parser.parse_args()

    if args.png_output_dir is not None:
        try:
            output_paths = save_workflow_pngs(
                args.png_output_dir,
                draw_method=args.png_draw_method,
            )
        except ImportError as exc:
            if args.png_draw_method == "pyppeteer":
                raise SystemExit(
                    "Pyppeteer is not installed. Install it with "
                    "`pip install pyppeteer`, then rerun the command."
                ) from exc
            raise
        except ValueError as exc:
            error_text = str(exc)
            if "mermaid.ink" in error_text:
                raise SystemExit(
                    "Could not reach mermaid.ink. Re-run with "
                    "`--png-draw-method pyppeteer` for local rendering."
                ) from exc
            raise

        for output_path in output_paths:
            print(f"Saved workflow PNG to {output_path}")
        return

    mermaid_output = build_generated_mermaid()

    if args.output_path is not None:
        args.output_path.write_text(mermaid_output, encoding="utf-8")
        print(f"Saved Mermaid workflow to {args.output_path}")
        return

    print(mermaid_output)


if __name__ == "__main__":
    main()
