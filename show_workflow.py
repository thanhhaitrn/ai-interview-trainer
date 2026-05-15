from __future__ import annotations

import argparse
from pathlib import Path

from app.interview_agent import interview_agent


def build_generated_mermaid() -> str:
    """Generate Mermaid text from the single-agent workflow steps."""
    steps = interview_agent.workflow_steps()
    lines = ["flowchart TD"]

    for index, step in enumerate(steps):
        lines.append(f'  N{index}["{step}"]')
        if index > 0:
            lines.append(f"  N{index - 1} --> N{index}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show Mermaid workflow for the single interview agent."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path where Mermaid text should be written.",
    )
    args = parser.parse_args()

    mermaid_output = build_generated_mermaid()

    if args.output_path is not None:
        args.output_path.write_text(mermaid_output, encoding="utf-8")
        print(f"Saved Mermaid workflow to {args.output_path}")
        return

    print(mermaid_output)


if __name__ == "__main__":
    main()
