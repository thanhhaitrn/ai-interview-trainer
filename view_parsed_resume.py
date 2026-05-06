"""Streamlit app for inspecting normalized resume JSON files."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st


LLM_DIR = Path("data/llm")


def list_resume_files(directory: Path = LLM_DIR) -> list[Path]:
    """Return normalized resume JSON files available for viewing."""
    if not directory.exists():
        return []

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".json"
    )


def load_resume(path: Path) -> dict:
    """Load one normalized resume JSON document from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def render_candidate(candidate: dict) -> None:
    """Render the top-level candidate summary fields."""
    st.subheader("Candidate")
    st.write(f"**Full name:** {candidate.get('full_name') or '-'}")
    st.write(f"**Headline:** {candidate.get('headline') or '-'}")


def render_section(section: dict) -> None:
    """Render one resume section and its extracted items."""
    section_name = section.get("section_name", "Untitled Section")
    items = section.get("items", [])

    with st.expander(section_name, expanded=True):
        if not items:
            st.caption("No items")
            return

        for index, item in enumerate(items, start=1):
            st.markdown(f"**Item {index}**")
            st.json(item)


def main() -> None:
    """Launch the Streamlit resume viewer UI."""
    st.set_page_config(page_title="Resume Viewer", layout="wide")
    st.title("Resume JSON Viewer")

    resume_files = list_resume_files()
    if not resume_files:
        st.error(f"No JSON files found in {LLM_DIR}")
        return

    selected_file = st.sidebar.selectbox(
        "Resume JSON file",
        resume_files,
        format_func=lambda path: path.name,
    )

    data = load_resume(selected_file)

    source = data.get("source", {})
    candidate = data.get("candidate", {})
    sections = data.get("sections", [])

    st.caption(f"Source file: {source.get('file_name', '-')}")
    st.caption(f"Page count: {source.get('page_count', '-')}")

    left_col, right_col = st.columns([3, 2])

    with left_col:
        render_candidate(candidate)
        st.subheader("Sections")
        for section in sections:
            render_section(section)

    with right_col:
        st.subheader("Raw JSON")
        st.json(data)


if __name__ == "__main__":
    main()
