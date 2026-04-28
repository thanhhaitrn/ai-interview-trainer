from __future__ import annotations

import json
from pathlib import Path

import streamlit as st


DEFAULT_JSON_PATH = Path("data/llm/resume1_llm.json")


def load_resume(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def render_candidate(candidate: dict) -> None:
    st.subheader("Candidate")
    st.write(f"**Full name:** {candidate.get('full_name') or '-'}")
    st.write(f"**Headline:** {candidate.get('headline') or '-'}")


def render_section(section: dict) -> None:
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
    st.set_page_config(page_title="Resume Viewer", layout="wide")
    st.title("Resume JSON Viewer")

    path_value = st.sidebar.text_input("Resume JSON path", str(DEFAULT_JSON_PATH))
    json_path = Path(path_value)

    if not json_path.exists():
        st.error(f"File not found: {json_path}")
        return

    data = load_resume(json_path)

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
