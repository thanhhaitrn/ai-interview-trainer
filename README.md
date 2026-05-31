# AI Interview Agent

## Overview
This project now keeps the active application code under `app/`:
- Parsing resume PDFs
- Normalizing parsed resume output
- Running the interview agent graph
- Viewing normalized resume JSON in Streamlit

## Requirements

- Python 3.10+
- `streamlit`
- `docling`

Install dependencies:
```bash
pip install -r requirements.txt
```

## Run

Parse a resume PDF:

```bash
python -m app.resume_system.parser data/resumes/raw/resume1.pdf --output-dir data/resumes/parsed
```

Normalize the parsed JSON:

```bash
python -m app.resume_system.resume_normalizer data/resumes/parsed/resume1_parsed.json --output-dir data/resumes/llm
```

Normalize from the app entrypoint:

```bash
python -m app.main normalize-resume data/resumes/parsed/resume1_parsed.json --output-dir data/resumes/llm
```

Or parse a PDF first, then normalize the parsed output:

```bash
python -m app.main prepare-resume data/resumes/raw/resume1.pdf
```

Show the interview graph workflow:

```bash
python -m app.main show-workflow
```

View the normalized JSON:

```bash
streamlit run view_parsed_resume.py
```


## Interview Question Generation and Evaluation
Local interview-agent project that generates interview questions and evaluates
candidate answers with one unified AI agent profile.

Input style:
- Preferred: structured `resume` + structured `job_description`
- Backward compatible: flat `cv_context` + `job_description_context`
- Optional advanced config:
  - `interview_config` for rubric-driven structured question generation
  - `evaluation_config` for weighted criteria, anchors, fairness, and evidence rules

## Structure

- `app/resume_system/`: parse PDFs and normalize Docling JSON into the app resume schema.
- `app/agent/`: prompt builders, LLM client, profile, and structured outputs.
- `app/graph/`: LangGraph interview workflow, schemas, state, and nodes.
- `app/main.py`: CLI for resume preparation, interactive interviews, and workflow inspection.
- `data/resumes/llm/`: LLM-ready resume JSON files used at runtime.
- `data/jobs/`: job description `.txt` files used at runtime.
- `tests/fixtures/`: structured request payloads used only by tests.

## Workflow Utilities

Print the workflow:

```bash
python -m app.main show-workflow
```

Save Mermaid text to a file:

```bash
python -m app.main show-workflow --output-path workflow.mmd
```
