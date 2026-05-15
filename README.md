# AI Interview Agent

## Resume Parser
This README covers only the following part of the project: 
- Parsing resume PDFs
- Normalizing the parsed output
- Viewing the result in Streamlit.

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
python src/resume_system/parser.py data/raw/resume1.pdf --output-dir data/parsed
```

Normalize the parsed JSON:

```bash
python src/resume_system/resume_normalizer.py data/parsed/resume1_parsed.json --output-dir data/llm
```

View the normalized JSON:

```bash
streamlit run app.py
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

- `generate_question.py`: read a question request JSON file, generate one
  question, and save the result JSON.
- `evaluate_answer.py`: read an evaluation request JSON file, evaluate one
  answer, and save the result JSON.
- `show_workflow.py`: print the project workflow in the terminal.
- `data/requests/`: sample input JSON files for both scripts.
- `data/questions/`: default output folder for generated questions.
- `data/evaluations/`: default output folder for answer evaluations.

## Run Like A Simple Script Project

Print the workflow:

```bash
python show_workflow.py
```

Save Mermaid text to a file:

```bash
python show_workflow.py --output-path workflow.mmd
```

Generate a question from a JSON request file:

```bash
python generate_question.py
```

That uses the default input file:

`data/requests/question_request.json`

The default request file now uses structured input:
- `resume` (same shape style as normalized resume output)
- `job_description` (role metadata, responsibilities, requirements, skills)
- `interview_config` (stage, techniques, competencies, constraints, fairness)

Evaluate an answer from a JSON request file:

```bash
python evaluate_answer.py
```

That uses the default input file:

`data/requests/evaluation_request.json`

This request also uses structured `resume` + `job_description`.
It also includes `evaluation_config` with weighted criteria and rating anchors.

You can also pass a custom input path and output folder:

```bash
python generate_question.py data/requests/question_request.json --output-dir data/questions
python evaluate_answer.py data/requests/evaluation_request.json --output-dir data/evaluations
```

Legacy examples remain available:
- `data/requests/question_request_legacy.json`
- `data/requests/evaluation_request_legacy.json`
