AI Interview Agent

Local interview-agent project that generates interview questions and evaluates
candidate answers with LangGraph-managed agent flows.

## Setup

1. Create a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and update values:
   - `OLLAMA_MODEL=<your_model_name>`
   - `OLLAMA_API_KEY=REPLACE_WITH_YOUR_OLLAMA_API_KEY`

## Structure

- `generate_question.py`: read a question request JSON file, generate one
  question, and save the result JSON.
- `evaluate_answer.py`: read an evaluation request JSON file, evaluate one
  answer, and save the result JSON.
- `show_workflow.py`: print the project workflow in the terminal.
- `data/requests/`: sample input JSON files for both scripts.
- `data/questions/`: default output folder for generated questions.
- `data/evaluations/`: default output folder for answer evaluations.

Print the workflow:

```bash
python show_workflow.py
```

Save Mermaid text to a file:

```bash
python show_workflow.py --output-path workflow.mmd
```

Save PNG workflow diagrams generated directly by `draw_mermaid_png()`:

```bash
python show_workflow.py --png-output-dir data/workflow
```

If `mermaid.ink` is blocked in your environment, use local rendering:

```bash
pip install pyppeteer
python show_workflow.py --png-output-dir data/workflow --png-draw-method pyppeteer
```

Generate a question from a JSON request file:

```bash
python generate_question.py
```

That uses the default input file:

`data/requests/question_request.json`

Evaluate an answer from a JSON request file:

```bash
python evaluate_answer.py
```

That uses the default input file:

`data/requests/evaluation_request.json`

You can also pass a custom input path and output folder:

```bash
python generate_question.py data/requests/question_request.json --output-dir data/questions
python evaluate_answer.py data/requests/evaluation_request.json --output-dir data/evaluations
```
