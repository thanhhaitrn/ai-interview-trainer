AI Interview Agent

Local interview-agent project that generates interview questions and evaluates
candidate answers with LangGraph-managed agent flows.

## Setup

1. Create a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and update values:
   - `OLLAMA_BASE_URL=http://localhost:11434`
   - `OLLAMA_MODEL=<your_model_name>`
   - `OLLAMA_API_KEY=REPLACE_WITH_YOUR_OLLAMA_API_KEY`

`OLLAMA_API_KEY` is optional for local Ollama, but required for cloud/gateway
setups that enforce bearer authentication.

4. Make sure Ollama is running and that the model exists:
   - `ollama serve`
   - `ollama pull <your_model_name>`

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

Save PNG workflow diagrams generated directly by `draw_mermaid_png()`:

```bash
python show_workflow.py --png-output-dir data/workflow
```

If `mermaid.ink` is blocked in your environment, use local rendering:

```bash
pip install pyppeteer
python show_workflow.py --png-output-dir data/workflow --png-draw-method pyppeteer
```

Notebook visualization (same style you requested):

```python
from IPython.display import Image, display
from app.agent_graph import question_graph

display(Image(question_graph.get_graph().draw_mermaid_png()))
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

## Notes

- The project is Ollama-only now (no mock provider mode).
- The scripts read `.env` by default for provider configuration.
- Generated files are written as formatted JSON for easy inspection.

## Real API Usage

1. Choose your Ollama mode:
- Local Ollama (no API key needed):
  - `OLLAMA_BASE_URL=http://localhost:11434`
  - `OLLAMA_API_KEY=` (leave empty)
- Ollama cloud or secured gateway:
  - Create a key at `https://ollama.com/settings/keys`
  - Put it in `.env` as `OLLAMA_API_KEY=<your_key>`
  - Set `OLLAMA_BASE_URL=https://ollama.com` (or your gateway URL)

2. Confirm Ollama connectivity:

```bash
curl "$OLLAMA_BASE_URL/api/tags"
```

3. Generate a question with your real model:

```bash
python generate_question.py
```

4. Evaluate an answer with your real model:

```bash
python evaluate_answer.py
```

5. Optional direct cloud check with bearer header:

```bash
curl https://ollama.com/api/tags -H "Authorization: Bearer $OLLAMA_API_KEY"
```
