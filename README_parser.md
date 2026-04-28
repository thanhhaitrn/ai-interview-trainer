# Resume Parsing Module

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
