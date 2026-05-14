import ollama

MODEL_NAME = "qwen3-embedding:0.6b"

RESUME_RETRIEVAL_INSTRUCTION = (
    "Given a job requirement or recruiter search query, retrieve resume chunks that" \
    "show relevant cnadidate skills, experience, projects, education, and achievements."
)

def format_query_with_instruction(query: str, instruction: str = RESUME_RETRIEVAL_INSTRUCTION) -> str:
    return f"Instruct: {instruction}\n Query:{query}" 

def embed_documents(texts: list[str]) -> list[list[float]]:
    """Use this for chunks stored in resume_chunks/job_chunks"""
    response = ollama.embed(
        model=MODEL_NAME,
        input=texts,
    )
    return response["embeddings"]

def embed_query(query: str) -> list[float]:
    """Use this when searching resume chunks with a JD/query. Query gets instruction"""
    instructed_query = format_query_with_instruction(query)

    response = ollama.embed(
        model=MODEL_NAME,
        input=instructed_query,
    )
    return response["embeddings"][0]