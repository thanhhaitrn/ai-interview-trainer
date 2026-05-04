"""Helpers for talking to Ollama-compatible APIs."""

import os
import requests
from dotenv import load_dotenv


# Load variables from `.env` into the current process before reading them.
load_dotenv()


def get_model_settings():
    """Read Ollama configuration from environment variables."""
    base_url = os.getenv("OLLAMA_BASE_URL")
    api_key = os.getenv("OLLAMA_API_KEY", "")
    model_name = os.getenv("OLLAMA_MODEL", "")

    if not base_url:
        raise ValueError("OLLAMA_BASE_URL is missing in .env")
    if not model_name:
        raise ValueError("OLLAMA_MODEL is missing in .env")

    return {
        "base_url": base_url,
        "api_key": api_key,
        "model_name": model_name,
    }


def call_ollama_api(prompt: str, task: str) -> str:
    """Send a prompt to an Ollama server using its `/api/generate` route."""
    # Pull the current settings each time so runtime changes are respected.
    settings = get_model_settings()

    # Ollama needs the exact model name to run.
    if not settings["model_name"]:
        raise ValueError("OLLAMA_MODEL is missing in .env")

    # Trim trailing slash so the final route does not contain `//`.
    base_url = settings["base_url"].rstrip("/")
    # Ollama generation requests are sent to `/api/generate`.
    url = f"{base_url}/api/generate"

    headers = {"Content-Type": "application/json"}
    # Local Ollama does not require auth, but cloud/gateway setups may.
    if settings["api_key"]:
        headers["Authorization"] = f"Bearer {settings['api_key']}"

    # Ask Ollama for a single non-streaming JSON-formatted response.
    payload = {
        "model": settings["model_name"],
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2
        }
    }

    # Make the network call and give the provider up to two minutes to answer.
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=120
    )

    # Raise an exception automatically for 4xx/5xx HTTP responses.
    response.raise_for_status()
    # Decode the provider response body from JSON into a Python object.
    data = response.json()

    # Ollama returns generated text inside `response`.
    if "response" not in data:
        raise ValueError("Ollama response is missing the 'response' field.")

    return data["response"]


def call_llm_api(prompt: str, task: str) -> str:
    """Call Ollama for both question generation and answer evaluation."""
    return call_ollama_api(prompt=prompt, task=task)
