# src/tools/utils.py
# Shared utility functions used across agents and tools.
# Functions:
#   call_llm(prompt, system_prompt, temperature=0)
#     — single entry point for all Ollama/Llama calls in the system
#     — temperature is 0 by default on every call — never change this default
#     — all LLM calls in the system go through here, never call Ollama directly
#   get_current_date()
#     — returns today's date for reasoning about policy effective dates
#   format_chunks_for_prompt(chunks)
#     — formats retrieved chunks into a clean string for injection into prompts
#   format_chunks_for_citation(chunks)
#     — formats chunks into citation references for the final answer
#   truncate_text(text, max_tokens)
#     — truncates text to stay within context window limits
#   clean_llm_json_response(response)
#     — strips markdown code fences and whitespace from LLM JSON responses
#     — use before every json.loads() call on an LLM response

import json
import re
import requests
from datetime import date
from config import LLM_MODEL, OLLAMA_BASE_URL


def call_llm(prompt: str, system_prompt: str = "", temperature: float = 0) -> str:
    """
    Single entry point for all Ollama/Llama calls in the system.
    Temperature is 0 by default on every call — never change this default.
    All LLM calls in the system go through here, never call Ollama directly.
    """
    payload = {
        "model": LLM_MODEL,
        "messages": [],
        "stream": False,
        "options": {"temperature": temperature}
    }

    if system_prompt:
        payload["messages"].append({"role": "system", "content": system_prompt})

    payload["messages"].append({"role": "user", "content": prompt})

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def get_current_date() -> str:
    """
    Returns today's date as a readable string.
    Used by the Reasoning Agent to reason about policy effective dates.
    """
    return date.today().strftime("%B %d, %Y")


def format_chunks_for_prompt(chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into a clean string for injection into prompts.
    Each chunk is labeled with its source document and section header.
    """
    if not chunks:
        return "No relevant policy sections were found."

    formatted = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("document_name", "Unknown Document")
        section = chunk.get("metadata", {}).get("section_header", "Unknown Section")
        text = chunk.get("text", "")
        formatted.append(f"[{i}] Source: {source} — {section}\n{text}")

    return "\n\n".join(formatted)


def format_chunks_for_citation(chunks: list[dict]) -> list[dict]:
    """
    Formats chunks into citation references for the final answer.
    Returns a list of dicts with document_name, section_header, and chunk_index.
    """
    citations = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        citations.append({
            "document_name": metadata.get("document_name", "Unknown Document"),
            "section_header": metadata.get("section_header", "Unknown Section"),
            "chunk_index": metadata.get("chunk_index", 0)
        })
    return citations


def truncate_text(text: str, max_chars: int = 3000) -> str:
    """
    Truncates text to stay within context window limits.
    Truncates at the last complete sentence within the limit where possible.
    """
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.8:
        return truncated[:last_period + 1] + " [truncated]"
    return truncated + " [truncated]"


def clean_llm_json_response(response: str) -> str:
    """
    Strips markdown code fences and whitespace from LLM JSON responses.
    Use before every json.loads() call on an LLM response.
    Handles ```json ... ``` and ``` ... ``` fences.
    """
    response = response.strip()
    response = re.sub(r"^```(?:json)?\s*", "", response)
    response = re.sub(r"\s*```$", "", response)
    return response.strip()


def safe_json_parse(response: str, fallback: dict = None) -> dict:
    """
    Cleans and parses a JSON response from the LLM.
    Returns fallback dict on failure instead of raising an exception.
    Always use this instead of bare json.loads() on LLM output.
    """
    if fallback is None:
        fallback = {}
    try:
        cleaned = clean_llm_json_response(response)
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return fallback