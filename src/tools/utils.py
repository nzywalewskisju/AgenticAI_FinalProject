# utils.py
# Shared utility functions used across agents and tools.
# call_llm routes inference to either Ollama (local) or OpenAI GPT-4o mini
# depending on the active provider setting in config. All LLM calls in
# the system go through here — never call Ollama or OpenAI directly.
#
# Functions: call_llm, _call_ollama, _call_openai, get_current_date,
#            format_chunks_for_prompt, format_chunks_for_citation,
#            truncate_text, clean_llm_json_response, safe_json_parse

import config
import json
import re
import requests
from datetime import date
from config import LLM_MODEL, OLLAMA_BASE_URL


def call_llm(prompt: str, system_prompt: str = "", temperature: float = 0) -> str:
    # Single entry point for all LLM calls. Routes to Ollama or OpenAI
    # based on the active provider setting in config.

    import config

    if config.ACTIVE_LLM_PROVIDER == "openai":
        return _call_openai(prompt, system_prompt, temperature)
    else:
        return _call_ollama(prompt, system_prompt, temperature)


def _call_ollama(prompt: str, system_prompt: str = "", temperature: float = 0) -> str:
    # Sends a prompt to the local Llama model via the Ollama REST API
    # and returns the response text.

    payload = {
        "model": config.LLM_MODEL,
        "messages": [],
        "stream": False,
        "options": {"temperature": temperature}
    }

    if system_prompt:
        payload["messages"].append({"role": "system", "content": system_prompt})
    payload["messages"].append({"role": "user", "content": prompt})

    response = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/chat",
        json=payload
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def _call_openai(prompt: str, system_prompt: str = "", temperature: float = 0) -> str:
    # Sends a prompt to GPT-4o mini via the OpenAI API and returns
    # the response text.

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package is not installed. "
            "Run: pip install openai"
        )

    import config
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message.content


def get_current_date() -> str:
    # Returns today's date as a readable string. Used by the reasoning
    # agent to reason about policy effective dates and contribution limits.

    return date.today().strftime("%B %d, %Y")


def format_chunks_for_prompt(chunks: list[dict]) -> str:
    # Formats a list of retrieved chunks into a numbered string with
    # source and section labels for injection into agent prompts.

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
    # Formats chunks into a list of citation dicts containing document
    # name, section header, and chunk index for the review agent.

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
    # Truncates text to stay within context window limits. Tries to
    # break at the last complete sentence within the limit.

    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.8:
        return truncated[:last_period + 1] + " [truncated]"
    return truncated + " [truncated]"


def clean_llm_json_response(response: str) -> str:
    # Strips markdown code fences and whitespace from LLM responses.
    # Must be called before every json.loads on LLM output.

    response = response.strip()
    response = re.sub(r"^```(?:json)?\s*", "", response)
    response = re.sub(r"\s*```$", "", response)
    return response.strip()


def safe_json_parse(response: str, fallback: dict = None) -> dict:
    # Cleans and parses a JSON response from the LLM. Returns the
    # fallback dict on failure instead of raising an exception.
    
    if fallback is None:
        fallback = {}
    try:
        cleaned = clean_llm_json_response(response)
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return fallback