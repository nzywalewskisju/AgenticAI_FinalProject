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