# utils.py
# shared utility functions used across multiple agents and tools
# functions defined here:
#   - get_current_date(): returns today's date for policy effective date reasoning
#   - format_chunks_for_prompt(chunks): formats retrieved chunks into readable prompt text
#   - truncate_text(text, max_length): safely shortens text that exceeds context windows
# add any small helper function here that does not belong to a specific tool category