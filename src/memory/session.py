# src/memory/session.py
# Short-term conversation memory — resets when the session ends.
# Stores the last N turns of dialogue (default 10) so the Reasoning Agent
# can understand follow-up questions without the user repeating context.
# Example: user asks "what about part-time employees?" — session memory provides
#   the prior question so the agent knows what "what about" refers to.
# Keyed by session_id. In-memory only — not persisted to disk.
# SessionMemory class + global session_memory instance used by the orchestrator.