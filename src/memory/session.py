# src/memory/session.py
# Short-term conversation memory — resets when the session ends.
# Stores the last N turns of dialogue (default 10) so the Reasoning Agent
# can understand follow-up questions without the user repeating context.
# Example: user asks "what about part-time employees?" — session memory provides
#   the prior question so the agent knows what "what about" refers to.
# Keyed by session_id. In-memory only — not persisted to disk.
# SessionMemory class + global session_memory instance used by the orchestrator.

from collections import defaultdict


MAX_TURNS = 10


class SessionMemory:
    """
    In-memory conversation history scoped by session_id.
    Each session stores up to MAX_TURNS turns (one turn = one user + one assistant message).
    Resets when the application restarts.
    """

    def __init__(self):
        self._sessions: dict[str, list[dict]] = defaultdict(list)

    def add_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        """
        Adds a completed conversation turn to the session history.
        Automatically trims to MAX_TURNS.
        """
        self._sessions[session_id].append({
            "role": "user",
            "content": user_message
        })
        self._sessions[session_id].append({
            "role": "assistant",
            "content": assistant_message
        })

        # Keep only the last MAX_TURNS * 2 messages (user + assistant pairs)
        max_messages = MAX_TURNS * 2
        if len(self._sessions[session_id]) > max_messages:
            self._sessions[session_id] = self._sessions[session_id][-max_messages:]

    def get_history(self, session_id: str) -> list[dict]:
        """
        Returns the full message history for a session.
        """
        return self._sessions.get(session_id, [])

    def get_context_string(self, session_id: str) -> str:
        """
        Returns the conversation history as a formatted string
        for injection into agent prompts.
        """
        history = self.get_history(session_id)
        if not history:
            return "No prior conversation in this session."

        lines = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")

        return "\n".join(lines)

    def clear_session(self, session_id: str) -> None:
        """
        Clears all memory for a session.
        Called when user clicks 'Clear Session' in the GUI.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions and len(self._sessions[session_id]) > 0


# Global instance used by the orchestrator
session_memory = SessionMemory()