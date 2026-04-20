# src/memory/profile.py
# Medium-term user profile memory — persists across sessions.
# Stores personal facts the user has stated about themselves, automatically
# extracted by the orchestrator without requiring explicit "remember this" commands.
# Examples of stored facts:
#   employment_duration, role, employment_type, department, ongoing_situations
# These facts are injected into the Reasoning Agent's context so the agent
# can apply policy to the user's specific situation without re-asking every session.
# Users can view and delete individual facts from the GUI profile panel.
# Persisted to data/profiles/{user_id}.json
# Does NOT store another employee's information or sensitive medical details.