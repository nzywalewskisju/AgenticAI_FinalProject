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

import json
import os
from datetime import datetime
from src.tools.utils import call_llm, safe_json_parse


PROFILES_DIR = "./data/profiles"


def _get_profile_path(user_id: str) -> str:
    return f"{PROFILES_DIR}/{user_id}.json"


def load_profile(user_id: str) -> dict:
    """
    Loads the user's profile from disk.
    Returns empty profile dict if none exists yet.
    """
    path = _get_profile_path(user_id)
    if not os.path.exists(path):
        return {"user_id": user_id, "facts": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_profile(user_id: str, profile: dict) -> None:
    """
    Saves the user's profile to disk.
    """
    os.makedirs(PROFILES_DIR, exist_ok=True)
    path = _get_profile_path(user_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)


def extract_and_update_profile(user_id: str, message: str) -> list[str]:
    """
    Automatically extracts personal facts from the user's message
    and merges them into the persisted profile.
    Returns a list of newly extracted fact labels so the GUI can notify the user.
    Called by the orchestrator on every incoming message.
    """
    system_prompt = """You are a fact extractor for an HR assistant.
Extract any personal employment facts the user mentions about themselves.
Only extract facts about the user themselves — not about other employees.
Look for: job title/role, years of employment, employment type (full-time/part-time/contractor),
department, location, any ongoing HR situations they mention (e.g. on leave, in a PIP).
Do not extract sensitive medical details.
If no facts are found, return an empty dict.
Respond only in JSON: {"facts": {"role": "...", "employment_duration": "...", ...}}
Use only the keys that are present. Do not invent keys."""

    response = call_llm(message, system_prompt=system_prompt)
    extracted = safe_json_parse(response, fallback={"facts": {}})
    new_facts = extracted.get("facts", {})

    if not new_facts:
        return []

    profile = load_profile(user_id)
    newly_added = []

    for key, value in new_facts.items():
        if value is None or str(value).strip().lower() in ("none", "", "null", "n/a"):
            continue
        if key not in profile["facts"] or profile["facts"][key] != value:
            profile["facts"][key] = value
            profile["last_updated"] = datetime.utcnow().isoformat()
            newly_added.append(f"{key.replace('_', ' ').title()}: {value}")

    if newly_added:
        save_profile(user_id, profile)

    return newly_added


def get_profile_context_string(user_id: str) -> str:
    """
    Returns the user's profile facts as a formatted string
    for injection into agent prompts.
    """
    profile = load_profile(user_id)
    facts = profile.get("facts", {})

    if not facts:
        return "No profile information known about this user yet."

    lines = ["Known facts about this user:"]
    for key, value in facts.items():
        label = key.replace("_", " ").title()
        lines.append(f"  - {label}: {value}")

    return "\n".join(lines)


def delete_profile_fact(user_id: str, fact_key: str) -> bool:
    """
    Removes a single fact from the user's profile.
    Called when user deletes a fact from the GUI profile panel.
    Returns True if deleted, False if key not found.
    """
    profile = load_profile(user_id)
    if fact_key in profile.get("facts", {}):
        del profile["facts"][fact_key]
        save_profile(user_id, profile)
        return True
    return False


def clear_profile(user_id: str) -> None:
    """
    Wipes all facts from the user's profile.
    """
    profile = load_profile(user_id)
    profile["facts"] = {}
    save_profile(user_id, profile)

def get_relevant_profile_context(user_id: str, query: str) -> str:
    """
    Returns only the profile facts relevant to the current query.
    Prevents facts from unrelated prior questions contaminating new answers.
    """
    profile = load_profile(user_id)
    facts = profile.get("facts", {})

    if not facts:
        return "No profile information known about this user yet."

    query_lower = query.lower()

    # Facts that are always relevant regardless of query topic
    always_relevant = ["role", "employment_type", "employment_duration", "department"]

    # Location only relevant for state-specific policy questions
    location_keywords = [
        "california", "state", "new york", "washington", "illinois",
        "georgia", "leave", "fmla", "parental", "sick", "pto",
        "family leave", "disability", "accommodation"
    ]
    location_relevant = any(k in query_lower for k in location_keywords)

    # Ongoing situations only relevant for related queries
    situation_keywords = [
        "leave", "pip", "complaint", "fmla", "accommodation",
        "performance", "investigation", "grievance"
    ]
    situation_relevant = any(k in query_lower for k in situation_keywords)

    relevant_facts = {}
    for key, value in facts.items():
        if key in always_relevant:
            relevant_facts[key] = value
        elif key == "location" and location_relevant:
            relevant_facts[key] = value
        elif key == "ongoing_situations" and situation_relevant:
            relevant_facts[key] = value

    if not relevant_facts:
        return "No relevant profile information for this query."

    lines = ["Known facts about this user:"]
    for key, value in relevant_facts.items():
        label = key.replace("_", " ").title()
        lines.append(f"  - {label}: {value}")

    return "\n".join(lines)