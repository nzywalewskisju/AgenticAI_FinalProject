# src/memory/__init__.py
# Memory package — three distinct layers of memory, each with different persistence.
#   session.py  — conversation turns, in-memory only, resets each session
#   profile.py  — user facts extracted from conversation, persists to disk per user
#   registry.py — document registry, persists to disk per user
# All memory is scoped by user_id — no data bleeds between users.