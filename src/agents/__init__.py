# src/agents/__init__.py
# Agents package.
# Agents contain reasoning and decision logic — tools do not.
# Hierarchy:
#   orchestrator.py — top-level router, session manager, profile extractor
#   reasoning.py    — ReAct loop, calls retrieval and document tools
#   review.py       — five checks: grounding, alignment, tone, advice, citations
#   governor.py     — PII detection, escalation, compliance stamp, audit logging
# Only the agent that owns a tool should call it.