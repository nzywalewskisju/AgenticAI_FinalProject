# src/tools/__init__.py
# Tools package.
# Raw functions only — no reasoning or decision logic lives here.
# Tools are owned by specific agents and should only be called by their owner:
#   retrieval.py  → Reasoning Sub-Agent
#   document.py   → Reasoning Sub-Agent
#   governance.py → Governor Sub-Agent
#   utils.py      → shared helpers, callable by any agent