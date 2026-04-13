# orchestrator.py
# the top-level agent — receives every user query and decides what happens next
# responsibilities:
#   - call classify_query to determine if the query is HR-related and in scope
#   - call session memory to attach prior conversation context
#   - call Governance Sub-Agent (pre-check) to screen the query before reasoning
#   - route valid queries to the Reasoning Sub-Agent
#   - receive the reviewed answer and return it to the user
#   - reject out-of-scope queries with a clear message before anything else runs
# does NOT do retrieval, reasoning, or review — only classification and routing