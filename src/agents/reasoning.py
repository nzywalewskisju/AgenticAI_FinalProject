# reasoning.py
# the Reasoning Sub-Agent — this is where the ReAct loop runs
# receives: user query + session context + cleared status from Governance pre-check
# ReAct loop:
#   Thought: what do i need to find to answer this?
#   Action: call a tool (retrieve_chunks, keyword_search, check_document_exists, etc.)
#   Observation: evaluate what came back
#   repeat until confident or until max iterations reached
# output: draft answer + list of chunks used as sources
# if confidence is too low after max iterations: return "no reliable information found"
# does NOT do final validation — that is review.py's job