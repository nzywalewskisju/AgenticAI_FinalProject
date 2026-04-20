# src/agents/review.py
# Review Sub-Agent — quality gate before the final answer reaches the user.
# Runs five checks in order. All five must pass.
#   1. verify_grounding
#      — every factual claim in the answer traces to a retrieved chunk
#      — grounding score must be >= 0.7, otherwise answer is rejected
#   2. check_policy_alignment
#      — the answer accurately reflects what the policy says
#      — catches cases where retrieval was correct but paraphrasing was not
#      — e.g. "may be eligible" cannot become "are entitled to"
#   3. check_tone
#      — answer uses appropriate sensitivity for HR topics
#      — catches dismissive or overly casual language on sensitive subjects
#   4. check_advice_applicability
#      — the agent actually applied policy to the user's specific situation
#      — rejects answers that only summarize policy without situational reasoning
#      — this is the check that enforces agent behaviour over chatbot behaviour
#   5. inject_citations
#      — attaches source document name and section to every factual claim
# If any check fails: reject → back to Reasoning Sub-Agent (max 2 retries total)
# If all pass: forward to Governor post-check