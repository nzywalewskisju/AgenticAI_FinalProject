# review.py
# the Review Sub-Agent — validates the draft answer before it reaches the user
# receives: draft answer + chunks the Reasoning Agent used
# runs four checks in order:
#   1. verify_grounding: every claim traces back to a retrieved chunk
#   2. check_policy_alignment: answer accurately reflects what the policy actually says
#   3. check_tone: response is appropriately sensitive given the topic
#   4. inject_citations: attach source document name and section to the final answer
# if grounding or alignment fails badly: reject and send back to Reasoning Agent
# if it passes: forward to Governance Sub-Agent for post-check and audit logging