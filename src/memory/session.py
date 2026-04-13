# session.py
# manages conversation history so follow-up questions have context
# responsibilities:
#   - store each conversation turn (user query + agent answer) by session/user ID
#   - retrieve the last N turns to include in the orchestrator's context
#   - clear session history when a conversation ends
# keep this simple for now — store history in memory as a list of dicts
# later improvement: persist to a file or database so history survives restarts