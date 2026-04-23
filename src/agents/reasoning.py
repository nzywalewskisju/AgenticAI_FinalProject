# src/agents/reasoning.py
# Reasoning Sub-Agent — implements the ReAct loop.
# This is what makes the system an agent rather than a retrieval chatbot.
# The agent dynamically decides which tools to call, in what order, and how many times.
# ReAct loop structure: Thought → Action → PAUSE → Observation → repeat
# Max iterations: MAX_REACT_TURNS (set in config.py)
# Tools available to this agent:
#   - check_policy_coverage (MUST be called before retrieve_chunks)
#   - retrieve_chunks
#   - keyword_search
#   - rerank_results
#   - get_current_date
#   - request_clarification
# The agent explicitly:
#   1. Extracts the facts of the user's situation from the query
#   2. Retrieves policy relevant to those facts
#   3. Reasons about the gap between the user's situation and policy requirements
#   4. Produces concrete, personalized advice — not just a policy summary
# Returns: {situation_facts, draft_answer, chunks_used, status, iterations}
# Status: "success" | "clarification" | "no_info" | "error"

import re
from config import MAX_REACT_TURNS, RERANK_SKIP_THRESHOLD
from src.tools.utils import call_llm, format_chunks_for_prompt, get_current_date
from src.tools.retrieval import retrieve_chunks, keyword_search, rerank_results
from src.tools.document import check_policy_coverage


ACTION_RE = re.compile(r"^Action:\s*([\w]+)\s*:\s*(.+)$", re.DOTALL | re.MULTILINE)

REASONING_SYSTEM_PROMPT = """You are an HR Policy Reasoning Agent. Your job is NOT to summarize policy.
Your job is to reason about the user's specific situation and give them concrete, actionable advice
grounded in the policy documents you retrieve.

You follow the ReAct pattern: Thought → Action → PAUSE → Observation → repeat.

RULES:
- You MUST call check_policy_coverage before calling retrieve_chunks.
- You MUST call retrieve_chunks or keyword_search on EVERY query — no exceptions.
- You MUST NOT produce an Answer until you have retrieved at least one chunk.
- You MUST extract the facts of the user's situation before retrieving anything.
- You MUST apply retrieved policy to the user's specific facts — not just quote the policy.
- You MUST NOT make up policy details. If you cannot find relevant policy, say so.
- You MUST NOT use hedging language: never say "typically", "usually", "generally", "I think", "probably".
- You MUST NOT use AND, OR, or quote operators in search queries — use plain natural language only.
- If the user's situation is unclear, call request_clarification.
- Every factual claim in your answer MUST come from retrieved chunks.
- If retrieve_chunks returns no results, you MUST try keyword_search with different plain terms.
- If keyword_search returns no results, try retrieve_chunks again with simpler terms.

SEARCH QUERY FORMAT:
- Use plain natural language — no quotes, no AND, no OR, no special operators
- Good: retrieve_chunks: 401k contribution limits catch-up
- Good: keyword_search: 401k catch-up contribution age
- Bad: retrieve_chunks: "401k plan limits" AND "catch-up contributions"
- Bad: keyword_search: "catch-up contribution" AND "age 62"

AVAILABLE ACTIONS:
- check_policy_coverage: <topic> — check if relevant policy exists before retrieving
- retrieve_chunks: <plain natural language query> — semantic search for relevant policy sections
- keyword_search: <plain terms> — exact keyword search for specific policy terms
- rerank_results: <query> — re-score current chunks for relevance (call after retrieving)
- get_current_date: now — get today's date for reasoning about effective dates
- request_clarification: <question> — ask the user for more detail before proceeding

FORMAT:
Thought: [your reasoning about what to do next]
Action: [action_name]: [input]
PAUSE

When you have enough information to give a complete, grounded answer:
Thought: I now have enough information to advise the user.
Answer: [your full advice here]

Your answer must include:
1. The facts of the user's situation as you understand them
2. The specific policy that applies
3. Concrete advice — what the user should do, what they are eligible for, what risks they face
4. Any deadlines, notice requirements, or steps they need to take"""


def _execute_action(action: str, action_input: str, user_id: str, chunks_used: list) -> str:
    """
    Executes a named action and returns the observation string.
    Updates chunks_used in place when retrieval actions return chunks.
    """
    action = action.strip().lower()

    # Sanitize query — strip quotes and boolean operators the model may have added
    action_input = (
        action_input.strip()
        .replace(" AND ", " ")
        .replace(" OR ", " ")
        .replace(" NOT ", " ")
        .replace('"', '')
        .replace("'", "")
        .strip()
    )

    print(f"[REASONING] Action: {action} | Input: {action_input}")

    if action == "check_policy_coverage":
        result = check_policy_coverage(action_input, user_id)
        print(f"[REASONING] Coverage result: {result}")
        return f"Policy coverage check: {result['reason']} Covered: {result['covered']}"

    elif action == "retrieve_chunks":
        chunks = retrieve_chunks(action_input, user_id)
        print(f"[REASONING] Retrieved {len(chunks)} chunks")
        if not chunks:
            return (
                "No relevant chunks found for this query. "
                "Try keyword_search with different plain terms, "
                "or try retrieve_chunks with a simpler, shorter query."
            )
        for c in chunks:
            if c not in chunks_used:
                chunks_used.append(c)
        return f"Retrieved {len(chunks)} chunks:\n\n{format_chunks_for_prompt(chunks)}"

    elif action == "keyword_search":
        chunks = keyword_search(action_input, user_id)
        print(f"[REASONING] Keyword search returned {len(chunks)} chunks")
        if not chunks:
            return (
                "No results found for keyword search. "
                "Try retrieve_chunks with a plain natural language query instead."
            )
        for c in chunks:
            if c not in chunks_used:
                chunks_used.append(c)
        return f"Keyword search returned {len(chunks)} chunks:\n\n{format_chunks_for_prompt(chunks)}"

    elif action == "rerank_results":
        if not chunks_used:
            return "No chunks to rerank. Retrieve chunks first."
        top_distance = chunks_used[0].get("distance", 1.0)
        if top_distance < RERANK_SKIP_THRESHOLD:
            return f"Top chunk is already highly relevant (distance {top_distance:.3f}). Skipping rerank."
        reranked = rerank_results(action_input, chunks_used)
        chunks_used.clear()
        chunks_used.extend(reranked)
        return f"Reranked {len(reranked)} chunks by relevance."

    elif action == "get_current_date":
        return f"Today's date is: {get_current_date()}"

    elif action == "request_clarification":
        return f"CLARIFICATION_NEEDED: {action_input}"

    else:
        return (
            f"Unknown action: {action}. "
            f"Available actions: check_policy_coverage, retrieve_chunks, "
            f"keyword_search, rerank_results, get_current_date, request_clarification."
        )


def run_reasoning_agent(
    query: str,
    user_id: str,
    session_context: str = "",
    profile_context: str = ""
) -> dict:
    """
    Runs the ReAct loop for the given query.
    Returns {situation_facts, draft_answer, chunks_used, status, iterations}
    """
    print(f"[REASONING] Starting ReAct loop for user: {user_id}")
    chunks_used = []

    context_block = ""
    if profile_context and "No profile" not in profile_context:
        context_block += f"\n\nUSER PROFILE:\n{profile_context}"
    if session_context and "No prior" not in session_context:
        context_block += f"\n\nPRIOR CONVERSATION:\n{session_context}"

    initial_prompt = f"""USER QUERY: {query}
{context_block}

Begin by extracting the facts of the user's situation, then check policy coverage,
then retrieve relevant policy, then reason about how the policy applies to their specific situation.
You MUST retrieve policy chunks before providing any Answer."""

    messages = [{"role": "user", "content": initial_prompt}]

    iterations = 0
    situation_facts = ""

    while iterations < MAX_REACT_TURNS:
        iterations += 1

        full_prompt = "\n\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in messages
        )

        response = call_llm(full_prompt, system_prompt=REASONING_SYSTEM_PROMPT)
        messages.append({"role": "assistant", "content": response})

        if iterations == 1 and "situation" in response.lower():
            situation_facts = response.split("Action:")[0].strip()

        # Check for clarification request
        if "CLARIFICATION_NEEDED:" in response:
            question = response.split("CLARIFICATION_NEEDED:")[-1].strip()
            return {
                "situation_facts": situation_facts,
                "draft_answer": question,
                "chunks_used": chunks_used,
                "status": "clarification",
                "iterations": iterations
            }

        # Check for final answer — but only allow if chunks have been retrieved
        if "Answer:" in response:
            if not chunks_used:
                print(f"[REASONING] Agent tried to answer without retrieving chunks — forcing retrieval")
                messages.append({
                    "role": "user",
                    "content": (
                        "You have not retrieved any policy documents yet. "
                        "You MUST call check_policy_coverage and then retrieve_chunks "
                        "or keyword_search before providing an Answer. "
                        "Do not answer from memory or training data."
                    )
                })
                continue
            answer = response.split("Answer:")[-1].strip()
            return {
                "situation_facts": situation_facts,
                "draft_answer": answer,
                "chunks_used": chunks_used,
                "status": "success",
                "iterations": iterations
            }

        # Parse and execute action
        action_match = ACTION_RE.search(response)
        if action_match:
            action = action_match.group(1)
            action_input = action_match.group(2)
            observation = _execute_action(action, action_input, user_id, chunks_used)
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            messages.append({
                "role": "user",
                "content": "Continue. Use an Action or provide your final Answer."
            })

    # Max turns reached without an answer
    return {
        "situation_facts": situation_facts,
        "draft_answer": "",
        "chunks_used": chunks_used,
        "status": "no_info",
        "iterations": iterations
    }