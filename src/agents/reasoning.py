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
import queue

from yaml import emit
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
- You MUST apply retrieved policy to the user's specific facts — not just quote the policy.
- You MUST NOT make up policy details. If you cannot find relevant policy, say so.
- You MUST NOT use hedging language: never say "typically", "usually", "generally", "I think", "probably".
- You MUST NOT use AND, OR, or quote operators in search queries — use plain natural language only.
- If the user's situation is unclear, call request_clarification.
- Every factual claim in your answer MUST come from retrieved chunks.
- If retrieve_chunks returns no results, try keyword_search with different plain terms.
- ALWAYS try keyword_search after retrieve_chunks to catch content semantic search may miss.

- User profile facts are for your reasoning only — never mention them in your Answer text.
- Your Answer should address the policy directly without restating who the user is.
- Each query is independent — never carry over eligibility determinations or policy conclusions from a previous question.
- Session history provides conversational context only — prior answers do not apply to new unrelated questions.

- For short or vague queries, expand them before retrieving using specific policy terms.
- For state-specific questions, always include the state name in your retrieval query.
- When a query mentions Professional Development Fund, tuition, MBA, or degree programs, retrieve using "professional development fund eligibility excluded degree programs".
- Never conflate the Professional Development Fund with the Tuition Reimbursement Program — they are separate programs with different rules.
- When previous attempt feedback mentions a contradiction, change your retrieval query completely and search for the specific policy that was contradicted.

- When a retrieved chunk says a policy is "effective January 1, 2025" it means it is currently active — never describe it as a future benefit.
- Never tell a user a benefit "will be available" if the retrieved chunk says it is already effective.
- When you have retrieved chunks, base your Answer ONLY on those chunks — never use training data to fill gaps.
- If retrieved chunks contain a specific dollar amount, date, or number that answers the query, state it exactly.
- If the chunks do not contain the answer, say "the retrieved policy does not specify this" rather than guessing.
- Never describe enrollment portals, forms, or deadlines unless they are explicitly stated in a retrieved chunk.
- NEVER reference chunks by number in your Answer. Do not say "policy chunk [1]" or "Source:" inline.

- When answering equipment damage questions, always check retrieved chunks for ALL damage categories — accidental, negligent, and theft each have different rules and financial consequences.
- When a chunk contains a table with multiple rows, extract ALL relevant rows not just the first matching one.
- Never make classification determinations that the policy assigns to IT, HR, or a manager. If the policy says "IT assessment required", tell the user IT will make that determination — do not decide for them.
- When multiple policy categories could apply to a situation, present all of them and explain which applies under which conditions.

- The current date is provided at the top of every query. Always use that date when reasoning about contribution limits, deadlines, or effective dates.
- Never assume a year from your training data — always use the date provided in the query.

SEARCH QUERY FORMAT:
- Use plain natural language — no quotes, no AND, no OR, no special operators
- Good: retrieve_chunks: 401k contribution limits catch-up
- Good: keyword_search: 401k catch-up contribution age
- Bad: retrieve_chunks: "401k plan limits" AND "catch-up contributions"
- Bad: keyword_search: "catch-up contribution" AND "age 62"

CRITICAL FORMAT RULE:
Every Action MUST be on a single line in exactly this format:
Action: tool_name: your input here

NEVER use Action: followed by bullet points or multiple lines.
NEVER use Action: as a label for reasoning notes.
If you are reasoning, use Thought: not Action:

AVAILABLE ACTIONS:
- check_policy_coverage: <topic> — check if relevant policy exists before retrieving
- retrieve_chunks: <plain natural language query> — semantic search for relevant policy sections
- keyword_search: <plain terms> — exact keyword search for specific policy terms
- rerank_results: <query> — re-score current chunks for relevance (call after retrieving)
- get_current_date: now — get today's date for reasoning about effective dates
- request_clarification: <question> — ask the user for more detail before proceeding

FORMAT:
Thought: [your reasoning about what to do next]
Action: tool_name: input on this same line
PAUSE

When you have enough information to give a complete, grounded answer:
Thought: I now have enough information to advise the user.
Answer: [your full advice here]

Your answer must include:
1. The facts of the user's situation as you understand them
2. The specific policy that applies
3. Concrete advice — what the user should do, what they are eligible for, what risks they face
4. Any deadlines, notice requirements, or steps they need to take"""


def _execute_action(
    action: str,
    action_input: str,
    user_id: str,
    chunks_used: list,
    retrieval_count: list,
    original_query: str = ""
) -> str:
    """
    Executes a named action and returns the observation string.
    Updates chunks_used in place when retrieval actions return chunks.
    Uses text-based deduplication to prevent the same chunk being added multiple times.
    retrieval_count is a single-element list used as a mutable counter.
    original_query is the original user query used for context-aware auto-triggers.
    """
    action = action.strip().lower()

    # Sanitize query — strip quotes, boolean operators, and underscores
    action_input = (
        action_input.strip()
        .replace("_", " ")
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
        if retrieval_count[0] >= 3:
            return (
                "Maximum retrievals reached. "
                "You have already retrieved chunks three times. "
                "You MUST now produce your Answer using the chunks you already have. "
                "Do not call retrieve_chunks or keyword_search again."
            )
        retrieval_count[0] += 1
        chunks = retrieve_chunks(action_input, user_id)
        print(f"[REASONING] Retrieved {len(chunks)} chunks (retrieval {retrieval_count[0]}/3)")
        if not chunks:
            return (
                "No relevant chunks found for this query. "
                "Try keyword_search with different plain terms, "
                "or produce your Answer using what you already have."
            )
        existing_texts = {c["text"] for c in chunks_used}
        new_count = 0
        for c in chunks:
            if c["text"] not in existing_texts:
                chunks_used.append(c)
                existing_texts.add(c["text"])
                new_count += 1

        action_lower = action_input.lower()
        query_lower = original_query.lower()

        # Auto keyword search for MBA/degree exclusion
        pd_fund_query = any(term in query_lower for term in [
            "professional development", "pd fund", "development fund"
        ])
        if any(term in action_lower for term in [
            "professional development", "pd fund", "mba", "degree", "tuition"
        ]) or pd_fund_query:
            from src.tools.retrieval import keyword_search as kw_search
            exclusion_chunks = kw_search("degree programs not covered MBA excluded", user_id)
            for c in exclusion_chunks:
                if c["text"] not in existing_texts:
                    chunks_used.append(c)
                    existing_texts.add(c["text"])
                    new_count += 1
            if exclusion_chunks:
                print(f"[REASONING] Auto-added {len(exclusion_chunks)} exclusion chunks")

        # Auto keyword search for pet insurance queries
        if any(term in action_lower for term in [
            "pet", "dog", "cat", "animal", "nationwide", "voluntary benefits"
        ]) or any(term in query_lower for term in ["pet", "dog", "cat", "parrot", "animal"]):
            print(f"[REASONING] Auto-triggering pet insurance keyword search")
            from src.tools.retrieval import keyword_search as kw_search
            pet_chunks = kw_search("pet insurance dogs cats nationwide voluntary benefits", user_id)
            for c in pet_chunks:
                if c["text"] not in existing_texts:
                    chunks_used.append(c)
                    existing_texts.add(c["text"])
                    new_count += 1
            if pet_chunks:
                print(f"[REASONING] Auto-added {len(pet_chunks)} pet insurance chunks")

        # Auto keyword search for SECURE 2.0 enhanced catch-up contribution
        is_401k_age_query = (
            any(term in query_lower for term in [
                "401k", "401(k)", "contribution", "catch-up", "catchup"
            ]) and
            any(age in query_lower for age in ["60", "61", "62", "63", "age"])
        )
        if is_401k_age_query or any(term in action_lower for term in [
            "401k", "catch-up", "catchup", "secure", "contribution limit"
        ]):
            from src.tools.retrieval import keyword_search as kw_search
            secure_chunks = kw_search(
                "SECURE 2.0 employees aged 60 61 62 63 enhanced catch-up 11250 effective January 2025", user_id
            )
            for c in secure_chunks:
                if c["text"] not in existing_texts:
                    chunks_used.append(c)
                    existing_texts.add(c["text"])
                    new_count += 1
            if secure_chunks:
                print(f"[REASONING] Auto-added {len(secure_chunks)} SECURE 2.0 chunks")
                for c in secure_chunks:
                    print(f"[REASONING] SECURE chunk: {c['text'][:200]}")

        return f"Retrieved {len(chunks)} chunks ({new_count} new):\n\n{format_chunks_for_prompt(chunks)}"

    elif action == "keyword_search":
        if retrieval_count[0] >= 3:
            return (
                "Maximum retrievals reached. "
                "You MUST now produce your Answer using the chunks you already have. "
                "Do not call retrieve_chunks or keyword_search again."
            )
        retrieval_count[0] += 1
        chunks = keyword_search(action_input, user_id)
        print(f"[REASONING] Keyword search returned {len(chunks)} chunks (retrieval {retrieval_count[0]}/3)")
        if not chunks:
            return (
                "No results found for keyword search. "
                "Try retrieve_chunks with a plain natural language query instead, "
                "or produce your Answer using what you already have."
            )
        existing_texts = {c["text"] for c in chunks_used}
        new_count = 0
        for c in chunks:
            if c["text"] not in existing_texts:
                chunks_used.append(c)
                existing_texts.add(c["text"])
                new_count += 1
        return f"Keyword search returned {len(chunks)} chunks ({new_count} new):\n\n{format_chunks_for_prompt(chunks)}"

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
            f"You have used all available tools. "
            f"Available actions: check_policy_coverage, retrieve_chunks, "
            f"keyword_search, rerank_results, get_current_date, request_clarification. "
            f"If you have enough information from the retrieved chunks, provide your Answer now. "
            f"Do not invent new actions."
        )


def run_reasoning_agent(
    query: str,
    user_id: str,
    session_context: str = "",
    profile_context: str = "",
    prior_chunks: list = None,
    status_queue=None
) -> dict:
    """
    Runs the ReAct loop for the given query.
    Retrieval is capped at 2 calls total to prevent excessive chunk accumulation.
    Accepts prior_chunks from previous attempts so the agent does not
    start from zero on retries.
    Returns {situation_facts, draft_answer, chunks_used, status, iterations}
    """
    def emit(msg: str):
        print(f"[REASONING] {msg}")
        if status_queue:
            status_queue.put(msg)

    emit(f"Starting analysis...")
    chunks_used = list(prior_chunks) if prior_chunks else []
    retrieval_count = [0]

    context_block = ""
    if profile_context and "No profile" not in profile_context:
        context_block += f"\n\nUSER PROFILE:\n{profile_context}"
    if session_context and "No prior" not in session_context:
        context_block += f"\n\nPRIOR CONVERSATION:\n{session_context}"

    prior_context = ""
    if chunks_used:
        prior_context = (
            f"\n\nNOTE: {len(chunks_used)} policy chunks were already retrieved in a previous attempt. "
            f"You may proceed directly to reasoning about the user's situation using those chunks, "
            f"or retrieve additional chunks if needed. Your retrieval limit is still 2 total."
        )

    initial_prompt = f"""USER QUERY: {query}
{context_block}{prior_context}

Begin by extracting the facts of the user's situation, then check policy coverage,
then retrieve relevant policy, then reason about how the policy applies to their specific situation.
You MUST retrieve policy chunks before providing any Answer unless prior chunks are already provided above.
Retrieve NO MORE THAN TWICE — after two retrievals you must produce your Answer."""

    messages = [{"role": "user", "content": initial_prompt}]

    iterations = 0
    situation_facts = ""
    blocked_answer_count = 0

    while iterations < MAX_REACT_TURNS:
        iterations += 1

        emit(f"Reasoning step {iterations} of {MAX_REACT_TURNS}...")

        full_prompt = "\n\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in messages
        )

        response = call_llm(full_prompt, system_prompt=REASONING_SYSTEM_PROMPT)
        messages.append({"role": "assistant", "content": response})

        if iterations == 1 and "situation" in response.lower():
            situation_facts = response.split("Action:")[0].strip()

        if "CLARIFICATION_NEEDED:" in response:
            question = response.split("CLARIFICATION_NEEDED:")[-1].strip()
            emit("Requesting clarification from user...")
            return {
                "situation_facts": situation_facts,
                "draft_answer": question,
                "chunks_used": chunks_used,
                "status": "clarification",
                "iterations": iterations
            }

        if "Answer:" in response:
            if not chunks_used:
                blocked_answer_count += 1
                emit("No policy documents retrieved yet — retrieving before answering...")
                print(f"[REASONING] Agent tried to answer without retrieving chunks — forcing retrieval (attempt {blocked_answer_count})")
                if blocked_answer_count >= 2:
                    print(f"[REASONING] Agent unable to retrieve chunks after {blocked_answer_count} attempts — breaking loop")
                    return {
                        "situation_facts": situation_facts,
                        "draft_answer": "",
                        "chunks_used": [],
                        "status": "no_info",
                        "iterations": iterations
                    }
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
            emit("Composing answer from retrieved policy...")
            raw_answer = response.split("Answer:")[-1].strip()
            answer_lines = []
            for line in raw_answer.split("\n"):
                stripped = line.strip()
                if (stripped.startswith("Action:") or
                    stripped.startswith("PAUSE") or
                    stripped.startswith("Note:") or
                    stripped.startswith("Thought:")):
                    break
                answer_lines.append(line)
            answer = "\n".join(answer_lines).strip()
            return {
                "situation_facts": situation_facts,
                "draft_answer": answer,
                "chunks_used": chunks_used,
                "status": "success",
                "iterations": iterations
            }

        action_match = ACTION_RE.search(response)
        if action_match:
            action = action_match.group(1)
            action_input = action_match.group(2)

            # Emit a human-readable status for each action
            action_display = action_input[:60].strip()
            if action == "check_policy_coverage":
                emit(f"Checking if policy exists for: {action_display}...")
            elif action == "retrieve_chunks":
                emit(f"Searching policy documents for: {action_display}...")
            elif action == "keyword_search":
                emit(f"Keyword search: {action_display}...")
            elif action == "rerank_results":
                emit(f"Ranking results by relevance...")
            elif action == "get_current_date":
                emit("Checking today's date...")
            elif action == "request_clarification":
                emit("Preparing clarification question...")
            else:
                emit(f"Running: {action}...")

            # If this is the last turn and we have chunks, force an Answer instead
            if iterations >= MAX_REACT_TURNS - 1 and chunks_used:
                emit("Final turn reached — composing answer from retrieved policy...")
                print(f"[REASONING] Last turn reached with chunks available — forcing Answer")
                force_prompt = (
                    f"You have retrieved {len(chunks_used)} policy chunks. "
                    f"This is your final turn. "
                    f"You MUST now write your Answer using the chunks you have. "
                    f"Do not call any more actions. "
                    f"Write your Answer now."
                )
                messages.append({"role": "user", "content": force_prompt})

                full_prompt = "\n\n".join(
                    f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                    for m in messages
                )
                final_response = call_llm(full_prompt, system_prompt=REASONING_SYSTEM_PROMPT)

                if "Answer:" in final_response:
                    raw_answer = final_response.split("Answer:")[-1].strip()
                    answer_lines = []
                    for line in raw_answer.split("\n"):
                        stripped = line.strip()
                        if (stripped.startswith("Action:") or
                            stripped.startswith("PAUSE") or
                            stripped.startswith("Note:") or
                            stripped.startswith("Thought:")):
                            break
                        answer_lines.append(line)
                    answer = "\n".join(answer_lines).strip()
                    return {
                        "situation_facts": situation_facts,
                        "draft_answer": answer,
                        "chunks_used": chunks_used,
                        "status": "success",
                        "iterations": iterations
                    }

            observation = _execute_action(
                action, action_input, user_id, chunks_used, retrieval_count,
                original_query=query
            )
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            print(f"[REASONING] No action parsed. Raw response:\n{response[:300]}")
            messages.append({
                "role": "user",
                "content": (
                    "Your last response did not contain a valid Action. "
                    "Remember: every Action must be on a single line like this:\n"
                    "Action: tool_name: your input here\n"
                    "Use an Action or provide your final Answer."
                )
            })

    emit("Max reasoning steps reached...")
    print(f"[REASONING] Max turns reached without Answer — returning no_info")
    return {
        "situation_facts": situation_facts,
        "draft_answer": "",
        "chunks_used": chunks_used,
        "status": "no_info",
        "iterations": iterations
    }