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
- User profile facts are for your reasoning only — NEVER include them in your Answer text.
- Do not start your answer by restating who the user is or what their role is.
- Your Answer should address the policy directly without introducing the user's profile.
- Do not say things like "The user is a data analyst" or "As a [role] at [company]" in your Answer.
- Session history is for conversational context only — do NOT apply facts from a previous unrelated question to the current question.
- If the current question is about a different topic than the previous question, treat it as fresh with no assumed context from prior answers.
- NEVER apply eligibility criteria, age ranges, or specific numbers from a previous answer to a new unrelated question.
- Each query is independent. Do NOT carry over facts, eligibility determinations, or policy conclusions from previous questions in the session history.
- Session history shows you what was asked before — it does not mean prior answers apply to the current question.
- Each query is independent. Do NOT carry over facts, eligibility determinations, or policy conclusions from previous questions in the session history.
- Session history shows you what was asked before — it does not mean prior answers apply to the current question.
- If the current question is about a different topic than the prior question, start fresh with new retrieval.
- For short or vague queries, expand them before retrieving. Example: "Am I eligible for the professional development fund?" should become "professional development fund eligibility requirements criteria".
- Always use specific policy terms from the query as your retrieval keywords.
- For state-specific questions, always include the state name in your retrieval query. Example: California parental leave should retrieve "California CFRA parental leave policy".
- When a query mentions the Professional Development Fund, ALWAYS retrieve section 7.7 specifically using the query "professional development fund eligibility covered expenses excluded".
- When a query mentions tuition, MBA, or degree programs, ALWAYS retrieve using "tuition reimbursement program degree MBA" to find the correct policy.
- Never conflate the Professional Development Fund with the Tuition Reimbursement Program — they are separate programs with different rules.
- When a query mentions MBA, degree program, or tuition, you MUST retrieve chunks using "tuition reimbursement program degree MBA excluded" before answering.
- When previous attempt feedback mentions a contradiction or wrong program, change your retrieval query completely — do not repeat the same search.
- If feedback says "Policy contradiction detected", retrieve the specific policy that was contradicted using different search terms.
- If the query mentions MBA, degree, tuition, or "not covered", you MUST call keyword_search with the term "degree programs not covered" as one of your searches.
- keyword_search with exact terms finds content that semantic search misses due to poor section headers.

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
    retrieval_count: list
) -> str:
    """
    Executes a named action and returns the observation string.
    Updates chunks_used in place when retrieval actions return chunks.
    Uses text-based deduplication to prevent the same chunk being added multiple times.
    retrieval_count is a single-element list used as a mutable counter.
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

        # Auto keyword search for exclusion language when PD Fund is mentioned
        action_lower = action_input.lower()
        if any(term in action_lower for term in [
            "professional development", "pd fund", "mba", "degree", "tuition"
        ]):
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
        ]):
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
                action, action_input, user_id, chunks_used, retrieval_count
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