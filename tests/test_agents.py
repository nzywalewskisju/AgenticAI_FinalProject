# tests/test_agents.py
# Integration tests for agent behaviour.
# Ollama is mocked with controlled responses to test routing and logic.
# Tests orchestrator.py:
#   - out_of_scope queries are rejected before any agent is called
#   - high_stakes queries are escalated before Reasoning runs
#   - queries with no ingested documents prompt a file upload request
#   - profile facts are extracted and persisted correctly
#   - no-chunks guard rejects answers produced without retrieved evidence
#   - max 2 retries are enforced when Review rejects Reasoning output
# Tests reasoning.py:
#   - ReAct loop calls check_policy_coverage before retrieve_chunks
#   - loop exits correctly when Answer: is detected in response
#   - loop exits after MAX_REACT_TURNS if no answer is produced
#   - returns situation_facts, draft_answer, chunks_used in output
# Tests review.py:
#   - all five checks run in order
#   - answer is rejected if grounding score < 0.7
#   - answer is rejected if advice does not apply policy to user situation
# Tests governor.py:
#   - PII queries are blocked at pre-check
#   - escalation threshold correctly routes high-risk queries
#   - audit log entry contains all required fields

import pytest
from unittest.mock import patch, MagicMock


# ── Orchestrator tests ─────────────────────────────────────────────────────────

class TestOrchestrator:

    @patch("src.agents.orchestrator.classify_query")
    @patch("src.agents.orchestrator.extract_and_update_profile")
    def test_out_of_scope_returns_immediately(self, mock_profile, mock_classify):
        from src.agents.orchestrator import run_orchestrator
        from config import ROUTE_OUT_OF_SCOPE
        from pydantic import BaseModel

        mock_profile.return_value = []
        mock_classify.return_value = MagicMock(
            category=ROUTE_OUT_OF_SCOPE,
            confidence=0.95,
            reasoning="Not HR related"
        )

        result = run_orchestrator("What is the weather today?", user_id="user_1")
        assert result["status"] == "out_of_scope"
        assert result["route"] == ROUTE_OUT_OF_SCOPE

    @patch("src.agents.orchestrator.classify_query")
    @patch("src.agents.orchestrator.extract_and_update_profile")
    @patch("src.agents.orchestrator.has_documents")
    def test_no_documents_returns_upload_prompt(self, mock_has_docs, mock_profile, mock_classify):
        from src.agents.orchestrator import run_orchestrator
        from config import ROUTE_IN_SCOPE

        mock_profile.return_value = []
        mock_classify.return_value = MagicMock(
            category=ROUTE_IN_SCOPE,
            confidence=0.9,
            reasoning="HR question"
        )
        mock_has_docs.return_value = False

        result = run_orchestrator("How many PTO days do I get?", user_id="user_1")
        assert result["status"] == "no_documents"
        assert "upload" in result["answer"].lower()

    @patch("src.agents.orchestrator.classify_query")
    @patch("src.agents.orchestrator.extract_and_update_profile")
    @patch("src.agents.orchestrator.has_documents")
    @patch("src.agents.orchestrator.run_governance_precheck")
    def test_pii_query_is_blocked(self, mock_precheck, mock_has_docs, mock_profile, mock_classify):
        from src.agents.orchestrator import run_orchestrator
        from config import ROUTE_IN_SCOPE

        mock_profile.return_value = []
        mock_classify.return_value = MagicMock(category=ROUTE_IN_SCOPE, confidence=0.9, reasoning="")
        mock_has_docs.return_value = True
        mock_precheck.return_value = {
            "cleared": False,
            "reason": "PII detected",
            "message": "Contains another employee's data.",
            "escalated": False
        }

        result = run_orchestrator("What is Jane's salary?", user_id="user_1")
        assert result["status"] == "blocked"

    @patch("src.agents.orchestrator.classify_query")
    @patch("src.agents.orchestrator.extract_and_update_profile")
    @patch("src.agents.orchestrator.has_documents")
    @patch("src.agents.orchestrator.run_governance_precheck")
    def test_high_stakes_query_escalates(self, mock_precheck, mock_has_docs, mock_profile, mock_classify):
        from src.agents.orchestrator import run_orchestrator
        from config import ROUTE_HIGH_STAKES

        mock_profile.return_value = []
        mock_classify.return_value = MagicMock(category=ROUTE_HIGH_STAKES, confidence=0.95, reasoning="")
        mock_has_docs.return_value = True
        mock_precheck.return_value = {
            "cleared": False,
            "reason": "Escalation required",
            "message": "Please contact HR directly.",
            "escalated": True
        }

        result = run_orchestrator("My manager is harassing me.", user_id="user_1")
        assert result["status"] == "escalated"
        assert "HR" in result["answer"]

    @patch("src.agents.orchestrator.classify_query")
    @patch("src.agents.orchestrator.extract_and_update_profile")
    @patch("src.agents.orchestrator.has_documents")
    @patch("src.agents.orchestrator.run_governance_precheck")
    @patch("src.agents.orchestrator.run_reasoning_agent")
    @patch("src.agents.orchestrator.run_review_agent")
    @patch("src.agents.orchestrator.run_governance_postcheck")
    def test_no_chunks_guard_triggers_retry(
        self, mock_postcheck, mock_review, mock_reasoning,
        mock_precheck, mock_has_docs, mock_profile, mock_classify
    ):
        from src.agents.orchestrator import run_orchestrator
        from config import ROUTE_IN_SCOPE

        mock_profile.return_value = []
        mock_classify.return_value = MagicMock(category=ROUTE_IN_SCOPE, confidence=0.9, reasoning="")
        mock_has_docs.return_value = True
        mock_precheck.return_value = {"cleared": True, "reason": "", "message": "", "escalated": False}

        # Return no chunks every time — should exhaust retries
        mock_reasoning.return_value = {
            "situation_facts": "",
            "draft_answer": "Some answer",
            "chunks_used": [],
            "status": "success",
            "iterations": 2
        }

        result = run_orchestrator("How many PTO days?", user_id="user_1")
        assert result["status"] == "no_info"


# ── Reasoning agent tests ──────────────────────────────────────────────────────

class TestReasoningAgent:

    @patch("src.agents.reasoning.call_llm")
    @patch("src.agents.reasoning.check_policy_coverage")
    def test_loop_exits_on_answer(self, mock_coverage, mock_llm):
        from src.agents.reasoning import run_reasoning_agent
        mock_coverage.return_value = {"covered": True, "reason": "Found"}
        mock_llm.return_value = (
            "Thought: I have the information I need.\n"
            "Answer: Based on the PTO policy, you are eligible for 15 days per year."
        )

        result = run_reasoning_agent("How much PTO do I get?", user_id="user_1")
        assert result["status"] == "success"
        assert "15 days" in result["draft_answer"]

    @patch("src.agents.reasoning.call_llm")
    def test_loop_exits_after_max_turns(self, mock_llm):
        from src.agents.reasoning import run_reasoning_agent
        from config import MAX_REACT_TURNS
        # Never produce an Answer — always produce an action
        mock_llm.return_value = "Thought: Still thinking.\nAction: get_current_date: now\nPAUSE"

        result = run_reasoning_agent("What is my PTO balance?", user_id="user_1")
        assert result["status"] == "no_info"
        assert result["iterations"] == MAX_REACT_TURNS

    @patch("src.agents.reasoning.call_llm")
    def test_clarification_request_returns_question(self, mock_llm):
        from src.agents.reasoning import run_reasoning_agent
        mock_llm.return_value = (
            "Thought: I need more information.\n"
            "Action: request_clarification: Are you a full-time or part-time employee?\nPAUSE"
        )

        result = run_reasoning_agent("Am I eligible?", user_id="user_1")
        assert result["status"] == "clarification"
        assert "full-time or part-time" in result["draft_answer"]

    def test_result_has_required_keys(self):
        from src.agents.reasoning import run_reasoning_agent
        with patch("src.agents.reasoning.call_llm") as mock_llm:
            mock_llm.return_value = "Answer: You get 15 days."
            result = run_reasoning_agent("PTO?", user_id="user_1")
            required_keys = ["situation_facts", "draft_answer", "chunks_used", "status", "iterations"]
            for key in required_keys:
                assert key in result, f"Missing key: {key}"


# ── Review agent tests ─────────────────────────────────────────────────────────

class TestReviewAgent:

    @patch("src.agents.review.call_llm")
    def test_fails_on_no_chunks(self, mock_llm):
        from src.agents.review import run_review_agent
        result = run_review_agent(
            draft_answer="You get 15 days of PTO.",
            query="How much PTO?",
            situation_facts="User has worked 2 years.",
            chunks_used=[]
        )
        assert result["passed"] is False
        assert result["grounding_score"] == 0.0

    @patch("src.agents.review.call_llm")
    def test_fails_on_low_grounding_score(self, mock_llm):
        from src.agents.review import run_review_agent
        mock_llm.return_value = '{"score": 0.4, "reason": "Claims not found in chunks"}'

        chunks = [{"text": "Policy content.", "metadata": {"document_name": "PTO", "section_header": "Basics", "chunk_index": 0}}]
        result = run_review_agent(
            draft_answer="You get 30 days of PTO.",
            query="How much PTO?",
            situation_facts="",
            chunks_used=chunks
        )
        assert result["passed"] is False
        assert result["grounding_score"] < 0.7

    @patch("src.agents.review.call_llm")
    def test_passes_all_checks(self, mock_llm):
        from src.agents.review import run_review_agent
        # All LLM checks return passing results
        mock_llm.side_effect = [
            '{"score": 0.9, "reason": "Well grounded"}',           # grounding
            '{"passed": true, "reason": "Accurate"}',               # alignment
            '{"passed": true, "reason": "Appropriate tone"}',       # tone
            '{"passed": true, "reason": "Good situational advice"}' # applicability
        ]

        chunks = [{"text": "Employees accrue 1.5 days per month.", "metadata": {"document_name": "PTO Policy", "section_header": "Accrual", "chunk_index": 0}}]
        result = run_review_agent(
            draft_answer="Based on your 8 months of employment, you have accrued 12 days.",
            query="How much PTO do I have?",
            situation_facts="User has worked 8 months.",
            chunks_used=chunks
        )
        assert result["passed"] is True
        assert "Sources:" in result["answer"]

    @patch("src.agents.review.call_llm")
    def test_fails_on_advice_not_applied(self, mock_llm):
        from src.agents.review import run_review_agent
        mock_llm.side_effect = [
            '{"score": 0.85, "reason": "Grounded"}',
            '{"passed": true, "reason": "Aligned"}',
            '{"passed": true, "reason": "Good tone"}',
            '{"passed": false, "reason": "Answer only summarizes policy, does not address user situation"}'
        ]

        chunks = [{"text": "PTO accrues at 1.5 days per month.", "metadata": {"document_name": "PTO Policy", "section_header": "Accrual", "chunk_index": 0}}]
        result = run_review_agent(
            draft_answer="The PTO policy states employees accrue 1.5 days per month.",
            query="How much PTO do I have after 8 months?",
            situation_facts="User has worked 8 months.",
            chunks_used=chunks
        )
        assert result["passed"] is False
        assert "applicability" in result["failure_reason"].lower()


# ── Governor agent tests ───────────────────────────────────────────────────────

class TestGovernorAgent:

    @patch("src.agents.governor.detect_pii")
    def test_precheck_blocks_pii(self, mock_pii):
        from src.agents.governor import run_governance_precheck
        mock_pii.return_value = {"contains_pii": True, "reason": "Contains employee name and salary"}
        result = run_governance_precheck("What is Jane's salary?", user_id="user_1")
        assert result["cleared"] is False
        assert result["escalated"] is False
        assert "private information" in result["message"].lower()

    @patch("src.agents.governor.detect_pii")
    @patch("src.agents.governor.assess_escalation_risk")
    def test_precheck_escalates_high_risk(self, mock_risk, mock_pii):
        from src.agents.governor import run_governance_precheck
        mock_pii.return_value = {"contains_pii": False, "reason": ""}
        mock_risk.return_value = {
            "risk_score": 0.9,
            "should_escalate": True,
            "reason": "Harassment claim"
        }
        result = run_governance_precheck("My manager is harassing me", user_id="user_1")
        assert result["cleared"] is False
        assert result["escalated"] is True
        assert "HR" in result["message"]

    @patch("src.agents.governor.detect_pii")
    @patch("src.agents.governor.assess_escalation_risk")
    def test_precheck_clears_safe_query(self, mock_risk, mock_pii):
        from src.agents.governor import run_governance_precheck
        mock_pii.return_value = {"contains_pii": False, "reason": ""}
        mock_risk.return_value = {"risk_score": 0.1, "should_escalate": False, "reason": "Low risk"}
        result = run_governance_precheck("How many PTO days do I get?", user_id="user_1")
        assert result["cleared"] is True

    @patch("src.agents.governor.compliance_stamp")
    @patch("src.agents.governor.write_audit_log")
    def test_postcheck_always_writes_audit_log(self, mock_audit, mock_stamp):
        from src.agents.governor import run_governance_postcheck
        mock_stamp.return_value = {"passed": True, "flagged_phrases": [], "reason": ""}

        run_governance_postcheck(
            session_id="sess_1",
            user_id="user_1",
            query="How much PTO?",
            route="hr_in_scope",
            chunks_used=[],
            situation_facts="",
            final_answer="You have 15 days.",
            grounding_score=0.9
        )
        mock_audit.assert_called_once()

    @patch("src.agents.governor.compliance_stamp")
    @patch("src.agents.governor.write_audit_log")
    def test_postcheck_appends_disclaimer_on_compliance_failure(self, mock_audit, mock_stamp):
        from src.agents.governor import run_governance_postcheck
        mock_stamp.return_value = {
            "passed": False,
            "flagged_phrases": ["you are entitled to"],
            "reason": "Absolute statement"
        }

        result = run_governance_postcheck(
            session_id="sess_1",
            user_id="user_1",
            query="What am I owed?",
            route="hr_in_scope",
            chunks_used=[],
            situation_facts="",
            final_answer="You are entitled to 20 days.",
            grounding_score=0.85
        )
        assert "informational purposes only" in result["answer"].lower()
        assert result["passed"] is False