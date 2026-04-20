# tests/test_tools.py
# Unit tests for all tools in isolation.
# Ollama and ChromaDB are mocked — these tests do not require a running Ollama instance.
# Tests retrieval.py:
#   - retrieve_chunks returns results above SIMILARITY_THRESHOLD only
#   - keyword_search returns results containing exact query terms
#   - rerank_results correctly sorts chunks by Llama-assigned score
# Tests document.py:
#   - check_policy_coverage returns True when relevant content exists
#   - check_policy_coverage returns False when collection is empty or off-topic
#   - registry functions correctly read, write, and delete from JSON
# Tests governance.py:
#   - detect_pii flags queries containing another employee's sensitive information
#   - assess_escalation_risk returns float between 0 and 1
#   - compliance_stamp flags absolute statements correctly
#   - write_audit_log appends valid JSON to audit_log.jsonl
# Tests utils.py:
#   - clean_llm_json_response strips code fences correctly
#   - truncate_text does not exceed max_tokens

import pytest
import json
import os
import tempfile
from unittest.mock import patch, MagicMock


# ── Utils tests ────────────────────────────────────────────────────────────────

class TestUtils:

    def test_clean_json_response_strips_json_fence(self):
        from src.tools.utils import clean_llm_json_response
        raw = '```json\n{"key": "value"}\n```'
        result = clean_llm_json_response(raw)
        assert result == '{"key": "value"}'

    def test_clean_json_response_strips_plain_fence(self):
        from src.tools.utils import clean_llm_json_response
        raw = '```\n{"key": "value"}\n```'
        result = clean_llm_json_response(raw)
        assert result == '{"key": "value"}'

    def test_clean_json_response_passthrough_clean(self):
        from src.tools.utils import clean_llm_json_response
        raw = '{"key": "value"}'
        result = clean_llm_json_response(raw)
        assert result == '{"key": "value"}'

    def test_safe_json_parse_valid(self):
        from src.tools.utils import safe_json_parse
        result = safe_json_parse('{"score": 0.8}')
        assert result == {"score": 0.8}

    def test_safe_json_parse_returns_fallback_on_invalid(self):
        from src.tools.utils import safe_json_parse
        result = safe_json_parse("not valid json at all", fallback={"error": True})
        assert result == {"error": True}

    def test_truncate_text_under_limit(self):
        from src.tools.utils import truncate_text
        text = "Short text."
        result = truncate_text(text, max_chars=100)
        assert result == text

    def test_truncate_text_over_limit(self):
        from src.tools.utils import truncate_text
        text = "word " * 1000
        result = truncate_text(text, max_chars=100)
        assert len(result) <= 120   # allow for truncation marker
        assert "[truncated]" in result

    def test_get_current_date_returns_string(self):
        from src.tools.utils import get_current_date
        result = get_current_date()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_chunks_for_prompt_empty(self):
        from src.tools.utils import format_chunks_for_prompt
        result = format_chunks_for_prompt([])
        assert "No relevant" in result

    def test_format_chunks_for_prompt_with_chunks(self):
        from src.tools.utils import format_chunks_for_prompt
        chunks = [{
            "text": "Employees get 15 days PTO.",
            "metadata": {"document_name": "PTO Policy", "section_header": "Entitlements"}
        }]
        result = format_chunks_for_prompt(chunks)
        assert "PTO Policy" in result
        assert "Entitlements" in result
        assert "Employees get 15 days PTO." in result


# ── Retrieval tests ────────────────────────────────────────────────────────────

class TestRetrieval:

    @patch("src.tools.retrieval._get_collection")
    @patch("src.tools.retrieval._embed_query")
    def test_retrieve_chunks_filters_by_threshold(self, mock_embed, mock_collection):
        from src.tools.retrieval import retrieve_chunks
        from config import SIMILARITY_THRESHOLD

        mock_embed.return_value = [0.1] * 768
        mock_col = MagicMock()
        mock_col.count.return_value = 5
        mock_col.query.return_value = {
            "documents": [["Good chunk", "Bad chunk"]],
            "metadatas": [[{"source_file": "a.pdf"}, {"source_file": "b.pdf"}]],
            "distances": [[0.2, 0.9]]   # 0.2 passes, 0.9 fails threshold
        }
        mock_collection.return_value = mock_col

        results = retrieve_chunks("PTO policy", user_id="user_1")
        assert len(results) == 1
        assert results[0]["text"] == "Good chunk"

    @patch("src.tools.retrieval._get_collection")
    @patch("src.tools.retrieval._embed_query")
    def test_retrieve_chunks_empty_collection(self, mock_embed, mock_collection):
        from src.tools.retrieval import retrieve_chunks
        mock_col = MagicMock()
        mock_col.count.return_value = 0
        mock_collection.return_value = mock_col

        results = retrieve_chunks("anything", user_id="user_1")
        assert results == []

    @patch("src.tools.retrieval._get_collection")
    def test_keyword_search_scores_relevance(self, mock_collection):
        from src.tools.retrieval import keyword_search
        mock_col = MagicMock()
        mock_col.count.return_value = 2
        mock_col.get.return_value = {
            "documents": [
                "PTO policy allows 15 vacation days per year.",
                "The dress code requires business casual attire."
            ],
            "metadatas": [{"source_file": "pto.pdf"}, {"source_file": "dress.pdf"}]
        }
        mock_collection.return_value = mock_col

        results = keyword_search("vacation days", user_id="user_1")
        assert len(results) > 0
        assert results[0]["text"] == "PTO policy allows 15 vacation days per year."

    @patch("src.tools.retrieval.call_llm")
    def test_rerank_results_sorts_by_score(self, mock_llm):
        from src.tools.retrieval import rerank_results
        mock_llm.return_value = '[{"index": 1, "score": 3}, {"index": 2, "score": 9}]'

        chunks = [
            {"text": "Less relevant chunk.", "metadata": {}},
            {"text": "Highly relevant chunk.", "metadata": {}}
        ]
        results = rerank_results("PTO policy", chunks)
        assert results[0]["text"] == "Highly relevant chunk."
        assert results[0]["rerank_score"] == 9

    @patch("src.tools.retrieval.call_llm")
    def test_rerank_drops_low_scores(self, mock_llm):
        from src.tools.retrieval import rerank_results
        mock_llm.return_value = '[{"index": 1, "score": 2}, {"index": 2, "score": 8}]'

        chunks = [
            {"text": "Irrelevant chunk.", "metadata": {}},
            {"text": "Relevant chunk.", "metadata": {}}
        ]
        results = rerank_results("PTO policy", chunks)
        assert len(results) == 1
        assert results[0]["text"] == "Relevant chunk."


# ── Document tools tests ───────────────────────────────────────────────────────

class TestDocumentTools:

    @patch("src.tools.document._get_collection")
    @patch("src.tools.document._embed_query")
    def test_check_policy_coverage_empty_collection(self, mock_embed, mock_collection):
        from src.tools.document import check_policy_coverage
        mock_col = MagicMock()
        mock_col.count.return_value = 0
        mock_collection.return_value = mock_col

        result = check_policy_coverage("PTO", user_id="user_1")
        assert result["covered"] is False
        assert "No documents" in result["reason"]

    @patch("src.tools.document._get_collection")
    @patch("src.tools.document._embed_query")
    def test_check_policy_coverage_found(self, mock_embed, mock_collection):
        from src.tools.document import check_policy_coverage
        mock_embed.return_value = [0.1] * 768
        mock_col = MagicMock()
        mock_col.count.return_value = 5
        mock_col.query.return_value = {
            "distances": [[0.2]],
            "documents": [["PTO policy content"]]
        }
        mock_collection.return_value = mock_col

        result = check_policy_coverage("vacation days", user_id="user_1")
        assert result["covered"] is True

    def test_registry_add_and_get(self, tmp_path, monkeypatch):
        from src.tools import document as doc_module
        monkeypatch.setattr(doc_module, "_get_registry_path", lambda uid: str(tmp_path / f"{uid}.json"))

        from src.tools.document import add_to_registry, get_registry
        record = add_to_registry("user_1", "/tmp/policy.pdf", chunk_count=10)
        assert record["file_name"] == "policy.pdf"
        assert record["chunk_count"] == 10

        registry = get_registry("user_1")
        assert len(registry) == 1

    def test_registry_no_duplicate_on_re_add(self, tmp_path, monkeypatch):
        from src.tools import document as doc_module
        monkeypatch.setattr(doc_module, "_get_registry_path", lambda uid: str(tmp_path / f"{uid}.json"))

        from src.tools.document import add_to_registry, get_registry
        add_to_registry("user_1", "/tmp/policy.pdf", chunk_count=10)
        add_to_registry("user_1", "/tmp/policy.pdf", chunk_count=10)

        registry = get_registry("user_1")
        assert len(registry) == 1

    @patch("src.tools.document._get_collection")
    def test_registry_remove(self, mock_collection, tmp_path, monkeypatch):
        from src.tools import document as doc_module
        monkeypatch.setattr(doc_module, "_get_registry_path", lambda uid: str(tmp_path / f"{uid}.json"))

        mock_col = MagicMock()
        mock_col.get.return_value = {"ids": []}
        mock_collection.return_value = mock_col

        from src.tools.document import add_to_registry, remove_from_registry, get_registry
        add_to_registry("user_1", "/tmp/policy.pdf", chunk_count=5)
        removed = remove_from_registry("user_1", "/tmp/policy.pdf")
        assert removed is True
        assert get_registry("user_1") == []


# ── Governance tools tests ─────────────────────────────────────────────────────

class TestGovernanceTools:

    @patch("src.tools.governance.call_llm")
    def test_detect_pii_flags_other_employee(self, mock_llm):
        from src.tools.governance import detect_pii
        mock_llm.return_value = '{"contains_pii": true, "reason": "Contains another employee name and salary"}'
        result = detect_pii("What is John Smith's salary?")
        assert result["contains_pii"] is True

    @patch("src.tools.governance.call_llm")
    def test_detect_pii_clears_self_reference(self, mock_llm):
        from src.tools.governance import detect_pii
        mock_llm.return_value = '{"contains_pii": false, "reason": "User asking about themselves"}'
        result = detect_pii("How many PTO days do I have left?")
        assert result["contains_pii"] is False

    def test_escalation_always_escalates_harassment(self):
        from src.tools.governance import assess_escalation_risk
        result = assess_escalation_risk("My manager is harassing me")
        assert result["should_escalate"] is True
        assert result["risk_score"] == 1.0

    def test_escalation_always_escalates_discrimination(self):
        from src.tools.governance import assess_escalation_risk
        result = assess_escalation_risk("I think I was passed over due to discrimination")
        assert result["should_escalate"] is True

    @patch("src.tools.governance.call_llm")
    def test_compliance_stamp_flags_absolute_statements(self, mock_llm):
        from src.tools.governance import compliance_stamp
        mock_llm.return_value = '{"legally_dangerous": true, "flagged_phrases": ["you are entitled to"], "reason": "Absolute entitlement claim"}'
        result = compliance_stamp("You are entitled to 20 days of PTO.")
        assert result["passed"] is False
        assert len(result["flagged_phrases"]) > 0

    @patch("src.tools.governance.call_llm")
    def test_compliance_stamp_flags_hedge_phrases(self, mock_llm):
        from src.tools.governance import compliance_stamp
        mock_llm.return_value = '{"legally_dangerous": false, "flagged_phrases": [], "reason": ""}'
        result = compliance_stamp("Typically, employees receive 15 days of PTO.")
        assert result["passed"] is False
        assert "typically" in result["flagged_phrases"]

    def test_write_audit_log_appends_valid_json(self, tmp_path, monkeypatch):
        import src.tools.governance as gov_module
        log_path = str(tmp_path / "audit_log.jsonl")
        monkeypatch.setattr(gov_module, "AUDIT_LOG_PATH", log_path)
        # Re-import to pick up monkeypatched path
        from src.tools.governance import write_audit_log
        write_audit_log(
            session_id="sess_1",
            user_id="user_1",
            query="How much PTO do I have?",
            route="hr_in_scope",
            escalated=False,
            chunks_used=[],
            situation_facts="User has been employed 2 years.",
            final_answer="You have 15 days remaining.",
            grounding_score=0.9,
            compliance_passed=True
        )
        with open(log_path, "r") as f:
            entry = json.loads(f.readline())
        assert entry["user_id"] == "user_1"
        assert entry["query"] == "How much PTO do I have?"
        assert entry["grounding_score"] == 0.9
        assert "entry_id" in entry
        assert "timestamp" in entry