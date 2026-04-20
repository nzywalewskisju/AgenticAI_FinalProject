# tests/test_ingestion.py
# Unit tests for the ingestion pipeline.
# Tests loader.py:
#   - PDF loading returns expected text and metadata structure
#   - DOCX loading returns expected text and metadata structure
#   - Heading detection correctly marks ## prefixes
#   - Unsupported file types raise a clear error
# Tests chunker.py:
#   - Heading-based chunking splits on correct boundaries
#   - Paragraph fallback activates when no headings are detected
#   - Fixed-size fallback activates when paragraphs exceed CHUNK_SIZE
#   - All chunks carry required metadata fields
# Tests embedder.py:
#   - Chunks are stored with correct metadata in ChromaDB
#   - Duplicate ingestion of the same file does not create duplicate chunks
#   - Collection is correctly scoped by user_id

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock


# ── Loader tests ───────────────────────────────────────────────────────────────

class TestLoader:

    def test_load_document_unsupported_type(self):
        from src.ingestion.loader import load_document
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document("document.txt", user_id="test_user")

    def test_load_document_routes_pdf(self):
        from src.ingestion.loader import load_document
        with patch("src.ingestion.loader.load_pdf") as mock_pdf:
            mock_pdf.return_value = [{"text": "test", "metadata": {}}]
            result = load_document("policy.pdf", user_id="test_user")
            mock_pdf.assert_called_once_with("policy.pdf", "test_user")

    def test_load_document_routes_docx(self):
        from src.ingestion.loader import load_document
        with patch("src.ingestion.loader.load_docx") as mock_docx:
            mock_docx.return_value = [{"text": "test", "metadata": {}}]
            result = load_document("policy.docx", user_id="test_user")
            mock_docx.assert_called_once_with("policy.docx", "test_user")

    def test_heading_detection_marks_all_caps(self):
        from src.ingestion.loader import _detect_and_mark_headings
        text = "VACATION POLICY\nEmployees accrue 1.5 days per month."
        result = _detect_and_mark_headings(text)
        assert "## VACATION POLICY" in result

    def test_heading_detection_marks_numbered_sections(self):
        from src.ingestion.loader import _detect_and_mark_headings
        text = "1.2 Leave Entitlements\nFull-time employees are eligible."
        result = _detect_and_mark_headings(text)
        assert "## 1.2 Leave Entitlements" in result

    def test_load_pdf_metadata_structure(self):
        from src.ingestion.loader import load_pdf
        mock_page = MagicMock()
        mock_page.page_content = "VACATION POLICY\nEmployees get 15 days."
        with patch("src.ingestion.loader.PyPDFLoader") as MockLoader:
            MockLoader.return_value.load.return_value = [mock_page]
            result = load_pdf("pto_policy.pdf", user_id="user_1")
            assert len(result) == 1
            assert result[0]["metadata"]["file_type"] == "pdf"
            assert result[0]["metadata"]["user_id"] == "user_1"
            assert result[0]["metadata"]["source_file"] == "pto_policy.pdf"
            assert "text" in result[0]

    def test_load_all_documents_skips_failed_files(self):
        from src.ingestion.loader import load_all_documents
        with patch("src.ingestion.loader.load_document") as mock_load:
            mock_load.side_effect = [
                [{"text": "good doc", "metadata": {}}],
                Exception("File not found")
            ]
            result = load_all_documents(["good.pdf", "bad.pdf"], user_id="user_1")
            assert len(result) == 1
            assert result[0]["text"] == "good doc"


# ── Chunker tests ──────────────────────────────────────────────────────────────

class TestChunker:

    def test_heading_based_chunking(self):
        from src.ingestion.chunker import chunk_document
        doc = {
            "text": "## Section One\nContent of section one.\n## Section Two\nContent of section two.",
            "metadata": {
                "source_file": "policy.pdf",
                "file_type": "pdf",
                "document_name": "Policy",
                "user_id": "user_1"
            }
        }
        chunks = chunk_document(doc)
        assert len(chunks) >= 2
        headers = [c["metadata"]["section_header"] for c in chunks]
        assert "Section One" in headers
        assert "Section Two" in headers

    def test_paragraph_fallback_when_no_headings(self):
        from src.ingestion.chunker import chunk_document
        doc = {
            "text": "First paragraph with some content here.\n\nSecond paragraph with different content.",
            "metadata": {
                "source_file": "policy.pdf",
                "file_type": "pdf",
                "document_name": "Policy",
                "user_id": "user_1"
            }
        }
        chunks = chunk_document(doc)
        assert len(chunks) >= 1
        assert "Paragraph" in chunks[0]["metadata"]["section_header"]

    def test_all_chunks_have_required_metadata(self):
        from src.ingestion.chunker import chunk_document
        doc = {
            "text": "## Test Section\nSome policy content here that is meaningful.",
            "metadata": {
                "source_file": "policy.pdf",
                "file_type": "pdf",
                "document_name": "Policy",
                "user_id": "user_1"
            }
        }
        chunks = chunk_document(doc)
        required_fields = ["source_file", "file_type", "document_name", "user_id", "section_header", "chunk_index"]
        for chunk in chunks:
            for field in required_fields:
                assert field in chunk["metadata"], f"Missing field: {field}"

    def test_fixed_size_fallback_for_large_sections(self):
        from src.ingestion.chunker import chunk_document
        from config import CHUNK_SIZE
        long_text = "Word " * (CHUNK_SIZE * 2)
        doc = {
            "text": f"## Big Section\n{long_text}",
            "metadata": {
                "source_file": "policy.pdf",
                "file_type": "pdf",
                "document_name": "Policy",
                "user_id": "user_1"
            }
        }
        chunks = chunk_document(doc)
        assert len(chunks) > 1

    def test_chunk_index_is_sequential(self):
        from src.ingestion.chunker import chunk_document
        doc = {
            "text": "## One\nContent.\n## Two\nContent.\n## Three\nContent.",
            "metadata": {
                "source_file": "policy.pdf",
                "file_type": "pdf",
                "document_name": "Policy",
                "user_id": "user_1"
            }
        }
        chunks = chunk_document(doc)
        indices = [c["metadata"]["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))


# ── Embedder tests ─────────────────────────────────────────────────────────────

class TestEmbedder:

    @patch("src.ingestion.embedder._embed_text")
    @patch("src.ingestion.embedder._get_collection")
    def test_stores_chunks_in_chromadb(self, mock_collection, mock_embed):
        from src.ingestion.embedder import embed_and_store
        mock_embed.return_value = [0.1] * 768
        mock_col = MagicMock()
        mock_col.get.return_value = {"ids": []}
        mock_collection.return_value = mock_col

        chunks = [{
            "text": "Test policy content.",
            "metadata": {"source_file": "policy.pdf", "chunk_index": 0, "user_id": "user_1"}
        }]
        count = embed_and_store(chunks, user_id="user_1")
        assert count == 1
        mock_col.add.assert_called_once()

    @patch("src.ingestion.embedder._embed_text")
    @patch("src.ingestion.embedder._get_collection")
    def test_skips_duplicate_chunks(self, mock_collection, mock_embed):
        from src.ingestion.embedder import embed_and_store
        mock_embed.return_value = [0.1] * 768
        mock_col = MagicMock()
        # Simulate chunk already existing
        mock_col.get.return_value = {"ids": ["policy.pdf__chunk_0"]}
        mock_collection.return_value = mock_col

        chunks = [{
            "text": "Already ingested content.",
            "metadata": {"source_file": "policy.pdf", "chunk_index": 0, "user_id": "user_1"}
        }]
        count = embed_and_store(chunks, user_id="user_1")
        assert count == 0
        mock_col.add.assert_not_called()

    def test_chunk_id_is_deterministic(self):
        from src.ingestion.embedder import _make_chunk_id
        metadata = {"source_file": "pto_policy.pdf", "chunk_index": 3}
        id1 = _make_chunk_id(metadata)
        id2 = _make_chunk_id(metadata)
        assert id1 == id2
        assert "pto_policy.pdf" in id1
        assert "3" in id1