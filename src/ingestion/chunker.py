# src/ingestion/chunker.py
# Responsible for splitting loaded documents into chunks for embedding.
# Uses a hybrid chunking strategy in this order:
#   1. Heading/section detection — splits on ## markers and numbered section headers
#   2. Paragraph fallback — splits on double newlines if no headings found
#   3. Fixed-size fallback — uses RecursiveCharacterTextSplitter if paragraphs are too large
# Each chunk carries metadata: source_file, file_type, document_name,
#   section_header, chunk_index, user_id
# Chunk size and overlap are set in config.py (CHUNK_SIZE, CHUNK_OVERLAP).
# Never called directly by agents — called by the ingestion pipeline only.

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """
    Splits text on ## heading markers.
    Returns list of (section_header, section_text) tuples.
    """
    pattern = re.compile(r"^## (.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))

    if not matches:
        return []

    sections = []
    for i, match in enumerate(matches):
        header = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append((header, section_text))

    return sections


def _split_by_paragraphs(text: str) -> list[tuple[str, str]]:
    """
    Splits text on double newlines when no headings are detected.
    Returns list of ("Paragraph N", paragraph_text) tuples.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    return [(f"Paragraph {i + 1}", p) for i, p in enumerate(paragraphs)]


def _apply_fixed_size_fallback(
    section_header: str,
    section_text: str
) -> list[tuple[str, str]]:
    """
    Applies RecursiveCharacterTextSplitter to a section that exceeds CHUNK_SIZE.
    Returns list of (section_header, chunk_text) tuples — header is preserved.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    sub_chunks = splitter.split_text(section_text)
    return [(section_header, chunk) for chunk in sub_chunks]


def chunk_document(document: dict) -> list[dict]:
    """
    Takes a single loaded document {text, metadata} and returns
    a list of chunk dicts ready for embedding.
    Applies hybrid chunking strategy: headings → paragraphs → merge fragments → fixed-size.
    Each chunk carries full metadata including section_header and chunk_index.
    """
    text = document["text"]
    metadata = document["metadata"]
    chunks = []

    # Step 1: try heading-based splitting
    sections = _split_by_headings(text)

    # Step 2: fall back to paragraph splitting if no headings found
    if not sections:
        sections = _split_by_paragraphs(text)

    # Step 3: merge short fragments — prevents table rows and PDF table
    # fragments from becoming isolated chunks that cannot be retrieved
    merged_sections = []
    for header, section_text in sections:
        if len(section_text.strip()) < 300 and merged_sections:
            # Merge into previous section instead of creating a standalone chunk
            prev_header, prev_text = merged_sections[-1]
            merged_sections[-1] = (prev_header, prev_text + "\n" + section_text)
        else:
            merged_sections.append((header, section_text))

    # Step 4: apply fixed-size fallback within any section that is too large
    final_sections = []
    for header, section_text in merged_sections:
        if len(section_text) > CHUNK_SIZE:
            sub = _apply_fixed_size_fallback(header, section_text)
            final_sections.extend(sub)
        else:
            final_sections.append((header, section_text))

    # Step 5: build chunk dicts with full metadata
    for i, (header, chunk_text) in enumerate(final_sections):
        if not chunk_text.strip():
            continue
        chunk_metadata = {
            **metadata,
            "section_header": header,
            "chunk_index": i
        }
        chunks.append({
            "text": chunk_text,
            "metadata": chunk_metadata
        })

    return chunks


def chunk_all_documents(documents: list[dict]) -> list[dict]:
    """
    Chunks all loaded documents.
    Returns a flat list of all chunk dicts across all documents.
    """
    all_chunks = []
    for document in documents:
        doc_chunks = chunk_document(document)
        all_chunks.extend(doc_chunks)
        print(f"[CHUNKER] {document['metadata']['document_name']}: {len(doc_chunks)} chunks")
    return all_chunks