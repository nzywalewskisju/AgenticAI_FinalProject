# src/ingestion/loader.py
# Responsible for reading raw files into text and metadata.
# Accepts a list of file paths — files can come from anywhere on the user's disk.
# Supports PDF (via LangChain PyPDFLoader) and DOCX (via python-docx).
# Preserves document structure by detecting and marking headings with ## prefix
# so the chunker can use them as section boundaries.
# Returns a list of dicts: {text, metadata} where metadata contains:
#   source_file, file_type, document_name, user_id
# Never called directly by agents — called by the ingestion pipeline only.

import os
from docx import Document
from langchain_community.document_loaders import PyPDFLoader


def _detect_and_mark_headings(text: str) -> str:
    """
    Scans text for likely heading patterns and prefixes them with ##
    so the chunker can detect section boundaries.
    Only marks lines that are clearly section headings — not table rows or fragments.
    """
    import re
    lines = text.split("\n")
    marked = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            marked.append(line)
            continue

        # Skip likely table fragments
        is_table_fragment = (
            stripped.endswith("|") or
            stripped.startswith("|") or
            stripped.count("|") > 1 or
            stripped.endswith("or") or
            stripped.endswith("and") or
            stripped.endswith("(") or
            stripped.endswith(",") or
            stripped.endswith("%") or
            stripped.endswith("/month") or
            stripped.endswith("/year") or
            stripped.endswith("$") or
            stripped.startswith("$") or
            len(stripped) < 8 or
            stripped.replace(".", "").replace(",", "").replace("$", "").replace("/", "").replace("-", "").replace(" ", "").isdigit()
        )

        if is_table_fragment:
            marked.append(line)
            continue

        # Only mark as heading if it matches strong structural patterns
        is_numbered = bool(re.match(r"^\d+(\.\d+)?\s+[A-Z]", stripped)) and len(stripped) < 80
        is_section_keyword = stripped.lower().startswith("section")
        is_all_caps_meaningful = stripped.isupper() and len(stripped) > 6 and len(stripped) < 60 and len(stripped.split()) >= 2

        if is_numbered or is_section_keyword or is_all_caps_meaningful:
            marked.append(f"## {stripped}")
        else:
            marked.append(line)

    return "\n".join(marked)


def load_pdf(file_path: str, user_id: str) -> list[dict]:
    """
    Loads a PDF file using LangChain's PyPDFLoader.
    Returns a list of {text, metadata} dicts — one per page, then merged.
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    full_text = "\n".join(page.page_content for page in pages)
    full_text = _detect_and_mark_headings(full_text)

    document_name = os.path.splitext(os.path.basename(file_path))[0].replace("_", " ").replace("-", " ").title()

    return [{
        "text": full_text,
        "metadata": {
            "source_file": os.path.basename(file_path),
            "file_type": "pdf",
            "document_name": document_name,
            "user_id": user_id
        }
    }]


def load_docx(file_path: str, user_id: str) -> list[dict]:
    """
    Loads a DOCX file using python-docx.
    Preserves heading structure by detecting paragraph styles.
    Returns a list of {text, metadata} dicts.
    """
    doc = Document(file_path)
    lines = []

    for para in doc.paragraphs:
        if not para.text.strip():
            continue
        if para.style.name.startswith("Heading"):
            lines.append(f"## {para.text.strip()}")
        else:
            lines.append(para.text.strip())

    full_text = "\n".join(lines)
    document_name = os.path.splitext(os.path.basename(file_path))[0].replace("_", " ").replace("-", " ").title()

    return [{
        "text": full_text,
        "metadata": {
            "source_file": os.path.basename(file_path),
            "file_type": "docx",
            "document_name": document_name,
            "user_id": user_id
        }
    }]


def load_document(file_path: str, user_id: str) -> list[dict]:
    """
    Routes a single file to the correct loader based on extension.
    Raises ValueError for unsupported file types.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return load_pdf(file_path, user_id)
    elif ext == ".docx":
        return load_docx(file_path, user_id)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported types: .pdf, .docx")


def load_all_documents(file_paths: list[str], user_id: str) -> list[dict]:
    """
    Loads all documents from a list of file paths.
    Skips files that fail to load and prints a warning.
    Returns a flat list of all {text, metadata} dicts across all files.
    """
    all_documents = []

    for file_path in file_paths:
        try:
            docs = load_document(file_path, user_id)
            all_documents.extend(docs)
            print(f"[LOADER] Loaded: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"[LOADER] Warning: could not load {file_path}: {e}")

    return all_documents