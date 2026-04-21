# config.py
# Central configuration file for the HR Policy Assistant.
# All settings, thresholds, model names, and paths are defined here.
# Never hardcode these values in any other file — always import from config.
# To tune the system, edit this file only.

import os
from dotenv import load_dotenv

load_dotenv()

# ── Model settings ─────────────────────────────────────────────────────────────
LLM_MODEL        = "llama3"
EMBEDDING_MODEL  = "nomic-embed-text"
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── ChromaDB ───────────────────────────────────────────────────────────────────
CHROMA_DB_PATH  = "./db"
COLLECTION_NAME = "hr_documents"

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K_RESULTS        = 3        # reduced from 5 — faster retrieval, negligible accuracy cost
# Cosine distance: 0 = identical, 1 = orthogonal.
# 0.35 means "must be at least moderately similar". Tune after ingesting real docs.
SIMILARITY_THRESHOLD = 0.35

# ── Ingestion ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 600
CHUNK_OVERLAP = 80

# ── Agent behaviour ────────────────────────────────────────────────────────────
MAX_REACT_TURNS      = 6
ESCALATION_THRESHOLD = 0.75

# ── Routing categories ─────────────────────────────────────────────────────────
ROUTE_IN_SCOPE     = "hr_in_scope"
ROUTE_HIGH_STAKES  = "high_stakes"
ROUTE_OUT_OF_SCOPE = "out_of_scope"

# ── Storage paths ──────────────────────────────────────────────────────────────
AUDIT_LOG_PATH = "./logs/audit_log.jsonl"
PROFILES_DIR   = "./data/profiles"
REGISTRY_DIR   = "./data/registry"
USERS_FILE     = "./data/users/users.json"

# ── Auth ───────────────────────────────────────────────────────────────────────
SECURITY_QUESTIONS = [
    "What city were you born in?",
    "What was your first pet's name?",
    "What street did you grow up on?",
    "What was the name of your first school?",
    "What is your mother's maiden name?"
]

# ── Reranking ──────────────────────────────────────────────────────────────────
# Chunks with a distance below this threshold are considered highly confident
# and will skip reranking entirely to save an LLM call.
RERANK_SKIP_THRESHOLD = 0.15

# ── Ensure required directories exist at import time ──────────────────────────
for _dir in ["./db", "./logs", "./data/profiles", "./data/registry", "./data/users"]:
    os.makedirs(_dir, exist_ok=True)