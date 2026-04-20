# src/ingestion/__init__.py
# Ingestion pipeline package.
# Contains loader.py, chunker.py, and embedder.py.
# Ingestion runs inline when a new file is uploaded by the user — not as a one-time script.
# Can run multiple times as users upload additional documents.
# All ingested data is scoped per user via user_id.