# gui.py
# Desktop GUI entry point for the HR Policy Assistant.
# Run with: python gui.py
# Launches a login dialog first, then the main application window.
# Login dialog handles: account creation, login, forgot password flow.
# Main window contains:
#   - File picker for uploading HR documents from anywhere on disk
#   - Document panel showing all ingested documents with per-doc clear buttons
#   - User profile panel showing remembered facts with ability to delete individual facts
#   - Query input field (Enter key or Submit button)
#   - Scrollable answer display with citations
#   - Status indicator: idle / ingesting / thinking / ready / error
#   - Clear Session button that resets conversation memory only
# All inference and storage is local — no external API calls at runtime.
# Errors from Ollama and ChromaDB are surfaced in the window, not the terminal.