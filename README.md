# PolicyPro — AI-Powered HR Policy Assistant

PolicyPro is a multi-agent AI system that gives employees instant, accurate, and cited answers to HR policy questions, grounded entirely in a company's own uploaded documents. It runs locally by default using Ollama and llama3.2, with optional GPT-4o mini support for improved reasoning quality.

Built as a final project for the Agentic AI course at Saint Joseph's University.

---

## Contributors

- Dennis Johnson
- Nick Zywalewski

---

## Requirements

- Python 3.10 or higher
- [Ollama](https://ollama.com) installed and running
- The following models pulled via Ollama:
- ollama pull llama3.2
- ollama pull nomic-embed-text
- An OpenAI API key (optional, only needed for GPT-4o mini mode)

---

## Installation

1. Clone the repository
2. Create and activate a virtual environment:
- Mac/Linux: python -m venv venv then source venv/bin/activate
- Windows: python -m venv venv then venv\Scripts\activate
3. Install dependencies: pip install -r requirements.txt
4. Create a .env file in the project root with the following:
- OPENAI_API_KEY=your_openai_key_here
- GMAIL_APP_PASSWORD=your_gmail_app_password_here
- ALERT_EMAIL=your_alert_email_here

---

## First Time Setup

On first launch, create an account through the login screen. You will be asked to set a username, password, and a security question for account recovery. Once logged in, upload your HR policy documents using the document panel on the left side of the interface. The first ingestion will take longer than subsequent ones as the embedding model processes all chunks for the first time. Once ingestion is complete you can begin asking questions immediately.

---

## Running the Application

Make sure Ollama is running before launching. If it does not start automatically, run the following in a separate terminal and leave it open:

ollama serve

Then launch the application:

python gui.py

To run in CLI mode instead:

python main.py

To reset a user's password from the terminal:

python main.py --reset-password username


## How It Works

1. Upload HR policy documents (PDF or DOCX) through the document panel
2. Documents are loaded, chunked, embedded, and stored in a local ChromaDB collection scoped to your account
3. Ask any HR policy question in the chat interface
4. The Orchestrator classifies the query and routes it through the agent pipeline
5. The Governor checks for prompt injection, PII, and escalation risk before reasoning begins
6. The Reasoning agent runs a ReAct loop, calling retrieval tools dynamically to find relevant policy chunks
7. The Review agent checks grounding, alignment, tone, and applicability before the answer is approved
8. The Governor runs a final compliance stamp and the answer reaches you with source citations

---

## Model Selection

The model selector in the top bar allows switching between:

- Llama (local): runs entirely on your machine via Ollama, no data leaves the device
- GPT-4o mini: faster and more accurate on complex reasoning, requires an OpenAI API key and sends query text to OpenAI

Embeddings always use nomic-embed-text locally regardless of which LLM is selected.

---

## Security and Compliance

- All queries are screened for prompt injection and PII before reasoning begins
- Sensitive topics like harassment, discrimination, and legal disputes are escalated to HR rather than answered
- Every interaction is logged to logs/audit_log.jsonl with the user, query, answer, and timestamp
- Email alerts are sent to the configured HR inbox for any security or escalation event
- Answers are scanned for legally dangerous absolute statements before reaching the user

---

## Known Limitations

- llama3.2 running on CPU is slow, expect several minutes per query depending on hardware
- PDF tables with complex formatting may not chunk correctly on first ingestion, clearing all documents and re-ingesting usually resolves this
- The system is only as accurate as the documents uploaded, outdated or incomplete documents will produce outdated or incomplete answers
- GPT-4o mini requires an active OpenAI account with billing enabled
- The system does not currently support file types other than PDF and DOCX

---

## Troubleshooting

- If ingestion returns 0 chunks, make sure Ollama is running by opening a terminal and running ollama serve, then try ingesting again
- If the application fails to launch, make sure the virtual environment is activated before running python gui.py
- If ChromaDB throws a corruption error during ingestion, delete the db folder from the project root and re-ingest your documents
- If you see a connection refused error, Ollama is not running, start it with ollama serve in a separate terminal and leave that window open
- If you are on Mac and Ollama does not start automatically on login, open the Ollama app from Applications and enable Launch at Login from the menu bar icon

---

## Notes

- The database and all user data are stored locally and never sent to any external server when using Llama
- Each user has their own isolated ChromaDB collection, documents ingested by one user are not visible to others
- If Ollama is not running, the application will fail to embed or retrieve, run ollama serve in a separate terminal before launching