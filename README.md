# Payment Document Chatbot

A local chatbot for payment documents. You can upload files, ask questions, and get answers with source citations.

It supports:

- UPI transaction CSV files
- Bank API logs
- Compliance documents
- Partnership/SLA documents

The app uses:

- FastAPI for the backend
- Postgres for app data
- OpenSearch for search and analytics
- A local OpenAI-compatible LLM for chat
- A simple browser UI

## Demo

<video src="data/assets/PaymentDocs_Chatbot.mp4" controls width="100%">
  Demo recording: data/assets/PaymentDocs_Chatbot.mp4
</video>

### Screenshots

![Payment document chatbot role selection](data/assets/Screenshot%202026-08-12%20at%2010.08.20%E2%80%AFAM.png)

![Payment document chatbot document upload](data/assets/Screenshot%202026-08-12%20at%2010.08.44%E2%80%AFAM.png)

![Payment document chatbot chat response](data/assets/Screenshot%202026-08-12%20at%2010.09.14%E2%80%AFAM.png)

![Payment document chatbot citations](data/assets/Screenshot%202026-08-12%20at%2010.09.18%E2%80%AFAM.png)

![Payment document chatbot history](data/assets/Screenshot%202026-08-12%20at%2010.09.24%E2%80%AFAM.png)

## Quick Start

Copy the environment file:

```bash
cp .env.example .env
```

Start Postgres and OpenSearch:

```bash
docker compose up -d postgres opensearch
```

Create and activate Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

Run database migration:

```bash
.venv/bin/alembic upgrade head
```

Start the app:

```bash
python -m server
```

Open:

```text
http://localhost:8000
```

API docs:

```text
http://localhost:8000/docs
```

## Run With Docker

To run everything with Docker:

```bash
docker compose --profile app up --build
```

The app will run at:

```text
http://localhost:8000
```

## Local LLM

The chatbot expects a local OpenAI-compatible model server.

Default URL:

```text
http://127.0.0.1:1234/v1
```

Default model:

```text
qwen3-1.7b
```

You can change these in `.env`.

Docker Compose starts a llama.cpp server automatically. It loads the model from Hugging Face:

```text
LLAMACPP_HF_REPO=Qwen/Qwen3-1.7B-GGUF:Q8_0
```

You can change this in `.env`.

## Important `.env` Values

For local development:

```text
DATABASE_URL=postgresql://paydoc:paydoc@localhost:5432/paydoc
OPENSEARCH_URL=http://localhost:9200
LOCAL_LLM_OPENAI_BASE_URL=http://127.0.0.1:1234/v1
```

In Docker, the app reads `.env`, but overrides these values so containers can talk to each other:

```text
DATABASE_URL=postgresql://paydoc:paydoc@postgres:5432/paydoc
OPENSEARCH_URL=http://opensearch:9200
LOCAL_LLM_OPENAI_BASE_URL=http://llamacpp:8080/v1
```

## How To Use

1. Open `http://localhost:8000`.
2. Choose a role from the left sidebar.
3. Upload documents.
4. Ask questions in the chat.

Available roles:

- Product Lead
- Tech Lead
- Compliance Lead
- Bank Alliance Lead

## Sample Data

Sample files are available in:

```text
data/samples/
```

Folders:

```text
data/samples/upi_transaction/      UPI transaction CSV files
data/samples/bank_api_log/         Bank API log JSONL files
data/samples/compliance_audit/     Compliance PDF files
data/samples/partnership_sla/      Partnership/SLA PDF files
```

Example upload:

```bash
curl -X POST http://localhost:8000/api/documents \
  -H "X-User-Role: product_lead" \
  -F "doc_type=upi_transaction" \
  -F "files=@data/samples/upi_transaction/paytm_upi_transactions_2026-08.csv"
```

## Document Types

When uploading a file, choose one document type:

- `upi_transaction`
- `bank_api_log`
- `compliance_audit`
- `partnership_sla`

## Search Behavior

The chatbot has two kinds of tools:

- SQL tools for exact numbers, counts, rates, sums, averages, date filters, and grouped reports.
- Vector search tools for fuzzy questions, document text, explanations, clauses, and context.

Example:

```text
What's the success rate of UPI transactions this month?
```

This should use SQL because it asks for an exact metric.

## Reset Everything

To clear Postgres and OpenSearch completely:

```bash
docker compose down -v --remove-orphans
docker compose up -d postgres opensearch
.venv/bin/alembic upgrade head
```

Warning: this deletes all local database and OpenSearch data.

## Useful Commands

Check containers:

```bash
docker compose ps
```

See app logs:

```bash
docker compose logs -f payment-document-chatbot
```

See OpenSearch logs:

```bash
docker compose logs -f opensearch
```

Validate Docker Compose:

```bash
docker compose --profile app config --quiet
```

Check Python syntax:

```bash
python3 -m py_compile server.py
```

## Main Files

```text
server.py                         App startup
api/                              API endpoints
frontend/                         Browser UI
database/models.py                Database tables
vector_db/knowledge_base.py       Upload indexing and search
vector_db/vector_search.py        Vector search logic
chatbot/stakeholder_handler.py    Role agents and tools
chatbot/response_generator.py     LLM prompt and response helpers
alembic/versions/                 Database migration
docker-compose.yml                Docker services
.env.example                      Example config
```

## Notes

- OpenSearch security is disabled for local development.
- The app uses demo header-based roles, not production login.
- Indexing can take time for large files.
- Do not put real secrets in `.env` when building Docker images.
