# AI Engineer Interview Assignment Submission

## Assignment Overview

This submission implements a multi-stakeholder payment document chatbot for:

- Product Lead
- Tech Lead
- Compliance Lead
- Bank Alliance Lead

The system ingests payment documents, indexes structured and unstructured content, routes questions to role-specific agents, and answers with source citations. It supports UPI transaction CSV files, bank API JSON/JSONL logs, compliance PDFs, partnership/SLA PDFs, and text documents.

Demo evidence is included in:

- `data/assets/PaymentDocs_Chatbot.mp4`
- `data/assets/Screenshot 2026-08-12 at 10.*.png` files

## Part 1: Document Understanding Pipeline

### 1.1 Model Selection and Fine-Tuning

Implemented files:

- `document_processor/document_loader.py`
- `document_processor/document_classifier.py`
- `document_processor/entity_extractor.py`
- `document_processor/model_fine_tuning.ipynb`

Model choices from the implementation and fine-tuning notebook:

| Requirement | Model Used | Local Artifact / Runtime | License Notes |
| --- | --- | --- | --- |
| Document classification | [`distilbert/distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased) | Fine-tuned to `data/models/distilbert-base-uncased-finetuned-document-classification` | Hugging Face model, Apache-2.0 |
| Named Entity Recognition | [`distilbert/distilbert-base-cased`](https://huggingface.co/distilbert/distilbert-base-cased) | Fine-tuned to `data/models/distilbert-base-cased-finetuned-entity-extraction` | Hugging Face model, Apache-2.0 |
| Text embedding | [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) | Loaded through Sentence Transformers using `EMBEDDING_MODEL_NAME` | Hugging Face model, MIT |
| Question answering / response generation | [`Qwen/Qwen3-1.7B-GGUF:Q8_0`](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF) | Served locally with llama.cpp as `qwen3-1.7b` | Hugging Face model, Apache-2.0 |

The classifier and entity extractor load local fine-tuned model directories from environment configuration:

```text
DOCUMENT_CLASSIFIER_MODEL_PATH=data/models/distilbert-base-uncased-finetuned-document-classification
ENTITY_EXTRACTOR_MODEL_PATH=data/models/distilbert-base-cased-finetuned-entity-extraction
```

Reasoning for this model set:

- The classification and NER models use DistilBERT because it is a lightweight distilled BERT family model that is faster and smaller than BERT while still being strong enough for supervised payment-document classification and entity extraction.
- `BAAI/bge-small-en-v1.5` is used for embeddings because it is a compact English embedding model suitable for local semantic search with 384-dimensional vectors.
- `Qwen/Qwen3-1.7B-GGUF:Q8_0` is used for chat because the 1.7B GGUF quantized model can run locally through llama.cpp, keeping document contents private and avoiding external LLM API calls.
- All base models are available on Hugging Face with permissive licenses for local/private prototype use. Apache-2.0 or MIT license notices should be retained when redistributing model artifacts.

Other strong model options considered:

- Larger encoder models such as `bert-base-uncased`, `roberta-base`, `deberta-v3-base`, and LayoutLM/DocFormer-style document models can improve accuracy on richer document understanding tasks.
- Larger embedding models such as `BAAI/bge-base-en-v1.5`, `BAAI/bge-large-en-v1.5`, or E5-large can improve retrieval quality.
- Larger local LLMs such as Qwen 7B/14B, Mistral 7B, or Llama-family 8B models can improve answer quality and reasoning depth.

These were intentionally not selected for this prototype because the target machine is a MacBook Air with 8 GB RAM. The chosen DistilBERT, BGE-small, and Qwen 1.7B GGUF stack keeps the system practical for a private/local demo without needing a cloud GPU or sending payment documents to external APIs.

Fine-tuning use cases:

- The document classifier is fine-tuned to map uploaded files into the assignment document classes: UPI transactions, bank integration logs, compliance circulars/audit reports, and partnership agreements.
- The NER model is fine-tuned to extract payment-document entities such as document/circular number, issue date, title/subject, agreement identifiers, bank names, transaction references, response codes, and SLA-related values.
- These supervised models reduce reliance on only prompt-based extraction and make ingestion more consistent before chunks are indexed.

Document loading supports:

- PDF extraction using PyMuPDF4LLM
- OCR fallback using Tesseract for scanned/low-text PDFs
- CSV ingestion using Pandas and LangChain `DataFrameLoader`
- JSON and JSONL ingestion using Pandas
- Plain text ingestion using LangChain `TextLoader`

### 1.2 Supervised Learning Implementation

Fine-tuning data is included under `data/fine_tuning/`.

Document classification dataset:

- `train.jsonl`: 160 examples
- `validation.jsonl`: 48 examples
- `test.jsonl`: 48 examples
- `label_map.json`: document type labels

Named entity recognition dataset:

- `train.jsonl`: 50 examples
- `validation.jsonl`: 10 examples
- `test.jsonl`: 10 examples

The datasets cover the required document types:

- UPI transaction logs
- Bank API integration responses
- Compliance audit reports
- Partnership/SLA documents

## Part 2: Vector Database and Knowledge Base

### 2.1 Vector Database Design

Implemented files:

- `vector_db/embedding_service.py`
- `vector_db/vector_search.py`
- `vector_db/knowledge_base.py`

OpenSearch is used for vector storage, keyword search, and SQL analytics. It was selected because it gives the chatbot a single local/private retrieval layer with:

- Hybrid search: combines semantic vector retrieval with keyword/BM25 matching for better recall on payment terms, codes, bank names, and regulatory phrases.
- SQL support: enables exact analytics directly over indexed transaction/log fields, such as success rates, latency averages, failure counts, and grouped reports.
- Typed metadata filters: supports role-specific filters on dates, banks, statuses, document type, page number, severity, and SLA/compliance fields.
- Operational simplicity: runs locally through Docker Compose and avoids a separate vector database plus a separate analytics engine.

The system creates one index per document type:

```text
payment_documents_upi_transaction
payment_documents_bank_api_log
payment_documents_compliance_audit
payment_documents_partnership_sla
```

Common chunk metadata:

- `document_id`
- `doc_type`
- `filename`
- `chunk_index`
- `page_number`
- `page_count`
- `content`
- `content_vector`
- `created_at`
- `updated_at`
- `indexed_at`

Typed metadata is added per document type. For example:

- UPI transactions: `transaction_id`, `timestamp`, `payer_bank`, `payee_bank`, `merchant_category`, `amount`, `status`, `response_code`, `settlement_status`, `risk_score`, `fraud_flag`
- Bank API logs: `bank`, `operation`, `request_id`, `trace_id`, `status`, `latency_ms`, `reconciliation_status`, `severity`, `gross_amount`, `fees`, `net_amount`
- Compliance PDFs: `circular_number`, `subject`, `issue_date`
- Partnership/SLA PDFs: `agreement_id`, `title`, `effective_date`, `expiry_date`

### 2.2 Knowledge Extraction

The ingestion pipeline:

1. Stores uploaded file metadata in Postgres.
2. Loads the document based on file type.
3. Uses the NER/document intelligence layer to extract document-level entities such as document number or circular number, issue date, title/subject, agreement id, effective date, and expiry date where available.
4. Splits content into markdown-aware chunks.
5. Adds the extracted entities to every chunk from that document so each retrieved chunk carries better context, freshness signals, and source identity.
6. Creates embeddings for each chunk.
7. Writes chunks and metadata to OpenSearch.
8. Updates indexing status as `inprogress`, `success`, or `failed`.

The extracted `document_id`, document/circular number, title/subject, and issue date are especially useful for compliance and SLA documents. They help the assistant identify newer circulars, cite the correct source, and avoid returning isolated text chunks without enough business context.

Search strategy:

- SQL search is used for exact analytics such as counts, success rates, totals, averages, trends, and grouped reports.
- Hybrid vector search is used for semantic questions, compliance clauses, SLA clauses, incident context, and supporting evidence.
- OpenSearch read-only SQL validation prevents unsafe write queries.
- Document updates and deletion are supported through the document API and remove both Postgres records and OpenSearch chunks.

## Part 3: Multi-Stakeholder Chatbot Development

### 3.1 Stakeholder-Specific Query Processing

Implemented files:

- `chatbot/query_router.py`
- `chatbot/stakeholder_handler.py`
- `chatbot/response_generator.py`

The chatbot uses a LangGraph router and one role-scoped LangChain agent per stakeholder.

| Stakeholder | Agent Behavior |
| --- | --- |
| Product Lead | Uses UPI transaction SQL for success rate, volume, failure rate, payment method, merchant, bank, and region analysis |
| Tech Lead | Uses bank API log SQL for integration failures, latency, severity, reconciliation, and error pattern analysis |
| Compliance Lead | Uses semantic search over compliance documents for regulatory requirements, audit controls, risk factors, and obligations |
| Bank Alliance Lead | Uses SLA document search plus bank integration SQL for partnership terms, SLA performance, uptime, latency, and violations |

Assignment sample queries supported:

- Product Lead: "What's the success rate of UPI transactions this month?"
- Product Lead: "Which payment methods are most popular?"
- Product Lead: "Show me transaction volume trends by region"
- Tech Lead: "Are there any API integration failures today?"
- Tech Lead: "What's the average response time for bank APIs?"
- Tech Lead: "Show me error patterns in payment processing"
- Compliance Lead: "Any suspicious transaction patterns detected?"
- Compliance Lead: "Are we meeting KYC requirements?"
- Compliance Lead: "Show me audit trail for high-value transactions"
- Bank Alliance Lead: "How is our SLA performance with Bank X?"
- Bank Alliance Lead: "What's the integration health score for new partnerships?"
- Bank Alliance Lead: "Any partnership agreement violations?"

### 3.2 Context-Aware Response Generation

Role handling is implemented with request headers:

```text
X-User-Role: product_lead | tech_lead | compliance_lead | bank_alliance_lead
X-User-Id: optional user id
X-Tenant-Id: optional tenant id
```

The middleware normalizes roles and stores user context on the request. Invalid or missing roles fall back to `product_lead`.

Response behavior:

- Domain questions must call a relevant SQL or vector search tool before answering.
- SQL is required for exact numbers, rates, percentages, averages, and grouped reports.
- Vector search is used for fuzzy evidence, clauses, regulatory text, and narrative context.
- Answers include citations from retrieved chunks or executed SQL queries.
- Chat history is saved in Postgres and passed back into later turns for conversation memory.
- Streaming responses are supported through Server-Sent Events.

## Part 4: System Integration and Deployment

### 4.1 API Design

Implemented files:

- `api/document_upload.py`
- `api/document_intelligence.py`
- `api/chat_endpoints.py`
- `api/auth_middleware.py`
- `api/schema.py`

Base URL:

```text
http://localhost:8000
```

FastAPI docs:

```text
http://localhost:8000/docs
```

#### Document Ingestion API

Upload documents:

```bash
curl -X POST http://localhost:8000/api/documents \
  -H "X-User-Role: product_lead" \
  -F "doc_type=upi_transaction" \
  -F "files=@data/samples/upi_transaction/paytm_upi_transactions_2026-08.csv"
```

List documents:

```bash
curl http://localhost:8000/api/documents
```

Download document:

```bash
curl -OJ http://localhost:8000/api/documents/1/download
```

Delete document:

```bash
curl -X DELETE http://localhost:8000/api/documents/1
```

#### Chat API

Non-streaming chat:

```bash
curl -X POST http://localhost:8000/api/chats \
  -H "Content-Type: application/json" \
  -H "X-User-Role: tech_lead" \
  -d '{
    "input": {"text": "Are there any API integration failures today?"},
    "stream": false
  }'
```

Streaming chat:

```bash
curl -N -X POST http://localhost:8000/api/chats \
  -H "Content-Type: application/json" \
  -H "X-User-Role: compliance_lead" \
  -d '{
    "input": {"text": "What compliance obligations are mentioned in recent UPI circulars?"},
    "stream": true
  }'
```

Chat history:

```bash
curl http://localhost:8000/api/chats
curl http://localhost:8000/api/chats/1/messages
```

#### Document Intelligence API

Classify document text:

```bash
curl -X POST http://localhost:8000/api/document-intelligence/classify \
  -F "text=UPI transaction failed with response code U16"
```

Extract entities:

```bash
curl -X POST http://localhost:8000/api/document-intelligence/extract-entities \
  -F "text=Transaction TXN123 failed at HDFC Bank with response code U30"
```

#### Error Handling and Status Codes

- `200`: successful read, chat, classification, or entity extraction response
- `201`: document upload accepted
- `204`: delete completed
- `404`: document, chat, message, or stored file not found
- `422`: invalid filename, missing file/text, unsupported document format, invalid pagination, or invalid payload
- `500`: infrastructure/model errors such as OpenSearch or local LLM unavailability

Rate limiting is not implemented in this demo. The FastAPI middleware layer is ready for a rate limiting middleware if required.

### 4.2 User Interface

Implemented files:

- `frontend/chat_interface.html`
- `frontend/role_selector.js`
- `frontend/document_viewer.js`

UI capabilities:

- Role selection for Product Lead, Tech Lead, Compliance Lead, and Bank Alliance Lead
- Document upload with document type selection
- Processing/indexing status display
- Chat interface with role-aware answers
- Source citation display
- Chat history support

## Deliverable 1: Working Chatbot System

Required structure is implemented:

```text
document_processor/
  model_fine_tuning.ipynb
  document_classifier.py
  entity_extractor.py
  document_loader.py
vector_db/
  embedding_service.py
  vector_search.py
  knowledge_base.py
chatbot/
  query_router.py
  stakeholder_handler.py
  response_generator.py
api/
  chat_endpoints.py
  document_upload.py
  document_intelligence.py
  auth_middleware.py
frontend/
  chat_interface.html
  role_selector.js
  document_viewer.js
```

## Deliverable 2: Technical Documentation

Architecture, model rationale, vector database schema, stakeholder query handling, API usage, authentication, and error handling are documented in this file and in `README.md`.

## Deliverable 3: Demo Script and Test Cases

### Demo Setup

```bash
cp .env.example .env
docker compose up -d postgres opensearch
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
.venv/bin/alembic upgrade head
python -m server
```

Open:

```text
http://localhost:8000
```

Docker-only run:

```bash
docker compose --profile app up --build
```

### Demo Recording and Screenshots

The demo assets are stored in `data/assets/`:

```text
PaymentDocs_Chatbot.mp4
Screenshot 2026-08-12 at 10.*.png
```

### Demonstration Scenarios

1. Open `http://localhost:8000`.
2. Select `Product Lead`.
3. Upload `data/samples/upi_transaction/paytm_upi_transactions_2026-08.csv`.
4. Ask: "What's the success rate of UPI transactions this month?"
5. Switch to `Tech Lead`.
6. Upload a file from `data/samples/bank_api_log/`.
7. Ask: "What's the average response time for bank APIs by bank?"
8. Switch to `Compliance Lead`.
9. Upload a PDF from `data/samples/compliance_audit/`.
10. Ask: "What compliance obligations are mentioned in recent UPI circulars?"
11. Switch to `Bank Alliance Lead`.
12. Upload a PDF from `data/samples/partnership_sla/`.
13. Ask: "Which SLA clauses mention uptime, escalation, or termination?"
14. Show citations and chat history.
15. Trigger edge cases with an invalid document id and an unsupported file extension.

### Test Dataset

Sample documents are included in `data/samples/`:

- `upi_transaction/`: 8 CSV files
- `bank_api_log/`: 8 JSONL files
- `compliance_audit/`: 46 PDF files
- `partnership_sla/`: 20 PDF files

Total sample documents: 82.

### Predefined Test Queries

| Scenario | Role | Query | Expected Behavior |
| --- | --- | --- | --- |
| UPI success rate | Product Lead | "What's the success rate of UPI transactions this month?" | Uses SQL against `payment_documents_upi_transaction` and returns total/success/rate |
| Popular payment methods | Product Lead | "Which payment methods are most popular?" | Uses SQL grouping over payment mode |
| Region trend | Product Lead | "Show me transaction volume trends by region" | Uses SQL grouping by region/state metadata |
| API failures | Tech Lead | "Are there any API integration failures today?" | Uses SQL filtering by status/severity/date |
| API latency | Tech Lead | "What's the average response time for bank APIs?" | Uses SQL average over `latency_ms` |
| Error patterns | Tech Lead | "Show me error patterns in payment processing" | Uses SQL breakdown and vector context if needed |
| Compliance lookup | Compliance Lead | "Are there obligations related to UPI Circle?" | Uses vector search over compliance PDFs with citations |
| Audit trail | Compliance Lead | "Show me audit trail for high-value transactions" | Uses relevant compliance/audit evidence and transaction context where indexed |
| SLA performance | Bank Alliance Lead | "How is our SLA performance with Bank X?" | Uses SLA search and bank API SQL metrics |
| Agreement violations | Bank Alliance Lead | "Any partnership agreement violations?" | Uses vector search over SLA clauses and available log evidence |
| Invalid role | Any | `X-User-Role: bad_role` | Falls back to Product Lead |
| Empty upload | Any | Upload request without file | Returns `422` |
| Missing document | Any | `GET /api/documents/999999` | Returns `404` |

## Submission Instructions Response

### 1. GitHub Repository

The repository contains the complete codebase, README, sample data, fine-tuning data, Docker setup, API layer, frontend, and this submission document.

### 2. Docker Setup

Docker files included:

- `Dockerfile`
- `docker-compose.yml`

Docker Compose runs:

- FastAPI application
- Postgres
- OpenSearch
- llama.cpp OpenAI-compatible model server

### 3. Demo Video

Demo video:

```text
data/assets/PaymentDocs_Chatbot.mp4
```

### 4. Live Demo

Run the app and open:

```text
http://localhost:8000
```

API docs:

```text
http://localhost:8000/docs
```

## Bonus Features Implemented

- Streaming chat responses using Server-Sent Events
- Conversation memory through persisted chat history
- Role-specific routing with separate tools per stakeholder
- Hybrid semantic and keyword search
- SQL analytics over OpenSearch for exact metrics
- Source citations for vector and SQL-backed answers
