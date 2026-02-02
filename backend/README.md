# Demo Chat Backend (FastAPI)

A beginner-friendly FastAPI service for the chat widget demo, now with a simple RAG knowledge base and admin upload API.

## Quick start (local)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open docs at http://localhost:8000/docs and test:

- GET /api/v1/health
- POST /api/v1/chat with body: `{ "session_id": "demo", "message": "Hi" }`

## Docker

```bash
docker build -t demo-chat-backend -f backend/Dockerfile .
docker run -p 8000:8000 demo-chat-backend
```

## RAG + Admin endpoints

- `POST /api/v1/admin/upload` (multipart file) — saves to `backend/knowledge_base/documents` and indexes.
- `GET /api/v1/admin/status` — shows chunk and index status.
- `POST /api/v1/chat/rag` — answers using indexed documents and LLM (LLM_API_KEY / E2E_API_TOKEN required).

Set environment variables:
- `LLM_API_KEY` (or `E2E_API_TOKEN`) — required
- `LLM_BASE_URL` (default matches app.py E2E endpoint)
- `LLM_MODEL` (default `llama3_1_8b_instruct`)
