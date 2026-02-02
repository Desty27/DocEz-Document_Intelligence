# Demo Chat Widget Frontend

A minimal HTML/JS page to verify the chat backend.

## Run locally

```bash
cd frontend
python -m http.server 8080
```

Open http://localhost:8080 and:

- For the basic tester (index.html), set API base to http://localhost:8000/api/v1 and use the buttons to send chat/health.
- For the Odisha demo (odisha-demo.html), click the floating chat bubble and ask questions; it calls the RAG demo endpoint at /api/v1/chat/rag.
- For admin uploads (admin.html), point to http://localhost:8000/api/v1, upload PDFs/CSVs/Excels/TXT, and check KB status.
