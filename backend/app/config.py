import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BASE_DIR / "knowledge_base"
DOCS_DIR = KB_DIR / "documents"
EMBEDDINGS_FILE = KB_DIR / "embeddings.npy"
INDEX_FILE = KB_DIR / "faiss_index.index"
CHUNKS_FILE = KB_DIR / "chunks.json"

# Ensure directories exist
DOCS_DIR.mkdir(parents=True, exist_ok=True)
KB_DIR.mkdir(parents=True, exist_ok=True)

# LLM defaults (align with app.py setup)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://infer.e2enetworks.net/project/p-5915/genai/llama3_1_8b_instruct/v1/")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3_1_8b_instruct")
LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("E2E_API_TOKEN", "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJGSjg2R2NGM2pUYk5MT2NvNE52WmtVQ0lVbWZZQ3FvcXRPUWVNZmJoTmxFIn0.eyJleHAiOjE3ODIxOTIwMjcsImlhdCI6MTc1MDY1NjAyNywianRpIjoiMTdmMWJhZGEtYTYyMS00ZTMwLWJmYTEtZDJhMjYzOTUzMjA4IiwiaXNzIjoiaHR0cDovL2dhdGV3YXkuZTJlbmV0d29ya3MuY29tL2F1dGgvcmVhbG1zL2FwaW1hbiIsImF1ZCI6ImFjY291bnQiLCJzdWIiOiJiMmMzNDZkMC04MmQwLTQxYzItOWVkNS1mNmY4Nzc0MjVlNDkiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJhcGltYW51aSIsInNlc3Npb25fc3RhdGUiOiJmNTc3N2QwMi1lYjE4LTRhYjktYmM0NS03ZjVkNmRjZmFjZmUiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbIiJdLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImFwaXVzZXIiLCJkZWZhdWx0LXJvbGVzLWFwaW1hbiJdfSwicmVzb3VyY2VfYWNjZXNzIjp7ImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoicHJvZmlsZSBlbWFpbCIsInNpZCI6ImY1Nzc3ZDAyLWViMTgtNGFiOS1iYzQ1LTdmNWQ2ZGNmYWNmZSIsImVtYWlsX3ZlcmlmaWVkIjpmYWxzZSwiaXNfcGFydG5lcl9yb2xlIjpmYWxzZSwibmFtZSI6IkF2aW5hc2ggU2luZ2ggIiwicHJpbWFyeV9lbWFpbCI6InN1cGVyYi5zdWppdEBnbWFpbC5jb20iLCJpc19wcmltYXJ5X2NvbnRhY3QiOmZhbHNlLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJhdmluYXNoLnNpbmdoLjI3MDUwNEBnbWFpbC5jb20iLCJnaXZlbl9uYW1lIjoiQXZpbmFzaCIsImZhbWlseV9uYW1lIjoiU2luZ2ggIiwiZW1haWwiOiJhdmluYXNoLnNpbmdoLjI3MDUwNEBnbWFpbC5jb20iLCJpc19pbmRpYWFpX3VzZXIiOmZhbHNlfQ.OL2wpdaRRcANjZY9Mx_DQqyoX1_tDAXwqVVGA5rVIGoTV7Bc1hWEy94L0-C3QLTyFJCn1ROd6pbS7rL1qsCcW_XGicfVYLx-PpYFMcAe4GTpxnI_3hn8a_ohjcy8-H6DQ39wXHrVGap8jXsED6V3OurdOw0S_3u8n5bnMw2CIqs"))

if not LLM_API_KEY:
    # Keep empty; the calling code will raise a clear error when missing
    LLM_API_KEY = ""

# Firebase Realtime Database URL for storing chunks (can be public or secured)
FIREBASE_RTD_DB = os.getenv("FIREBASE_RTD_DB", "https://gesil-chunk-default-rtdb.asia-southeast1.firebasedatabase.app/")
