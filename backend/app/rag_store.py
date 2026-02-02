import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from app import config


class KnowledgeBase:
    """Lightweight RAG store that mirrors the Streamlit app behavior."""

    def __init__(self) -> None:
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.index = None
        self.chunks: List[Dict] = []
        self._load_from_disk()

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        # Improved chunking: split into sentence-like boundaries and assemble chunks
        if not text:
            return []

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Split on sentence boundaries (simple heuristic)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks: List[str] = []
        current = []
        current_len = 0

        for sent in sentences:
            s = sent.strip()
            if not s:
                continue
            s_len = len(s)
            if current_len + s_len <= chunk_size or not current:
                current.append(s)
                current_len += s_len + 1
            else:
                chunks.append(" ".join(current))
                # start new chunk but include overlap from last sentence(s)
                if overlap > 0:
                    # keep last sentence as overlap if it's shorter than overlap
                    overlap_buf = []
                    buf_len = 0
                    for part in reversed(current):
                        overlap_buf.insert(0, part)
                        buf_len += len(part) + 1
                        if buf_len >= overlap:
                            break
                    current = overlap_buf[:]  # type: ignore
                    current_len = sum(len(p) + 1 for p in current)
                else:
                    current = []
                    current_len = 0
                current.append(s)
                current_len += s_len + 1

        if current:
            chunks.append(" ".join(current))

        return chunks

    @staticmethod
    def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(how="all")
        df = df.fillna("")
        return df

    def _extract_pdf(self, path: Path) -> str:
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)

    def _extract_excel(self, path: Path) -> str:
        xl = pd.ExcelFile(path)
        parts = []
        for sheet in xl.sheet_names:
            df = self._clean_df(pd.read_excel(path, sheet_name=sheet))
            if df.empty:
                continue
            parts.append(f"Sheet: {sheet}\nColumns: {', '.join(df.columns.astype(str))}\n")
            for _, row in df.iterrows():
                row_items = [f"{col}: {val}" for col, val in row.items() if str(val).strip()]
                if row_items:
                    parts.append(" | ".join(row_items))
        return "\n".join(parts)

    def _extract_csv(self, path: Path) -> str:
        df = self._clean_df(pd.read_csv(path))
        if df.empty:
            return ""
        parts = [f"Columns: {', '.join(df.columns.astype(str))}"]
        for _, row in df.iterrows():
            row_items = [f"{col}: {val}" for col, val in row.items() if str(val).strip()]
            if row_items:
                parts.append(" | ".join(row_items))
        return "\n".join(parts)

    def _ingest_file(self, path: Path) -> List[Dict]:
        ext = path.suffix.lower()
        file_type = ext.lstrip(".") if ext else "unknown"
        text = ""
        if ext == ".pdf":
            text = self._extract_pdf(path)
        elif ext in {".xlsx", ".xls"}:
            text = self._extract_excel(path)
        elif ext == ".csv":
            text = self._extract_csv(path)
        else:
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                text = path.read_text(errors="ignore")

        chunks = []
        if text:
            for chunk in self._chunk_text(text, chunk_size=600, overlap=120):
                chunks.append({"filename": path.name, "chunk": chunk, "file_type": file_type})
        return chunks

    def rebuild_documents(self, data_dir: Path) -> Tuple[int, int]:
        data_dir.mkdir(parents=True, exist_ok=True)
        files = [p for p in data_dir.iterdir() if p.is_file()]
        all_chunks: List[Dict] = []

        for path in files:
            try:
                all_chunks.extend(self._ingest_file(path))
            except Exception as exc:
                print(f"Failed to ingest {path.name}: {exc}")

        # Replace current chunks entirely
        self.chunks = all_chunks

        # Push full list to Firebase
        try:
            fb_url = config.FIREBASE_RTD_DB.rstrip("/") + "/chunks.json"
            requests.put(fb_url, json=self.chunks, timeout=10)
        except Exception as exc:
            print(f"Warning: failed to sync chunks to Firebase: {exc}")

        if not self.chunks:
            # Clear local artifacts if empty
            try:
                if config.EMBEDDINGS_FILE.exists():
                    config.EMBEDDINGS_FILE.unlink()
                if config.INDEX_FILE.exists():
                    config.INDEX_FILE.unlink()
                with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
                    json.dump([], f)
            except Exception as exc:
                print(f"Warning: failed to clear local artifacts: {exc}")
            self.index = None
            return 0, 0

        # Build embeddings and index from scratch
        embeddings = self.model.encode([c["chunk"] for c in self.chunks])
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.float32(embeddings))

        try:
            np.save(config.EMBEDDINGS_FILE, embeddings)
            with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)
            faiss.write_index(self.index, str(config.INDEX_FILE))
        except Exception as exc:
            print(f"Warning: failed to persist rebuilt artifacts: {exc}")

        return len(self.chunks), len(self.chunks)

    def delete_file(self, filename: str) -> Tuple[int, int]:
        target = config.DOCS_DIR / Path(filename).name
        if target.exists() and target.is_file():
            try:
                target.unlink()
            except Exception as exc:
                print(f"Warning: failed to delete file {target}: {exc}")

        # Rebuild from remaining files
        return self.rebuild_documents(config.DOCS_DIR)

    def delete_all(self) -> Tuple[int, int]:
        # Remove all files on disk
        try:
            for p in config.DOCS_DIR.glob("*"):
                if p.is_file():
                    p.unlink()
        except Exception as exc:
            print(f"Warning: failed to delete all files: {exc}")

        self.chunks = []
        self.index = None
        try:
            if config.EMBEDDINGS_FILE.exists():
                config.EMBEDDINGS_FILE.unlink()
            if config.INDEX_FILE.exists():
                config.INDEX_FILE.unlink()
            with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
        except Exception as exc:
            print(f"Warning: failed to clear local artifacts: {exc}")

        # Push empty to Firebase
        try:
            fb_url = config.FIREBASE_RTD_DB.rstrip("/") + "/chunks.json"
            requests.put(fb_url, json=[], timeout=10)
        except Exception as exc:
            print(f"Warning: failed to sync empty chunks to Firebase: {exc}")

        return 0, 0

    def ingest_documents(self, data_dir: Path) -> Tuple[int, int]:
        data_dir.mkdir(parents=True, exist_ok=True)
        files = [p for p in data_dir.iterdir() if p.is_file()]
        existing_files = {c.get("filename", "") for c in self.chunks}
        new_chunks: List[Dict] = []

        for path in files:
            if path.name in existing_files:
                continue
            try:
                new_chunks.extend(self._ingest_file(path))
            except Exception as exc:
                print(f"Failed to ingest {path.name}: {exc}")

        if not new_chunks:
            return 0, 0

        # Persist new chunks to Firebase RTDB (append semantics)
        try:
            fb_url = config.FIREBASE_RTD_DB.rstrip("/") + "/chunks.json"
            resp = requests.get(fb_url, timeout=10)
            if resp.status_code == 200 and resp.content:
                try:
                    remote = resp.json() or []
                except Exception:
                    remote = []
            else:
                remote = []

            if isinstance(remote, dict):
                # if stored as map, convert to list
                remote = list(remote.values())

            combined = remote + new_chunks
            # push combined list back
            requests.put(fb_url, json=combined, timeout=10)
            # Update local chunks list
            self.chunks = combined
        except Exception as exc:
            print(f"Warning: failed to sync chunks to Firebase: {exc}")
            # fall back to local storage append
            self.chunks.extend(new_chunks)

        # Build embeddings for new chunks and update index
        embeddings = self.model.encode([c["chunk"] for c in new_chunks])

        if self.chunks and config.EMBEDDINGS_FILE.exists():
            # Append to existing local embeddings and index
            try:
                all_embeddings = np.load(config.EMBEDDINGS_FILE)
                all_embeddings = np.vstack([all_embeddings, embeddings])
                np.save(config.EMBEDDINGS_FILE, all_embeddings)
                if self.index is None and config.INDEX_FILE.exists():
                    self.index = faiss.read_index(str(config.INDEX_FILE))
                if self.index is not None:
                    self.index.add(np.float32(embeddings))
            except Exception:
                # If anything fails, rebuild from scratch below
                self.chunks = self.chunks  # keep chunks
                embeddings = self.model.encode([c["chunk"] for c in self.chunks])
                self.index = faiss.IndexFlatL2(embeddings.shape[1])
                self.index.add(np.float32(embeddings))
                np.save(config.EMBEDDINGS_FILE, embeddings)
        else:
            # Fresh index from all chunks
            all_chunks = self.chunks
            all_embeddings = self.model.encode([c["chunk"] for c in all_chunks])
            self.index = faiss.IndexFlatL2(all_embeddings.shape[1])
            self.index.add(np.float32(all_embeddings))
            np.save(config.EMBEDDINGS_FILE, all_embeddings)

        # Save local chunks copy
        try:
            with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"Warning: failed to write local chunks file: {exc}")

        try:
            faiss.write_index(self.index, str(config.INDEX_FILE))
        except Exception as exc:
            print(f"Warning: failed to write faiss index: {exc}")

        return len(new_chunks), len(self.chunks)

    def _load_from_disk(self) -> None:
        try:
            # Try local chunks file first
            if config.CHUNKS_FILE.exists():
                with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
            else:
                # Fallback to Firebase RTDB
                try:
                    fb_url = config.FIREBASE_RTD_DB.rstrip("/") + "/chunks.json"
                    resp = requests.get(fb_url, timeout=10)
                    if resp.status_code == 200 and resp.content:
                        data = resp.json()
                        if isinstance(data, dict):
                            # convert map to list
                            data = list(data.values())
                        self.chunks = data or []
                        # write local copy
                        with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
                            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
                except Exception as exc:
                    print(f"Warning: could not fetch chunks from Firebase: {exc}")

            if config.INDEX_FILE.exists() and config.EMBEDDINGS_FILE.exists():
                try:
                    self.index = faiss.read_index(str(config.INDEX_FILE))
                except Exception as exc:
                    print(f"Warning: could not load faiss index: {exc}")
                    # rebuild index from embeddings file if possible
                    try:
                        all_embeddings = np.load(config.EMBEDDINGS_FILE)
                        self.index = faiss.IndexFlatL2(all_embeddings.shape[1])
                        self.index.add(np.float32(all_embeddings))
                    except Exception:
                        self.index = None
        except Exception as exc:
            print(f"Warning: could not load KB from disk: {exc}")

    def query(self, text: str, top_k: int = 5) -> Tuple[str, List[str]]:
        if not self.chunks or self.index is None:
            return "Knowledge base is empty. Please upload documents first.", []

        query_vec = self.model.encode([text])
        scores, idx = self.index.search(np.float32(query_vec), k=min(top_k, len(self.chunks)))
        selected = [self.chunks[i] for i in idx[0] if i < len(self.chunks)]

        # Build a source-prefixed context so the LLM can cite file names
        parts = []
        for s in selected:
            parts.append(f"Source: {s.get('filename', 'unknown')}\n---\n{s.get('chunk', '')}")

        context = "\n\n".join(parts)
        sources = list({c["filename"] for c in selected})

        prompt = (
            "You are an assistant that should PREFER using the provided context to answer.\n"
            "Cite filenames when using their content. If the answer is not present in the context, say:\n"
            "'I don't know based on the provided documents.' then offer to help search or rephrase.\n\n"
            f"Context:\n{context}\n\nQuestion: {text}\nAnswer:"
        )

        return prompt, sources


kb = KnowledgeBase()


def get_kb() -> KnowledgeBase:
    return kb
