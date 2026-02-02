from collections import Counter
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, UploadFile, File, HTTPException

from app import config
from app.rag_store import get_kb
from app.models import AdminStatus, UploadResponse, FileStat

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/status", response_model=AdminStatus)
def kb_status() -> AdminStatus:
    kb = get_kb()
    chunk_count = len(kb.chunks)

    file_chunks: Dict[str, int] = {}
    type_counts = Counter()
    files: list[FileStat] = []

    for c in kb.chunks:
        fname = c.get("filename", "unknown")
        file_type = c.get("file_type") or Path(fname).suffix.lstrip(".") or "unknown"
        file_chunks[fname] = file_chunks.get(fname, 0) + 1
        type_counts[file_type] += 1

    for fname, cnt in file_chunks.items():
        ft = Path(fname).suffix.lstrip(".") or "unknown"
        files.append(FileStat(name=fname, chunks=cnt, file_type=ft))

    file_count = len(file_chunks)
    status_text = "Ready - {files} files, {chunks} chunks".format(files=file_count, chunks=chunk_count)

    return AdminStatus(
        documents_dir=str(config.DOCS_DIR),
        chunk_count=chunk_count,
        index_present=1 if config.INDEX_FILE.exists() else 0,
        file_count=file_count,
        type_counts=dict(type_counts),
        files=files,
        processing=False,
        status_text=status_text,
    )


@router.post("/upload")
def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".xlsx", ".xls", ".csv", ".txt"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    dest = config.DOCS_DIR / file.filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        f.write(file.file.read())

    kb = get_kb()
    new_chunks, total_chunks = kb.ingest_documents(config.DOCS_DIR)

    return UploadResponse(
        saved=dest.name,
        new_chunks=int(new_chunks),
        total_chunks=int(total_chunks),
    )


@router.post("/rebuild", response_model=AdminStatus)
def rebuild_index() -> AdminStatus:
    kb = get_kb()
    kb.rebuild_documents(config.DOCS_DIR)
    return kb_status()


@router.delete("/file/{filename}", response_model=AdminStatus)
def delete_file(filename: str) -> AdminStatus:
    if not filename:
        raise HTTPException(status_code=400, detail="Filename required")
    kb = get_kb()
    exists = any((c.get("filename") == filename) for c in kb.chunks) or (config.DOCS_DIR / filename).exists()
    if not exists:
        raise HTTPException(status_code=404, detail="File not found")
    kb.delete_file(filename)
    return kb_status()


@router.delete("/all", response_model=AdminStatus)
def delete_all() -> AdminStatus:
    kb = get_kb()
    kb.delete_all()
    return kb_status()
