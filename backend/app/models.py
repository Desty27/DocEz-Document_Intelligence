from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: str = Field(examples=["user", "assistant"])
    content: str

class ChatRequest(BaseModel):
    session_id: str = Field(description="Client-generated session identifier")
    message: str
    context: Optional[List[str]] = None

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    history: List[ChatMessage]
    sources: List[str] = Field(default_factory=list, description="Placeholder for citations")
    followups: List[str] = Field(default_factory=list, description="Suggested follow-up questions")


class ChatFullRequest(BaseModel):
    session_id: str = Field(description="Client-generated session identifier")
    message: str
    history: Optional[List[ChatMessage]] = Field(default=None, description="Optional recent conversation history")

class HealthResponse(BaseModel):
    status: str = "ok"


class FileStat(BaseModel):
    name: str
    chunks: int
    file_type: str = "unknown"


class AdminStatus(BaseModel):
    documents_dir: str
    chunk_count: int
    index_present: int
    file_count: int = 0
    type_counts: Dict[str, int] = Field(default_factory=dict)
    files: List[FileStat] = Field(default_factory=list)
    processing: bool = False
    status_text: str = ""


class UploadResponse(BaseModel):
    saved: str
    new_chunks: int
    total_chunks: int
