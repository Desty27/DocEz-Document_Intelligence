import json
from typing import List, Optional
from fastapi import APIRouter
from app.models import ChatRequest, ChatResponse, ChatMessage, ChatFullRequest
from app.rag_store import get_kb
from app.llm_client import complete

router = APIRouter(prefix="/chat/rag", tags=["rag"])


_GREETINGS = {"hi", "hello", "hey", "namaskar", "namaste", "good morning", "good afternoon", "good evening"}


def _format_history(history: Optional[List[ChatMessage]]) -> str:
    if not history:
        return "(no prior conversation)"
    parts = []
    for turn in history[-5:]:  # keep recent few for brevity
        parts.append(f"{turn.role}: {turn.content}")
    return "\n".join(parts)


@router.post("", response_model=ChatResponse)
def rag_chat(req: ChatRequest) -> ChatResponse:
    # Simple intent handling: respond to short greetings without calling LLM
    text = (req.message or "").strip()
    low = text.lower()
    if low in _GREETINGS or (len(text) <= 3 and any(c.isalpha() for c in text)):
        reply = "Namaskar! How can I help you today? Ask about citizen services, helplines, or tourism in Odisha."
        history = [
            ChatMessage(role="user", content=req.message),
            ChatMessage(role="assistant", content=reply),
        ]
        return ChatResponse(session_id=req.session_id, reply=reply, history=history, sources=[])

    kb = get_kb()
    # Request more candidates to improve recall
    prompt, sources = kb.query(req.message, top_k=10)
    reply = complete(prompt)

    history = [
        ChatMessage(role="user", content=req.message),
        ChatMessage(role="assistant", content=reply),
    ]

    return ChatResponse(
        session_id=req.session_id,
        reply=reply,
        history=history,
        sources=sources,
    )


@router.post("/full", response_model=ChatResponse)
def rag_chat_full(req: ChatFullRequest) -> ChatResponse:
    text = (req.message or "").strip()
    low = text.lower()
    if low in _GREETINGS or (len(text) <= 3 and any(c.isalpha() for c in text)):
        reply = "Namaskar! How can I help you today? Ask about citizen services, helplines, or tourism in Odisha."
        history = [ChatMessage(role="user", content=req.message), ChatMessage(role="assistant", content=reply)]
        return ChatResponse(session_id=req.session_id, reply=reply, history=history, sources=[], followups=[])

    kb = get_kb()
    prompt, sources = kb.query(req.message, top_k=10)

    history_str = _format_history(req.history)
    context_only = prompt.split("Context:\n", 1)[-1]
    rich_prompt = (
        "You are an Odisha assistant. Use ONLY the provided context and recent conversation to answer.\n"
        "If the answer is not present, say 'I don't know based on the provided documents.' and offer to help rephrase.\n"
        "Then suggest up to 3 concise follow-up questions.\n"
        "Return a JSON object with keys 'answer' (string) and 'followups' (array of strings).\n\n"
        f"Context:\n{context_only}\n\n"
        f"Conversation:\n{history_str}\n\n"
        f"User question: {text}\n"
        "Answer JSON:" 
    )

    raw = complete(rich_prompt)
    answer = raw
    followups: list[str] = []

    try:
        parsed = json.loads(raw)
        answer = parsed.get("answer", answer)
        followups = parsed.get("followups", []) or []
    except Exception:
        # Try to salvage followups if model returned plain text with bullets
        lines = [ln.strip("- ") for ln in raw.splitlines() if ln.strip()]
        if lines:
            answer = lines[0]
            followups = lines[1:4]

    history = [ChatMessage(role="user", content=req.message), ChatMessage(role="assistant", content=answer)]

    return ChatResponse(
        session_id=req.session_id,
        reply=answer,
        history=history,
        sources=sources,
        followups=followups,
    )
