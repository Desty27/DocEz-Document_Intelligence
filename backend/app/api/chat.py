from fastapi import APIRouter, Depends
from app.models import ChatRequest, ChatResponse, ChatMessage
from app.state import SessionStore, get_store

router = APIRouter(prefix="/chat", tags=["chat"])

def _generate_reply(message: str) -> str:
    # Placeholder reply logic for demo purposes
    return f"Echo: {message}"

@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, store: SessionStore = Depends(get_store)) -> ChatResponse:
    reply = _generate_reply(req.message)

    store.add_message(req.session_id, "user", req.message)
    store.add_message(req.session_id, "assistant", reply)

    history = [ChatMessage(**m) for m in store.get_history(req.session_id)]

    return ChatResponse(
        session_id=req.session_id,
        reply=reply,
        history=history,
        sources=[],
    )

@router.get("/history", response_model=list[ChatMessage])
def get_history(session_id: str, store: SessionStore = Depends(get_store)) -> list[ChatMessage]:
    return [ChatMessage(**m) for m in store.get_history(session_id)]
