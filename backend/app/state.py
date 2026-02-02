from typing import Dict, List

class SessionStore:
    """Very small in-memory store for chat history. Suitable for demos only."""

    def __init__(self) -> None:
        self._store: Dict[str, List[Dict[str, str]]] = {}

    def add_message(self, session_id: str, role: str, content: str) -> None:
        history = self._store.setdefault(session_id, [])
        history.append({"role": role, "content": content})

    def get_history(self, session_id: str, limit: int = 20) -> List[Dict[str, str]]:
        history = self._store.get(session_id, [])
        return history[-limit:]

store = SessionStore()


def get_store() -> SessionStore:
    return store
