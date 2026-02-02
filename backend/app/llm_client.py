from typing import Optional
import requests
from app import config


def complete(prompt: str, temperature: float = 0.4, max_tokens: int = 400) -> str:
    if not config.LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY (or E2E_API_TOKEN) is not set.")

    base = config.LLM_BASE_URL.rstrip("/")
    url = f"{base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
    except Exception as exc:
        raise RuntimeError(f"LLM request failed: {exc}")

    if resp.status_code != 200:
        raise RuntimeError(f"LLM returned {resp.status_code}: {resp.text}")

    try:
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(f"Failed to parse LLM response JSON: {exc}")

    # Expecting OpenAI-like response shape
    try:
        choice = data.get("choices", [])[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        content = None
        if message:
            content = message.get("content")
        if not content:
            content = choice.get("text") if isinstance(choice, dict) else None
        return (content or "").strip()
    except Exception as exc:
        raise RuntimeError(f"Unexpected LLM response shape: {exc}")
