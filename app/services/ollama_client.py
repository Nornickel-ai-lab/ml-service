import httpx

from app.config import settings

BATCH_SIZE = 32


def _base_url() -> str:
    return settings.ollama_base_url.rstrip("/")


def is_available() -> bool:
    if not settings.ollama_enabled or not settings.ollama_base_url:
        return False
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{_base_url()}/api/tags")
            return response.status_code == 200
    except httpx.HTTPError:
        return False


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    with httpx.Client(timeout=300.0) as client:
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            response = client.post(
                f"{_base_url()}/api/embed",
                json={"model": settings.ollama_embed_model, "input": batch},
            )
            response.raise_for_status()
            vectors.extend(response.json()["embeddings"])
    return vectors


def completion(prompt: str, temperature: float = 0.2) -> str:
    return chat("", prompt, temperature=temperature)


def chat(system: str, user: str, temperature: float = 0.2) -> str:
    messages: list[dict[str, str]] = []
    if system.strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.append({"role": "user", "content": user})
    with httpx.Client(timeout=600.0) as client:
        response = client.post(
            f"{_base_url()}/api/chat",
            json={
                "model": settings.ollama_llm_model,
                "messages": messages,
                "stream": False,
                "keep_alive": "10m",
                "options": {
                    "temperature": temperature,
                    "num_predict": settings.ollama_num_predict,
                    "num_ctx": settings.ollama_num_ctx,
                },
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
