from fastapi import APIRouter

from app.config import settings
from app.services import gigachat_client, ollama_client

router = APIRouter(prefix="/providers", tags=["providers"])


@router.get("")
def list_providers() -> dict:
    ollama_available = ollama_client.is_available()
    gigachat_ready = gigachat_client.is_available()
    return {
        "default": settings.default_ml_provider if settings.default_ml_provider != "cloud" else "gigachat",
        "providers": [
            {
                "id": "gigachat",
                "label": "GigaChat API",
                "available": gigachat_ready,
                "gigachat_configured": gigachat_ready,
                "llm_model": settings.gigachat_llm_model,
                "embed_model": settings.gigachat_embed_model,
            },
            {
                "id": "ollama",
                "label": "Ollama",
                "available": ollama_available,
                "llm_model": settings.ollama_llm_model,
                "embed_model": settings.ollama_embed_model,
            },
        ],
    }


def semantic_dims(provider: str) -> int:
    if provider == "ollama":
        return settings.ollama_embedding_dims
    return settings.gigachat_embedding_dims
