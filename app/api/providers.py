from fastapi import APIRouter

from app.config import settings
from app.services import ollama_client

router = APIRouter(prefix="/providers", tags=["providers"])


@router.get("")
def list_providers() -> dict:
    ollama_available = ollama_client.is_available()
    return {
        "default": settings.default_ml_provider,
        "providers": [
            {
                "id": "cloud",
                "label": "Облако",
                "available": True,
                "mock": settings.mock_yandex,
            },
            {
                "id": "ollama",
                "label": "Локально (Ollama)",
                "available": ollama_available,
                "llm_model": settings.ollama_llm_model,
                "embed_model": settings.ollama_embed_model,
            },
        ],
    }


def semantic_dims(provider: str) -> int:
    if provider == "ollama":
        return settings.ollama_embedding_dims
    return settings.embedding_dims
