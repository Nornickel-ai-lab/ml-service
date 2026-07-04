from fastapi import APIRouter

from app.config import settings
from app.services import cloud_ml, ollama_client
from app.services import yandex_client

router = APIRouter(prefix="/providers", tags=["providers"])


@router.get("")
def list_providers() -> dict:
    ollama_available = ollama_client.is_available()
    yandex_ready = yandex_client.credentials_configured()
    cloud_available = yandex_ready or cloud_ml.cloud_uses_mock()
    return {
        "default": settings.default_ml_provider,
        "providers": [
            {
                "id": "cloud",
                "label": "Yandex API",
                "available": cloud_available,
                "yandex_configured": yandex_ready,
                "mock_fallback": cloud_ml.cloud_uses_mock(),
                "llm_model": settings.yandex_model,
                "embed_doc_model": settings.yandex_embed_doc_model,
                "embed_query_model": settings.yandex_embed_query_model,
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
    return settings.embedding_dims
