from fastapi import FastAPI

from app.api.contradictions import router as contradictions_router
from app.api.embed import router as embed_router
from app.api.extract import router as extract_router
from app.api.providers import router as providers_router
from app.api.query import router as query_router
from app.config import settings
from app.api.rerank import router as rerank_router
from app.ocr.router import router as ocr_router
from app.services import gigachat_client, ocr_service, rerank_service
from app.services import ollama_client

app = FastAPI(title="rdmap-ml-service", version="0.1.0")

app.include_router(providers_router)
app.include_router(extract_router)
app.include_router(embed_router)
app.include_router(contradictions_router)
app.include_router(query_router)
app.include_router(ocr_router)
app.include_router(rerank_router)


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "gigachat_configured": gigachat_client.credentials_configured(),
        "default_ml_provider": settings.default_ml_provider,
        "ollama_available": ollama_client.is_available(),
        "ollama_llm_model": settings.ollama_llm_model,
        "ollama_embed_model": settings.ollama_embed_model,
        "gigachat_llm_model": settings.yandex_llm_model,
        "gigachat_embed_model": settings.yandex_embed_doc_model,
        "ocr_ready": ocr_service.model_loaded() if settings.ocr_enabled else False,
        "ocr_enabled": settings.ocr_enabled,
        "rerank_ready": rerank_service.model_loaded() if settings.rerank_enabled else False,
        "rerank_model": settings.rerank_model,
        "rerank_enabled": settings.rerank_enabled,
        "ollama_enabled": settings.ollama_enabled,
    }
