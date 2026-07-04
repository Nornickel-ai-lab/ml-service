from fastapi import FastAPI

from app.api.embed import router as embed_router
from app.api.query import router as query_router
from app.config import settings
from app.ocr.router import router as ocr_router
from app.services import ocr_service
from app.services.yandex_client import credentials_configured

app = FastAPI(title="rdmap-ml-service", version="0.1.0")

app.include_router(embed_router)
app.include_router(query_router)
app.include_router(ocr_router)


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "yandex_configured": credentials_configured(),
        "mock_yandex": settings.mock_yandex,
        "ocr_ready": ocr_service.model_loaded(),
    }
