from fastapi import APIRouter, HTTPException

from app.schemas.rerank import RerankRequest, RerankResponse
from app.services import rerank_service

router = APIRouter(prefix="/rerank", tags=["rerank"])


@router.post("", response_model=RerankResponse)
def rerank_passages(request: RerankRequest) -> RerankResponse:
    try:
        return rerank_service.rerank(request)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"rerank failed: {exc}") from exc
