from fastapi import APIRouter, HTTPException

from app.schemas.embed import EmbedRequest, EmbedResponse
from app.services import embed_service

router = APIRouter(tags=["embed"])


@router.post("/embed", response_model=EmbedResponse)
def embed_texts(request: EmbedRequest) -> EmbedResponse:
    try:
        return embed_service.embed(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
