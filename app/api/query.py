from fastapi import APIRouter, HTTPException

from app.schemas.query import SynthesizeRequest, SynthesizeResponse
from app.services import synthesize_service

router = APIRouter(prefix="/query", tags=["query"])


@router.post("/synthesize", response_model=SynthesizeResponse)
def synthesize_answer(request: SynthesizeRequest) -> SynthesizeResponse:
    try:
        return synthesize_service.synthesize(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
