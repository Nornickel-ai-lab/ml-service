from fastapi import APIRouter, HTTPException

from app.schemas.parse import ParseRequest, ParseResponse
from app.schemas.query import SynthesizeRequest, SynthesizeResponse
from app.services import parse_service, synthesize_service

router = APIRouter(prefix="/query", tags=["query"])


@router.post("/parse", response_model=ParseResponse)
def parse_query(request: ParseRequest) -> ParseResponse:
    try:
        return parse_service.parse_query(request)
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/synthesize", response_model=SynthesizeResponse)
def synthesize_answer(request: SynthesizeRequest) -> SynthesizeResponse:
    try:
        return synthesize_service.synthesize(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
