from fastapi import APIRouter, HTTPException

from app.schemas.extract import ExtractRequest, ExtractResponse
from app.services import extract_service

router = APIRouter(tags=["extract"])


@router.post("/extract", response_model=ExtractResponse)
def extract_entities(request: ExtractRequest) -> ExtractResponse:
    try:
        return extract_service.extract(request)
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
