from fastapi import APIRouter

from app.schemas.extract import ExtractRequest, ExtractResponse
from app.services import extract_service

router = APIRouter(tags=["extract"])


@router.post("/extract", response_model=ExtractResponse)
def extract_entities(request: ExtractRequest) -> ExtractResponse:
    return extract_service.extract(request)
