from fastapi import APIRouter

from app.schemas.contradictions import CompareRequest, CompareResponse
from app.services import contradiction_service

router = APIRouter(prefix="/contradictions", tags=["contradictions"])


@router.post("/compare", response_model=CompareResponse)
def compare_contradictions(request: CompareRequest) -> CompareResponse:
    return contradiction_service.compare(request)
