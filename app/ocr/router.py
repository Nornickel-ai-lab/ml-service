from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.ocr import OcrParseResponse
from app.services import ocr_service

router = APIRouter(prefix="/ocr", tags=["ocr"])


@router.get("/health")
def ocr_health() -> dict[str, str | bool]:
    return {"status": "ok", "model_loaded": ocr_service.model_loaded()}


@router.post("/parse", response_model=OcrParseResponse)
async def parse_pdf(file: UploadFile) -> OcrParseResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="pdf required")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="empty file")
    try:
        result = ocr_service.parse_pdf_bytes(data)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="ocr failed") from exc
    pages = result["pages"]
    return OcrParseResponse(pages=pages, full_text=result["full_text"])
