from fastapi import APIRouter

router = APIRouter(prefix="/ocr", tags=["ocr"])


@router.get("/health")
def ocr_health() -> dict[str, str | bool]:
    return {"status": "ok", "model_loaded": False}


@router.post("/parse")
def parse_stub() -> dict[str, str]:
    return {"status": "not_implemented"}
