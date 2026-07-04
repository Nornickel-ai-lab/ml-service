from pathlib import Path

from fastapi import HTTPException

from app.config import settings
from app.schemas.extract import ExtractRequest, ExtractResponse
from app.schemas.parse import ParseRequest, ParseResponse
from app.services import mock_extract, mock_parse, ollama_client, yandex_client
from app.services.llm_json import parse_model_with_retry
from app.services.provider import resolve_provider

EXTRACT_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "extract.txt"
PARSE_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "parse.txt"
MAX_EXTRACT_CHARS = 6000


def extract(request: ExtractRequest) -> ExtractResponse:
    if settings.mock_yandex:
        return mock_extract.extract_mock(
            text=request.text,
            document_id=request.document_id,
            title=request.title,
        )
    try:
        return _extract_yandex(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"extract parse failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _extract_yandex(request: ExtractRequest) -> ExtractResponse:
    template = EXTRACT_PROMPT_PATH.read_text(encoding="utf-8")
    text = request.text[:MAX_EXTRACT_CHARS]
    prompt = template.format(text=text)

    def generate() -> str:
        return yandex_client.completion(prompt, temperature=0.1)

    parsed = parse_model_with_retry(generate, ExtractResponse)
    if request.document_id:
        for index, conclusion in enumerate(parsed.conclusions):
            if not conclusion.id:
                parsed.conclusions[index] = conclusion.model_copy(
                    update={"id": f"{request.document_id}_c{index}"},
                )
    return parsed
