from pathlib import Path

from fastapi import HTTPException

from app.schemas.extract import ExtractRequest, ExtractResponse
from app.services import gigachat_client, mock_extract, ollama_client
from app.services.llm_json import parse_model_with_retry
from app.services.provider import resolve_provider

EXTRACT_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "extract.txt"
SYSTEM_PROMPT = "Извлекай структурированные данные из технических текстов. Отвечай только валидным JSON."
MAX_EXTRACT_CHARS = 6000


def extract(request: ExtractRequest) -> ExtractResponse:
    provider = resolve_provider(request.provider)
    if provider == "ollama":
        return _extract_ollama(request)
    try:
        gigachat_client.ensure_ready()
        return _extract_gigachat(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"extract parse failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _extract_gigachat(request: ExtractRequest) -> ExtractResponse:
    prompt = _build_prompt(request)

    def generate() -> str:
        return gigachat_client.completion(prompt, temperature=0.1)

    return _finalize_response(parse_model_with_retry(generate, ExtractResponse), request)


def _extract_ollama(request: ExtractRequest) -> ExtractResponse:
    user_prompt = _build_prompt(request)

    def generate() -> str:
        return ollama_client.chat(SYSTEM_PROMPT, user_prompt, temperature=0.1)

    try:
        parsed = parse_model_with_retry(generate, ExtractResponse)
    except ValueError:
        return mock_extract.extract_mock(
            text=request.text,
            document_id=request.document_id,
            title=request.title,
        )
    return _finalize_response(parsed, request)


def _build_prompt(request: ExtractRequest) -> str:
    template = EXTRACT_PROMPT_PATH.read_text(encoding="utf-8")
    text = request.text[:MAX_EXTRACT_CHARS]
    return template.format(text=text)


def _finalize_response(parsed: ExtractResponse, request: ExtractRequest) -> ExtractResponse:
    if not request.document_id:
        return parsed
    for index, conclusion in enumerate(parsed.conclusions):
        if not conclusion.id:
            parsed.conclusions[index] = conclusion.model_copy(
                update={"id": f"{request.document_id}_c{index}"},
            )
    return parsed
