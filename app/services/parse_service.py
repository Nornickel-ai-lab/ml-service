from pathlib import Path

from fastapi import HTTPException

from app.schemas.parse import ParseRequest, ParseResponse
from app.services import gigachat_client, mock_parse, ollama_client
from app.services.gigachat_fallback import log_gigachat_fallback, ollama_fallback_enabled
from app.services.llm_json import parse_model_with_retry
from app.services.provider import resolve_provider

PARSE_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "parse.txt"
SYSTEM_PROMPT = "Ты анализируешь технические запросы. Отвечай только валидным JSON."


def parse_query(request: ParseRequest) -> ParseResponse:
    use_llm = request.use_llm if request.use_llm is not None else settings.query_parse_llm
    if not use_llm:
        return mock_parse.parse_mock(request.query)
    provider = resolve_provider(request.provider)
    if provider == "ollama":
        return _parse_ollama(request)
    try:
        gigachat_client.ensure_ready()
        return _parse_gigachat(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"query parse failed: {exc}") from exc
    except RuntimeError as exc:
        if ollama_fallback_enabled():
            log_gigachat_fallback("parse", str(exc))
            return _parse_ollama(request)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _parse_gigachat(request: ParseRequest) -> ParseResponse:
    template = PARSE_PROMPT_PATH.read_text(encoding="utf-8")
    prompt = template.format(query=request.query)

    def generate() -> str:
        return gigachat_client.completion(prompt, temperature=0.1)

    return parse_model_with_retry(generate, ParseResponse)


def _parse_ollama(request: ParseRequest) -> ParseResponse:
    template = PARSE_PROMPT_PATH.read_text(encoding="utf-8")
    user_prompt = template.format(query=request.query)

    def generate() -> str:
        return ollama_client.chat(SYSTEM_PROMPT, user_prompt, temperature=0.1)

    try:
        return parse_model_with_retry(generate, ParseResponse)
    except ValueError:
        return mock_parse.parse_mock(request.query)
