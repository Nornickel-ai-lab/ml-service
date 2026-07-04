from pathlib import Path
import re

from fastapi import HTTPException

from app.config import settings
from app.schemas.query import PassageInput, SourceOutput, SynthesizeRequest, SynthesizeResponse
from app.services import gigachat_client, ollama_client
from app.services.provider import resolve_provider

from app.services.relevance import is_insufficient_answer, is_weak_retrieval

PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "synthesize.txt"
SYSTEM_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "synthesize_system.txt"

META_MARKERS = (
    "фрагменты содержат",
    "разных языках",
    "формат",
    "приветств",
    "благодарност",
    "не комментиру",
    "если данных мало",
)


def synthesize(request: SynthesizeRequest) -> SynthesizeResponse:
    request = request.model_copy(update={"passages": _merge_passages_by_document(request.passages)})
    weak, relevance = is_weak_retrieval(request.query, request.passages)
    if weak:
        return _weak_retrieval_response(request, relevance)
    provider = resolve_provider(request.provider)
    if provider == "ollama":
        return _synthesize_ollama(request)
    try:
        gigachat_client.ensure_ready()
        return _synthesize_gigachat(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _merge_passages_by_document(passages: list[PassageInput]) -> list[PassageInput]:
    merged: dict[str, PassageInput] = {}
    order: list[str] = []
    for passage in passages:
        doc_id = passage.document_id
        chunk = passage.chunk_text.strip()
        if not doc_id:
            continue
        if doc_id not in merged:
            merged[doc_id] = passage.model_copy(update={"chunk_text": chunk})
            order.append(doc_id)
            continue
        existing = merged[doc_id]
        text = existing.chunk_text
        if chunk and chunk not in text:
            text = f"{text}\n\n{chunk}" if text else chunk
        score = passage.score
        if existing.score is not None and score is not None:
            score = max(existing.score, score)
        elif existing.score is not None:
            score = existing.score
        merged[doc_id] = existing.model_copy(update={"chunk_text": text, "score": score})
    return [merged[doc_id] for doc_id in order]


def _weak_retrieval_response(request: SynthesizeRequest, relevance: float) -> SynthesizeResponse:
    lines = [
        "**В базе нет достаточно близких материалов по запросу**",
        "",
        "Уточните вопрос: укажите объект, процесс или параметры из ваших документов.",
        "Например добавьте название фабрики, тип воды или целевой показатель.",
    ]
    confidence = max(0.15, min(0.35, relevance * 0.6))
    return SynthesizeResponse(answer_md="\n".join(lines), sources=[], confidence=confidence)


def _synthesize_gigachat(request: SynthesizeRequest) -> SynthesizeResponse:
    prompt = _build_prompt(request, max_passages=len(request.passages), max_chars=1200)
    answer = gigachat_client.completion(prompt, temperature=0.1)
    if is_insufficient_answer(answer):
        return SynthesizeResponse(answer_md=answer, sources=[], confidence=0.28)
    sources = [_to_source(passage, index) for index, passage in enumerate(request.passages)]
    confidence = 0.72
    return SynthesizeResponse(answer_md=answer, sources=sources, confidence=confidence)


def _synthesize_ollama(request: SynthesizeRequest) -> SynthesizeResponse:
    passages = request.passages[: settings.ollama_max_passages]
    if not passages:
        return SynthesizeResponse(
            answer_md="Нет релевантных фрагментов в индексе",
            sources=[],
            confidence=0.2,
        )
    cleaned = [_clean_passage(passage) for passage in passages]
    user_prompt = _build_user_prompt(request.query, cleaned, settings.ollama_passage_chars)
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    answer = ollama_client.chat(system_prompt, user_prompt, temperature=0.1)
    if _looks_like_meta(answer):
        answer = _extractive_answer(request.query, cleaned)
        confidence = 0.55
    elif is_insufficient_answer(answer):
        confidence = 0.28
    else:
        confidence = 0.72
    sources = []
    if confidence >= 0.45:
        sources = [_to_source(passage, index) for index, passage in enumerate(cleaned)]
    return SynthesizeResponse(answer_md=answer, sources=sources, confidence=confidence)


def _build_user_prompt(query: str, passages: list[PassageInput], max_chars: int) -> str:
    blocks = []
    for index, passage in enumerate(passages, start=1):
        blocks.append(f"[{index}] {passage.title}\n{passage.chunk_text[:max_chars]}")
    return f"Вопрос:\n{query}\n\nФрагменты:\n" + ("\n\n".join(blocks) if blocks else "нет данных")


def _clean_passage(passage: PassageInput) -> PassageInput:
    text = passage.chunk_text
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    return PassageInput(
        document_id=passage.document_id,
        title=passage.title,
        chunk_text=text.strip(),
        geo=passage.geo,
        year=passage.year,
    )


def _looks_like_meta(answer: str) -> bool:
    lowered = answer.lower().strip()
    if len(lowered) < 40:
        return True
    return any(marker in lowered for marker in META_MARKERS)


def _extractive_answer(query: str, passages: list[PassageInput]) -> str:
    lines = [f"**{query}**", ""]
    lines.append("По найденным фрагментам:")
    for index, passage in enumerate(passages, start=1):
        excerpt = passage.chunk_text[:280].strip()
        if excerpt:
            lines.append(f"{index}. **{passage.title}** — {excerpt}")
    if len(lines) <= 2:
        return f"По запросу «{query}» в индексе недостаточно данных для вывода."
    return "\n".join(lines)


def _build_prompt(
    request: SynthesizeRequest,
    max_passages: int,
    max_chars: int,
) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    blocks = []
    for index, passage in enumerate(request.passages[:max_passages], start=1):
        blocks.append(f"[{index}] {passage.title}\n{passage.chunk_text[:max_chars]}")
    return template.format(
        query=request.query,
        passages="\n\n".join(blocks) if blocks else "нет данных",
    )


def _to_source(passage: PassageInput, index: int, relevance: float | None = None) -> SourceOutput:
    if passage.score is not None:
        score = min(0.95, max(0.25, passage.score / 5.0))
    elif relevance is not None:
        score = max(0.25, 0.55 - index * 0.05) * max(relevance, 0.35)
    else:
        score = max(0.5, 0.9 - index * 0.05)
    return SourceOutput(
        document_id=passage.document_id,
        title=passage.title,
        chunk_text=passage.chunk_text[:1500],
        confidence=score,
        geo=passage.geo,
        year=passage.year,
    )
