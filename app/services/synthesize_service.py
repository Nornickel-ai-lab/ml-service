from pathlib import Path
import re

from fastapi import HTTPException

from app.config import settings
from app.schemas.query import (
    PassageInput,
    SourceGroupOutput,
    SourceOutput,
    SynthesizeRequest,
    SynthesizeResponse,
    SynthesizeStructuredOutput,
)
from app.services import gigachat_client, ollama_client
from app.services.gigachat_fallback import log_gigachat_fallback, ollama_fallback_enabled
from app.services.provider import resolve_provider

from app.services.llm_json import parse_model_with_retry
from app.services.relevance import is_insufficient_answer, is_weak_retrieval

PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "synthesize.txt"
STRUCTURED_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "synthesize_structured.txt"
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
    relevance_query = (request.relevance_query or request.query).strip()
    weak, relevance = is_weak_retrieval(relevance_query, request.passages)
    if weak and not request.passages:
        return _weak_retrieval_response(request, relevance)
    provider = resolve_provider(request.provider)
    if provider == "ollama":
        return _synthesize_ollama(request)
    try:
        gigachat_client.ensure_ready()
        return _synthesize_gigachat(request)
    except RuntimeError as exc:
        if ollama_fallback_enabled():
            log_gigachat_fallback("synthesize", str(exc))
            return _synthesize_ollama(request)
        if request.passages:
            return _extractive_fallback(request, str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _extractive_fallback(request: SynthesizeRequest, reason: str) -> SynthesizeResponse:
    passages = request.passages[: settings.gigachat_max_passages]
    cleaned = [_clean_passage(passage) for passage in passages]
    answer = _extractive_answer(request.query, cleaned)
    return SynthesizeResponse(
        answer_md=answer,
        sources=[_to_source(passage, index) for index, passage in enumerate(cleaned)],
        confidence=0.55,
        groups=_build_groups(cleaned),
        gaps=[f"GigaChat временно недоступен: {reason[:120]}"],
    )


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
        merged[doc_id] = existing.model_copy(update={"chunk_text": text, "score": score, "page_num": existing.page_num or passage.page_num})
    return [merged[doc_id] for doc_id in order]


def _weak_retrieval_response(request: SynthesizeRequest, relevance: float) -> SynthesizeResponse:
    query_short = request.query.strip()
    if len(query_short) > 160:
        query_short = query_short[:157] + "…"
    lines = [
        "**В загруженном корпусе нет материалов, которые напрямую отвечают на запрос**",
        "",
        f"Запрос: *{query_short}*",
        "",
        "Найденные фрагменты не содержат подтверждённых данных по формулировке запроса.",
        "",
        "**Что можно сделать:** уточнить материал или процесс, снять фильтры по годам и географии "
        "или загрузить профильные документы по теме.",
    ]
    confidence = max(0.12, min(0.32, relevance * 0.5))
    gaps = [
        "Нет подтверждённых источников с методами или режимами по формулировке запроса.",
        "Семантический поиск нашёл только слабо связанные фрагменты.",
    ]
    recommendations = [
        "Сформулируйте вопрос по конкретному материалу или процессу из загруженного корпуса.",
        "Проверьте фильтры: год, география, материал, процесс.",
    ]
    return SynthesizeResponse(
        answer_md="\n".join(lines),
        sources=[],
        confidence=confidence,
        gaps=gaps,
        recommendations=recommendations,
    )


def _build_groups(passages: list[PassageInput]) -> list[SourceGroupOutput]:
    buckets: dict[str, list[str]] = {}
    for passage in passages:
        label = passage.geo or "unknown"
        if label == "RU":
            title = "Отечественная практика"
        elif label == "foreign":
            title = "Зарубежная практика"
        elif label == "world":
            title = "Мировая практика"
        else:
            title = "Источники"
        buckets.setdefault(title, [])
        if passage.title not in buckets[title]:
            buckets[title].append(passage.title)
    return [
        SourceGroupOutput(title=title, summary=f"{len(titles)} источник(ов)", source_titles=titles)
        for title, titles in buckets.items()
    ]


def _synthesize_gigachat(request: SynthesizeRequest) -> SynthesizeResponse:
    passages = request.passages[: settings.gigachat_max_passages]
    request = request.model_copy(update={"passages": passages})
    template = STRUCTURED_PROMPT_PATH.read_text(encoding="utf-8")
    blocks = []
    for index, passage in enumerate(request.passages, start=1):
        blocks.append(f"[{index}] {passage.title}\n{passage.chunk_text[:1200]}")
    prompt = template.format(
        query=request.query,
        passages="\n\n".join(blocks) if blocks else "нет данных",
    )

    def generate() -> str:
        return gigachat_client.completion(prompt, temperature=0.1)

    try:
        structured = parse_model_with_retry(generate, SynthesizeStructuredOutput)
        sources = [_to_source(passage, index) for index, passage in enumerate(request.passages)]
        if is_insufficient_answer(structured.answer_md):
            return SynthesizeResponse(answer_md=structured.answer_md, sources=[], confidence=0.28)
        return SynthesizeResponse(
            answer_md=structured.answer_md,
            sources=sources,
            confidence=0.72,
            groups=structured.groups or _build_groups(request.passages),
            gaps=structured.gaps,
            recommendations=structured.recommendations,
        )
    except ValueError:
        pass

    legacy_prompt = _build_prompt(request, max_passages=len(request.passages), max_chars=1200)
    answer = gigachat_client.completion(legacy_prompt, temperature=0.1)
    if is_insufficient_answer(answer):
        return SynthesizeResponse(answer_md=answer, sources=[], confidence=0.28)
    sources = [_to_source(passage, index) for index, passage in enumerate(request.passages)]
    return SynthesizeResponse(
        answer_md=answer,
        sources=sources,
        confidence=0.72,
        groups=_build_groups(request.passages),
    )


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
    return SynthesizeResponse(
        answer_md=answer,
        sources=sources,
        confidence=confidence,
        groups=_build_groups(cleaned) if confidence >= 0.45 else [],
    )


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
        page_num=passage.page_num,
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
        page_num=passage.page_num,
    )
