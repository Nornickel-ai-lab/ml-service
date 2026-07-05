import re

from app.schemas.contradictions import CompareRequest, CompareResponse
from app.schemas.extract import NumericConstraint
from app.services import gigachat_client
from app.services.llm_json import parse_model_with_retry

NUMERIC_PATTERN = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(мг/л|mg/l|т/сут|т/мес|м/с|мм/с|%)",
    re.IGNORECASE,
)

CONTRADICTION_MARKERS = [
    "однако",
    "в то время как",
    "противореч",
    "различа",
    "не совпада",
]

PROMPT_PATH = __import__("pathlib").Path(__file__).resolve().parents[2] / "prompts" / "contradictions.txt"


class CompareStructuredOutput(CompareResponse):
    pass


def compare(request: CompareRequest) -> CompareResponse:
    try:
        if gigachat_client.credentials_configured():
            return _compare_gigachat(request)
    except (RuntimeError, ValueError):
        pass
    return _compare_heuristic(request)


def _compare_gigachat(request: CompareRequest) -> CompareResponse:
    gigachat_client.ensure_ready()
    template = PROMPT_PATH.read_text(encoding="utf-8")
    prompt = template.format(
        topic=request.topic,
        title_a=request.passage_a.title,
        text_a=request.passage_a.text[:2000],
        title_b=request.passage_b.title,
        text_b=request.passage_b.text[:2000],
    )

    def generate() -> str:
        return gigachat_client.completion(prompt, temperature=0.1)

    parsed = parse_model_with_retry(generate, CompareStructuredOutput)
    return CompareResponse(
        is_contradiction=parsed.is_contradiction,
        summary=parsed.summary,
        confidence=parsed.confidence,
    )


def _compare_heuristic(request: CompareRequest) -> CompareResponse:
    values_a = _numbers(request.passage_a.text)
    values_b = _numbers(request.passage_b.text)
    marker_hit = any(
        marker in request.passage_a.text.lower() or marker in request.passage_b.text.lower()
        for marker in CONTRADICTION_MARKERS
    )
    value_conflict = _values_conflict(values_a, values_b)
    topic_terms = [term for term in request.topic.lower().split() if len(term) >= 4]
    if topic_terms and not _texts_share_terms(
        request.passage_a.text,
        request.passage_b.text,
        topic_terms,
    ):
        return CompareResponse(
            is_contradiction=False,
            summary=f"Фрагменты не относятся к одной теме «{request.topic}»",
            confidence=0.2,
        )
    is_contradiction = marker_hit or value_conflict or _topic_numeric_conflict(
        request.topic,
        request.passage_a.text,
        request.passage_b.text,
    )
    summary = (
        f"Разные данные по теме «{request.topic}» в «{request.passage_a.title}» и «{request.passage_b.title}»"
        if is_contradiction
        else f"Явных расхождений по теме «{request.topic}» не найдено"
    )
    confidence = 0.82 if is_contradiction else 0.35
    return CompareResponse(
        is_contradiction=is_contradiction,
        summary=summary,
        confidence=confidence,
    )


def _numbers(text: str) -> list[float]:
    return [float(item.group(1).replace(",", ".")) for item in NUMERIC_PATTERN.finditer(text)]


def _values_conflict(left: list[float], right: list[float]) -> bool:
    if not left or not right:
        return False
    for a in left[:5]:
        for b in right[:5]:
            if a != 0 and b != 0:
                ratio = max(a, b) / min(a, b)
                if ratio >= 1.25:
                    return True
    return False


def _topic_numeric_conflict(topic: str, text_a: str, text_b: str) -> bool:
    topic_lower = topic.lower()
    if topic_lower not in text_a.lower() or topic_lower not in text_b.lower():
        return False
    return _values_conflict(_numbers(text_a), _numbers(text_b))


def _texts_share_terms(text_a: str, text_b: str, terms: list[str]) -> bool:
    blob_a = text_a.lower()
    blob_b = text_b.lower()
    hits_a = sum(1 for term in terms if term in blob_a)
    hits_b = sum(1 for term in terms if term in blob_b)
    return hits_a >= 1 and hits_b >= 1


def constraints_conflict(
    left: list[NumericConstraint],
    right: list[NumericConstraint],
) -> bool:
    for a in left:
        for b in right:
            if a.unit != b.unit:
                continue
            if a.parameter[:20].lower() in b.parameter.lower() or b.parameter[:20].lower() in a.parameter.lower():
                if a.value != b.value:
                    return True
    return False
