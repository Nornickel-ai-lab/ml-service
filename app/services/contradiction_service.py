import re

from app.schemas.contradictions import CompareRequest, CompareResponse
from app.schemas.extract import NumericConstraint

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


def compare(request: CompareRequest) -> CompareResponse:
    values_a = _numbers(request.passage_a.text)
    values_b = _numbers(request.passage_b.text)
    marker_hit = any(
        marker in request.passage_a.text.lower() or marker in request.passage_b.text.lower()
        for marker in CONTRADICTION_MARKERS
    )
    value_conflict = _values_conflict(values_a, values_b)
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
