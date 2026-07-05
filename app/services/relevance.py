import re

from app.config import settings
from app.schemas.query import PassageInput

STOPWORDS = {
    "какие",
    "какой",
    "какая",
    "какое",
    "которые",
    "если",
    "для",
    "при",
    "или",
    "чтобы",
    "также",
    "между",
    "через",
    "где",
    "когда",
    "можно",
    "нужно",
    "должен",
    "есть",
    "все",
    "всё",
    "данной",
    "данный",
    "данная",
    "методы",
    "метода",
    "метод",
    "подходят",
    "подходит",
    "подходящие",
    "подходящий",
}

ANCHOR_MIN_LEN = 8

INSUFFICIENT_ANSWER_MARKERS = (
    "данных недостаточно",
    "уточните запрос",
    "уточните вопрос",
    "не содержат",
    "не отвечают",
    "не отвечает",
    "прямой информации",
    "предлагаю уточнить",
    "недостаточно данных",
    "нет данных",
    "не нашёл",
    "не нашел",
    "не найден",
    "в индексе недостаточно",
    "близких материалов",
    "не могу ответить",
)

DOMAIN_GENERIC = {
    "покажите",
    "описаны",
    "практике",
    "считается",
    "оптимальной",
    "технические",
    "решения",
    "организации",
    "эксперименты",
    "публикации",
    "распределению",
    "последние",
    "способы",
    "применялись",
    "рубежом",
    "показатели",
    "параметрам",
    "источниках",
    "подтверждённые",
    "подтвержденные",
    "указанным",
}


def _terms_raw(query: str) -> list[str]:
    words = re.findall(r"[\w\u0400-\u04FF]+", query.lower())
    result: list[str] = []
    for word in words:
        if len(word) < 4 or word in STOPWORDS or word.isdigit():
            continue
        result.append(word)
    return result


def _anchor_terms(query: str) -> list[str]:
    return [term for term in _terms_raw(query) if len(term) >= ANCHOR_MIN_LEN]


def _terms(query: str) -> list[str]:
    words = re.findall(r"[\w\u0400-\u04FF]+", query.lower())
    result: list[str] = []
    for word in words:
        if len(word) < 5 or word in STOPWORDS or word in DOMAIN_GENERIC or word.isdigit():
            continue
        result.append(word)
    if len(result) >= 2:
        return result
    fallback: list[str] = []
    for word in words:
        if len(word) < 4 or word in STOPWORDS or word in DOMAIN_GENERIC or word.isdigit():
            continue
        fallback.append(word)
    return fallback


def _term_hits(term: str, text: str) -> bool:
    if term in text:
        return True
    if len(term) == 4 and term[:3] in text:
        return True
    if len(term) >= 8:
        return term[:7] in text
    if len(term) >= 6:
        return term[:5] in text
    return False


def retrieval_relevance(query: str, passages: list[PassageInput]) -> float:
    terms = _terms(query)
    if not terms:
        return 1.0 if passages else 0.0
    if not passages:
        return 0.0
    blob = "\n".join(passage.chunk_text.lower() for passage in passages)
    hits = sum(1 for term in terms if _term_hits(term, blob))
    return hits / len(terms)


def is_insufficient_answer(answer: str) -> bool:
    lowered = answer.lower().strip()
    if len(lowered) < 15:
        return False
    return any(marker in lowered for marker in INSUFFICIENT_ANSWER_MARKERS)


def is_weak_retrieval(query: str, passages: list[PassageInput]) -> tuple[bool, float]:
    if not passages:
        return True, 0.0

    ratio = retrieval_relevance(query, passages)
    top_score = max((p.score or 0.0) for p in passages)
    ratio_threshold = settings.weak_relevance_ratio
    es_min = settings.weak_es_score_min

    if top_score >= es_min:
        return False, ratio

    blob = "\n".join(passage.chunk_text.lower() for passage in passages)
    anchors = _anchor_terms(query)
    if anchors:
        anchor_hits = sum(1 for term in anchors if _term_hits(term, blob))
        if anchor_hits == 0 and ratio < ratio_threshold:
            return True, min(ratio, 0.25)

    if ratio >= ratio_threshold:
        return False, ratio

    terms = _terms(query)
    head = [term for term in sorted(terms, key=len, reverse=True) if len(term) >= ANCHOR_MIN_LEN][:3]
    head_hits = sum(1 for term in head if _term_hits(term, blob))
    if head and head_hits >= 1 and ratio >= ratio_threshold * 0.7:
        return False, ratio

    if top_score >= es_min * 0.8 and ratio >= ratio_threshold * 0.5:
        return False, ratio

    return True, ratio
