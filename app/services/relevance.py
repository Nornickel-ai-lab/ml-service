import re

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
    "исходная",
    "исходной",
    "содержит",
    "требуемый",
    "требуемое",
    "данной",
    "данный",
    "данная",
    "воды",
    "воде",
    "воду",
    "методы",
    "метода",
    "метод",
    "подходят",
    "подходит",
    "подходящие",
    "подходящий",
}

RELEVANCE_RATIO_THRESHOLD = 0.4

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
    "сульфаты",
    "сульфатов",
    "хлориды",
    "хлоридов",
    "хлорид",
    "сухой",
    "остаток",
    "остатка",
    "кальция",
    "магния",
    "натрия",
    "промсточные",
    "промсточных",
    "ионы",
    "ионов",
    "раствор",
    "раствора",
    "очистка",
    "очистки",
    "обработка",
    "обработки",
}


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
    terms = sorted(_terms(query), key=len, reverse=True)
    head = [term for term in terms if len(term) >= 8][:3]
    blob = "\n".join(passage.chunk_text.lower() for passage in passages)
    head_hits = sum(1 for term in head if _term_hits(term, blob))
    if ratio < RELEVANCE_RATIO_THRESHOLD:
        return True, ratio
    if head and len(terms) >= 3 and head_hits < 2:
        return True, ratio
    return False, ratio
