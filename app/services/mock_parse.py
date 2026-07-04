import re
from datetime import datetime

from app.schemas.parse import NumericFilter, ParseEntity, ParseResponse, QueryFilters
from app.services import mock_extract

INTENT_MARKERS = {
    "comparison": ["сравн", " vs ", "против", "вариант"],
    "parameter_search": ["мг/л", "mg/l", "≤", ">=", "т/сут", "концентрац", "температур"],
    "overview": ["обзор", "все публикац", "покажите все", "какие метод"],
}

GEO_ALIASES = {
    "RU": ["росси", "отечествен", " рф ", "ru"],
    "foreign": ["зарубеж", "foreign", "миров", "chile", "чili", "китай", "china"],
}

OPERATOR_MAP = {
    "<=": "lte",
    "≤": "lte",
    ">=": "gte",
    "≥": "gte",
    "<": "lt",
    ">": "gt",
    "=": "eq",
}


def parse_mock(query: str) -> ParseResponse:
    lowered = query.lower()
    intent = _detect_intent(lowered)
    entities = _detect_entities(lowered)
    filters = QueryFilters(
        geo=_detect_geo(lowered),
        year_from=_detect_year_from(query, lowered),
        year_to=_detect_year_to(query),
        numeric=_detect_numeric(query),
    )
    search_text = _build_search_text(query, entities)
    return ParseResponse(
        intent=intent,
        entities=entities,
        filters=filters,
        search_text=search_text,
    )


def _detect_intent(lowered: str) -> str:
    for intent, markers in INTENT_MARKERS.items():
        if any(marker in lowered for marker in markers):
            return intent
    return "literature_search"


def _detect_entities(lowered: str) -> list[ParseEntity]:
    entities: list[ParseEntity] = []
    for name in mock_extract.MATERIALS:
        if name.lower() in lowered:
            entities.append(ParseEntity(type="Material", name=name))
    for name in mock_extract.PROCESSES:
        if name.lower() in lowered:
            entities.append(ParseEntity(type="Process", name=name))
    equipment = ["католит", "анод", "ванн", "диафрагм", "пвп"]
    for name in equipment:
        if name in lowered:
            entities.append(ParseEntity(type="Equipment", name=name))
    deduped: list[ParseEntity] = []
    seen: set[tuple[str, str]] = set()
    for item in entities:
        key = (item.type, item.name.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:8]


def _detect_geo(lowered: str) -> list[str]:
    found: list[str] = []
    for geo, markers in GEO_ALIASES.items():
        if any(marker in lowered for marker in markers):
            found.append(geo)
    return found


def _detect_year_from(query: str, lowered: str) -> int | None:
    last_years = re.search(r"последн(?:ие|их)\s+(\d+)\s+лет", lowered)
    if last_years is not None:
        return datetime.now().year - int(last_years.group(1))
    from_year = re.search(r"с\s+((19|20)\d{2})", query)
    if from_year is not None:
        return int(from_year.group(1))
    return None


def _detect_year_to(query: str) -> int | None:
    to_year = re.search(r"до\s+((19|20)\d{2})", query)
    if to_year is not None:
        return int(to_year.group(1))
    return None


def _detect_numeric(query: str) -> list[NumericFilter]:
    items: list[NumericFilter] = []
    for constraint in mock_extract._parse_numeric_constraints(query):
        operator = OPERATOR_MAP.get(constraint.operator, "eq")
        if operator not in ("lt", "lte", "gt", "gte", "eq"):
            operator = "eq"
        items.append(
            NumericFilter(
                parameter=constraint.parameter,
                operator=operator,
                value=constraint.value,
                unit=constraint.unit,
            )
        )
    return items[:10]


def _build_search_text(query: str, entities: list[ParseEntity]) -> str:
    if not entities:
        return query.strip()
    names = [entity.name for entity in entities]
    return " ".join(names)
