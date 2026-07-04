import re
import uuid

from app.schemas.extract import (
    ConclusionItem,
    EntityItem,
    ExtractResponse,
    NumericConstraint,
    RelationItem,
)

MATERIALS = [
    "никель",
    "медь",
    "золото",
    "серебро",
    "католит",
    "анод",
    "кобальт",
    "nickel",
    "copper",
]

PROCESSES = [
    "электроэкстракция",
    "выщелачивание",
    "обогащение",
    "плавка",
    "циркуляция",
    "electrowinning",
    "leaching",
]

EXPERT_MARKERS = ["автор", "проф", "докт", "канд", "researcher", "author"]
FACILITY_MARKERS = ["лаборатор", "завод", "установк", "нии", "plant", "facility", "lab"]
EXPERIMENT_MARKERS = ["опыт", "эксперимент", "испытан", "experiment", "trial"]

GEO_MARKERS = {
    "россия": "RU",
    "отечествен": "RU",
    "зарубеж": "foreign",
    "foreign": "foreign",
    "chile": "foreign",
    "чili": "foreign",
    "china": "foreign",
    "китай": "foreign",
}

NUMERIC_PATTERN = re.compile(
    r"(?P<param>[а-яa-zA-Z_\-\s]{3,40}?)\s*"
    r"(?P<op><=|>=|<|>|=)?\s*"
    r"(?P<value>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>(?:мг/л|mg/l|т/сут|т/мес|°C|℃|м/с|мм/с|%|долл\./т))",
    re.IGNORECASE,
)

YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


def extract_mock(text: str, document_id: str | None = None, title: str | None = None) -> ExtractResponse:
    lowered = text.lower()
    doc_id = document_id or str(uuid.uuid4())
    entities: list[EntityItem] = []
    relations: list[RelationItem] = []

    found_materials = [name for name in MATERIALS if name in lowered]
    found_processes = [name for name in PROCESSES if name in lowered]

    for name in found_materials[:5]:
        entities.append(EntityItem(type="Material", name=name))
    for name in found_processes[:5]:
        entities.append(EntityItem(type="Process", name=name))

    for marker in EXPERIMENT_MARKERS:
        if marker in lowered:
            entities.append(EntityItem(type="Experiment", name=f"эксперимент: {marker}"))
            break
    for marker in EXPERT_MARKERS:
        if marker in lowered:
            entities.append(EntityItem(type="Expert", name=f"эксперт ({marker})"))
            break
    for marker in FACILITY_MARKERS:
        if marker in lowered:
            entities.append(EntityItem(type="Facility", name=f"объект ({marker})"))
            break

    if found_processes and found_materials:
        relations.append(
            RelationItem(
                type="uses_material",
                from_name=found_processes[0],
                to_name=found_materials[0],
            )
        )

    numeric_constraints = _parse_numeric_constraints(text)
    conclusions = _parse_conclusions(text, doc_id)
    geo = _detect_geo(lowered)
    year = _detect_year(text)

    return ExtractResponse(
        entities=entities,
        relations=relations,
        conclusions=conclusions,
        numeric_constraints=numeric_constraints,
        geo=geo,
        year=year,
        process=found_processes[0] if found_processes else None,
        material=found_materials[0] if found_materials else None,
    )


def _parse_numeric_constraints(text: str) -> list[NumericConstraint]:
    items: list[NumericConstraint] = []
    for match in NUMERIC_PATTERN.finditer(text):
        param = " ".join(match.group("param").split())
        op = match.group("op") or "="
        value = float(match.group("value").replace(",", "."))
        unit = match.group("unit")
        if len(param) < 3:
            continue
        items.append(
            NumericConstraint(
                parameter=param[:80],
                operator=op,
                value=value,
                unit=unit,
            )
        )
    return items[:20]


def _parse_conclusions(text: str, document_id: str) -> list[ConclusionItem]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    items: list[ConclusionItem] = []
    for index, sentence in enumerate(sentences):
        cleaned = sentence.strip()
        if len(cleaned) < 40:
            continue
        if not re.search(r"\d|никел|мед|католит|скорост|концентрац|т/|мг", cleaned, re.IGNORECASE):
            continue
        items.append(
            ConclusionItem(
                id=f"{document_id}_c{index}",
                text=cleaned[:500],
                confidence=0.75,
            )
        )
        if len(items) >= 8:
            break
    return items


def _detect_geo(lowered: str) -> str | None:
    for marker, geo in GEO_MARKERS.items():
        if marker in lowered:
            return geo
    return None


def _detect_year(text: str) -> int | None:
    match = YEAR_PATTERN.search(text)
    if match is None:
        return None
    return int(match.group(0))
