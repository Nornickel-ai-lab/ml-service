from typing import Literal

from pydantic import BaseModel, Field


class NumericFilter(BaseModel):
    parameter: str
    operator: Literal["lt", "lte", "gt", "gte", "eq"]
    value: float
    unit: str


class QueryFilters(BaseModel):
    geo: list[str] = Field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None
    numeric: list[NumericFilter] = Field(default_factory=list)


class ParseEntity(BaseModel):
    type: str
    name: str


class ParseRequest(BaseModel):
    query: str = Field(min_length=1)
    provider: Literal["gigachat", "ollama"] | None = None
    use_llm: bool | None = None


class ParseResponse(BaseModel):
    intent: str
    entities: list[ParseEntity]
    filters: QueryFilters
    search_text: str
