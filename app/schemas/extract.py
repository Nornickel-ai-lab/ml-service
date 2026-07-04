from typing import Literal

from pydantic import BaseModel, Field


class EntityItem(BaseModel):
    type: str
    name: str


class RelationItem(BaseModel):
    type: str
    from_name: str
    to_name: str


class NumericConstraint(BaseModel):
    parameter: str
    operator: str
    value: float
    unit: str


class ConclusionItem(BaseModel):
    id: str
    text: str
    confidence: float = 0.7
    geo: str | None = None
    year: int | None = None


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1)
    document_id: str | None = None
    title: str | None = None
    provider: Literal["gigachat", "ollama"] | None = None


class ExtractResponse(BaseModel):
    entities: list[EntityItem]
    relations: list[RelationItem]
    conclusions: list[ConclusionItem]
    numeric_constraints: list[NumericConstraint]
    geo: str | None = None
    year: int | None = None
    process: str | None = None
    material: str | None = None
