from typing import Literal

from pydantic import BaseModel, Field


class PassageInput(BaseModel):
    document_id: str
    title: str
    chunk_text: str
    geo: str | None = None
    year: int | None = None
    score: float | None = None
    page_num: int | None = None


class SynthesizeRequest(BaseModel):
    query: str
    passages: list[PassageInput] = Field(default_factory=list)
    provider: Literal["gigachat", "ollama"] | None = None
    relevance_query: str | None = None


class SourceOutput(BaseModel):
    document_id: str
    title: str
    chunk_text: str
    confidence: float
    geo: str | None = None
    year: int | None = None
    page_num: int | None = None


class SourceGroupOutput(BaseModel):
    title: str
    summary: str = ""
    source_titles: list[str] = Field(default_factory=list)


class SynthesizeResponse(BaseModel):
    answer_md: str
    sources: list[SourceOutput]
    confidence: float
    groups: list[SourceGroupOutput] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class SynthesizeStructuredOutput(BaseModel):
    answer_md: str
    groups: list[SourceGroupOutput] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
