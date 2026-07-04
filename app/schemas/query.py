from typing import Literal

from pydantic import BaseModel, Field


class PassageInput(BaseModel):
    document_id: str
    title: str
    chunk_text: str
    geo: str | None = None
    year: int | None = None


class SynthesizeRequest(BaseModel):
    query: str
    passages: list[PassageInput] = Field(default_factory=list)
    provider: Literal["cloud", "ollama"] | None = None


class SourceOutput(BaseModel):
    document_id: str
    title: str
    chunk_text: str
    confidence: float
    geo: str | None = None
    year: int | None = None


class SynthesizeResponse(BaseModel):
    answer_md: str
    sources: list[SourceOutput]
    confidence: float
