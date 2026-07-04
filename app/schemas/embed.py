from typing import Literal

from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1)
    mode: Literal["query", "passage"] = "passage"
    provider: Literal["gigachat", "ollama"] | None = None


class EmbedResponse(BaseModel):
    vectors: list[list[float]]
    dims: int
