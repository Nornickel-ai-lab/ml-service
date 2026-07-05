from pydantic import BaseModel, Field


class RerankPassageInput(BaseModel):
    id: str
    text: str = Field(min_length=1)


class RerankRequest(BaseModel):
    query: str = Field(min_length=1)
    passages: list[RerankPassageInput] = Field(min_length=1, max_length=64)
    top_k: int = Field(default=10, ge=1, le=64)


class RerankResultItem(BaseModel):
    id: str
    score: float
    rank: int


class RerankResponse(BaseModel):
    results: list[RerankResultItem]
