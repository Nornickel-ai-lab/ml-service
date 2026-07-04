from pydantic import BaseModel, Field


class ComparePassage(BaseModel):
    document_id: str
    title: str
    text: str


class CompareRequest(BaseModel):
    topic: str = Field(min_length=1)
    passage_a: ComparePassage
    passage_b: ComparePassage


class CompareResponse(BaseModel):
    is_contradiction: bool
    summary: str
    confidence: float
