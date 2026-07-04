from pydantic import BaseModel


class OcrPage(BaseModel):
    page_num: int
    text: str


class OcrParseResponse(BaseModel):
    pages: list[OcrPage]
    full_text: str
