from pathlib import Path

from app.config import settings
from app.schemas.query import PassageInput, SourceOutput, SynthesizeRequest, SynthesizeResponse
from app.services import mock_yandex, yandex_client

PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "synthesize.txt"


def synthesize(request: SynthesizeRequest) -> SynthesizeResponse:
    if settings.mock_yandex:
        return mock_yandex.mock_synthesize(request)

    template = PROMPT_PATH.read_text(encoding="utf-8")
    blocks = []
    for index, passage in enumerate(request.passages, start=1):
        blocks.append(
            f"[{index}] {passage.title}\n{passage.chunk_text[:1200]}"
        )
    prompt = template.format(
        query=request.query,
        passages="\n\n".join(blocks) if blocks else "нет данных",
    )
    answer = yandex_client.completion(prompt, temperature=0.2)
    sources = [_to_source(passage, index) for index, passage in enumerate(request.passages)]
    confidence = 0.75 if request.passages else 0.2
    return SynthesizeResponse(answer_md=answer, sources=sources, confidence=confidence)


def _to_source(passage: PassageInput, index: int) -> SourceOutput:
    score = max(0.5, 0.9 - index * 0.05)
    return SourceOutput(
        document_id=passage.document_id,
        title=passage.title,
        chunk_text=passage.chunk_text[:500],
        confidence=score,
        geo=passage.geo,
        year=passage.year,
    )
