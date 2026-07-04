import hashlib
import math
import random

from app.schemas.query import PassageInput, SourceOutput, SynthesizeRequest, SynthesizeResponse


def mock_vector(text: str, dims: int) -> list[float]:
    seed = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    rng = random.Random(seed)
    values = [rng.uniform(-1.0, 1.0) for _ in range(dims)]
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return [value / norm for value in values]


def mock_embed_texts(texts: list[str], dims: int) -> list[list[float]]:
    return [mock_vector(text, dims) for text in texts]


def mock_synthesize(request: SynthesizeRequest) -> SynthesizeResponse:
    if not request.passages:
        return SynthesizeResponse(
            answer_md="Нет релевантных фрагментов в индексе",
            sources=[],
            confidence=0.2,
        )

    lines = [f"**{request.query}**", ""]
    for index, passage in enumerate(request.passages[:5], start=1):
        excerpt = passage.chunk_text[:240].strip()
        lines.append(f"{index}. {passage.title}: {excerpt}")

    sources = [
        _to_source(passage, index)
        for index, passage in enumerate(request.passages[:5])
    ]
    return SynthesizeResponse(
        answer_md="\n".join(lines),
        sources=sources,
        confidence=0.72,
    )


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
