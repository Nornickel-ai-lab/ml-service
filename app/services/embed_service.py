import math

from app.config import settings
from app.schemas.embed import EmbedRequest, EmbedResponse
from app.services import yandex_client

ROLE_SLOTS = 8
ROLE_MAP = {"researcher": 0, "admin": 1}


def embed(request: EmbedRequest) -> EmbedResponse:
    model_name = (
        settings.yandex_embed_query_model
        if request.mode == "query"
        else settings.yandex_embed_doc_model
    )
    semantic = yandex_client.embed_texts(request.texts, model_name)
    roles = request.access_roles or ["researcher", "admin"]
    vectors = [
        _with_role_suffix(vec, roles if request.mode == "passage" else roles[:1])
        for vec in semantic
    ]
    return EmbedResponse(vectors=vectors, dims=settings.embedding_dims)


def _with_role_suffix(semantic: list[float], roles: list[str]) -> list[float]:
    suffix = [0.0] * ROLE_SLOTS
    for role in roles:
        idx = ROLE_MAP.get(role)
        if idx is not None and idx < ROLE_SLOTS:
            suffix[idx] = 1.0
    weight = settings.role_vector_weight
    combined = semantic + [value * weight for value in suffix]
    norm = math.sqrt(sum(value * value for value in combined))
    if norm == 0:
        return combined
    return [value / norm for value in combined]
