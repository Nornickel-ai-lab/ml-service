import math

from app.config import settings
from app.schemas.embed import EmbedRequest, EmbedResponse
from app.services import mock_yandex, ollama_client, yandex_client
from app.services.provider import resolve_provider

ROLE_SLOTS = 8
ROLE_MAP = {"researcher": 0, "admin": 1}


def embed(request: EmbedRequest) -> EmbedResponse:
    provider = resolve_provider(request.provider)
    if provider == "ollama":
        semantic = ollama_client.embed_texts(request.texts)
        base_dims = settings.ollama_embedding_dims
    elif settings.mock_yandex:
        semantic = mock_yandex.mock_embed_texts(request.texts, settings.embedding_dims)
        base_dims = settings.embedding_dims
    else:
        model_name = (
            settings.yandex_embed_query_model
            if request.mode == "query"
            else settings.yandex_embed_doc_model
        )
        semantic = yandex_client.embed_texts(request.texts, model_name)
        base_dims = settings.embedding_dims
    roles = request.access_roles or ["researcher", "admin"]
    vectors = [
        _with_role_suffix(vec, roles if request.mode == "passage" else roles[:1])
        for vec in semantic
    ]
    return EmbedResponse(vectors=vectors, dims=base_dims + ROLE_SLOTS)


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
