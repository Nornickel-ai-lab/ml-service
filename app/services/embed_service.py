from app.config import settings
from app.schemas.embed import EmbedRequest, EmbedResponse
from app.services import gigachat_client, ollama_client
from app.services.gigachat_fallback import log_gigachat_fallback, ollama_fallback_enabled
from app.services.provider import resolve_provider


def embed(request: EmbedRequest) -> EmbedResponse:
    provider = resolve_provider(request.provider)
    if provider == "ollama":
        semantic = ollama_client.embed_texts(request.texts)
        base_dims = settings.ollama_embedding_dims
        return EmbedResponse(vectors=semantic, dims=base_dims)
    try:
        gigachat_client.ensure_ready()
        semantic = gigachat_client.embed_texts(request.texts, mode=request.mode)
        base_dims = len(semantic[0]) if semantic else settings.gigachat_embedding_dims
        return EmbedResponse(vectors=semantic, dims=base_dims)
    except RuntimeError as exc:
        if ollama_fallback_enabled():
            log_gigachat_fallback("embed", str(exc))
            semantic = ollama_client.embed_texts(request.texts)
            return EmbedResponse(vectors=semantic, dims=settings.ollama_embedding_dims)
        raise
