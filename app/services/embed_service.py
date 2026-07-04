from app.config import settings
from app.schemas.embed import EmbedRequest, EmbedResponse
from app.services import gigachat_client, ollama_client
from app.services.provider import resolve_provider


def embed(request: EmbedRequest) -> EmbedResponse:
    provider = resolve_provider(request.provider)
    if provider == "ollama":
        semantic = ollama_client.embed_texts(request.texts)
        base_dims = settings.ollama_embedding_dims
    else:
        gigachat_client.ensure_ready()
        semantic = gigachat_client.embed_texts(request.texts, mode=request.mode)
        base_dims = len(semantic[0]) if semantic else settings.gigachat_embedding_dims
    return EmbedResponse(vectors=semantic, dims=base_dims)
