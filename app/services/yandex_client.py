from yandex_ai_studio_sdk import AIStudio

from app.config import settings

_sdk: AIStudio | None = None


def get_sdk() -> AIStudio:
    global _sdk
    if _sdk is None:
        if not settings.yandex_api_key or not settings.yandex_folder_id:
            raise RuntimeError("yandex credentials missing")
        _sdk = AIStudio(folder_id=settings.yandex_folder_id, auth=settings.yandex_api_key)
    return _sdk


def completion(prompt: str, temperature: float = 0.2) -> str:
    sdk = get_sdk()
    model = sdk.models.completions(settings.yandex_model).configure(temperature=temperature)
    result = model.run(prompt)
    for alt in result:
        return alt.text if hasattr(alt, "text") else str(alt)
    return ""


def embed_texts(texts: list[str], model_name: str) -> list[list[float]]:
    sdk = get_sdk()
    model = sdk.models.text_embeddings(model_name)
    vectors: list[list[float]] = []
    for text in texts:
        result = model.run(text)
        if hasattr(result, "embedding"):
            vectors.append(list(result.embedding))
        elif hasattr(result, "__iter__") and not isinstance(result, str):
            vectors.append(list(result))
        else:
            vectors.append(list(result))
    return vectors
