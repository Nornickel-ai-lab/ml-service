from yandex_ai_studio_sdk import AIStudio
from yandex_ai_studio_sdk._exceptions import AioRpcError

from app.config import settings

_sdk: AIStudio | None = None


def credentials_configured() -> bool:
    return bool(
        settings.yandex_folder_id
        and (settings.yandex_api_key or settings.yandex_iam_token),
    )


def _auth_value() -> str:
    if settings.yandex_iam_token:
        return settings.yandex_iam_token
    if settings.yandex_api_key:
        return settings.yandex_api_key
    raise RuntimeError("yandex credentials missing")


def _map_yandex_error(exc: Exception) -> RuntimeError:
    if isinstance(exc, AioRpcError):
        message = str(exc)
        if "PERMISSION_DENIED" in message:
            return RuntimeError(
                "yandex permission denied: check folder_id, billing, and role ai.languageModels.user for the API key service account",
            )
        return RuntimeError(f"yandex api error: {message[:240]}")
    if isinstance(exc, RuntimeError):
        return exc
    return RuntimeError(str(exc))


def get_sdk() -> AIStudio:
    global _sdk
    if _sdk is None:
        if not credentials_configured():
            raise RuntimeError("yandex credentials missing")
        _sdk = AIStudio(folder_id=settings.yandex_folder_id, auth=_auth_value())
    return _sdk


def completion(prompt: str, temperature: float = 0.2) -> str:
    try:
        sdk = get_sdk()
        model = sdk.models.completions(settings.yandex_model).configure(temperature=temperature)
        result = model.run(prompt)
        for alt in result:
            return alt.text if hasattr(alt, "text") else str(alt)
        return ""
    except Exception as exc:
        raise _map_yandex_error(exc) from exc


def embed_texts(texts: list[str], model_name: str) -> list[list[float]]:
    try:
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
    except Exception as exc:
        raise _map_yandex_error(exc) from exc
