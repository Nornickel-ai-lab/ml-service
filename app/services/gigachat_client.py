import httpx

from app.config import settings

FOUNDATION_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1"


def credentials_configured() -> bool:
    return bool(settings.yandex_api_key.strip() and settings.yandex_folder_id.strip())


def is_available() -> bool:
    if not credentials_configured():
        return False
    try:
        completion("ok", temperature=0.0)
        return True
    except RuntimeError:
        return False


def ensure_ready() -> None:
    if not credentials_configured():
        raise RuntimeError(
            "yandex credentials missing: set YANDEX_API_KEY and YANDEX_FOLDER_ID in server-service/.env",
        )


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Api-Key {settings.yandex_api_key.strip()}",
        "x-folder-id": settings.yandex_folder_id.strip(),
        "Content-Type": "application/json",
    }


def _model_uri(kind: str, name: str) -> str:
    folder = settings.yandex_folder_id.strip()
    if kind == "gpt":
        return f"gpt://{folder}/{name}"
    return f"emb://{folder}/{name}"


def chat(system: str, user: str, temperature: float = 0.2) -> str:
    messages: list[dict[str, str]] = []
    if system.strip():
        messages.append({"role": "system", "text": system.strip()})
    messages.append({"role": "user", "text": user})
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{FOUNDATION_URL}/completion",
            headers=_headers(),
            json={
                "modelUri": _model_uri("gpt", settings.yandex_llm_model),
                "completionOptions": {
                    "stream": False,
                    "temperature": temperature,
                    "maxTokens": 4000,
                },
                "messages": messages,
            },
        )
        if response.status_code >= 400:
            raise RuntimeError(f"yandex chat failed: {response.status_code} {response.text[:240]}")
        data = response.json()
        return data["result"]["alternatives"][0]["message"]["text"]


def completion(prompt: str, temperature: float = 0.2) -> str:
    return chat("", prompt, temperature=temperature)


def embed_texts(texts: list[str], mode: str = "passage") -> list[list[float]]:
    model = (
        settings.yandex_embed_query_model
        if mode == "query"
        else settings.yandex_embed_doc_model
    )
    vectors: list[list[float]] = []
    with httpx.Client(timeout=120.0) as client:
        for text in texts:
            response = client.post(
                f"{FOUNDATION_URL}/textEmbedding",
                headers=_headers(),
                json={
                    "modelUri": _model_uri("emb", model),
                    "text": text,
                },
            )
            if response.status_code >= 400:
                raise RuntimeError(f"yandex embed failed: {response.status_code} {response.text[:240]}")
            vectors.append(response.json()["embedding"])
    return vectors
