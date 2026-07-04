import base64
import re
import time
import uuid

import httpx

from app.config import settings

BATCH_SIZE = 90
QUERY_INSTRUCTION = "Given a technical research question, retrieve a relevant passage.\nQuery: "
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

_token: str | None = None
_token_expires_at: float = 0.0


def _authorization_key() -> str:
    raw = settings.gigachat_credentials.strip()
    if raw:
        if _UUID_RE.match(raw):
            raise RuntimeError(
                "gigachat: GIGACHAT_CREDENTIALS looks like Client ID only; "
                "paste Authorization Key from developers.sber.ru or set "
                "GIGACHAT_CLIENT_ID + GIGACHAT_CLIENT_SECRET",
            )
        return raw
    client_id = settings.gigachat_client_id.strip()
    client_secret = settings.gigachat_client_secret.strip()
    if client_id and client_secret:
        payload = f"{client_id}:{client_secret}".encode()
        return base64.b64encode(payload).decode()
    return ""


def credentials_configured() -> bool:
    if settings.gigachat_credentials.strip():
        return True
    return bool(settings.gigachat_client_id.strip() and settings.gigachat_client_secret.strip())


def ensure_ready() -> None:
    if not credentials_configured():
        raise RuntimeError(
            "gigachat credentials missing: set GIGACHAT_CREDENTIALS or "
            "GIGACHAT_CLIENT_ID + GIGACHAT_CLIENT_SECRET in server-service/.env",
        )


def _verify_ssl() -> bool:
    return settings.gigachat_verify_ssl


def _get_access_token() -> str:
    global _token, _token_expires_at
    if _token and time.time() < _token_expires_at - 60:
        return _token
    ensure_ready()
    with httpx.Client(verify=_verify_ssl(), timeout=30.0) as client:
        response = client.post(
            settings.gigachat_auth_url,
            headers={
                "Authorization": f"Basic {_authorization_key()}",
                "RqUID": str(uuid.uuid4()),
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            data={"scope": settings.gigachat_scope},
        )
        if response.status_code >= 400:
            raise RuntimeError(f"gigachat auth failed: {response.status_code} {response.text[:240]}")
        payload = response.json()
        _token = payload["access_token"]
        expires_in = payload.get("expires_in")
        if expires_in is not None:
            _token_expires_at = time.time() + float(expires_in)
        else:
            _token_expires_at = float(payload.get("expires_at", time.time() + 1800))
        return _token


def _api_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_get_access_token()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def chat(system: str, user: str, temperature: float = 0.2) -> str:
    messages: list[dict[str, str]] = []
    if system.strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.append({"role": "user", "content": user})
    with httpx.Client(verify=_verify_ssl(), timeout=120.0) as client:
        response = client.post(
            f"{settings.gigachat_api_url.rstrip('/')}/chat/completions",
            headers=_api_headers(),
            json={
                "model": settings.gigachat_llm_model,
                "messages": messages,
                "temperature": temperature,
                "stream": False,
            },
        )
        if response.status_code >= 400:
            raise RuntimeError(f"gigachat chat failed: {response.status_code} {response.text[:240]}")
        data = response.json()
        return data["choices"][0]["message"]["content"]


def completion(prompt: str, temperature: float = 0.2) -> str:
    return chat("", prompt, temperature=temperature)


def embed_texts(texts: list[str], mode: str = "passage") -> list[list[float]]:
    prepared = list(texts)
    if mode == "query" and prepared:
        prepared[0] = QUERY_INSTRUCTION + prepared[0]
    vectors: list[list[float]] = []
    with httpx.Client(verify=_verify_ssl(), timeout=300.0) as client:
        for start in range(0, len(prepared), BATCH_SIZE):
            batch = prepared[start : start + BATCH_SIZE]
            response = client.post(
                f"{settings.gigachat_api_url.rstrip('/')}/embeddings",
                headers=_api_headers(),
                json={"model": settings.gigachat_embed_model, "input": batch},
            )
            if response.status_code >= 400:
                raise RuntimeError(f"gigachat embed failed: {response.status_code} {response.text[:240]}")
            data = response.json()
            items = sorted(data["data"], key=lambda item: item["index"])
            for item in items:
                vectors.append(item["embedding"])
    return vectors
