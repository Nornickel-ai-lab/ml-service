from app.config import settings
from app.services import yandex_client


def cloud_uses_mock() -> bool:
    if yandex_client.credentials_configured():
        return False
    return settings.mock_yandex


def ensure_yandex_ready() -> None:
    if yandex_client.credentials_configured():
        return
    if settings.mock_yandex:
        return
    raise RuntimeError(
        "yandex api not configured: set YANDEX_API_KEY and YANDEX_FOLDER_ID in .env",
    )
