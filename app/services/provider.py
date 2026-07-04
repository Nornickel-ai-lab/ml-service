from typing import Literal

from app.config import settings

MlProvider = Literal["gigachat", "ollama"]


def resolve_provider(override: str | None) -> MlProvider:
    if override == "cloud":
        override = "gigachat"
    if override in ("gigachat", "ollama"):
        return override
    default = settings.default_ml_provider
    if default == "cloud":
        return "gigachat"
    return default
