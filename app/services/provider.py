from typing import Literal

from app.config import settings

MlProvider = Literal["cloud", "ollama"]


def resolve_provider(override: str | None) -> MlProvider:
    if override in ("cloud", "ollama"):
        return override
    return settings.default_ml_provider
