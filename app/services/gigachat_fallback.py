import logging

from app.services import ollama_client

logger = logging.getLogger(__name__)


def ollama_fallback_enabled() -> bool:
    return ollama_client.is_available()


def log_gigachat_fallback(operation: str, detail: str) -> None:
    logger.warning("gigachat %s failed (%s), falling back to ollama", operation, detail[:160])
