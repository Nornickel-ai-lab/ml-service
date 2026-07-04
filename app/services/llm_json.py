import json
import re
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def extract_json_text(raw: str) -> str:
    stripped = raw.strip()
    block = JSON_BLOCK_PATTERN.search(stripped)
    if block is not None:
        return block.group(1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def parse_model(raw: str, model: type[T]) -> T:
    payload = json.loads(extract_json_text(raw))
    return model.model_validate(payload)


def parse_model_with_retry(
    generate: Callable[[], str],
    model: type[T],
) -> T:
    last_error: Exception | None = None
    for _ in range(2):
        try:
            return parse_model(generate(), model)
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            last_error = exc
    if last_error is None:
        raise ValueError("llm json parse failed")
    raise ValueError(str(last_error)) from last_error
