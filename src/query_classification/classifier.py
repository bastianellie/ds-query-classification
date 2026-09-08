"""The core single-text classifier: one text in, validated labels out.

Wraps a LiteLLM completion call with structured-output support, a graceful
fallback to plain JSON mode for models that don't support ``json_schema``, and
bounded retries.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import litellm
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Common env vars that hold an endpoint/base URL, in priority order. litellm has
# no single "endpoint" key shared across providers, so we bridge the most common
# names here. The first one that is set (and non-empty) wins.
_API_BASE_ENV_VARS = (
    "LITELLM_API_BASE",
    "AZURE_API_BASE",
    "AZURE_OPENAI_ENDPOINT",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
)


def resolve_api_base() -> str | None:
    """Return the first configured endpoint env var, or None if none are set.

    Returning None (rather than "") lets litellm fall back to its own
    provider-specific resolution instead of trying to use an empty endpoint.
    """
    for name in _API_BASE_ENV_VARS:
        value = os.getenv(name)
        if value:
            return value
    return None


class Classifier:
    """Classify a single text against a dynamically-built schema.

    Holds the run configuration (model, system prompt, schema) so it can be
    constructed once and reused across many texts.
    """

    def __init__(
        self,
        model_id: str,
        system_prompt: str,
        classification_model: type[BaseModel],
        max_retries: int = 3,
        retry_delay: float = 5.0,
        api_base: str | None = None,
        temperature: float | None = None,
    ) -> None:
        if max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {max_retries}")
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.classification_model = classification_model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        # Resolve from the common endpoint env vars when not given explicitly.
        # Kept as None (never "") so litellm can fall back to its own resolution.
        self.api_base = api_base if api_base is not None else resolve_api_base()
        self.temperature = temperature

    def _completion_kwargs(self, messages: list[dict]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self.model_id, "messages": messages}
        # Only pass api_base when actually set; an empty/None value makes litellm
        # attempt an empty endpoint and raise instead of using its own defaults.
        if self.api_base:
            kwargs["api_base"] = self.api_base
        # Only pass temperature when actually set, so every existing call site that
        # doesn't set it keeps relying on the provider's own default temperature.
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        return kwargs

    def _complete(self, messages: list[dict]) -> str:
        kwargs = self._completion_kwargs(messages)
        try:
            # Preferred: structured output with json_schema (not all models support this).
            response = litellm.completion(
                response_format=self.classification_model, **kwargs
            )
        except litellm.BadRequestError:
            # Fallback: json_object mode - the system prompt describes the schema.
            response = litellm.completion(
                response_format={"type": "json_object"}, **kwargs
            )
        return response.choices[0].message.content

    def classify(self, text: str) -> dict[str, Any]:
        """Classify ``text``, returning a dict of {category: [labels]}.

        Raises the last exception if all retry attempts fail.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text},
        ]

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                content = self._complete(messages)
                result = self.classification_model.model_validate_json(content)
                return result.model_dump(mode="json")
            except Exception as e:  # noqa: BLE001 - retried and re-raised below
                last_exc = e
                if attempt < self.max_retries:
                    logger.warning(
                        "Attempt %d/%d failed: %s. Retrying in %gs...",
                        attempt,
                        self.max_retries,
                        e,
                        self.retry_delay,
                    )
                    time.sleep(self.retry_delay)
                else:
                    logger.warning(
                        "Attempt %d/%d failed: %s. Giving up.",
                        attempt,
                        self.max_retries,
                        e,
                    )

        assert last_exc is not None
        raise last_exc
