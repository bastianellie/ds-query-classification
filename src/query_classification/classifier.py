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


# --- Cerebus / Portkey gateway support -------------------------------------
#
# Cerebus is an internal LLM gateway (built on Portkey) some teams route
# through instead of a direct provider. It has two addressing modes:
#   - "direct" (the default): a directly-hosted model (OpenAI, Gemini, ...),
#               addressed by an `@workspace/model` slug — or a bare model
#               name — passed through as-is via `--model`/`--critic-model`/
#               etc., sent with an `x-portkey-provider: openai` header. No
#               extra per-model configuration is needed for this mode.
#   - "azure":  an Azure-backed model, addressed by a Portkey Config ID
#               (sent as the `x-portkey-config` header) — opt-in via
#               `CEREBUS_MODE=azure`, since a Config ID is workspace- and
#               model-specific and has no sensible default.
# Either way, litellm treats the call as a generic OpenAI-compatible endpoint
# (the `openai/` model prefix), so the gateway/provider distinction lives
# entirely in which URL + headers this module attaches, not in litellm's own
# routing. Enabled via `--cerebus` or `DEFAULT_LLM_PROVIDER=cerebus` (wired in
# cli.py/experiment.py) — a run that asks for neither never touches any of
# this. The gateway URLs below default to this org's shared nonprod Cerebus
# endpoints (same values used by other internal tools) so that, combined with
# the "direct" mode default, most setups need only `DEFAULT_LLM_PROVIDER=cerebus`
# plus an optional `CEREBUS_API_KEY` — matching the minimal-footprint
# convention used elsewhere internally. Every default below is still
# independently overridable via its own env var.
_CEREBUS_GATEWAY_AZURE_URL_DEFAULT = "https://gw.az.nonprod.cerebus.tio.elsevier.systems/v1"
_CEREBUS_GATEWAY_DIRECT_URL_DEFAULT = "https://gw.nonprod.cerebus.tio.elsevier.systems/v1"
_CEREBUS_AWS_PROFILE_DEFAULT = "kd-nonprod"
_CEREBUS_AWS_REGION_DEFAULT = "us-east-1"
_CEREBUS_SECRET_ID_DEFAULT = "shared_genai/portkey-nonprod"
_CEREBUS_SECRET_KEY_DEFAULT = "sciencedirect_portkey_api_key"

_cerebus_api_key_cache: str | None = None


def cerebus_enabled_via_env() -> bool:
    """True if `DEFAULT_LLM_PROVIDER=cerebus` (case-insensitive) — an
    alternative to passing `--cerebus` explicitly, for a "set it in .env and
    forget it" setup. Either one turns Cerebus routing on."""
    return os.getenv("DEFAULT_LLM_PROVIDER", "").strip().lower() == "cerebus"


def _cerebus_key_help_message(aws_profile: str, underlying: Exception | None = None) -> str:
    msg = (
        "No Cerebus API key is available: CEREBUS_API_KEY is not set, and it "
        "could not be fetched from AWS Secrets Manager.\n"
        f"  - To use an explicit key, set CEREBUS_API_KEY in .env.\n"
        f"  - To use the AWS fallback, make sure you have an active SSO "
        f"session: aws sso login --profile {aws_profile}\n"
        "  - Also check CEREBUS_SECRET_ID/CEREBUS_SECRET_KEY/CEREBUS_AWS_REGION "
        "if your secret isn't at the default coordinates."
    )
    if underlying is not None:
        # Only the exception's type, never str(underlying): a botocore/SDK
        # error's message text is not a trusted, sanitized value and must
        # never be assumed safe to display or log verbatim (mirrors
        # dataset_io.py's/induction.py's sanitized-failure convention).
        msg += f"\n  (underlying error type: {type(underlying).__name__})"
    return msg


def _resolve_cerebus_api_key() -> str:
    """Resolve the Cerebus/Portkey service key.

    Resolution order: (1) the ``CEREBUS_API_KEY`` env var; (2) AWS Secrets
    Manager, via ``CEREBUS_AWS_PROFILE``/``CEREBUS_AWS_REGION`` and the
    JSON secret at ``CEREBUS_SECRET_ID``'s ``CEREBUS_SECRET_KEY`` field.
    Only the AWS-fallback path is cached in-process (the env-var path is a
    single ``os.getenv`` call, cheap enough not to need it) — never persisted
    to disk. Raises ``RuntimeError`` with actionable remediation steps if
    neither source yields a key.
    """
    global _cerebus_api_key_cache

    env_key = os.getenv("CEREBUS_API_KEY")
    if env_key:
        return env_key
    if _cerebus_api_key_cache:
        return _cerebus_api_key_cache

    aws_profile = os.getenv("CEREBUS_AWS_PROFILE", _CEREBUS_AWS_PROFILE_DEFAULT)
    aws_region = os.getenv("CEREBUS_AWS_REGION", _CEREBUS_AWS_REGION_DEFAULT)
    secret_id = os.getenv("CEREBUS_SECRET_ID", _CEREBUS_SECRET_ID_DEFAULT)
    secret_key = os.getenv("CEREBUS_SECRET_KEY", _CEREBUS_SECRET_KEY_DEFAULT)

    try:
        import boto3  # optional dependency — see the `cerebus` extra
    except ModuleNotFoundError as e:
        if e.name != "boto3":
            # A transitive dependency of boto3 is missing, not boto3 itself —
            # misreporting this as "install boto3" would send someone in
            # circles reinstalling a package that's already present.
            raise
        raise RuntimeError(
            "AWS Secrets Manager fallback requires the optional 'boto3' "
            "dependency, which is not installed. Install it with: "
            "pip install '.[cerebus]' — or set CEREBUS_API_KEY directly."
        ) from e

    try:
        import json as _json

        session = boto3.Session(profile_name=aws_profile)
        client = session.client("secretsmanager", region_name=aws_region)
        response = client.get_secret_value(SecretId=secret_id)
        key = _json.loads(response["SecretString"]).get(secret_key)
        if not isinstance(key, str) or not key:
            raise ValueError(f"secret '{secret_id}' has no non-empty string key '{secret_key}'")
    except Exception as e:  # noqa: BLE001 - any AWS/auth failure funnels into one clear error
        raise RuntimeError(_cerebus_key_help_message(aws_profile, e)) from e

    _cerebus_api_key_cache = key
    return key


def reject_insecure_cerebus_endpoint(api_base: str | None) -> None:
    """Refuse a non-HTTPS endpoint override under ``--cerebus``.

    The Cerebus API key and Portkey auth headers are attached to whatever
    ``api_base`` is in effect (FR-1.4 lets an explicit ``--api-base`` win over
    the resolved gateway URL) — sending them to a plain-HTTP or otherwise
    malformed endpoint would expose the gateway credential in transit. This
    is a minimal guard, not a full allowlist: it only rejects the clearly
    unsafe case (no TLS), since the value is the user's own CLI flag, not
    externally-attacker-controlled input.
    """
    if api_base is None:
        return
    if not api_base.lower().startswith("https://"):
        raise ValueError(
            f"--api-base must use https:// when --cerebus is set (got {api_base!r}) "
            "— the gateway API key and auth headers would otherwise be sent in "
            "the clear."
        )


def build_cerebus_completion_kwargs() -> dict[str, Any]:
    """Resolve the Cerebus gateway configuration from the environment.

    Returns ``{"api_base": ..., "api_key": ..., "extra_headers": {...}}``,
    ready to merge into a ``Classifier`` constructor call. ``CEREBUS_MODE``
    defaults to ``"direct"`` (no per-model configuration needed) and the
    gateway URLs default to this org's shared nonprod endpoints — so the
    common case needs no ``CEREBUS_*`` variables beyond an optional
    ``CEREBUS_API_KEY``. Raises ``ValueError`` naming the invalid setting if
    ``CEREBUS_MODE`` is set to something other than ``"azure"``/``"direct"``,
    or if `azure` mode's required ``CEREBUS_CONFIG_ID`` (which has no
    sensible default — it's workspace- and model-specific) is missing —
    validated *before* resolving the API key, so a configuration mistake
    surfaces without waiting on (or masking behind) a slow/failing AWS call.
    """
    mode = os.getenv("CEREBUS_MODE", "direct")
    if mode not in ("azure", "direct"):
        raise ValueError(
            f"CEREBUS_MODE must be 'azure' or 'direct', got {mode!r}. Set it in .env."
        )

    if mode == "azure":
        api_base = os.getenv("CEREBUS_GATEWAY_AZURE_URL", _CEREBUS_GATEWAY_AZURE_URL_DEFAULT)
        config_id = os.getenv("CEREBUS_CONFIG_ID")
        if not config_id:
            raise ValueError("CEREBUS_CONFIG_ID is required when CEREBUS_MODE=azure")
    else:
        api_base = os.getenv("CEREBUS_GATEWAY_DIRECT_URL", _CEREBUS_GATEWAY_DIRECT_URL_DEFAULT)
        config_id = None

    api_key = _resolve_cerebus_api_key()
    headers = {"x-portkey-api-key": api_key}
    if mode == "azure":
        headers["x-portkey-config"] = config_id
    else:
        headers["x-portkey-provider"] = "openai"

    return {"api_base": api_base, "api_key": api_key, "extra_headers": headers}


def cerebus_model_id(model_id: str) -> str:
    """Prefix ``model_id`` with litellm's ``openai/`` custom-provider marker,
    if not already present — required for litellm to treat the Cerebus
    gateway's ``api_base`` as a generic OpenAI-compatible endpoint rather than
    trying to resolve ``model_id`` against a native provider."""
    return model_id if model_id.startswith("openai/") else f"openai/{model_id}"


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
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
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
        # Explicit api_key/extra_headers are for gateway setups (e.g. Cerebus/
        # Portkey, see build_cerebus_completion_kwargs) that need a specific
        # key and custom auth headers rather than litellm's own per-provider
        # env-var resolution. Unset by default, so every existing call site
        # keeps relying on litellm's native credential handling.
        self.api_key = api_key
        self.extra_headers = extra_headers

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
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        return kwargs

    def _complete(self, messages: list[dict]) -> str:
        kwargs = self._completion_kwargs(messages)
        return self._attempt_completion(kwargs, allow_temperature_drop=True)

    def _attempt_completion(
        self, kwargs: dict[str, Any], *, allow_temperature_drop: bool
    ) -> str:
        try:
            # Preferred: structured output with json_schema (not all models support this).
            response = litellm.completion(
                response_format=self.classification_model, **kwargs
            )
        except litellm.UnsupportedParamsError:
            # litellm.UnsupportedParamsError is a subclass of BadRequestError, so
            # this must be checked before that broader except below - otherwise
            # it would be misdiagnosed as "structured output unsupported" and
            # retried in json_object mode with the same (still-unsupported)
            # temperature, failing again for the same reason instead of fixing it.
            # Some newer/reasoning models reject an explicit temperature outright
            # while reasoning is active - only their fixed default (usually 1) is
            # accepted. Retry once with it dropped, rather than burning every
            # outer retry attempt (classify()'s own loop) on a failure retrying
            # alone can never fix. Only the --critics sampling role ever sets
            # self.temperature, so this is a no-op for every other role.
            if not allow_temperature_drop or "temperature" not in kwargs:
                raise
            return self._attempt_completion(
                {k: v for k, v in kwargs.items() if k != "temperature"},
                allow_temperature_drop=False,
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
