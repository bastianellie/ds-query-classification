"""The core single-text classifier: one text in, validated labels out.

Wraps a LiteLLM completion call with structured-output support, a graceful
fallback to plain JSON mode for models that don't support ``json_schema``, and
bounded retries.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator, NamedTuple

import litellm
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# Per-thread active track_usage() sink, if any -- module-level (not per-instance)
# because only one track_usage() scope is ever active per thread at a time in
# this codebase's actual call sites (pipeline.py's plain-mode wrapper,
# BatchRunner's top-level call). See Classifier.track_usage().
_thread_local = threading.local()


class _UsageSink:
    """Accumulates usage for the single classify() call(s) made inside one
    ``with classifier.track_usage() as sink:`` block on the thread that
    created it. ``usage`` is ``None`` until at least one response-producing
    completion attempt is captured."""

    def __init__(self) -> None:
        self.usage: tuple[int, int] | None = None

    def _add(self, prompt_tokens: int, completion_tokens: int) -> None:
        if self.usage is None:
            self.usage = (prompt_tokens, completion_tokens)
        else:
            p, c = self.usage
            self.usage = (p + prompt_tokens, c + completion_tokens)

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


class FailureKind(NamedTuple):
    """Two independent axes describing what can be done about a classifier
    failure.

    ``retryable``: repeating the identical request may succeed.
    ``isolable``: the request is answerable, but this attempt wasn't; a
    caller that reshapes its input (e.g. sends less of it, or attributes the
    failure to a subset of a batch) may succeed.

    The two axes are independent -- a failure can be both, neither, or
    exactly one. ``pydantic.ValidationError`` is deliberately both: LLM
    output is stochastic, so an identical retry often succeeds, and a
    persistent failure is still attributable to whatever input provoked it.
    Collapsing this into a single three-way class would force a choice
    between those two truths; keeping the axes independent doesn't.
    """

    retryable: bool
    isolable: bool


class TruncatedResponseError(Exception):
    """The completion's ``finish_reason`` was ``"length"``: the model ran
    out of its output allowance before finishing. Classified isolable, not
    retryable -- an identical request under an identical cap will truncate
    again."""


class PolicyRefusalError(Exception):
    """The completion's ``finish_reason`` was ``"content_filter"``: the
    provider refused to answer. Classified neither retryable nor isolable --
    the content itself was the problem, not the request's size or shape."""


class EmptyResponseError(Exception):
    """The completion returned no usable content: either no ``choices`` at
    all, or a choice whose ``message.content`` is ``None`` and which wasn't
    already explained by a ``finish_reason`` check. Classified retryable,
    not isolable -- an empty completion is the shape a transient provider
    hiccup takes, and repeating the identical request may well succeed."""


_DROPPABLE_PARAMS: tuple[str, ...] = ("max_tokens", "temperature")
"""Parameters ``_attempt_completion`` drops, in this order, when a model
raises ``litellm.UnsupportedParamsError`` for the structured-output request.
Read by attribute lookup at call time (not bound as a default argument or
copied at import), so a test can ``monkeypatch`` it. ``max_tokens`` goes
first because it is only a guard -- proceeding without it reproduces
today's unbatched behavior exactly -- whereas ``temperature`` is
semantically meaningful and should survive if ``max_tokens`` alone was
rejected. These are not the only parameters ``_completion_kwargs`` emits
(``api_base``/``api_key``/``extra_headers`` are also conditional), just the
only ones safe to drop and retry without changing the request's meaning."""


def _classify_failure_verbose(exc: Exception) -> tuple[FailureKind, bool]:
    """Return ``(kind, matched)`` for ``exc``. ``matched`` is ``True`` when
    ``exc`` hit one of the named entries below, ``False`` when it fell
    through to the safe default. The public ``classify_failure`` discards
    ``matched``; ``_log_classified_failure`` needs it, because a defaulted
    ``FailureKind(False, False)`` must be distinguishable in logs from an
    *explicitly* non-retryable, non-isolable type such as
    ``BudgetExceededError`` -- both produce the same pair, but only one of
    them means "this exception type was never seen before".

    This is a flat, ordered chain -- never base-class grouping. Two
    hierarchy facts, verified against the installed litellm, force this
    shape: ``litellm.APIError`` has no subclasses and ``litellm.
    APIStatusError`` doesn't exist, so there is no usable server/client base
    class to catch against; and ``litellm.Timeout`` is *not* a subclass of
    ``litellm.APIConnectionError`` (they are siblings under the openai SDK's
    hierarchy, not litellm's own), so folding one into the other would
    silently misclassify it. The five ``BadRequestError`` subclasses
    (``ContextWindowExceededError``, ``ContentPolicyViolationError``,
    ``LiteLLMUnknownProvider``, ``UnsupportedParamsError``,
    ``ImageFetchError``) are placed before the plain-``BadRequestError``
    entry for the same reason ``_attempt_completion``'s own
    ``UnsupportedParamsError`` handling always has: a subclass carries a
    different meaning than its base and must be caught first, or
    ``isinstance`` against the base would swallow it.
    """
    retryable = FailureKind(True, False)
    isolable_only = FailureKind(False, True)
    fatal = FailureKind(False, False)
    both = FailureKind(True, True)

    # fmt: off
    chain: list[tuple[type[Exception] | tuple[type[Exception], ...], FailureKind]] = [
        ((litellm.RateLimitError, litellm.RouterRateLimitError,
          litellm.RouterRateLimitErrorBasic), retryable),
        ((litellm.InternalServerError, litellm.ServiceUnavailableError,
          litellm.BadGatewayError), retryable),
        (litellm.APIConnectionError, retryable),
        (litellm.Timeout, retryable),
        ((ValidationError, litellm.APIResponseValidationError,
          litellm.JSONSchemaValidationError), both),
        (litellm.ContextWindowExceededError, isolable_only),          # BadRequestError subclass
        (TruncatedResponseError, isolable_only),
        (litellm.ContentPolicyViolationError, fatal),                 # BadRequestError subclass
        (PolicyRefusalError, fatal),
        (litellm.LiteLLMUnknownProvider, fatal),                      # BadRequestError subclass
        ((litellm.AuthenticationError, litellm.PermissionDeniedError,
          litellm.NotFoundError), fatal),
        (litellm.UnsupportedParamsError, fatal),                      # BadRequestError subclass
        (litellm.ImageFetchError, fatal),                             # BadRequestError subclass
        (litellm.BudgetExceededError, fatal),
        (EmptyResponseError, retryable),
        ((litellm.BadRequestError, litellm.InvalidRequestError,
          litellm.UnprocessableEntityError), fatal),                  # BadRequestError itself
        ((litellm.OpenAIError, litellm.BaseLLMException,
          litellm.MockException), fatal),
    ]
    # fmt: on

    for exc_types, kind in chain:
        if isinstance(exc, exc_types):
            return kind, True
    return fatal, False


def classify_failure(exc: Exception) -> FailureKind:
    """Map ``exc`` to a ``FailureKind`` -- total over any exception. An
    unrecognized type is treated as ``FailureKind(False, False)``, the safe
    default: it neither retries nor invites a caller to reshape and resend.
    """
    return _classify_failure_verbose(exc)[0]


def _log_classified_failure(
    exc: Exception, kind: FailureKind, matched: bool, will_retry: bool
) -> None:
    """The sole logging call site for a classified failure in this module --
    both ``_attempt_completion``'s fallback guard and ``classify``'s retry
    loop route through here exactly once per failure (never both for the
    same occurrence; see ``_attempt_completion``'s fallback guard, which
    classifies but does not log). Logs only the exception's type name, both
    axis values, whether the type matched a named rule or fell through to
    the default, and whether the call will retry -- never the exception's
    own message text, which could contain a response body, credential,
    header, or endpoint value. This mirrors the existing sanitization
    convention in ``debate.py``'s ``_sanitize_error`` and this module's own
    ``_cerebus_key_help_message``, both of which log only
    ``type(exc).__name__``.
    """
    logger.warning(
        "%s classified retryable=%s isolable=%s (%s); %s",
        type(exc).__name__,
        kind.retryable,
        kind.isolable,
        "matched" if matched else "default branch",
        "retrying" if will_retry else "giving up",
    )


def _extract_content(response: Any) -> str:
    """Turn a successful ``litellm.completion()`` response into the ``str``
    ``classify()`` expects, checking for an unusable response **in this
    exact order** before falling through to a normal return. The ordering is
    normative, not stylistic: the conditions overlap on real responses and
    carry opposite classifications, so checking them in the wrong order
    silently produces the wrong one.

    1. Empty ``choices`` -- checked first only because every later check
       indexes ``choices[0]``.
    2. ``finish_reason == "length"`` -- raises ``TruncatedResponseError``.
    3. ``finish_reason == "content_filter"`` -- raises ``PolicyRefusalError``.
    4. ``message.content is None`` -- raises ``EmptyResponseError``.

    Checks 2 and 3 must precede check 4: a reasoning model that spends its
    entire output allowance on reasoning tokens returns
    ``finish_reason="length"`` *with* ``content=None`` -- classifying that
    as an empty response (retryable) would re-send an identical request
    that will truncate again. A content-filtered response likewise often
    carries no content; classifying that as retryable would re-send
    policy-violating content on every retry attempt.

    litellm's own ``map_finish_reason`` (verified against the installed
    version) normalizes ``"guardrail_intervened"`` to ``"content_filter"``
    before a response ever reaches here, and normalizes ``"eos"``,
    ``"finish_reason_unspecified"``, ``"malformed_function_call"``, and any
    value it doesn't recognize to ``"stop"`` -- so only ``"length"`` and
    ``"content_filter"`` need an explicit check; nothing else can signal a
    failure litellm doesn't already know about.
    """
    if not response.choices:
        raise EmptyResponseError()
    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise TruncatedResponseError()
    if choice.finish_reason == "content_filter":
        raise PolicyRefusalError()
    if choice.message.content is None:
        raise EmptyResponseError()
    return choice.message.content


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
        max_tokens: int | None = None,
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
        # Unset by default: a generous output cap is a runaway guard batched calls
        # set explicitly (spec 5, FR-2.5), never sized to squeeze a response, so no
        # existing unbatched call site changes behavior by leaving it unset.
        self.max_tokens = max_tokens
        self._usage_lock = threading.Lock()
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

    def usage_totals(self) -> tuple[int, int]:
        """(prompt_tokens, completion_tokens) accumulated across every real
        completion call this instance has made, regardless of whether a
        ``track_usage()`` context was active for any of them."""
        with self._usage_lock:
            return (self._total_prompt_tokens, self._total_completion_tokens)

    @contextmanager
    def track_usage(self) -> Iterator[_UsageSink]:
        """Context manager: ``with classifier.track_usage() as sink:
        classifier.classify(text)``, then read ``sink.usage ->
        tuple[int, int] | None``. ``classify()``'s own call signature and
        arguments are never touched -- this simply makes a thread-local sink
        available for ``_capture_usage_from_response`` to feed while the
        block is active. Not expected to nest in practice, but the previous
        sink (if any) is saved and restored regardless."""
        sink = _UsageSink()
        previous = getattr(_thread_local, "sink", None)
        _thread_local.sink = sink
        try:
            yield sink
        finally:
            _thread_local.sink = previous

    def _capture_usage_from_response(self, response: Any) -> None:
        """Called once, at the single point ``_attempt_completion`` obtains a
        response worth reading -- immediately before ``_extract_content``.
        Reads ``getattr(response, "usage", None)``; if both
        ``prompt_tokens``/``completion_tokens`` are present and not ``None``,
        adds them to this instance's running total and, if a
        ``track_usage()`` context is active on this thread, to that sink too.
        A response with no usable usage contributes nothing to either --
        never an error."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if prompt_tokens is None or completion_tokens is None:
            return
        with self._usage_lock:
            self._total_prompt_tokens += prompt_tokens
            self._total_completion_tokens += completion_tokens
        sink = getattr(_thread_local, "sink", None)
        if sink is not None:
            sink._add(prompt_tokens, completion_tokens)

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
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        return kwargs

    def _complete(self, messages: list[dict]) -> str:
        kwargs = self._completion_kwargs(messages)
        return self._attempt_completion(kwargs)

    def _attempt_completion(self, kwargs: dict[str, Any]) -> str:
        try:
            # Preferred: structured output with json_schema (not all models support this).
            response = litellm.completion(
                response_format=self.classification_model, **kwargs
            )
        except litellm.UnsupportedParamsError as exc:
            # litellm.UnsupportedParamsError is a subclass of BadRequestError, so
            # this must be checked before that broader except below - otherwise
            # it would be misdiagnosed as "structured output unsupported" and
            # retried in json_object mode with the same (still-unsupported)
            # parameter, failing again for the same reason instead of fixing it.
            # Some newer/reasoning models reject an explicit temperature outright
            # while reasoning is active - only their fixed default (usually 1) is
            # accepted. Drop parameters from _DROPPABLE_PARAMS one at a time,
            # rather than burning every outer retry attempt (classify()'s own
            # loop) on a failure retrying alone can never fix. `kwargs` itself
            # records what's already been dropped (a removed key is simply
            # absent), so re-evaluating from the top of the ladder each time
            # naturally skips already-dropped entries without needing separate
            # bookkeeping.
            for param in _DROPPABLE_PARAMS:
                if param in kwargs:
                    return self._attempt_completion(
                        {k: v for k, v in kwargs.items() if k != param}
                    )
            # Ladder exhausted (or was never applicable to this model/role) --
            # this may be a rejection of the response_format itself rather than
            # a parameter we control, so fall through to the JSON-mode fallback
            # using kwargs as they stand now (every droppable entry that was
            # present has already been removed).
            response = self._fallback_completion(kwargs, exc)
        except (
            litellm.ContextWindowExceededError,
            litellm.ContentPolicyViolationError,
            litellm.ImageFetchError,
            litellm.LiteLLMUnknownProvider,
        ):
            # None of these four is about the response schema, so a schema
            # fallback cannot help: an oversized request stays oversized, a
            # policy-violating request stays violating, and the other two are
            # provider/config problems a retried request wouldn't fix either.
            # Must be listed before the plain BadRequestError catch below,
            # since all four are BadRequestError subclasses (verified against
            # the installed litellm) and would otherwise be swallowed by it.
            raise
        except litellm.BadRequestError as exc:
            response = self._fallback_completion(kwargs, exc)
        self._capture_usage_from_response(response)
        return _extract_content(response)

    def _fallback_completion(
        self, kwargs: dict[str, Any], first_attempt_exc: Exception
    ) -> Any:
        """The JSON-mode fallback: the system prompt already describes the
        schema, so a model that rejected structured output (a plain
        ``BadRequestError``) or that ran out of ``_DROPPABLE_PARAMS`` entries
        to drop for an ``UnsupportedParamsError`` gets one more attempt here,
        asking for plain JSON instead of a ``json_schema`` response format.

        Guarded: any failure here is classified (never logged -- ``classify``'s
        loop is this module's sole logging call site, see
        ``_log_classified_failure``'s docstring) and re-raised as the
        fallback's own exception, chained ``from`` the first attempt's
        exception -- never the original object unchanged, and never a
        wrapper. A caller needs the fallback's own type (e.g. spec 5's
        batching layer catching ``ContextWindowExceededError`` by type), and
        the first attempt's exception is preserved as ``__cause__`` so the
        reason the fallback was entered at all stays diagnosable.
        """
        try:
            return litellm.completion(response_format={"type": "json_object"}, **kwargs)
        except Exception as fallback_exc:  # noqa: BLE001 - classified (not logged) and re-raised, chained, below
            classify_failure(fallback_exc)
            raise fallback_exc from first_attempt_exc

    def classify(self, text: str) -> dict[str, Any]:
        """Classify ``text``, returning a dict of {category: [labels]}.

        Retries only failures ``classify_failure`` marks ``retryable`` (see
        ``FailureKind``'s docstring) -- up to ``max_retries`` attempts, exactly
        as before. A non-retryable failure raises immediately, on whichever
        attempt it occurs, with no sleep and no further attempts: retrying a
        credential error or a truncated response cannot succeed, so there is
        nothing to wait for.
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
            except Exception as e:  # noqa: BLE001 - classified below, then either retried or re-raised
                last_exc = e
                kind, matched = _classify_failure_verbose(e)
                if not kind.retryable:
                    _log_classified_failure(e, kind, matched, will_retry=False)
                    raise
                if attempt < self.max_retries:
                    _log_classified_failure(e, kind, matched, will_retry=True)
                    time.sleep(self.retry_delay)
                else:
                    _log_classified_failure(e, kind, matched, will_retry=False)

        assert last_exc is not None
        raise last_exc
