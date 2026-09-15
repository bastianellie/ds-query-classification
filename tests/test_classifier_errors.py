"""Tests for classifier.py's failure-classification and completion-path
rewrite (spec 6). Direct imports of ``classify_failure``/``FailureKind``/the
response-failure exception types/``Classifier._complete``/
``Classifier._attempt_completion`` are an accepted INV-7 exception -- see
``spec/ARCHITECTURE.md``.

No test invokes a real ``litellm.completion`` -- every LLM call is faked,
reusing ``tests/test_cerebus.py``'s ``_FakeMessage``/``_FakeChoice``/
``_FakeResponse`` fixtures rather than duplicating them (no ``conftest.py``
exists in this repo to share fixtures otherwise).
"""

from __future__ import annotations

import time

import httpx
import litellm
import pytest
from pydantic import ValidationError

_FAKE_HTTPX_REQUEST = httpx.Request("POST", "https://example.invalid")
_FAKE_HTTPX_RESPONSE = httpx.Response(status_code=400, request=_FAKE_HTTPX_REQUEST)

import query_classification.classifier as classifier_module
from query_classification.classifier import (
    Classifier,
    EmptyResponseError,
    FailureKind,
    PolicyRefusalError,
    TruncatedResponseError,
    _classify_failure_verbose,
    _log_classified_failure,
    classify_failure,
)
from tests.test_cerebus import _FakeChoice, _FakeMessage, _FakeResponse, _classifier


# --- FR-1.1 / FR-1.2: classification -----------------------------------

# One instance of every concrete type FR-1.2 names, with its expected pair.
# Kept as (constructor, expected FailureKind) rather than a single dict so a
# construction failure for one type doesn't hide the others' results.
def _mk_rate_limit():
    return litellm.RateLimitError(message="rate limited", model="m", llm_provider="openai")


def _mk_router_rate_limit():
    return litellm.RouterRateLimitError(
        model="m", cooldown_time=1.0, enable_pre_call_checks=False, cooldown_list=[]
    )


def _mk_router_rate_limit_basic():
    return litellm.RouterRateLimitErrorBasic(model="m")


def _mk_internal_server():
    return litellm.InternalServerError(message="oops", model="m", llm_provider="openai")


def _mk_service_unavailable():
    return litellm.ServiceUnavailableError(message="down", model="m", llm_provider="openai")


def _mk_bad_gateway():
    return litellm.BadGatewayError(message="bad gateway", model="m", llm_provider="openai")


def _mk_api_connection():
    return litellm.APIConnectionError(message="conn", model="m", llm_provider="openai")


def _mk_timeout():
    return litellm.Timeout(message="timeout", model="m", llm_provider="openai")


def _mk_pydantic_validation():
    from pydantic import BaseModel

    class _M(BaseModel):
        x: str

    try:
        _M.model_validate_json("{not valid json")
        raise AssertionError("expected model_validate_json to raise ValidationError")
    except ValidationError as e:
        return e


def _mk_api_response_validation():
    return litellm.APIResponseValidationError(message="bad response", model="m", llm_provider="openai")


def _mk_json_schema_validation():
    return litellm.JSONSchemaValidationError(
        model="m", llm_provider="openai", raw_response="{}", schema="{}"
    )


def _mk_context_window():
    return litellm.ContextWindowExceededError(message="too long", model="m", llm_provider="openai")


def _mk_truncated():
    return TruncatedResponseError()


def _mk_content_policy():
    return litellm.ContentPolicyViolationError(message="blocked", model="m", llm_provider="openai")


def _mk_policy_refusal():
    return PolicyRefusalError()


def _mk_unknown_provider():
    return litellm.LiteLLMUnknownProvider(model="m", custom_llm_provider="nope")


def _mk_authentication():
    return litellm.AuthenticationError(message="bad key", model="m", llm_provider="openai")


def _mk_permission_denied():
    return litellm.PermissionDeniedError(
        message="denied", model="m", llm_provider="openai", response=_FAKE_HTTPX_RESPONSE
    )


def _mk_not_found():
    return litellm.NotFoundError(message="no such model", model="m", llm_provider="openai")


def _mk_unsupported_params():
    return litellm.UnsupportedParamsError(message="nope", model="m", llm_provider="openai")


def _mk_image_fetch():
    return litellm.ImageFetchError(message="fetch failed", model="m", llm_provider="openai")


def _mk_budget_exceeded():
    return litellm.BudgetExceededError(current_cost=10.0, max_budget=5.0)


def _mk_empty_response():
    return EmptyResponseError()


def _mk_bad_request():
    return litellm.BadRequestError(message="bad", model="m", llm_provider="openai")


def _mk_invalid_request():
    return litellm.InvalidRequestError(message="invalid", model="m", llm_provider="openai")


def _mk_unprocessable_entity():
    return litellm.UnprocessableEntityError(
        message="unprocessable", model="m", llm_provider="openai", response=_FAKE_HTTPX_RESPONSE
    )


def _mk_openai_error():
    return litellm.OpenAIError("generic openai error")


def _mk_mock_exception():
    return litellm.MockException(
        status_code=500, message="mock", llm_provider="openai", model="m"
    )


_RETRYABLE = FailureKind(True, False)
_ISOLABLE_ONLY = FailureKind(False, True)
_FATAL = FailureKind(False, False)
_BOTH = FailureKind(True, True)

_TABLE_CASES = [
    (_mk_rate_limit, _RETRYABLE),
    (_mk_router_rate_limit, _RETRYABLE),
    (_mk_router_rate_limit_basic, _RETRYABLE),
    (_mk_internal_server, _RETRYABLE),
    (_mk_service_unavailable, _RETRYABLE),
    (_mk_bad_gateway, _RETRYABLE),
    (_mk_api_connection, _RETRYABLE),
    (_mk_timeout, _RETRYABLE),
    (_mk_pydantic_validation, _BOTH),
    (_mk_api_response_validation, _BOTH),
    (_mk_json_schema_validation, _BOTH),
    (_mk_context_window, _ISOLABLE_ONLY),
    (_mk_truncated, _ISOLABLE_ONLY),
    (_mk_content_policy, _FATAL),
    (_mk_policy_refusal, _FATAL),
    (_mk_unknown_provider, _FATAL),
    (_mk_authentication, _FATAL),
    (_mk_permission_denied, _FATAL),
    (_mk_not_found, _FATAL),
    (_mk_unsupported_params, _FATAL),
    (_mk_image_fetch, _FATAL),
    (_mk_budget_exceeded, _FATAL),
    (_mk_empty_response, _RETRYABLE),
    (_mk_bad_request, _FATAL),
    (_mk_invalid_request, _FATAL),
    (_mk_unprocessable_entity, _FATAL),
    (_mk_openai_error, _FATAL),
    (_mk_mock_exception, _FATAL),
]


@pytest.mark.parametrize("make_exc,expected", _TABLE_CASES, ids=lambda v: getattr(v, "__name__", str(v)))
def test_classify_failure_every_named_type(make_exc, expected):
    assert classify_failure(make_exc()) == expected


def test_classify_failure_default_for_unrecognized_type():
    assert classify_failure(RuntimeError("mystery")) == _FATAL


def test_classify_failure_verbose_reports_matched_for_named_types():
    kind, matched = _classify_failure_verbose(_mk_budget_exceeded())
    assert kind == _FATAL
    assert matched is True


def test_classify_failure_verbose_reports_unmatched_for_default():
    kind, matched = _classify_failure_verbose(RuntimeError("mystery"))
    assert kind == _FATAL
    assert matched is False


# --- FR-1.2's hierarchy traps, pinned against the installed litellm -----

def test_hierarchy_trap_bad_request_subclasses_do_not_collapse():
    """The five BadRequestError subclasses must each classify per their own
    row, not fall through to the plain-BadRequestError row's value."""
    assert classify_failure(_mk_context_window()) == _ISOLABLE_ONLY
    assert classify_failure(_mk_bad_request()) == _FATAL
    # Confirms the two are genuinely distinguishable, not coincidentally equal.
    assert _ISOLABLE_ONLY != _FATAL


def test_hierarchy_trap_budget_exceeded_is_not_an_api_error():
    assert not issubclass(litellm.BudgetExceededError, litellm.APIError)
    assert classify_failure(_mk_budget_exceeded()) == _FATAL


def test_hierarchy_trap_bad_gateway_and_timeout_are_retryable_but_not_folded():
    assert classify_failure(_mk_bad_gateway()) == _RETRYABLE
    assert classify_failure(_mk_timeout()) == _RETRYABLE
    assert not issubclass(litellm.BadGatewayError, litellm.InternalServerError)
    assert not issubclass(litellm.Timeout, litellm.APIConnectionError)


def test_hierarchy_trap_authentication_and_invalid_request_do_not_match_bad_request_branch():
    assert not issubclass(litellm.AuthenticationError, litellm.BadRequestError)
    assert not issubclass(litellm.InvalidRequestError, litellm.BadRequestError)
    assert classify_failure(_mk_authentication()) == _FATAL
    assert classify_failure(_mk_invalid_request()) == _FATAL


# --- FR-3.2: logging (unit-level; end-to-end assertions live in
#     test_classify_retries_and_logging.py-equivalent tests added by Task 4) -

def test_log_classified_failure_emits_expected_content(caplog):
    import logging as _logging

    with caplog.at_level(_logging.WARNING):
        _log_classified_failure(_mk_authentication(), _FATAL, matched=True, will_retry=False)
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "AuthenticationError" in message
    assert "retryable=False" in message
    assert "isolable=False" in message
    assert "giving up" in message


def test_log_classified_failure_no_secret_values(caplog):
    import logging as _logging

    secret_message = "api_key=sk-should-not-appear endpoint=https://secret.example"
    exc = litellm.AuthenticationError(message=secret_message, model="m", llm_provider="openai")
    with caplog.at_level(_logging.WARNING):
        _log_classified_failure(exc, _FATAL, matched=True, will_retry=False)
    combined = "\n".join(r.getMessage() for r in caplog.records)
    assert "sk-should-not-appear" not in combined
    assert "secret.example" not in combined


# --- FR-2.4 / AR-2.2: ordered response checks, via _attempt_completion --

def _complete_with_fake_response(monkeypatch, response):
    clf = _classifier()
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: response)
    return clf._attempt_completion(clf._completion_kwargs([]))


def test_finish_reason_length_raises_truncated(monkeypatch):
    response = _FakeResponse(content=None, finish_reason="length")
    with pytest.raises(TruncatedResponseError):
        _complete_with_fake_response(monkeypatch, response)
    assert classify_failure(TruncatedResponseError()) == _ISOLABLE_ONLY


def test_finish_reason_content_filter_raises_policy_refusal(monkeypatch):
    response = _FakeResponse(content=None, finish_reason="content_filter")
    with pytest.raises(PolicyRefusalError):
        _complete_with_fake_response(monkeypatch, response)
    assert classify_failure(PolicyRefusalError()) == _FATAL


def test_finish_reason_stop_with_valid_content_returns_normally(monkeypatch):
    response = _FakeResponse(content='{"c": ["x"]}', finish_reason="stop")
    result = _complete_with_fake_response(monkeypatch, response)
    assert result == '{"c": ["x"]}'


def test_empty_choices_raises_empty_response(monkeypatch):
    response = _FakeResponse(content=None, choices=[])
    with pytest.raises(EmptyResponseError):
        _complete_with_fake_response(monkeypatch, response)
    assert classify_failure(EmptyResponseError()) == _RETRYABLE


def test_content_none_with_stop_raises_empty_response(monkeypatch):
    response = _FakeResponse(content=None, finish_reason="stop")
    with pytest.raises(EmptyResponseError):
        _complete_with_fake_response(monkeypatch, response)


def test_overlap_length_with_none_content_raises_truncated_not_empty(monkeypatch):
    """finish_reason='length' + content=None is exactly the shape a
    reasoning model returns when it exhausts its output budget on reasoning
    tokens. Must classify as truncation (isolable), not as an empty response
    (retryable) -- retrying an identical, identically-capped request would
    truncate again."""
    response = _FakeResponse(content=None, finish_reason="length")
    with pytest.raises(TruncatedResponseError):
        _complete_with_fake_response(monkeypatch, response)


def test_overlap_content_filter_with_none_content_raises_policy_not_empty(monkeypatch):
    """finish_reason='content_filter' + content=None must classify as a
    policy refusal (fatal), not as an empty response (retryable) -- retrying
    would re-send the same policy-violating content."""
    response = _FakeResponse(content=None, finish_reason="content_filter")
    with pytest.raises(PolicyRefusalError):
        _complete_with_fake_response(monkeypatch, response)


# --- FR-2.1: the fallback triggers only where it can help ---------------

@pytest.mark.parametrize(
    "make_exc",
    [
        lambda: litellm.ContextWindowExceededError(message="too long", model="m", llm_provider="openai"),
        lambda: litellm.ContentPolicyViolationError(message="blocked", model="m", llm_provider="openai"),
    ],
)
def test_excluded_types_never_reach_fallback(monkeypatch, make_exc):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        raise make_exc()

    monkeypatch.setattr(litellm, "completion", fake_completion)
    clf = _classifier()
    with pytest.raises(type(make_exc())):
        clf._complete([{"role": "user", "content": "hi"}])
    assert len(calls) == 1  # no fallback attempt


def test_plain_bad_request_error_reaches_fallback(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        if kwargs.get("response_format") == {"type": "json_object"}:
            return _FakeResponse(_content_json())
        raise litellm.BadRequestError(message="bad", model="m", llm_provider="openai")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    clf = _classifier()
    content = clf._complete([{"role": "user", "content": "hi"}])
    assert content == _content_json()
    assert len(calls) == 2
    assert calls[1]["response_format"] == {"type": "json_object"}


def _content_json():
    import json

    return json.dumps({"c": ["x"]})


# --- FR-2.2: the fallback call is guarded and chains, never wraps -------

def test_fallback_failure_is_chained_not_wrapped(monkeypatch):
    call_count = {"n": 0}

    def fake_completion(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise litellm.BadRequestError(message="bad", model="m", llm_provider="openai")
        raise litellm.RateLimitError(message="limited", model="m", llm_provider="openai")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    clf = _classifier()
    with pytest.raises(litellm.RateLimitError) as exc_info:
        clf._complete([{"role": "user", "content": "hi"}])
    assert isinstance(exc_info.value.__cause__, litellm.BadRequestError)
    assert call_count["n"] == 2


# --- FR-2.5: the ordered ladder, proven with an explicit 3-call sequence -

def test_ladder_drops_two_entries_in_order_across_three_calls(monkeypatch):
    """A 2-call 'reject the first, accept the second' sequence can only ever
    prove ONE drop happened before success -- it cannot show that BOTH
    ladder entries drop in order. This test proves both: call 1 (both
    params present) fails, call 2 (first entry dropped, second still
    present) fails again, call 3 (both dropped) succeeds."""
    monkeypatch.setattr(classifier_module, "_DROPPABLE_PARAMS", ("temperature", "top_p"))
    calls = []

    def fake_completion(**kwargs):
        calls.append(dict(kwargs))
        if "temperature" in kwargs or "top_p" in kwargs:
            raise litellm.UnsupportedParamsError(
                message="nope", model="m", llm_provider="openai"
            )
        return _FakeResponse(_content_json())

    monkeypatch.setattr(litellm, "completion", fake_completion)
    clf = _classifier(temperature=0.5)
    kwargs = clf._completion_kwargs([{"role": "user", "content": "hi"}])
    kwargs["top_p"] = 0.9
    content = clf._attempt_completion(kwargs)

    assert content == _content_json()
    assert len(calls) == 3
    assert "temperature" in calls[0] and "top_p" in calls[0]
    assert "temperature" not in calls[1] and "top_p" in calls[1]
    assert "temperature" not in calls[2] and "top_p" not in calls[2]


def test_ladder_skips_entry_absent_from_kwargs_without_an_extra_call(monkeypatch):
    monkeypatch.setattr(classifier_module, "_DROPPABLE_PARAMS", ("temperature", "top_p"))
    calls = []

    def fake_completion(**kwargs):
        calls.append(dict(kwargs))
        if "top_p" in kwargs:
            raise litellm.UnsupportedParamsError(message="nope", model="m", llm_provider="openai")
        return _FakeResponse(_content_json())

    monkeypatch.setattr(litellm, "completion", fake_completion)
    clf = _classifier()  # no temperature set -- only top_p is present
    kwargs = clf._completion_kwargs([{"role": "user", "content": "hi"}])
    kwargs["top_p"] = 0.9
    content = clf._attempt_completion(kwargs)

    assert content == _content_json()
    assert len(calls) == 2  # ladder skips "temperature" (absent), drops "top_p", succeeds


def test_fallback_kwargs_after_exhaustion_contain_no_ladder_params(monkeypatch):
    monkeypatch.setattr(classifier_module, "_DROPPABLE_PARAMS", ("temperature",))
    calls = []

    def fake_completion(**kwargs):
        calls.append(dict(kwargs))
        if kwargs.get("response_format") == {"type": "json_object"}:
            return _FakeResponse(_content_json())
        raise litellm.UnsupportedParamsError(message="nope", model="m", llm_provider="openai")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    clf = _classifier(temperature=0.5)
    content = clf._complete([{"role": "user", "content": "hi"}])
    assert content == _content_json()
    # 3 calls: (1) structured, temperature present, fails; (2) structured,
    # temperature dropped (the one-entry ladder is now exhausted), fails
    # again; (3) the fallback, which succeeds.
    assert len(calls) == 3
    assert "temperature" in calls[0]
    assert "temperature" not in calls[1]
    assert "temperature" not in calls[2]
    assert calls[2]["response_format"] == {"type": "json_object"}


# --- FR-3.1: classify() retries on the retryable axis only --------------

def test_classify_retries_retryable_failure_up_to_max_retries(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))

    def fake_completion(**kwargs):
        raise litellm.RateLimitError(message="limited", model="m", llm_provider="openai")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    clf = _classifier(max_retries=3, retry_delay=1.0)
    with pytest.raises(litellm.RateLimitError):
        clf.classify("hi")
    assert len(sleeps) == 2  # 3 attempts, sleeping between attempts 1->2 and 2->3


def test_classify_retries_validation_error_up_to_max_retries(monkeypatch):
    """Regression guard: pydantic.ValidationError is both retryable and
    isolable (FR-1.1), and classify() must still retry it exactly as before
    -- this is the retry the whole redesign exists to preserve."""
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))

    def fake_completion(**kwargs):
        return _FakeResponse(content="{not valid json")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    clf = _classifier(max_retries=3, retry_delay=1.0)
    with pytest.raises(ValidationError):
        clf.classify("hi")
    assert len(sleeps) == 2


def test_classify_fails_fast_on_non_retryable_failure(monkeypatch):
    calls = []
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))

    def fake_completion(**kwargs):
        calls.append(kwargs)
        raise litellm.AuthenticationError(message="bad key", model="m", llm_provider="openai")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    clf = _classifier(max_retries=2, retry_delay=1.0)  # >=2 so the assertion isn't trivial
    with pytest.raises(litellm.AuthenticationError):
        clf.classify("hi")
    assert len(calls) == 1
    assert sleeps == []


def test_classify_fails_fast_on_truncated_response(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))

    def fake_completion(**kwargs):
        return _FakeResponse(content=None, finish_reason="length")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    clf = _classifier(max_retries=2, retry_delay=1.0)
    with pytest.raises(TruncatedResponseError):
        clf.classify("hi")
    assert sleeps == []


# --- FR-3.2: end-to-end logging assertions, through classify() ----------

def test_classify_logs_authentication_error_naming_type_axes_and_no_retry(monkeypatch, caplog):
    import logging as _logging

    def fake_completion(**kwargs):
        raise litellm.AuthenticationError(message="bad key", model="m", llm_provider="openai")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    clf = _classifier(max_retries=2)
    with caplog.at_level(_logging.WARNING):
        with pytest.raises(litellm.AuthenticationError):
            clf.classify("hi")
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "AuthenticationError" in message
    assert "retryable=False" in message
    assert "isolable=False" in message
    assert "giving up" in message


def test_classify_logs_unrecognized_type_as_default_branch(monkeypatch, caplog):
    import logging as _logging

    def fake_completion(**kwargs):
        raise RuntimeError("mystery failure")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    clf = _classifier(max_retries=2)
    with caplog.at_level(_logging.WARNING):
        with pytest.raises(RuntimeError):
            clf.classify("hi")
    assert len(caplog.records) == 1
    assert "default branch" in caplog.records[0].getMessage()


def test_classify_never_logs_configured_secrets(monkeypatch, caplog):
    import logging as _logging

    secret_key = "sk-super-secret-value"
    secret_header = "x-portkey-api-key"
    secret_endpoint = "https://gateway.secret.example"

    def fake_completion(**kwargs):
        assert kwargs.get("api_key") == secret_key
        raise litellm.AuthenticationError(message="bad key", model="m", llm_provider="openai")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    clf = _classifier(
        max_retries=1,
        api_base=secret_endpoint,
        api_key=secret_key,
        extra_headers={secret_header: "value"},
    )
    with caplog.at_level(_logging.WARNING):
        with pytest.raises(litellm.AuthenticationError):
            clf.classify("hi")
    combined = "\n".join(r.getMessage() for r in caplog.records)
    assert secret_key not in combined
    assert secret_endpoint not in combined


def test_classify_logs_exactly_once_for_a_chained_fallback_failure(monkeypatch, caplog):
    """FR-2.2's scenario, exercised end to end: a plain BadRequestError on
    the structured-output call routes to the fallback, which fails with a
    RateLimitError. Because _attempt_completion's fallback guard classifies
    but does not log (classify()'s loop is the sole log site), this must
    produce exactly one log line, not two."""
    import logging as _logging

    call_count = {"n": 0}

    def fake_completion(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise litellm.BadRequestError(message="bad", model="m", llm_provider="openai")
        raise litellm.RateLimitError(message="limited", model="m", llm_provider="openai")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    clf = _classifier(max_retries=1)
    with caplog.at_level(_logging.WARNING):
        with pytest.raises(litellm.RateLimitError):
            clf.classify("hi")
    assert len(caplog.records) == 1
    assert "RateLimitError" in caplog.records[0].getMessage()


# --- Task 1 baseline: nothing about the completion path has changed yet --

def test_full_suite_baseline_untouched():
    """Sanity marker for this task: Classifier's public behavior (fixtures
    below) is exercised end-to-end by test_cerebus.py; this file only adds
    new, additive tests."""
    assert _FakeResponse(content="{}").choices[0].message.content == "{}"
