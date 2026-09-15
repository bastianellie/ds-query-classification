"""Offline tests for --batch (spec 5): schema builder, Classifier.max_tokens,
budget resolution, batch sizing/payload assembly, and the BatchRunner. No
live network/provider calls — LLM calls are faked, matching AGENTS.md's
testing policy.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from query_classification.categories import Category, Label
from query_classification.classifier import Classifier
from query_classification.schema import build_batch_model, build_classification_model


def _category(name="c"):
    return Category(name=name, description="d", labels=[Label(value="x", description="d")])


def _classifier(**overrides):
    cat = _category()
    model = build_classification_model([cat])
    kwargs = dict(model_id="openai/gpt-4o-mini", system_prompt="sp", classification_model=model)
    kwargs.update(overrides)
    return Classifier(**kwargs)


# ---------------------------------------------------------------------------
# FR-2.1 / AR-2.1 / AR-2.2: build_batch_model
# ---------------------------------------------------------------------------


def test_build_batch_model_accepts_exactly_n_results():
    row_model = build_classification_model([_category()])
    batch_model = build_batch_model(row_model, 3)
    instance = batch_model(
        result_1={"c": ["x"]}, result_2={"c": ["x"]}, result_3={"c": ["x"]}
    )
    assert instance.result_1.c == ["x"]


def test_build_batch_model_raises_missing_for_short_response():
    row_model = build_classification_model([_category()])
    batch_model = build_batch_model(row_model, 3)
    with pytest.raises(Exception) as exc_info:
        batch_model(result_1={"c": ["x"]}, result_2={"c": ["x"]})
    assert "missing" in str(exc_info.value)


def test_build_batch_model_raises_extra_forbidden_for_long_response():
    row_model = build_classification_model([_category()])
    batch_model = build_batch_model(row_model, 3)
    with pytest.raises(Exception) as exc_info:
        batch_model(
            result_1={"c": ["x"]},
            result_2={"c": ["x"]},
            result_3={"c": ["x"]},
            result_4={"c": ["x"]},
        )
    assert "extra_forbidden" in str(exc_info.value)


def test_build_batch_model_raises_extra_forbidden_for_unexpected_nested_field():
    row_model = build_classification_model([_category()])
    batch_model = build_batch_model(row_model, 3)
    with pytest.raises(Exception) as exc_info:
        batch_model(
            result_1={"c": ["x"], "bogus": "y"},
            result_2={"c": ["x"]},
            result_3={"c": ["x"]},
        )
    assert "extra_forbidden" in str(exc_info.value)


def test_build_classification_model_output_still_ignores_unexpected_fields():
    row_model = build_classification_model([_category()])
    instance = row_model(c=["x"], bogus="ignored")
    assert instance.c == ["x"]
    assert not hasattr(instance, "bogus")


def test_build_batch_model_zero_raises_value_error():
    row_model = build_classification_model([_category()])
    with pytest.raises(ValueError):
        build_batch_model(row_model, 0)


def test_build_batch_model_strict_serialization_has_one_min_items():
    from litellm.utils import type_to_response_format_param

    row_model = build_classification_model([_category()])
    batch_model = build_batch_model(row_model, 3)
    schema = type_to_response_format_param(batch_model)
    schema_str = json.dumps(schema)
    assert schema_str.count("minItems") == 1
    assert '"strict": true' in schema_str or '"strict":true' in schema_str.replace(" ", "")


def test_build_batch_model_does_not_mutate_row_model():
    row_model = build_classification_model([_category()])
    fields_before = set(row_model.model_fields)
    build_batch_model(row_model, 2)
    assert set(row_model.model_fields) == fields_before
    assert row_model.model_config.get("extra") != "forbid"


# ---------------------------------------------------------------------------
# FR-2.5: Classifier.max_tokens
# ---------------------------------------------------------------------------


def test_classifier_completion_kwargs_omits_max_tokens_by_default():
    clf = _classifier()
    kwargs = clf._completion_kwargs([])
    assert "max_tokens" not in kwargs


def test_classifier_completion_kwargs_includes_max_tokens_when_set():
    clf = _classifier(max_tokens=512)
    kwargs = clf._completion_kwargs([])
    assert kwargs["max_tokens"] == 512


def test_droppable_params_drops_max_tokens_before_temperature(monkeypatch):
    import litellm

    calls = []

    def fake_completion(**kwargs):
        calls.append(dict(kwargs))
        if "max_tokens" in kwargs:
            raise litellm.UnsupportedParamsError(
                message="doesn't support max_tokens", model=kwargs.get("model", ""), llm_provider="openai"
            )
        if "temperature" in kwargs:
            raise litellm.UnsupportedParamsError(
                message="doesn't support temperature", model=kwargs.get("model", ""), llm_provider="openai"
            )

        class _Msg:
            content = json.dumps({"c": ["x"]})

        class _Choice:
            message = _Msg()
            finish_reason = "stop"

        class _Resp:
            choices = [_Choice()]

        return _Resp()

    monkeypatch.setattr(litellm, "completion", fake_completion)
    clf = _classifier(max_tokens=256, temperature=0.7)
    content = clf._complete([{"role": "user", "content": "hi"}])
    assert content == json.dumps({"c": ["x"]})
    assert len(calls) == 3
    assert "max_tokens" in calls[0] and "temperature" in calls[0]
    assert "max_tokens" not in calls[1] and "temperature" in calls[1]
    assert "max_tokens" not in calls[2] and "temperature" not in calls[2]
