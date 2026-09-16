"""Tests for spec 7 (cost-and-timing-logging): Classifier's usage capture /
`track_usage()` context manager (Task 1), and `cost.py`'s pricing/
computation/stats/timer/atomic-write primitives (Task 2). No live network/
provider calls -- every LLM call is faked, reusing `tests/test_cerebus.py`'s
`_classifier`/`_FakeChoice`/`_FakeMessage`/`_FakeResponse` fixtures rather
than duplicating them (no `conftest.py` exists in this repo to share
fixtures otherwise), matching `tests/test_classifier_errors.py`'s own
convention.
"""

from __future__ import annotations

import json
import threading
import time

import litellm
import pytest

from query_classification import batching as batching_module
from query_classification import cost as cost_module
from query_classification.cost import (
    CostPerToken,
    PricingCache,
    QueryCostCollector,
    Timer,
    compute_cost,
    expand_batch_costs_to_query_samples,
    stats_from_samples,
    uniform_estimate,
    write_json_atomic,
)
from tests.test_cerebus import _FakeChoice, _FakeMessage, _FakeResponse, _classifier


class _FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeResponseWithUsage(_FakeResponse):
    def __init__(self, content, prompt_tokens, completion_tokens, finish_reason="stop"):
        super().__init__(content, finish_reason=finish_reason)
        self.usage = _FakeUsage(prompt_tokens, completion_tokens)


def _valid_content():
    return json.dumps({"c": ["x"]})


# ---------------------------------------------------------------------------
# Task 1: Classifier usage capture / track_usage()
# ---------------------------------------------------------------------------


def test_capture_usage_from_real_response_updates_instance_total(monkeypatch):
    def fake_completion(**kwargs):
        return _FakeResponseWithUsage(_valid_content(), 100, 50)

    monkeypatch.setattr(litellm, "completion", fake_completion)
    clf = _classifier()
    clf.classify("hi")
    assert clf.usage_totals() == (100, 50)


def test_response_with_no_usage_attribute_contributes_nothing(monkeypatch):
    def fake_completion(**kwargs):
        return _FakeResponse(_valid_content())  # no `usage` attribute at all

    monkeypatch.setattr(litellm, "completion", fake_completion)
    clf = _classifier()
    clf.classify("hi")
    assert clf.usage_totals() == (0, 0)


def test_unsupported_params_retry_reports_only_the_successful_attempts_usage(monkeypatch):
    monkeypatch.setattr(
        __import__("query_classification.classifier", fromlist=["_"]),
        "_DROPPABLE_PARAMS",
        ("temperature",),
    )
    calls = {"n": 0}

    def fake_completion(**kwargs):
        calls["n"] += 1
        if "temperature" in kwargs:
            raise litellm.UnsupportedParamsError(
                message="nope", model="m", llm_provider="openai"
            )
        return _FakeResponseWithUsage(_valid_content(), 10, 5)

    monkeypatch.setattr(litellm, "completion", fake_completion)
    clf = _classifier(temperature=0.5)
    clf.classify("hi")
    # The first (rejected) attempt raised before ever obtaining a response --
    # it contributes nothing. Only the successful retry's usage is captured.
    assert clf.usage_totals() == (10, 5)
    assert calls["n"] == 2


def test_retries_across_validation_failures_sum_every_response_producing_attempt(monkeypatch):
    calls = {"n": 0}

    def fake_completion(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            return _FakeResponseWithUsage(json.dumps({"bogus": "y"}), 10, 5)
        return _FakeResponseWithUsage(_valid_content(), 10, 5)

    monkeypatch.setattr(litellm, "completion", fake_completion)
    clf = _classifier(max_retries=3, retry_delay=0.0)
    clf.classify("hi")
    assert clf.usage_totals() == (30, 15)
    assert calls["n"] == 3


def test_track_usage_sink_reports_only_calls_made_inside_the_with_block(monkeypatch):
    def fake_completion(**kwargs):
        return _FakeResponseWithUsage(_valid_content(), 20, 10)

    monkeypatch.setattr(litellm, "completion", fake_completion)
    clf = _classifier()
    clf.classify("outside the block")  # not tracked
    with clf.track_usage() as sink:
        clf.classify("inside the block")
    assert sink.usage == (20, 10)
    assert clf.usage_totals() == (40, 20)  # instance total includes both calls


def test_two_threads_each_see_only_their_own_calls_usage(monkeypatch):
    def fake_completion(**kwargs):
        time.sleep(0.01)
        return _FakeResponseWithUsage(_valid_content(), 7, 3)

    monkeypatch.setattr(litellm, "completion", fake_completion)
    clf = _classifier()
    results = {}

    def worker(name):
        with clf.track_usage() as sink:
            clf.classify("hi")
        results[name] = sink.usage

    threads = [threading.Thread(target=worker, args=(n,)) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results["a"] == (7, 3)
    assert results["b"] == (7, 3)
    assert clf.usage_totals() == (14, 6)


def test_classify_with_no_active_track_usage_context_is_unaffected(monkeypatch):
    def fake_completion(**kwargs):
        return _FakeResponseWithUsage(_valid_content(), 1, 1)

    monkeypatch.setattr(litellm, "completion", fake_completion)
    clf = _classifier()
    result = clf.classify("hi")
    assert result == {"c": ["x"]}


def test_wholesale_mocked_classify_wrapped_in_track_usage_observes_none(monkeypatch):
    """The exact shape tests/test_experiment.py's fake_classify fixture uses:
    a wholesale monkeypatch of Classifier.classify itself, with a narrow
    (self, text) signature. track_usage() must not raise -- it simply
    observes no usage, since the real completion path never ran."""

    def fake_classify(self, text):
        return {"c": ["x"]}

    monkeypatch.setattr(
        __import__("query_classification.classifier", fromlist=["_"]).Classifier,
        "classify",
        fake_classify,
    )
    clf = _classifier()
    with clf.track_usage() as sink:
        clf.classify("hi")
    assert sink.usage is None


# ---------------------------------------------------------------------------
# Task 2: cost.py primitives
# ---------------------------------------------------------------------------


def test_pricing_cache_resolves_workspace_slug_with_openai_prefix(monkeypatch):
    def fake_get_model_info(candidate):
        if candidate == "gpt-4o-mini":
            return {"input_cost_per_token": 1.5e-07, "output_cost_per_token": 6e-07}
        raise Exception("unmapped")

    monkeypatch.setattr(litellm, "get_model_info", fake_get_model_info)
    cache = PricingCache()
    result = cache.resolve("openai/@my-workspace/gpt-4o-mini")
    assert result.input == 1.5e-07
    assert result.output == 6e-07
    assert result.resolved_input_model == "gpt-4o-mini"
    assert result.resolved_output_model == "gpt-4o-mini"


def test_pricing_cache_chunk_down_to_a_real_model(monkeypatch):
    def fake_get_model_info(candidate):
        if candidate == "gpt-4o":
            return {"input_cost_per_token": 2.5e-06, "output_cost_per_token": 1e-05}
        raise Exception("unmapped")

    monkeypatch.setattr(litellm, "get_model_info", fake_get_model_info)
    cache = PricingCache()
    result = cache.resolve("gpt-4o-custom-suffix")
    assert result.input == 2.5e-06
    assert result.output == 1e-05
    assert result.resolved_input_model == "gpt-4o"


def test_pricing_cache_total_failure_returns_none_none(monkeypatch):
    def fake_get_model_info(candidate):
        raise Exception("unmapped")

    monkeypatch.setattr(litellm, "get_model_info", fake_get_model_info)
    cache = PricingCache()
    result = cache.resolve("totally-unknown-model")
    assert result.input is None
    assert result.output is None
    assert result.resolved_input_model is None
    assert result.resolved_output_model is None


def test_pricing_cache_resolves_same_model_id_only_once(monkeypatch):
    calls = {"n": 0}

    def fake_get_model_info(candidate):
        calls["n"] += 1
        return {"input_cost_per_token": 1e-06, "output_cost_per_token": 2e-06}

    monkeypatch.setattr(litellm, "get_model_info", fake_get_model_info)
    cache = PricingCache()
    cache.resolve("gpt-4o-mini")
    cache.resolve("gpt-4o-mini")
    assert calls["n"] == 1


def test_pricing_cache_candidate_walk_matches_batching_module(monkeypatch):
    seen = {"cost": [], "batching": []}

    def fake_get_model_info(candidate):
        raise Exception("unmapped")

    monkeypatch.setattr(litellm, "get_model_info", fake_get_model_info)
    model_id = "openai/@my-workspace/gpt-4o-mini-preview-2024"
    assert cost_module._candidate_ids(model_id) == batching_module._candidate_ids(model_id)


def test_pricing_cache_real_installed_pricing_table_shape():
    # At most one smoke test against the real litellm table (no monkeypatch),
    # per FR-6.1's brittleness note -- shape only, no pinned value.
    cache = PricingCache()
    result = cache.resolve("gpt-4o-mini")
    assert isinstance(result.input, float)
    assert isinstance(result.output, float)


def test_compute_cost_arithmetic():
    cpt = CostPerToken(input=1e-06, output=2e-06, resolved_input_model="m", resolved_output_model="m")
    assert compute_cost(100, 50, cpt) == pytest.approx(100 * 1e-06 + 50 * 2e-06)


def test_compute_cost_none_when_pricing_unresolved():
    cpt = CostPerToken(input=None, output=2e-06, resolved_input_model=None, resolved_output_model="m")
    assert compute_cost(100, 50, cpt) is None
    cpt2 = CostPerToken(input=1e-06, output=None, resolved_input_model="m", resolved_output_model=None)
    assert compute_cost(100, 50, cpt2) is None


def test_stats_from_samples_count_zero():
    assert stats_from_samples([]) == {"mean_usd": None, "stddev_usd": None, "count": 0}


def test_stats_from_samples_count_one():
    result = stats_from_samples([0.05])
    assert result["mean_usd"] == pytest.approx(0.05)
    assert result["stddev_usd"] is None
    assert result["count"] == 1


def test_stats_from_samples_count_n():
    result = stats_from_samples([1.0, 2.0, 3.0])
    assert result["mean_usd"] == pytest.approx(2.0)
    assert result["stddev_usd"] == pytest.approx(1.0)
    assert result["count"] == 3


def test_stats_from_samples_one_none_poisons_mean_and_stddev_not_count():
    result = stats_from_samples([1.0, None, 3.0])
    assert result["mean_usd"] is None
    assert result["stddev_usd"] is None
    assert result["count"] == 3


def test_uniform_estimate_stddev_always_zero_not_none_at_count_one():
    result = uniform_estimate(total_cost=0.03, count=1)
    assert result["mean_usd"] == pytest.approx(0.03)
    assert result["stddev_usd"] == 0.0
    assert result["count"] == 1


def test_uniform_estimate_count_zero():
    result = uniform_estimate(total_cost=None, count=0)
    assert result == {"mean_usd": None, "stddev_usd": None, "count": 0}


def test_uniform_estimate_count_n():
    result = uniform_estimate(total_cost=1.0, count=4)
    assert result["mean_usd"] == pytest.approx(0.25)
    assert result["stddev_usd"] == 0.0
    assert result["count"] == 4


def test_expand_batch_costs_to_query_samples_mixed_none():
    result = expand_batch_costs_to_query_samples([(2, 0.10), (3, None)])
    assert result == [0.05, 0.05, None, None, None]


def test_timer_reports_at_least_injected_sleep_duration():
    timer = Timer()
    time.sleep(0.05)
    assert timer.elapsed_seconds() >= 0.05


def test_write_json_atomic_leaves_no_partial_file_on_mid_write_failure(tmp_path, monkeypatch):
    target = tmp_path / "out.json"

    class _Unserializable:
        pass

    with pytest.raises(TypeError):
        write_json_atomic(target, {"bad": _Unserializable()})
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_write_json_atomic_writes_valid_json(tmp_path):
    target = tmp_path / "out.json"
    write_json_atomic(target, {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1}


def test_write_json_atomic_rejects_non_finite_values(tmp_path):
    target = tmp_path / "out.json"
    with pytest.raises(ValueError):
        write_json_atomic(target, {"a": float("nan")})
    assert not target.exists()


def test_query_cost_collector_records_and_reports_rows_attempted():
    collector = QueryCostCollector()
    collector.set_rows_attempted(5)
    collector.record(0.01)
    collector.record(None)
    assert collector.rows_attempted() == 5
    assert collector.samples() == [0.01, None]


def test_query_cost_collector_rows_attempted_none_before_set():
    collector = QueryCostCollector()
    assert collector.rows_attempted() is None
