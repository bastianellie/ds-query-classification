"""Offline tests for the --models multi-model voting mode (no live network
calls). Covers spec 4's FR-1.x/AR-1.x Verify conditions via fake, duck-typed
Classifier-shaped objects (any object exposing ``.classify(text) -> dict``).

These tests call ``run_multi_model`` directly rather than through
``classify_csv`` — the end-to-end/restore tests that need the real
``classify_csv`` (which only gains its ``models`` parameter in the pipeline.py
task) live in this same file, appended once that task lands.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from query_classification import Category, Label
from query_classification.multi_model import run_multi_model


@pytest.fixture
def sentiment_category():
    return Category(
        name="sentiment",
        description="Overall sentiment.",
        labels=[
            Label(value="positive", description="Happy."),
            Label(value="negative", description="Sad."),
        ],
    )


@pytest.fixture
def urgency_category():
    return Category(
        name="urgency",
        description="How urgent the request is.",
        labels=[
            Label(value="high", description="Urgent."),
            Label(value="low", description="Not urgent."),
        ],
    )


@pytest.fixture
def categories(sentiment_category):
    return [sentiment_category]


class FakeModel:
    """Returns a fixed {category: [labels]} dict every call, or raises."""

    def __init__(self, responses=None, raises=False):
        self.responses = responses or {}
        self.raises = raises
        self.calls = 0

    def classify(self, text):
        self.calls += 1
        if self.raises:
            raise RuntimeError("model boom sk-fake-secret-abc123")
        return dict(self.responses)


class DelayedFakeModel(FakeModel):
    """Sleeps before returning, to prove completion order doesn't affect
    the plurality tie-break (which must depend only on --models list
    order)."""

    def __init__(self, responses, delay):
        super().__init__(responses)
        self.delay = delay

    def classify(self, text):
        time.sleep(self.delay)
        return super().classify(text)


class ConcurrencyProbeModel:
    """Records the maximum number of classifiers in flight at once, via a
    shared counter/lock, to prove all N models run concurrently rather than
    serially."""

    def __init__(self, state, delay=0.05):
        self.state = state
        self.delay = delay

    def classify(self, text):
        with self.state["lock"]:
            self.state["current"] += 1
            self.state["max"] = max(self.state["max"], self.state["current"])
        time.sleep(self.delay)
        with self.state["lock"]:
            self.state["current"] -= 1
        return {"sentiment": ["positive"]}


# --- Basic dispatch -----------------------------------------------------


def test_calls_every_model_exactly_once(categories):
    models = {
        "a": FakeModel({"sentiment": ["positive"]}),
        "b": FakeModel({"sentiment": ["positive"]}),
        "c": FakeModel({"sentiment": ["negative"]}),
    }
    run_multi_model("hi", categories, models, allow_new_labels=False)
    assert [m.calls for m in models.values()] == [1, 1, 1]


def test_bounded_concurrency_all_models_run_at_once():
    state = {"lock": threading.Lock(), "current": 0, "max": 0}
    models = {f"m{i}": ConcurrencyProbeModel(state) for i in range(4)}
    categories = [Category(name="sentiment", description="d", labels=[Label(value="positive", description="p")])]
    run_multi_model("hi", categories, models, allow_new_labels=False)
    assert state["max"] == 4


# --- Vote tally and tie-break --------------------------------------------


def test_vote_tally_and_plurality_winner(categories):
    models = {
        "a": FakeModel({"sentiment": ["positive"]}),
        "b": FakeModel({"sentiment": ["positive"]}),
        "c": FakeModel({"sentiment": ["negative"]}),
    }
    result = run_multi_model("hi", categories, models, allow_new_labels=False)
    assert result["sentiment"] == ["positive"]
    assert json.loads(result["sentiment_votes"]) == {"positive": 2, "negative": 1}


def test_tie_break_by_list_order_regardless_of_completion_order(categories):
    # "a" is listed first but resolves SLOWER than "b" — the tie must still
    # go to "a" because tallying is index-ordered, not completion-ordered.
    models = {
        "a": DelayedFakeModel({"sentiment": ["positive"]}, delay=0.08),
        "b": DelayedFakeModel({"sentiment": ["negative"]}, delay=0.0),
    }
    result = run_multi_model("hi", categories, models, allow_new_labels=False)
    assert result["sentiment"] == ["positive"]


def test_none_label_bucketing_merges_new_label_suggestions(categories):
    models = {
        "a": FakeModel({"sentiment": ["none - joyful"]}),
        "b": FakeModel({"sentiment": ["none - upbeat"]}),
        "c": FakeModel({"sentiment": ["negative"]}),
    }
    result = run_multi_model("hi", categories, models, allow_new_labels=True)
    votes = json.loads(result["sentiment_votes"])
    assert votes == {"none (new label)": 2, "negative": 1}
    # representative label list is one model's own (unbucketed) suggestion
    assert result["sentiment"] in (["none - joyful"], ["none - upbeat"])


# --- Audit columns --------------------------------------------------------


def test_by_model_audit_column_has_every_successful_model(categories):
    models = {
        "a": FakeModel({"sentiment": ["positive"]}),
        "b": FakeModel({"sentiment": ["positive"]}),
        "c": FakeModel({"sentiment": ["negative"]}),
    }
    result = run_multi_model("hi", categories, models, allow_new_labels=False)
    by_model = json.loads(result["sentiment_by_model"])
    assert by_model == {"a": ["positive"], "b": ["positive"], "c": ["negative"]}


# --- Partial and total failure -------------------------------------------


def test_partial_failure_excludes_failed_model_and_sanitizes_error(categories):
    models = {
        "a": FakeModel({"sentiment": ["positive"]}),
        "b": FakeModel({"sentiment": ["positive"]}),
        "c": FakeModel(raises=True),
    }
    result = run_multi_model("hi", categories, models, allow_new_labels=False)
    assert json.loads(result["sentiment_by_model"]) == {"a": ["positive"], "b": ["positive"]}
    assert json.loads(result["sentiment_votes"]) == {"positive": 2}
    errors = json.loads(result["sentiment_model_errors"])
    assert errors == {"c": "RuntimeError: Classification call failed"}
    assert "sk-fake-secret-abc123" not in result["sentiment_model_errors"]
    assert "boom" not in result["sentiment_model_errors"]


def test_total_failure_never_raises_and_records_every_model(categories):
    models = {
        "a": FakeModel(raises=True),
        "b": FakeModel(raises=True),
        "c": FakeModel(raises=True),
    }
    result = run_multi_model("hi", categories, models, allow_new_labels=False)
    assert result["sentiment"] is None
    assert result["sentiment_votes"] == "{}"
    assert result["sentiment_by_model"] == "{}"
    errors = json.loads(result["sentiment_model_errors"])
    assert set(errors) == {"a", "b", "c"}
    assert all(v == "RuntimeError: Classification call failed" for v in errors.values())


def test_malformed_category_fails_whole_row_for_that_model(sentiment_category, urgency_category):
    categories = [sentiment_category, urgency_category]
    models = {
        "a": FakeModel({"sentiment": ["positive"], "urgency": ["low"]}),
        "b": FakeModel({"sentiment": ["positive"], "urgency": ["low"]}),
        "c": FakeModel(raises=True),  # simulates schema validation failing atomically
    }
    result = run_multi_model("hi", categories, models, allow_new_labels=False)
    for cat_name in ("sentiment", "urgency"):
        by_model = json.loads(result[f"{cat_name}_by_model"])
        assert "c" not in by_model
        errors = json.loads(result[f"{cat_name}_model_errors"])
        assert "c" in errors
