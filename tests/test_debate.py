"""Offline tests for the --critics debate pipeline (no live network calls).

Covers spec 1's FR-1.x/FR-2.x Verify conditions via fake, duck-typed
Classifier-shaped objects (any object exposing ``.classify(text) -> dict``).
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import pytest

from query_classification import Category, Classifier, Label, classify_csv
from query_classification import debate
from query_classification.prompts import build_critic_prompt, build_reconciler_prompt
from query_classification.schema import build_critic_model, build_reconciler_model


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
def categories(sentiment_category):
    return [sentiment_category]


class FakeSampler:
    """Returns a fixed sequence of top-label lists, one per call, optionally
    raising for entries set to None. Tracks logical submission order via the
    caller's own indexing (see FakeIndexedSampler for order-sensitive tests)."""

    def __init__(self, label_lists):
        self.label_lists = list(label_lists)
        self.calls = 0

    def classify(self, text):
        i = self.calls
        self.calls += 1
        val = self.label_lists[i]
        if val is None:
            raise RuntimeError("sample failed")
        return {"sentiment": val}


class FakeIndexedSampler:
    """A sampler whose per-call delay is controlled by an external map keyed
    by call-entry order, used to prove run_debate's result is correct even
    when samples complete out of the order they were submitted in."""

    def __init__(self, label_lists, delays):
        self.label_lists = list(label_lists)
        self.delays = list(delays)
        self.calls = 0

    def classify(self, text):
        i = self.calls
        self.calls += 1
        time.sleep(self.delays[i])
        return {"sentiment": self.label_lists[i]}


class FakeCritic:
    def __init__(self, verdict=None, raises=False):
        self.verdict = verdict
        self.raises = raises
        self.calls = []

    def classify(self, text):
        self.calls.append(text)
        if self.raises:
            raise RuntimeError("critic boom sk-fake-secret-abc123")
        return self.verdict


class FakeReconciler:
    def __init__(self, verdict=None, raises=False):
        self.verdict = verdict
        self.raises = raises
        self.calls = []

    def classify(self, text):
        self.calls.append(text)
        if self.raises:
            raise RuntimeError("reconciler boom sk-fake-secret-abc123")
        return self.verdict


# --- Feature 1: sampling, voting, routing ------------------------------------


def test_sampling_runs_calls_classifier_n_times(categories):
    sampler = FakeSampler([["positive"]] * 5)
    debate.run_debate(
        "hi",
        categories,
        sampler,
        {"sentiment": FakeCritic()},
        {"sentiment": FakeReconciler()},
        sampling_runs=5,
        consensus_threshold=4,
        allow_new_labels=False,
    )
    assert sampler.calls == 5


def test_vote_tally_on_top_label(categories):
    # threshold=3 so the 3-2 split bypasses debate cleanly, isolating the
    # tally check from escalation/critic behavior.
    sampler = FakeSampler([["positive"], ["positive"], ["positive"], ["negative"], ["negative"]])
    result = debate.run_debate(
        "hi", categories, sampler, {"sentiment": FakeCritic()}, {"sentiment": FakeReconciler()},
        sampling_runs=5, consensus_threshold=3, allow_new_labels=False,
    )
    assert json.loads(result["sentiment_votes"]) == {"positive": 3, "negative": 2}


def test_consensus_bypass_skips_critic(categories):
    sampler = FakeSampler([["positive"], ["positive"], ["positive"], ["positive"], ["negative"]])
    critic = FakeCritic()
    reconciler = FakeReconciler()
    result = debate.run_debate(
        "hi", categories, sampler, {"sentiment": critic}, {"sentiment": reconciler},
        sampling_runs=5, consensus_threshold=4, allow_new_labels=False,
    )
    assert result["sentiment"] == ["positive"]
    assert result["sentiment_challenged"] is False
    assert critic.calls == []
    assert reconciler.calls == []


def test_escalation_includes_leading_and_runner_up_in_critic_message(categories):
    sampler = FakeSampler([["positive"], ["positive"], ["positive"], ["negative"], ["negative"]])
    critic = FakeCritic(verdict={"challenges": False, "proposed_label": None, "argument": "no compelling alternative"})
    result = debate.run_debate(
        "hi", categories, sampler, {"sentiment": critic}, {"sentiment": FakeReconciler()},
        sampling_runs=5, consensus_threshold=4, allow_new_labels=False,
    )
    assert result["sentiment"] == ["positive"]
    assert len(critic.calls) == 1
    assert "positive" in critic.calls[0] and "negative" in critic.calls[0]


def test_no_runner_up_fallback_when_too_few_distinct_candidates(categories):
    # 2 sampling failures leave 3 successes, all agreeing — can't reach
    # consensus_threshold=4 numerically, but there's nothing to escalate with.
    sampler = FakeSampler([["positive"], None, None, ["positive"], ["positive"]])
    critic = FakeCritic()
    result = debate.run_debate(
        "hi", categories, sampler, {"sentiment": critic}, {"sentiment": FakeReconciler()},
        sampling_runs=5, consensus_threshold=4, allow_new_labels=False,
    )
    assert result["sentiment"] == ["positive"]
    assert critic.calls == []


def test_all_samples_failed_raises(categories):
    sampler = FakeSampler([None, None, None, None, None])
    with pytest.raises(RuntimeError):
        debate.run_debate(
            "hi", categories, sampler, {"sentiment": FakeCritic()}, {"sentiment": FakeReconciler()},
            sampling_runs=5, consensus_threshold=4, allow_new_labels=False,
        )


def test_partial_sample_failures_still_produce_a_tally(categories):
    sampler = FakeSampler([["positive"], None, ["positive"], None, ["negative"]])
    result = debate.run_debate(
        "hi", categories, sampler, {"sentiment": FakeCritic(verdict={"challenges": False, "proposed_label": None, "argument": "x"})},
        {"sentiment": FakeReconciler()}, sampling_runs=5, consensus_threshold=4, allow_new_labels=False,
    )
    assert json.loads(result["sentiment_votes"]) == {"positive": 2, "negative": 1}


def test_allow_new_labels_true_buckets_invented_suggestions_together(categories):
    sampler = FakeSampler([["none - urgency"], ["none - other"], ["positive"], ["positive"], ["positive"]])
    result = debate.run_debate(
        "hi", categories, sampler, {"sentiment": FakeCritic()}, {"sentiment": FakeReconciler()},
        sampling_runs=5, consensus_threshold=2, allow_new_labels=True,
    )
    assert json.loads(result["sentiment_votes"]) == {"none (new label)": 2, "positive": 3}


def test_allow_new_labels_false_keeps_invented_suggestions_distinct(categories):
    sampler = FakeSampler([["none - urgency"], ["none - other"], ["positive"], ["positive"], ["positive"]])
    result = debate.run_debate(
        "hi", categories, sampler, {"sentiment": FakeCritic()}, {"sentiment": FakeReconciler()},
        sampling_runs=5, consensus_threshold=2, allow_new_labels=False,
    )
    votes = json.loads(result["sentiment_votes"])
    assert votes == {"none - urgency": 1, "none - other": 1, "positive": 3}


def test_result_correct_regardless_of_completion_order(categories):
    # The first-submitted sample sleeps longest, so it completes LAST despite
    # being submitted first — the tally/representative selection must still
    # reflect each sample's own (index, value) pairing correctly.
    sampler = FakeIndexedSampler(
        label_lists=[["positive"], ["positive"], ["positive"], ["negative"], ["negative"]],
        delays=[0.05, 0.0, 0.0, 0.0, 0.0],
    )
    result = debate.run_debate(
        "hi", categories, sampler, {"sentiment": FakeCritic()}, {"sentiment": FakeReconciler()},
        sampling_runs=5, consensus_threshold=3, allow_new_labels=False,
    )
    assert json.loads(result["sentiment_votes"]) == {"positive": 3, "negative": 2}
    assert result["sentiment_initial"] == ["positive"]


# --- Feature 2: Critic challenge & Reconciliation ----------------------------


def test_critic_decline_preserves_leading_and_records_rationale(categories):
    sampler = FakeSampler([["positive"], ["positive"], ["positive"], ["negative"], ["negative"]])
    critic = FakeCritic(verdict={"challenges": False, "proposed_label": None, "argument": "no compelling alternative"})
    result = debate.run_debate(
        "hi", categories, sampler, {"sentiment": critic}, {"sentiment": FakeReconciler()},
        sampling_runs=5, consensus_threshold=4, allow_new_labels=False,
    )
    assert result["sentiment"] == ["positive"]
    assert result["sentiment_challenged"] is False
    assert result["sentiment_critic_argument"] == "no compelling alternative"
    assert result["sentiment_reconciled"] is False


def test_critic_challenge_triggers_reconciliation(categories):
    sampler = FakeSampler([["positive"], ["positive"], ["positive"], ["negative"], ["negative"]])
    critic = FakeCritic(verdict={"challenges": True, "proposed_label": "negative", "argument": "evidence points negative"})
    reconciler = FakeReconciler(verdict={"labels": ["negative"], "reasoning": "reconciled to negative"})
    result = debate.run_debate(
        "hi", categories, sampler, {"sentiment": critic}, {"sentiment": reconciler},
        sampling_runs=5, consensus_threshold=4, allow_new_labels=False,
    )
    assert result["sentiment"] == ["negative"]
    assert result["sentiment_challenged"] is True
    assert result["sentiment_reconciled"] is True
    assert result["sentiment_reconciler_reasoning"] == "reconciled to negative"
    assert "negative" in result["sentiment_critic_argument"]


def test_none_top_label_is_challengeable(categories):
    sampler = FakeSampler([["none"], ["none"], ["none"], ["positive"], ["positive"]])
    critic = FakeCritic(verdict={"challenges": True, "proposed_label": "positive", "argument": "clear positive sentiment"})
    reconciler = FakeReconciler(verdict={"labels": ["positive"], "reasoning": "agrees with critic"})
    result = debate.run_debate(
        "hi", categories, sampler, {"sentiment": critic}, {"sentiment": reconciler},
        sampling_runs=5, consensus_threshold=4, allow_new_labels=False,
    )
    assert result["sentiment"] == ["positive"]


def test_critic_failure_degrades_to_leading_with_sanitized_error(categories):
    sampler = FakeSampler([["positive"], ["positive"], ["positive"], ["negative"], ["negative"]])
    critic = FakeCritic(raises=True)
    result = debate.run_debate(
        "hi", categories, sampler, {"sentiment": critic}, {"sentiment": FakeReconciler()},
        sampling_runs=5, consensus_threshold=4, allow_new_labels=False,
    )
    assert result["sentiment"] == ["positive"]
    assert result["sentiment_challenged"] is False
    assert result["sentiment_reconciled"] is False
    assert result["sentiment_debate_error"] == "RuntimeError: Critic call failed"
    assert "sk-fake-secret-abc123" not in result["sentiment_debate_error"]


def test_reconciler_failure_degrades_to_leading_but_records_challenge(categories):
    sampler = FakeSampler([["positive"], ["positive"], ["positive"], ["negative"], ["negative"]])
    critic = FakeCritic(verdict={"challenges": True, "proposed_label": "negative", "argument": "x"})
    reconciler = FakeReconciler(raises=True)
    result = debate.run_debate(
        "hi", categories, sampler, {"sentiment": critic}, {"sentiment": reconciler},
        sampling_runs=5, consensus_threshold=4, allow_new_labels=False,
    )
    assert result["sentiment"] == ["positive"]
    assert result["sentiment_challenged"] is True
    assert result["sentiment_reconciled"] is False
    assert result["sentiment_debate_error"] == "RuntimeError: Reconciler call failed"
    assert "sk-fake-secret-abc123" not in result["sentiment_debate_error"]


def test_multiple_escalated_categories_debate_concurrently(sentiment_category):
    urgency = Category(
        name="urgency",
        description="How urgent.",
        labels=[Label(value="high", description="h"), Label(value="low", description="l")],
    )
    cats = [sentiment_category, urgency]

    class MultiSampler:
        def __init__(self):
            self.i = 0
            self.rows = [
                {"sentiment": ["positive"], "urgency": ["high"]},
                {"sentiment": ["positive"], "urgency": ["high"]},
                {"sentiment": ["positive"], "urgency": ["low"]},
                {"sentiment": ["negative"], "urgency": ["low"]},
                {"sentiment": ["negative"], "urgency": ["low"]},
            ]

        def classify(self, text):
            row = self.rows[self.i]
            self.i += 1
            return row

    critics = {
        "sentiment": FakeCritic(verdict={"challenges": False, "proposed_label": None, "argument": "no compelling alternative"}),
        "urgency": FakeCritic(verdict={"challenges": False, "proposed_label": None, "argument": "no compelling alternative"}),
    }
    reconcilers = {"sentiment": FakeReconciler(), "urgency": FakeReconciler()}
    result = debate.run_debate(
        "hi", cats, MultiSampler(), critics, reconcilers,
        sampling_runs=5, consensus_threshold=4, allow_new_labels=False,
    )
    assert result["sentiment"] == ["positive"]
    assert result["urgency"] == ["low"]
    assert len(critics["sentiment"].calls) == 1
    assert len(critics["urgency"].calls) == 1


# --- Full audit trail + pipeline integration ---------------------------------


def test_full_audit_trail_columns_present_only_with_critics(categories):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        input_csv = d / "in.csv"
        pd.DataFrame({"text": ["a"]}).to_csv(input_csv, index=False)

        class SimpleClassifier:
            def classify(self, text):
                return {"sentiment": ["positive"]}

        out_plain = d / "out_plain.csv"
        classify_csv(input_csv, "text", SimpleClassifier(), categories, output_path=out_plain)
        assert list(pd.read_csv(out_plain).columns) == ["text", "sentiment"]

        out_critics = d / "out_critics.csv"
        sampler = FakeSampler([["positive"]] * 5)
        classify_csv(
            input_csv, "text", sampler, categories, output_path=out_critics,
            critics=True,
            critic_classifiers={"sentiment": FakeCritic()},
            reconciler_classifiers={"sentiment": FakeReconciler()},
            sampling_runs=5, consensus_threshold=4,
        )
        cols = list(pd.read_csv(out_critics).columns)
        for suffix in debate.AUDIT_COLUMN_SUFFIXES:
            assert f"sentiment{suffix}" in cols


def test_restore_requires_votes_column_not_just_category_value(categories):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        input_csv = d / "in.csv"
        df = pd.DataFrame({"text": ["a"], "sentiment": ["['positive']"]})
        df.to_csv(input_csv, index=False)

        class BoomSampler:
            def classify(self, text):
                return {"sentiment": ["positive"]}

        # sentiment is filled but sentiment_votes is absent -> not "done" -> reprocessed.
        out = d / "out.csv"
        classify_csv(
            input_csv, "text", BoomSampler(), categories, output_path=out, restore=True,
            critics=True,
            critic_classifiers={"sentiment": FakeCritic()},
            reconciler_classifiers={"sentiment": FakeReconciler()},
            sampling_runs=1, consensus_threshold=1,
        )
        result = pd.read_csv(out)
        assert pd.notna(result.loc[0, "sentiment_votes"])


def test_restore_from_separate_completed_output_skips_done_rows(categories):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        input_csv = d / "in.csv"
        pd.DataFrame({"text": ["a", "b"]}).to_csv(input_csv, index=False)
        out = d / "out.csv"
        pd.DataFrame(
            {
                "text": ["a", "b"],
                "sentiment": ["['positive']", "['negative']"],
                "sentiment_initial": ["['positive']", "['negative']"],
                "sentiment_votes": ['{"positive": 5}', '{"negative": 5}'],
                "sentiment_challenged": [False, False],
                "sentiment_critic_argument": ["", ""],
                "sentiment_reconciled": [False, False],
                "sentiment_reconciler_reasoning": ["", ""],
                "sentiment_debate_error": ["", ""],
            }
        ).to_csv(out, index=False)

        class BoomSampler:
            def classify(self, text):
                raise AssertionError("should not be called: row already done")

        classify_csv(
            input_csv, "text", BoomSampler(), categories, output_path=out, restore=True,
            critics=True,
            critic_classifiers={"sentiment": FakeCritic()},
            reconciler_classifiers={"sentiment": FakeReconciler()},
            sampling_runs=5, consensus_threshold=4,
        )  # must not raise


def test_stale_audit_values_reset_on_fresh_non_restore_run(categories):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        input_csv = d / "in.csv"
        pd.DataFrame(
            {
                "text": ["a"],
                "sentiment": ["['stale']"],
                "sentiment_votes": ['{"stale": 99}'],
                "sentiment_challenged": [True],
            }
        ).to_csv(input_csv, index=False)
        out = d / "out.csv"
        sampler = FakeSampler([["positive"]] * 5)
        classify_csv(
            input_csv, "text", sampler, categories, output_path=out,
            critics=True,
            critic_classifiers={"sentiment": FakeCritic()},
            reconciler_classifiers={"sentiment": FakeReconciler()},
            sampling_runs=5, consensus_threshold=4,
        )
        result = pd.read_csv(out)
        assert "stale" not in result.loc[0, "sentiment_votes"]


def test_missing_collaborator_category_raises_before_any_call(categories):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        input_csv = d / "in.csv"
        pd.DataFrame({"text": ["a"]}).to_csv(input_csv, index=False)

        class BoomSampler:
            def classify(self, text):
                raise AssertionError("should not be called")

        with pytest.raises(ValueError, match="critic_classifiers"):
            classify_csv(
                input_csv, "text", BoomSampler(), categories,
                critics=True, critic_classifiers={}, reconciler_classifiers={"sentiment": FakeReconciler()},
                sampling_runs=5, consensus_threshold=4,
            )


def test_invalid_consensus_threshold_raises_before_any_call(categories):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        input_csv = d / "in.csv"
        pd.DataFrame({"text": ["a"]}).to_csv(input_csv, index=False)

        class BoomSampler:
            def classify(self, text):
                raise AssertionError("should not be called")

        with pytest.raises(ValueError, match="consensus_threshold"):
            classify_csv(
                input_csv, "text", BoomSampler(), categories,
                critics=True,
                critic_classifiers={"sentiment": FakeCritic()},
                reconciler_classifiers={"sentiment": FakeReconciler()},
                sampling_runs=3, consensus_threshold=5,
            )


def test_category_name_collides_with_generated_audit_column_raises(sentiment_category):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        input_csv = d / "in.csv"
        pd.DataFrame({"text": ["a"]}).to_csv(input_csv, index=False)
        colliding = Category(name="sentiment_votes", description="d", labels=[Label(value="x", description="d")])
        cats = [sentiment_category, colliding]

        class BoomSampler:
            def classify(self, text):
                raise AssertionError("should not be called")

        with pytest.raises(ValueError, match="sentiment_votes"):
            classify_csv(
                input_csv, "text", BoomSampler(), cats,
                critics=True,
                critic_classifiers={"sentiment": FakeCritic(), "sentiment_votes": FakeCritic()},
                reconciler_classifiers={"sentiment": FakeReconciler(), "sentiment_votes": FakeReconciler()},
                sampling_runs=5, consensus_threshold=4,
            )


# --- CLI-level: model routing, validation, resource guard --------------------


def test_cli_default_and_overridden_model_routing(categories, monkeypatch, tmp_path):
    from query_classification import cli

    # This test asserts on exact, unprefixed model-id strings, so it must not
    # be affected by a developer's real .env possibly setting
    # DEFAULT_LLM_PROVIDER=cerebus. main() calls load_dotenv(override=True),
    # which re-reads the real .env file and would silently re-set that var
    # (clobbering a plain monkeypatch.delenv) and prefix every constructed
    # model id here — no-op load_dotenv so only this test's own env applies.
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("DEFAULT_LLM_PROVIDER", raising=False)

    input_csv = tmp_path / "in.csv"
    pd.DataFrame({"text": ["hello"]}).to_csv(input_csv, index=False)

    constructed = []
    original_classifier = Classifier

    class RecordingClassifier(original_classifier):
        def __init__(self, *args, **kwargs):
            constructed.append(kwargs.get("model_id") or (args[0] if args else None))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(cli, "Classifier", RecordingClassifier)
    monkeypatch.setattr(
        cli, "classify_csv", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classify.py",
            "--input", str(input_csv),
            "--column", "text",
            "--output", str(tmp_path / "out.csv"),
            "--critics",
            "--model", "model-main",
            "--critic-model", "model-critic",
        ],
    )
    cli.main()
    # sampling classifier uses --model; critic uses --critic-model; reconciler
    # falls back to --model since --reconciler-model wasn't given.
    assert "model-main" in constructed
    assert "model-critic" in constructed
    assert constructed.count("model-main") >= 2  # sampling + reconciler(s)


def test_cli_invalid_sampling_temperature_exits_before_any_llm_call(monkeypatch, tmp_path, capsys):
    from query_classification import cli

    input_csv = tmp_path / "in.csv"
    pd.DataFrame({"text": ["hello"]}).to_csv(input_csv, index=False)

    def boom(*a, **kw):
        raise AssertionError("classify_csv should not be reached")

    monkeypatch.setattr(cli, "classify_csv", boom)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classify.py",
            "--input", str(input_csv),
            "--column", "text",
            "--critics",
            "--sampling-temperature", "-1",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert exc_info.value.code == 1
    assert "sampling-temperature" in capsys.readouterr().out


def test_cli_resource_guard_fires_for_critics_even_with_explicit_main_resources(monkeypatch, tmp_path):
    from query_classification import cli, resources

    input_csv = tmp_path / "in.csv"
    pd.DataFrame({"text": ["hello"]}).to_csv(input_csv, index=False)

    monkeypatch.setattr(resources, "RESOURCES_DIR", tmp_path / "nonexistent")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classify.py",
            "--input", str(input_csv),
            "--column", "text",
            "--critics",
            "--categories", str(resources.DEFAULT_CATEGORIES_FILE),
            "--system-prompt", str(resources.DEFAULT_SYSTEM_PROMPT_FILE),
        ],
    )
    with pytest.raises(SystemExit):
        cli.main()


# --- Schema/prompt builders (direct-import exceptions per ADR-002) ----------


def test_build_critic_model_validator(sentiment_category):
    model = build_critic_model(sentiment_category)
    model.model_validate({"challenges": False, "proposed_label": None, "argument": "no compelling alternative"})
    model.model_validate({"challenges": True, "proposed_label": "negative", "argument": "x"})
    with pytest.raises(Exception):
        model.model_validate({"challenges": True, "proposed_label": None, "argument": "x"})
    with pytest.raises(Exception):
        model.model_validate({"challenges": False, "proposed_label": "negative", "argument": "x"})


def test_build_reconciler_model_enforces_label_bounds(sentiment_category):
    model = build_reconciler_model(sentiment_category)
    model.model_validate({"labels": ["positive"], "reasoning": "x"})
    with pytest.raises(Exception):
        model.model_validate({"labels": ["a", "b", "c", "d"], "reasoning": "x"})


def test_build_critic_and_reconciler_prompts_are_nonempty(sentiment_category):
    critic_prompt = build_critic_prompt(
        sentiment_category.name, sentiment_category.description, "positive; negative"
    )
    reconciler_prompt = build_reconciler_prompt(
        sentiment_category.name, sentiment_category.description, "positive; negative"
    )
    assert sentiment_category.name in critic_prompt
    assert sentiment_category.name in reconciler_prompt


def test_sampling_temperature_included_in_completion_kwargs():
    from query_classification.schema import build_classification_model

    model = build_classification_model([])
    c = Classifier("m", "sp", model, temperature=0.7)
    assert c._completion_kwargs([])["temperature"] == 0.7
    c_default = Classifier("m", "sp", model)
    assert "temperature" not in c_default._completion_kwargs([])


# --- classify_csv's classifier/models public-API validation (spec 4, AR-1.2) -


class DummyModel:
    def classify(self, text):
        return {"sentiment": ["positive"]}


def test_classify_csv_rejects_neither_classifier_nor_models(categories):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        input_csv = d / "in.csv"
        pd.DataFrame({"text": ["a"]}).to_csv(input_csv, index=False)
        out = d / "out.csv"
        with pytest.raises(ValueError, match="exactly one of classifier or models"):
            classify_csv(input_csv, "text", None, categories, output_path=out)
        assert not out.exists()


def test_classify_csv_rejects_both_classifier_and_models(categories):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        input_csv = d / "in.csv"
        pd.DataFrame({"text": ["a"]}).to_csv(input_csv, index=False)
        out = d / "out.csv"
        models = {"a": DummyModel(), "b": DummyModel()}
        with pytest.raises(ValueError, match="mutually exclusive"):
            classify_csv(
                input_csv, "text", DummyModel(), categories, output_path=out, models=models
            )
        assert not out.exists()


def test_classify_csv_rejects_critics_with_models(categories):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        input_csv = d / "in.csv"
        pd.DataFrame({"text": ["a"]}).to_csv(input_csv, index=False)
        out = d / "out.csv"
        models = {"a": DummyModel(), "b": DummyModel()}
        with pytest.raises(ValueError, match="critics and models are mutually exclusive"):
            classify_csv(
                input_csv, "text", None, categories, output_path=out,
                critics=True, models=models,
                critic_classifiers={"sentiment": FakeCritic()},
                reconciler_classifiers={"sentiment": FakeReconciler()},
            )
        assert not out.exists()


def test_classify_csv_rejects_empty_models_dict(categories):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        input_csv = d / "in.csv"
        pd.DataFrame({"text": ["a"]}).to_csv(input_csv, index=False)
        out = d / "out.csv"
        with pytest.raises(ValueError, match="at least 2 entries"):
            classify_csv(input_csv, "text", None, categories, output_path=out, models={})
        assert not out.exists()


def test_classify_csv_rejects_single_model(categories):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        input_csv = d / "in.csv"
        pd.DataFrame({"text": ["a"]}).to_csv(input_csv, index=False)
        out = d / "out.csv"
        with pytest.raises(ValueError, match="at least 2 entries"):
            classify_csv(
                input_csv, "text", None, categories, output_path=out,
                models={"a": DummyModel()},
            )
        assert not out.exists()


def test_classify_csv_rejects_whitespace_only_model_key(categories):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        input_csv = d / "in.csv"
        pd.DataFrame({"text": ["a"]}).to_csv(input_csv, index=False)
        out = d / "out.csv"
        models = {"a": DummyModel(), "   ": DummyModel()}
        with pytest.raises(ValueError, match="non-empty"):
            classify_csv(input_csv, "text", None, categories, output_path=out, models=models)
        assert not out.exists()


def test_classify_csv_models_mode_rejects_cross_category_audit_collision():
    # A category literally named "sentiment_by_model" collides with the
    # "sentiment" category's own generated audit column (INV-8).
    colliding_categories = [
        Category(name="sentiment", description="d", labels=[Label(value="positive", description="p")]),
        Category(name="sentiment_by_model", description="d", labels=[Label(value="x", description="x")]),
    ]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        input_csv = d / "in.csv"
        pd.DataFrame({"text": ["a"]}).to_csv(input_csv, index=False)
        out = d / "out.csv"
        models = {"a": DummyModel(), "b": DummyModel()}
        with pytest.raises(ValueError, match="both generate column"):
            classify_csv(
                input_csv, "text", None, colliding_categories, output_path=out, models=models
            )
        assert not out.exists()
