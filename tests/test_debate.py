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
from query_classification import debate, multi_model
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


# --- CLI-level: --models mode (spec 4, FR-1.1/FR-1.2) ------------------------


def _write_sentiment_categories_json(path):
    """A minimal, self-contained categories file (mirrors the ``categories``
    fixture's "sentiment" category) for --models CLI-level tests that need to
    control exactly which keys a fake ``Classifier.classify`` must return."""
    payload = {
        "categories": [
            {
                "name": "sentiment",
                "description": "Overall sentiment.",
                "labels": [
                    {"value": "positive", "description": "Happy."},
                    {"value": "negative", "description": "Sad."},
                ],
            }
        ]
    }
    path.write_text(json.dumps(payload))


def _neutralize_real_dotenv(monkeypatch, cli):
    # See test_cli_default_and_overridden_model_routing's comment: main() calls
    # load_dotenv(override=True), which re-reads this repo's real .env (which
    # sets DEFAULT_LLM_PROVIDER=cerebus) and would clobber a plain monkeypatch
    # env value. No-op load_dotenv so only each test's own env applies, and
    # explicitly clear the two vars these tests care about either way.
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("DEFAULT_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("CEREBUS_MODE", raising=False)


def test_cli_models_and_critics_mutually_exclusive_exits_before_any_llm_call(
    monkeypatch, tmp_path, capsys
):
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
            "--models", "a", "b",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert exc_info.value.code == 1
    assert "--models" in capsys.readouterr().out


def test_cli_models_single_value_overrides_model_silently(monkeypatch, tmp_path):
    """spec 4 FR-1.1 points 3/9: a single-value --models silently overrides
    --model and resolves to ordinary single-classifier mode -- it must not
    require a second value (this used to be an error; the redesign made
    plain `--models a` behave exactly like `--model a`)."""
    from query_classification import cli

    _neutralize_real_dotenv(monkeypatch, cli)

    input_csv = tmp_path / "in.csv"
    pd.DataFrame({"text": ["hello"]}).to_csv(input_csv, index=False)

    constructed = []
    original_classifier = Classifier

    class RecordingClassifier(original_classifier):
        def __init__(self, *args, **kwargs):
            constructed.append(kwargs)
            super().__init__(*args, **kwargs)

    calls = {}

    def fake_classify_csv(*a, **kw):
        calls.update(kw)

    monkeypatch.setattr(cli, "Classifier", RecordingClassifier)
    monkeypatch.setattr(cli, "classify_csv", fake_classify_csv)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classify.py",
            "--input", str(input_csv),
            "--column", "text",
            "--output", str(tmp_path / "out.csv"),
            "--models", "only-one",
        ],
    )
    cli.main()

    assert len(constructed) == 1
    assert constructed[0]["model_id"] == "only-one"
    assert calls["models"] is None
    assert calls["classifier"] is not None


def test_cli_models_duplicate_value_uses_occurrence_suffixed_keys(monkeypatch, tmp_path):
    """spec 4 AR-1.3: --models given 2+ values allows duplicates -- this used
    to be an error; the redesign preserves duplicates and disambiguates
    repeated model ids with occurrence-suffixed classifier keys ("a#1",
    "a#2"), while the once-only id keeps its bare key."""
    from query_classification import cli

    _neutralize_real_dotenv(monkeypatch, cli)

    input_csv = tmp_path / "in.csv"
    pd.DataFrame({"text": ["hello"]}).to_csv(input_csv, index=False)

    calls = {}

    def fake_classify_csv(*a, **kw):
        calls.update(kw)

    monkeypatch.setattr(cli, "classify_csv", fake_classify_csv)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classify.py",
            "--input", str(input_csv),
            "--column", "text",
            "--output", str(tmp_path / "out.csv"),
            "--models", "a", "a", "b",
        ],
    )
    cli.main()

    assert calls["classifier"] is None
    assert set(calls["models"].keys()) == {"a#1", "a#2", "b"}


def test_cli_models_empty_value_exits_before_any_llm_call(monkeypatch, tmp_path, capsys):
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
            "--models", "a", "   ",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert exc_info.value.code == 1
    assert "--models" in capsys.readouterr().out


def test_cli_models_with_cerebus_azure_mode_exits_before_any_llm_call(
    monkeypatch, tmp_path, capsys
):
    from query_classification import cli

    _neutralize_real_dotenv(monkeypatch, cli)
    monkeypatch.setenv("CEREBUS_MODE", "azure")

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
            "--models", "a", "b",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert exc_info.value.code == 1
    assert "--models" in capsys.readouterr().out


def test_cli_models_mode_end_to_end_writes_audit_columns(monkeypatch, tmp_path):
    from query_classification import cli

    _neutralize_real_dotenv(monkeypatch, cli)

    input_csv = tmp_path / "in.csv"
    pd.DataFrame({"text": ["hello", "world"]}).to_csv(input_csv, index=False)
    categories_file = tmp_path / "categories.json"
    _write_sentiment_categories_json(categories_file)
    out = tmp_path / "out.csv"

    def fake_classify(self, text):
        return {"sentiment": ["positive"]}

    monkeypatch.setattr(Classifier, "classify", fake_classify)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classify.py",
            "--input", str(input_csv),
            "--column", "text",
            "--output", str(out),
            "--categories", str(categories_file),
            "--models", "model-a", "model-b",
        ],
    )
    cli.main()

    cols = list(pd.read_csv(out).columns)
    for suffix in multi_model.AUDIT_COLUMN_SUFFIXES:
        assert f"sentiment{suffix}" in cols
    assert "sentiment" in cols


def test_cli_models_mode_honors_allow_new_labels_carve_out(monkeypatch, tmp_path):
    """Spec 4 FR-1.2's carve-out: unlike plain mode (which forces
    allow_new_labels=True regardless of the flag), --models mode -- a
    --critics sibling, not a plain-mode variant -- must honor
    --allow-new-labels exactly like --critics does."""
    from query_classification import cli

    _neutralize_real_dotenv(monkeypatch, cli)

    input_csv = tmp_path / "in.csv"
    pd.DataFrame({"text": ["hello"]}).to_csv(input_csv, index=False)

    def _system_prompts_for(extra_args):
        constructed = []
        original_classifier = Classifier

        class RecordingClassifier(original_classifier):
            def __init__(self, *args, **kwargs):
                constructed.append(kwargs)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(cli, "Classifier", RecordingClassifier)
        monkeypatch.setattr(cli, "classify_csv", lambda *a, **kw: None)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "classify.py",
                "--input", str(input_csv),
                "--column", "text",
                "--output", str(tmp_path / "out.csv"),
                "--models", "a", "b",
                *extra_args,
            ],
        )
        cli.main()
        assert len(constructed) == 2
        prompts = {kw["system_prompt"] for kw in constructed}
        assert len(prompts) == 1  # identical prompt shared by every model
        return prompts.pop()

    forbidden_prompt = _system_prompts_for([])
    assert "Do not invent a new label." in forbidden_prompt

    allowed_prompt = _system_prompts_for(["--allow-new-labels"])
    assert "Do not invent a new label." not in allowed_prompt


def test_cli_models_mode_cerebus_routes_all_models_through_same_gateway_config(
    monkeypatch, tmp_path
):
    from query_classification import cli

    _neutralize_real_dotenv(monkeypatch, cli)

    fake_gateway = {
        "api_base": "https://gw.example.test/v1",
        "api_key": "fake-cerebus-key",
        "extra_headers": {
            "x-portkey-api-key": "fake-cerebus-key",
            "x-portkey-provider": "openai",
        },
    }
    monkeypatch.setattr(cli, "build_cerebus_completion_kwargs", lambda: fake_gateway)

    input_csv = tmp_path / "in.csv"
    pd.DataFrame({"text": ["hello"]}).to_csv(input_csv, index=False)

    constructed = []
    original_classifier = Classifier

    class RecordingClassifier(original_classifier):
        def __init__(self, *args, **kwargs):
            constructed.append(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(cli, "Classifier", RecordingClassifier)
    monkeypatch.setattr(cli, "classify_csv", lambda *a, **kw: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classify.py",
            "--input", str(input_csv),
            "--column", "text",
            "--output", str(tmp_path / "out.csv"),
            "--cerebus",
            "--models", "a", "b",
        ],
    )
    cli.main()

    assert len(constructed) == 2
    model_ids = {kw["model_id"] for kw in constructed}
    # _model_id() applies cerebus_model_id()'s "openai/" prefix to every
    # --models entry, exactly as it already does for --critic-model/
    # --reconciler-model.
    assert model_ids == {"openai/a", "openai/b"}
    for kw in constructed:
        assert kw["api_base"] == fake_gateway["api_base"]
        assert kw["api_key"] == fake_gateway["api_key"]
        assert kw["extra_headers"] == fake_gateway["extra_headers"]


# --- _resolve_classifier_models: Shared Resolution Truth Table (spec 4) ---


def _resolve(monkeypatch, argv, cerebus_mode=None):
    from query_classification import cli

    if cerebus_mode is None:
        monkeypatch.delenv("CEREBUS_MODE", raising=False)
    else:
        monkeypatch.setenv("CEREBUS_MODE", cerebus_mode)
    ns = cli.build_parser().parse_args(["--input", "unused.csv", "--column", "text", *argv])
    return cli._resolve_classifier_models(ns)


def test_resolve_models_alone_ignores_default_n_classifiers(monkeypatch):
    """The regression guard: `--models a b c` alone (no --n-classifiers) must
    resolve to exactly [a, b, c] -- this is the exact case that an earlier
    draft of this redesign silently broke by requiring --n-classifiers to
    match len(--models)."""
    assert _resolve(monkeypatch, ["--models", "a", "b", "c"]) == ["a", "b", "c"]


def test_resolve_models_ignores_n_classifiers_even_when_explicitly_conflicting(monkeypatch):
    """--n-classifiers plays no role at all once --models has 2+ values --
    not even as a validation check -- so an explicitly conflicting value is
    silently ignored rather than raising."""
    resolved = _resolve(monkeypatch, ["--models", "a", "b", "c", "--n-classifiers", "99"])
    assert resolved == ["a", "b", "c"]


def test_resolve_n_classifiers_zero_always_errors_even_with_models(monkeypatch):
    """The `--n-classifiers >= 1` floor is universal -- it is the one check
    that still applies even when --models has 2+ values and would otherwise
    make --n-classifiers irrelevant."""
    with pytest.raises(ValueError, match="--n-classifiers"):
        _resolve(monkeypatch, ["--models", "a", "b", "c", "--n-classifiers", "0"])


def test_resolve_model_replication_with_cerebus_azure_mode_succeeds_for_same_model(monkeypatch):
    """CEREBUS_MODE=azure only blocks 2+ DISTINCT resolved models -- a single
    model replicated via --n-classifiers must still be allowed."""
    resolved = _resolve(
        monkeypatch, ["--model", "a", "--n-classifiers", "3"], cerebus_mode="azure"
    )
    assert resolved == ["a", "a", "a"]


def test_resolve_models_distinct_with_cerebus_azure_mode_errors(monkeypatch):
    with pytest.raises(ValueError, match="CEREBUS_MODE=azure"):
        _resolve(monkeypatch, ["--models", "a", "b"], cerebus_mode="azure")


def test_build_classifier_dict_collision_guard_rejects_pathological_model_id(monkeypatch):
    """AR-1.3's collision guard: if a supplied model id already looks like an
    occurrence-suffixed key another position would independently produce
    (e.g. a literal "a#1" alongside two plain "a"s), raise a clear error
    instead of silently letting one classifier overwrite the other."""
    from query_classification import cli

    with pytest.raises(ValueError, match="collision"):
        cli._build_classifier_dict(["a", "a", "a#1"], lambda m: m)


# --- Repeated-instance behavior end-to-end (spec 4 FR-1.2/FR-1.3/FR-1.7) --


def test_cli_n_classifiers_replication_calls_every_instance_exactly_once(monkeypatch, tmp_path):
    """spec 4 FR-1.2 for the --n-classifiers replication case: --model x
    --n-classifiers 5 must construct 5 independent classifier instances and
    call every one of them exactly once per row -- not fewer, e.g. from an
    accidental dict-collapse bug when all 5 share the same model id."""
    from query_classification import cli

    _neutralize_real_dotenv(monkeypatch, cli)

    input_csv = tmp_path / "in.csv"
    pd.DataFrame({"text": ["hello"]}).to_csv(input_csv, index=False)
    categories_file = tmp_path / "categories.json"
    _write_sentiment_categories_json(categories_file)
    out = tmp_path / "out.csv"

    call_count = {"n": 0}

    def fake_classify(self, text):
        call_count["n"] += 1
        return {"sentiment": ["positive"]}

    monkeypatch.setattr(Classifier, "classify", fake_classify)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classify.py",
            "--input", str(input_csv),
            "--column", "text",
            "--output", str(out),
            "--categories", str(categories_file),
            "--model", "same-model",
            "--n-classifiers", "5",
        ],
    )
    cli.main()

    assert call_count["n"] == 5
    cols = list(pd.read_csv(out).columns)
    for suffix in multi_model.AUDIT_COLUMN_SUFFIXES:
        assert f"sentiment{suffix}" in cols


def test_cli_n_classifiers_repeated_instance_tie_break_by_construction_order(monkeypatch, tmp_path):
    """spec 4 FR-1.3 for the new same-model-repeated case this redesign adds
    (the original implementation only ever tested distinct-model ties): when
    two classifier instances share a model id, the plurality tie-break must
    still go to the earlier-constructed occurrence ("same-model#1"), even
    when it resolves SLOWER than the later occurrence ("same-model#2")."""
    from query_classification import cli

    _neutralize_real_dotenv(monkeypatch, cli)

    input_csv = tmp_path / "in.csv"
    pd.DataFrame({"text": ["hello"]}).to_csv(input_csv, index=False)
    categories_file = tmp_path / "categories.json"
    _write_sentiment_categories_json(categories_file)
    out = tmp_path / "out.csv"

    construction_order = []
    original_init = Classifier.__init__

    def recording_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        construction_order.append(id(self))

    def fake_classify(self, text):
        idx = construction_order.index(id(self))
        if idx == 0:
            time.sleep(0.08)
            return {"sentiment": ["positive"]}
        return {"sentiment": ["negative"]}

    monkeypatch.setattr(Classifier, "__init__", recording_init)
    monkeypatch.setattr(Classifier, "classify", fake_classify)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classify.py",
            "--input", str(input_csv),
            "--column", "text",
            "--output", str(out),
            "--categories", str(categories_file),
            "--model", "same-model",
            "--n-classifiers", "2",
        ],
    )
    cli.main()

    result = pd.read_csv(out)
    assert result.loc[0, "sentiment"] == "['positive']"


def test_cli_n_classifiers_repeated_instance_partial_failure_keeps_suffixed_keys_distinct(
    monkeypatch, tmp_path
):
    """spec 4 AR-1.3/FR-1.7: when one occurrence of a repeated model fails,
    the audit trail must attribute the failure to its own occurrence-
    suffixed key ("same-model#1") while the other occurrence ("same-
    model#2") is recorded as a normal success -- proving the two occurrences
    remain independently addressable, not merged."""
    from query_classification import cli

    _neutralize_real_dotenv(monkeypatch, cli)

    input_csv = tmp_path / "in.csv"
    pd.DataFrame({"text": ["hello"]}).to_csv(input_csv, index=False)
    categories_file = tmp_path / "categories.json"
    _write_sentiment_categories_json(categories_file)
    out = tmp_path / "out.csv"

    construction_order = []
    original_init = Classifier.__init__

    def recording_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        construction_order.append(id(self))

    def fake_classify(self, text):
        idx = construction_order.index(id(self))
        if idx == 0:
            raise RuntimeError("model boom")
        return {"sentiment": ["positive"]}

    monkeypatch.setattr(Classifier, "__init__", recording_init)
    monkeypatch.setattr(Classifier, "classify", fake_classify)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classify.py",
            "--input", str(input_csv),
            "--column", "text",
            "--output", str(out),
            "--categories", str(categories_file),
            "--model", "same-model",
            "--n-classifiers", "2",
        ],
    )
    cli.main()

    result = pd.read_csv(out)
    errors = json.loads(result.loc[0, "sentiment_model_errors"])
    by_model = json.loads(result.loc[0, "sentiment_by_model"])
    assert set(errors.keys()) == {"same-model#1"}
    assert set(by_model.keys()) == {"same-model#2"}
