"""Offline tests for the experiment runner (no live network or Hub calls).

Covers spec 2's FR-1.x/FR-2.x/FR-3.x Verify conditions. HF loading is
exercised via a fake ``datasets`` module (injected through ``sys.modules``)
whose function/class signatures mirror the real ones closely enough that a
wrong call would fail; LLM calls use fake ``Classifier``-shaped objects
exposing ``.classify(text) -> dict`` (dataset_io/induction tests) or a
monkeypatched ``Classifier.classify`` (experiment.py CLI-level tests, since
``experiment.py`` constructs real ``Classifier`` instances itself).

Traceability (test function -> spec Verify condition):
  FR-1.1  test_load_local_split_train_only / test_load_local_split_test_only /
          test_classify_missing_test_file_errors
  FR-1.2  test_validate_hf_id_* / test_load_hf_splits_missing_split_lists_available /
          test_load_hf_splits_resolved_sha_best_effort
  FR-1.3  test_load_local_split_verbatim_numeric_labels /
          test_resolve_hf_labels_unsupported_feature_type
  FR-1.4  test_missing_column_errors_list_available / test_text_label_column_must_differ
  FR-1.5  test_load_hf_splits_missing_extra_hint
  FR-1.6  test_split_availability_* (five cases)
  FR-1.7  test_project_and_filter_drops_empty_rows_once
  FR-1.8  test_resolve_hf_labels_mixed_withheld / test_run_end_to_end_withheld_test_labels /
          test_unseen_test_labels_recorded
  FR-2.1  test_induce_rng_exact_sequence / test_induce_seed_sensitivity_over_cap
  FR-2.3  test_induce_prompt_size_preflight / test_induce_truncation_marker
  FR-2.4  test_induce_rejects_missing_or_invented_labels / test_induce_rejects_duplicate_label
  FR-2.5  test_induce_rejects_reserved_sentinel / test_classify_rejects_supplied_sentinel
  FR-2.6  test_induce_sanitizes_failure
  FR-2.7  test_induce_rejects_degenerate_input
  FR-3.1  test_induce_subcommand_writes_categories_only
  FR-3.2  test_classify_rejects_multi_category_file
  FR-3.3  test_run_end_to_end
  FR-3.4  test_run_dir_overwrite_preserves_unrelated_files / test_overwrite_same_path_categories_noop /
          test_run_dir_refuses_nonempty_without_overwrite
  FR-3.5  test_plain_mode_closed_vocabulary / test_limit_truncates_before_classifying /
          test_run_subcommand_has_induction_flags
  FR-3.6  test_gold_column_rename_and_collision / test_critics_audit_column_collision /
          test_collision_against_projected_away_column
  FR-3.7  (this file)
  FR-3.8  test_completeness_check_partial_failure / test_completeness_check_allow_partial
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from query_classification.categories import Category, Label, load_categories
from query_classification.classifier import Classifier
from query_classification.dataset_io import (
    load_hf_splits,
    load_local_split,
    project_and_filter,
    resolve_hf_labels,
    validate_hf_id,
)
from query_classification.experiment import build_parser, main
from query_classification.induction import InductionOutcome, induce
from query_classification.prompts import build_induction_prompt
from query_classification.schema import build_induction_model


# ---------------------------------------------------------------------------
# Fake `datasets` module
# ---------------------------------------------------------------------------


class FakeClassLabel:
    """Mirrors real datasets.ClassLabel: .names is the ordered label list;
    int2str raises ValueError on a negative index, exactly like the real one."""

    def __init__(self, names):
        self.names = list(names)

    def int2str(self, i):
        if i < 0:
            raise ValueError(f"Invalid negative index: {i}")
        return self.names[i]


class FakeValue:
    def __init__(self, dtype):
        self.dtype = dtype


class FakeDataset:
    def __init__(self, table: dict, features: dict):
        self._table = dict(table)  # column name -> list of raw values
        self.features = features

    @property
    def column_names(self):
        return list(self._table.keys())

    def __getitem__(self, column):
        return self._table[column]

    def remove_columns(self, columns):
        remaining = {k: v for k, v in self._table.items() if k not in columns}
        remaining_features = {k: v for k, v in self.features.items() if k not in columns}
        return FakeDataset(remaining, remaining_features)

    def to_pandas(self):
        return pd.DataFrame(self._table)


class FakeDatasetDict(dict):
    pass


def _fake_load_dataset(path, name=None, revision=None):
    """Signature mirrors datasets.load_dataset(path, name=, revision=,
    split=None) with split intentionally omitted by callers."""
    return _FAKE_REGISTRY[path]


_FAKE_REGISTRY: dict[str, FakeDatasetDict] = {}


@pytest.fixture
def fake_datasets_module(monkeypatch):
    """Inject a fake `datasets` module into sys.modules for the duration of a
    test, with real-matching ClassLabel/Value semantics."""
    fake_mod = types.ModuleType("datasets")
    fake_mod.ClassLabel = FakeClassLabel
    fake_mod.Value = FakeValue
    fake_mod.load_dataset = _fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)
    _FAKE_REGISTRY.clear()
    yield _FAKE_REGISTRY
    _FAKE_REGISTRY.clear()


@pytest.fixture(autouse=True)
def _no_real_datasets_import_leak(monkeypatch):
    """Belt-and-suspenders: ensure a stray real `datasets` import never
    silently satisfies a test meant to exercise the fake/missing-extra path."""
    yield


# ---------------------------------------------------------------------------
# dataset_io.py — HF id validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_id",
    ["./missing", "/etc/passwd", "foo.git", "a/b/c", "csv", "imagefolder", ""],
)
def test_validate_hf_id_rejects(bad_id):
    with pytest.raises(ValueError):
        validate_hf_id(bad_id)


@pytest.mark.parametrize("good_id", ["squad", "nyu-mll/glue"])
def test_validate_hf_id_accepts(good_id):
    validate_hf_id(good_id)  # must not raise


# ---------------------------------------------------------------------------
# dataset_io.py — resolve_hf_labels
# ---------------------------------------------------------------------------


def test_resolve_hf_labels_classlabel_withheld(fake_datasets_module):
    ds = FakeDataset(
        {"label": [-1, 0, 1]}, {"label": FakeClassLabel(["neg", "pos"])}
    )
    assert resolve_hf_labels(ds, "label") == [None, "neg", "pos"]


def test_resolve_hf_labels_mixed_withheld(fake_datasets_module):
    ds = FakeDataset(
        {"label": [0, -1, 1, -1]}, {"label": FakeClassLabel(["neg", "pos"])}
    )
    assert resolve_hf_labels(ds, "label") == ["neg", None, "pos", None]


def test_resolve_hf_labels_string_value_passthrough(fake_datasets_module):
    ds = FakeDataset({"label": ["a", "b"]}, {"label": FakeValue("string")})
    assert resolve_hf_labels(ds, "label") == ["a", "b"]


def test_resolve_hf_labels_unsupported_feature_type(fake_datasets_module):
    ds = FakeDataset({"label": [1.0, 2.0]}, {"label": FakeValue("float32")})
    with pytest.raises(ValueError, match="unsupported feature type"):
        resolve_hf_labels(ds, "label")


# ---------------------------------------------------------------------------
# dataset_io.py — load_hf_splits
# ---------------------------------------------------------------------------


def _register_fake_hf_dataset(registry, hf_id, splits: dict):
    registry[hf_id] = FakeDatasetDict(splits)


def test_load_hf_splits_missing_split_lists_available(fake_datasets_module):
    _register_fake_hf_dataset(
        fake_datasets_module,
        "owner/name",
        {"train": FakeDataset({"text": ["a"], "label": [0]}, {"label": FakeClassLabel(["x"])})},
    )
    with pytest.raises(ValueError, match=r"Available splits: \['train'\]"):
        load_hf_splits("owner/name", None, None, "train", "test", "text", "label")


def test_load_hf_splits_only_fetches_requested_role(fake_datasets_module):
    _register_fake_hf_dataset(
        fake_datasets_module,
        "owner/name",
        {"train": FakeDataset({"text": ["a"], "label": [0]}, {"label": FakeClassLabel(["x"])})},
    )
    train_df, test_df, meta = load_hf_splits(
        "owner/name", None, None, "train", None, "text", "label"
    )
    assert train_df is not None
    assert test_df is None
    assert meta["source_column_names"]["test"] is None
    assert meta["source_column_names"]["train"] == ["text", "label"]


def test_load_hf_splits_overwrites_label_column_with_resolved_values(fake_datasets_module):
    _register_fake_hf_dataset(
        fake_datasets_module,
        "owner/name",
        {
            "train": FakeDataset(
                {"text": ["a", "b", "c"], "label": [-1, 0, 1], "extra": [1, 2, 3]},
                {"label": FakeClassLabel(["neg", "pos"])},
            )
        },
    )
    train_df, _, meta = load_hf_splits(
        "owner/name", None, None, "train", None, "text", "label"
    )
    assert list(train_df["label"]) == [None, "neg", "pos"]
    assert list(train_df.columns) == ["text", "label"]  # projected, 'extra' dropped
    assert meta["source_column_names"]["train"] == ["text", "label", "extra"]


def test_load_hf_splits_resolved_sha_best_effort(fake_datasets_module, monkeypatch):
    _register_fake_hf_dataset(
        fake_datasets_module,
        "owner/name",
        {"train": FakeDataset({"text": ["a"], "label": [0]}, {"label": FakeClassLabel(["x"])})},
    )

    class _FailingHfApi:
        def dataset_info(self, *a, **k):
            raise RuntimeError("network unreachable")

    monkeypatch.setattr("query_classification.dataset_io.HfApi", _FailingHfApi)
    _, _, meta = load_hf_splits("owner/name", None, "main", "train", None, "text", "label")
    assert meta["resolved_revision"] == "main"  # falls back to requested revision, never raises


def test_load_hf_splits_missing_extra_hint(monkeypatch):
    monkeypatch.delitem(sys.modules, "datasets", raising=False)
    monkeypatch.setattr(
        "builtins.__import__",
        _raise_module_not_found_for_datasets(__import__),
    )
    with pytest.raises(ModuleNotFoundError, match=r"pip install '\.\[hf\]'"):
        load_hf_splits("owner/name", None, None, "train", None, "text", "label")


def _raise_module_not_found_for_datasets(real_import):
    def _fake_import(name, *args, **kwargs):
        if name == "datasets":
            raise ModuleNotFoundError("No module named 'datasets'")
        return real_import(name, *args, **kwargs)

    return _fake_import


# ---------------------------------------------------------------------------
# dataset_io.py — load_local_split / project_and_filter
# ---------------------------------------------------------------------------


def test_load_local_split_verbatim_numeric_labels(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("text,label\nhello,001\nworld,002\n")
    df, cols = load_local_split(str(p), "text", "label")
    assert list(df["label"]) == ["001", "002"]
    assert cols == ["text", "label"]


def test_load_local_split_preserves_literal_na_text(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text('text,label\n"NA",pos\n"None",neg\n')
    df, _ = load_local_split(str(p), "text", "label")
    assert list(df["text"]) == ["NA", "None"]


def test_load_local_split_missing_column_raises(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="not found"):
        load_local_split(str(p), "text", "label")


def test_project_and_filter_drops_empty_rows_once():
    df = pd.DataFrame(
        {"text": ["hello", "", "  ", "world"], "label": ["pos", "pos", "neg", "neg"]}
    )
    filtered, dropped = project_and_filter(df, "text", "label", is_train_role=True)
    assert dropped == [1, 2]
    assert len(filtered) == 2


def test_project_and_filter_label_only_dropped_for_train_role():
    df = pd.DataFrame({"text": ["hello", "world"], "label": ["pos", ""]})
    filtered, dropped = project_and_filter(df, "text", "label", is_train_role=False)
    assert dropped == []  # test role: missing label doesn't drop the row
    assert len(filtered) == 2


# ---------------------------------------------------------------------------
# induction.py
# ---------------------------------------------------------------------------


class FakeInductionClassifier:
    def __init__(self, response=None, error=None, responses=None, n_labels=None):
        # ``responses`` (a list) is consumed one per call, in order -- for
        # tests that need successive calls to return different responses.
        # ``response`` (singular) keeps returning the same value every call,
        # for everything else.
        self.response = response
        self.error = error
        self.responses = responses
        self.calls = []
        # A real Classifier always exposes ``classification_model`` (the
        # schema it was built with) -- induce()'s arity guard reads it, so
        # this fake must too. Inferred from the response's own
        # ``description_i`` keys when not given explicitly, so the ~15
        # existing call sites in this file don't all need an explicit
        # ``n_labels=`` just to satisfy the guard.
        if n_labels is None:
            sample = (responses[0] if responses else response) or {}
            n_labels = sum(1 for k in sample if k.startswith("description_")) or None
        self.classification_model = build_induction_model(n_labels) if n_labels else None

    def classify(self, text):
        self.calls.append(text)
        if self.error is not None:
            raise self.error
        if self.responses is not None:
            return self.responses[len(self.calls) - 1]
        return self.response


def _induction_response(labels):
    """Build a positional induction response for ``len(labels)`` labels.
    ``labels`` only determines the count and each description's readable
    content -- position, not label identity, is what induce() actually maps
    by (AR-2.1); pass labels already in sorted order when a test cares about
    which description lands on which label."""
    return {
        "category_description": "desc",
        **{f"description_{i}": f"{lbl} desc" for i, lbl in enumerate(labels, start=1)},
    }


def _train_df(rows):
    return pd.DataFrame(rows, columns=["text", "label"])


def test_induce_rng_exact_sequence():
    # Two over-cap labels (5 and 4 candidates), one at-cap label (2 == cap).
    rows = (
        [(f"neg{i}", "neg") for i in range(5)]
        + [(f"neu{i}", "neu") for i in range(2)]
        + [(f"pos{i}", "pos") for i in range(4)]
    )
    df = _train_df(rows)
    classifier = FakeInductionClassifier(response=_induction_response(["neg", "neu", "pos"]))
    outcome = induce(
        df, "text", "label", "cat", classifier,
        seed=42, examples_per_label=2, max_example_chars=100, max_prompt_chars=100_000,
    )
    import random

    ref = random.Random(42)
    neg_positions = list(range(5))
    pos_positions = list(range(5, 5 + 4))  # positions 5..8 in the ORIGINAL frame order
    # induce() groups by label in original row order, so recompute per-label position lists:
    neg_idx = [i for i, r in enumerate(rows) if r[1] == "neg"]
    neu_idx = [i for i, r in enumerate(rows) if r[1] == "neu"]
    pos_idx = [i for i, r in enumerate(rows) if r[1] == "pos"]
    expected_neg = ref.sample(neg_idx, 2)
    # neu is at-cap (2 == 2): consumes no RNG state, uses stable order (all of it)
    expected_pos = ref.sample(pos_idx, 2)

    got_neg = [e["position"] for e in outcome.sampled_examples["neg"]]
    got_neu = [e["position"] for e in outcome.sampled_examples["neu"]]
    got_pos = [e["position"] for e in outcome.sampled_examples["pos"]]
    assert got_neg == expected_neg
    assert got_neu == neu_idx
    assert got_pos == expected_pos

    # Reproducibility: same seed, same result.
    classifier2 = FakeInductionClassifier(response=_induction_response(["neg", "neu", "pos"]))
    outcome2 = induce(
        df, "text", "label", "cat", classifier2,
        seed=42, examples_per_label=2, max_example_chars=100, max_prompt_chars=100_000,
    )
    assert outcome2.sampled_examples == outcome.sampled_examples


def test_induce_below_cap_label_contributes_all_its_examples():
    # A label with FEWER examples than --examples-per-label's cap must
    # contribute exactly all of them, in stable source order -- not just the
    # at-or-above-cap cases test_induce_rng_exact_sequence already covers.
    df = _train_df([("p1", "pos"), ("p2", "pos"), ("p3", "pos"), ("n1", "neg")])
    classifier = FakeInductionClassifier(response=_induction_response(["neg", "pos"]))
    outcome = induce(
        df, "text", "label", "cat", classifier,
        seed=0, examples_per_label=20, max_example_chars=100, max_prompt_chars=100_000,
    )
    assert len(outcome.sampled_examples["pos"]) == 3
    assert [e["position"] for e in outcome.sampled_examples["pos"]] == [0, 1, 2]


def test_induce_at_cap_label_consumes_no_rng_state():
    # Growing an over-cap label's pool must not perturb an at-or-below-cap
    # label's own selection.
    rows_a = [(f"neg{i}", "neg") for i in range(5)] + [("neu0", "neu"), ("neu1", "neu")]
    rows_b = [(f"neg{i}", "neg") for i in range(7)] + [("neu0", "neu"), ("neu1", "neu")]
    for rows in (rows_a, rows_b):
        df = _train_df(rows)
        classifier = FakeInductionClassifier(response=_induction_response(["neg", "neu"]))
        outcome = induce(
            df, "text", "label", "cat", classifier,
            seed=7, examples_per_label=2, max_example_chars=100, max_prompt_chars=100_000,
        )
        neu_idx = [i for i, r in enumerate(rows) if r[1] == "neu"]
        assert [e["position"] for e in outcome.sampled_examples["neu"]] == neu_idx


# --- FR-2.1: --induction-examples total-budget sizing mode -----------------


def test_induce_requires_exactly_one_sizing_mode():
    df = _train_df([("a", "pos"), ("b", "neg")])
    classifier = FakeInductionClassifier(response=_induction_response(["pos", "neg"]))
    with pytest.raises(ValueError, match="examples_per_label"):
        induce(
            df, "text", "label", "cat", classifier,
            seed=0, max_example_chars=100, max_prompt_chars=100_000,
        )
    assert classifier.calls == []
    with pytest.raises(ValueError, match="examples_per_label"):
        induce(
            df, "text", "label", "cat", classifier,
            seed=0, examples_per_label=5, induction_examples=10,
            max_example_chars=100, max_prompt_chars=100_000,
        )
    assert classifier.calls == []


def test_induce_budget_quotas_sum_to_exactly_n_largest_remainder():
    rows = [(f"a{i}", "a") for i in range(999)] + [(f"b{i}", "b") for i in range(999)] + [
        (f"c{i}", "c") for i in range(999)
    ]
    df = _train_df(rows)
    classifier = FakeInductionClassifier(response=_induction_response(["a", "b", "c"]))
    outcome = induce(
        df, "text", "label", "cat", classifier,
        seed=0, induction_examples=100, max_example_chars=100, max_prompt_chars=100_000,
    )
    quotas = {label: len(entries) for label, entries in outcome.sampled_examples.items()}
    assert quotas == {"a": 34, "b": 33, "c": 33}
    assert sum(quotas.values()) == 100


def test_induce_budget_n_equals_n_labels_gives_every_label_one():
    df = _train_df([("a1", "a"), ("b1", "b"), ("c1", "c")])
    classifier = FakeInductionClassifier(response=_induction_response(["a", "b", "c"]))
    outcome = induce(
        df, "text", "label", "cat", classifier,
        seed=0, induction_examples=3, max_example_chars=100, max_prompt_chars=100_000,
    )
    quotas = {label: len(entries) for label, entries in outcome.sampled_examples.items()}
    assert quotas == {"a": 1, "b": 1, "c": 1}


def test_induce_budget_redistributes_shortfall_to_labels_with_spare_rows():
    # "a" has only 2 usable rows; a budget of 10 over 3 labels must still
    # select 10 in total, with the freed 8 slots re-divided among "b"/"c".
    rows = [("a0", "a"), ("a1", "a")] + [(f"b{i}", "b") for i in range(999)] + [
        (f"c{i}", "c") for i in range(999)
    ]
    df = _train_df(rows)
    classifier = FakeInductionClassifier(response=_induction_response(["a", "b", "c"]))
    outcome = induce(
        df, "text", "label", "cat", classifier,
        seed=0, induction_examples=10, max_example_chars=100, max_prompt_chars=100_000,
    )
    quotas = {label: len(entries) for label, entries in outcome.sampled_examples.items()}
    assert quotas["a"] == 2
    assert sum(quotas.values()) == 10


def test_induce_budget_cascading_shortfall_across_multiple_scarce_labels():
    rows = [("a0", "a")] + [("b0", "b"), ("b1", "b")] + [(f"c{i}", "c") for i in range(999)]
    df = _train_df(rows)
    classifier = FakeInductionClassifier(response=_induction_response(["a", "b", "c"]))
    outcome = induce(
        df, "text", "label", "cat", classifier,
        seed=0, induction_examples=20, max_example_chars=100, max_prompt_chars=100_000,
    )
    quotas = {label: len(entries) for label, entries in outcome.sampled_examples.items()}
    assert quotas == {"a": 1, "b": 2, "c": 17}


def test_induce_budget_all_labels_scarce_sums_to_available_rows():
    df = _train_df([("a0", "a"), ("a1", "a"), ("b0", "b"), ("b1", "b"), ("b2", "b"),
                     ("c0", "c"), ("c1", "c"), ("c2", "c"), ("c3", "c")])
    classifier = FakeInductionClassifier(response=_induction_response(["a", "b", "c"]))
    outcome = induce(
        df, "text", "label", "cat", classifier,
        seed=0, induction_examples=100, max_example_chars=100, max_prompt_chars=100_000,
    )
    quotas = {label: len(entries) for label, entries in outcome.sampled_examples.items()}
    assert quotas == {"a": 2, "b": 3, "c": 4}


def test_induce_budget_below_label_count_rejected_with_zero_calls():
    df = _train_df([("a", "x"), ("b", "y"), ("c", "z")])
    classifier = FakeInductionClassifier(response=_induction_response(["x", "y", "z"]))
    with pytest.raises(ValueError, match="induction-examples"):
        induce(
            df, "text", "label", "cat", classifier,
            seed=0, induction_examples=2, max_example_chars=100, max_prompt_chars=100_000,
        )
    assert classifier.calls == []


def test_induce_budget_mode_is_seed_deterministic_but_not_cross_label_independent():
    rows_a = [(f"a{i}", "a") for i in range(10)] + [(f"b{i}", "b") for i in range(10)]
    rows_b = [(f"a{i}", "a") for i in range(20)] + [(f"b{i}", "b") for i in range(10)]

    def _positions(rows, seed=3):
        df = _train_df(rows)
        classifier = FakeInductionClassifier(response=_induction_response(["a", "b"]))
        outcome = induce(
            df, "text", "label", "cat", classifier,
            seed=seed, induction_examples=8, max_example_chars=100, max_prompt_chars=100_000,
        )
        return {lbl: [e["position"] for e in entries] for lbl, entries in outcome.sampled_examples.items()}

    # Determinism: same seed/split/flags -> identical example set.
    assert _positions(rows_a) == _positions(rows_a)
    # NOT cross-label independent (unlike --examples-per-label): growing "a"'s
    # pool changes "b"'s own quota/selection too, since quotas are computed
    # jointly. This is the guarantee --induction-examples explicitly gives up.
    assert _positions(rows_a)["b"] != _positions(rows_b)["b"]


# --- AR-2.1's arity guard (internal defensive check, spec 2 Spec Deviations #2) --


def test_induce_arity_guard_passes_for_a_correctly_sized_model():
    df = _train_df([("a", "pos"), ("b", "neg"), ("c", "neu")])
    classifier = FakeInductionClassifier(response=_induction_response(["pos", "neg", "neu"]), n_labels=3)
    outcome = induce(
        df, "text", "label", "cat", classifier,
        seed=0, examples_per_label=5, max_example_chars=100, max_prompt_chars=100_000,
    )
    assert len(outcome.category.labels) == 3


def test_induce_arity_guard_rejects_a_wrong_sized_model():
    # Only this direction actually proves the guard uses an exact field-set
    # comparison, not a substring/count check that would also match
    # "category_description" itself (verified live: a naive "description" in
    # name predicate counts 4 fields for a 3-label model).
    df = _train_df([("a", "pos"), ("b", "neg"), ("c", "neu")])
    classifier = FakeInductionClassifier(response=_induction_response(["pos", "neg"]), n_labels=2)
    with pytest.raises(ValueError, match="internal inconsistency|expected"):
        induce(
            df, "text", "label", "cat", classifier,
            seed=0, examples_per_label=5, max_example_chars=100, max_prompt_chars=100_000,
        )
    assert classifier.calls == []


def test_induce_positional_mapping_survives_out_of_order_completion_risk():
    """Distinguishable per-position descriptions must land on the correctly
    SORTED label, not the order labels happened to appear in the source
    rows -- this is what would catch an off-by-one or an accidental re-sort."""
    df = _train_df([("z1", "zeta"), ("a1", "alpha"), ("m1", "mu")])  # source order != sorted order
    classifier = FakeInductionClassifier(
        response={"category_description": "d", "description_1": "D-ALPHA", "description_2": "D-MU", "description_3": "D-ZETA"}
    )
    outcome = induce(
        df, "text", "label", "cat", classifier,
        seed=0, examples_per_label=5, max_example_chars=100, max_prompt_chars=100_000,
    )
    by_value = {lbl.value: lbl.description for lbl in outcome.category.labels}
    assert by_value == {"alpha": "D-ALPHA", "mu": "D-MU", "zeta": "D-ZETA"}


def test_induce_truncation_marker():
    df = _train_df([("0123456789extra", "pos"), ("short", "pos")])
    classifier = FakeInductionClassifier(response=_induction_response(["pos"]))
    outcome = induce(
        df, "text", "label", "cat", classifier,
        seed=0, examples_per_label=5, max_example_chars=10, max_prompt_chars=100_000,
    )
    texts = {e["text"] for e in outcome.sampled_examples["pos"]}
    assert "0123456789…[truncated]" in texts
    assert "short" in texts


def test_induce_prompt_size_preflight():
    df = _train_df([("a" * 50, "pos"), ("b" * 50, "neg")])
    classifier = FakeInductionClassifier(response=_induction_response(["pos", "neg"]))
    with pytest.raises(ValueError, match="max_prompt_chars"):
        induce(
            df, "text", "label", "cat", classifier,
            seed=0, examples_per_label=5, max_example_chars=100, max_prompt_chars=5,
        )
    assert classifier.calls == []  # zero calls before the preflight check


@pytest.mark.parametrize("bad_label", ["none", "NONE", "  none  ", "none - foo", "None - Bar"])
def test_induce_rejects_reserved_sentinel(bad_label):
    df = _train_df([("a", bad_label), ("b", "other")])
    classifier = FakeInductionClassifier(response=_induction_response([bad_label, "other"]))
    with pytest.raises(ValueError, match="reserved"):
        induce(
            df, "text", "label", "cat", classifier,
            seed=0, examples_per_label=5, max_example_chars=100, max_prompt_chars=100_000,
        )
    assert classifier.calls == []


def test_induce_sanitizes_failure():
    df = _train_df([("a", "pos"), ("b", "neg")])
    classifier = FakeInductionClassifier(error=RuntimeError("secret-token=abc123"), n_labels=2)
    with pytest.raises(RuntimeError) as exc_info:
        induce(
            df, "text", "label", "cat", classifier,
            seed=0, examples_per_label=5, max_example_chars=100, max_prompt_chars=100_000,
        )
    assert "secret-token" not in str(exc_info.value)


@pytest.mark.parametrize(
    "df",
    [pd.DataFrame(columns=["text", "label"]), _train_df([("a", "pos")])],
)
def test_induce_rejects_degenerate_input(df):
    classifier = FakeInductionClassifier(response=_induction_response(["pos"]))
    if df.empty:
        with pytest.raises(ValueError, match="empty"):
            induce(
                df, "text", "label", "cat", classifier,
                seed=0, examples_per_label=5, max_example_chars=100, max_prompt_chars=100_000,
            )
    else:
        # A single distinct label is allowed (not degenerate).
        outcome = induce(
            df, "text", "label", "cat", classifier,
            seed=0, examples_per_label=5, max_example_chars=100, max_prompt_chars=100_000,
        )
        assert isinstance(outcome, InductionOutcome)


def test_induce_category_round_trips():
    df = _train_df([("a", "pos"), ("b", "neg")])
    classifier = FakeInductionClassifier(response=_induction_response(["pos", "neg"]))
    outcome = induce(
        df, "text", "label", "cat", classifier,
        seed=0, examples_per_label=5, max_example_chars=100, max_prompt_chars=100_000,
    )
    assert Category.model_validate(outcome.category.model_dump()) == outcome.category


def _parse_induction_payload(sent):
    from query_classification.induction import _DATA_END, _DATA_START

    start = sent.index(_DATA_START) + len(_DATA_START)
    end = sent.index(_DATA_END)
    return json.loads(sent[start:end])


def test_induce_single_call_sees_all_labels_grouped():
    df = _train_df([("a1", "neg"), ("a2", "neg"), ("b1", "neu"), ("c1", "pos"), ("c2", "pos")])
    classifier = FakeInductionClassifier(response=_induction_response(["neg", "neu", "pos"]))
    induce(
        df, "text", "label", "cat", classifier,
        seed=0, examples_per_label=5, max_example_chars=100, max_prompt_chars=100_000,
    )
    assert len(classifier.calls) == 1
    sent = classifier.calls[0]
    for label in ("neg", "neu", "pos"):
        assert f'"{label}"' in sent


def test_induce_payload_is_an_ordered_positional_array_not_an_object():
    """Direct Verify condition for FR-2.2/AR-2.2: the request payload must be
    an array with explicit 1-based positions in sorted-label order, not an
    object keyed by label value -- an implementation that merely preserves
    dict insertion order would pass every other test in this file but fail
    this one."""
    df = _train_df([("a1", "neg"), ("a2", "neg"), ("b1", "neu"), ("c1", "pos"), ("c2", "pos")])
    classifier = FakeInductionClassifier(response=_induction_response(["neg", "neu", "pos"]))
    induce(
        df, "text", "label", "cat", classifier,
        seed=0, examples_per_label=5, max_example_chars=100, max_prompt_chars=100_000,
    )
    payload = _parse_induction_payload(classifier.calls[0])
    assert isinstance(payload["labels"], list)
    assert [entry["label"] for entry in payload["labels"]] == ["neg", "neu", "pos"]
    assert [entry["position"] for entry in payload["labels"]] == [1, 2, 3]
    for entry in payload["labels"]:
        assert entry["examples"], f"label {entry['label']!r} has no examples"


# ---------------------------------------------------------------------------
# schema.py / prompts.py additions
# ---------------------------------------------------------------------------


def test_build_induction_model_positional_required_fields():
    for n in (1, 3):
        model = build_induction_model(n)
        expected_fields = {"category_description"} | {f"description_{i}" for i in range(1, n + 1)}
        assert set(model.model_fields) == expected_fields
        payload = {"category_description": "d", **{f"description_{i}": f"d{i}" for i in range(1, n + 1)}}
        validated = model.model_validate(payload)
        assert validated.description_1 == "d1"


def test_build_induction_model_rejects_short_response():
    model = build_induction_model(3)
    with pytest.raises(Exception):
        model.model_validate(
            {"category_description": "d", "description_1": "d1", "description_2": "d2"}
        )


def test_build_induction_model_rejects_zero_labels():
    with pytest.raises(ValueError, match="n_labels"):
        build_induction_model(0)


def test_build_induction_model_serialization_stays_strict_compatible():
    """Regression guard for AR-2.1's design choice: required positional fields,
    not a length-bounded list. Cannot prove PROVIDER rejection of a bounded
    list (that would need a live call) -- only that the local schema keeps
    the shape verified against the installed litellm/pydantic stack: strict
    mode, every description field required, and no minItems/maxItems (which
    strict structured output does not support and would silently degrade
    every induction call to JSON-object-mode fallback)."""
    from litellm.utils import type_to_response_format_param

    model = build_induction_model(3)
    serialized = type_to_response_format_param(model)
    schema = serialized["json_schema"]
    assert serialized["json_schema"]["strict"] is True
    required = schema["schema"]["required"]
    assert {"category_description", "description_1", "description_2", "description_3"} == set(required)
    assert "minItems" not in json.dumps(schema)
    assert "maxItems" not in json.dumps(schema)


def test_build_induction_prompt_non_empty_no_placeholders():
    text = build_induction_prompt()
    assert text.strip()
    assert "{" not in text or "}" not in text.replace("{{", "").replace("}}", "")


def test_build_induction_prompt_states_positional_contract_not_label_echo():
    """Regression guard for the risk that the schema changes but the prompt
    doesn't: Classifier falls back to JSON-object mode when structured output
    is unavailable, and in that mode the prompt text is the ONLY statement of
    the description_i field names -- a stale prompt would silently break only
    against a live provider, invisible to every offline test."""
    text = build_induction_prompt()
    assert "description_" in text
    assert "exact name" not in text


# ---------------------------------------------------------------------------
# experiment.py — CLI-level, run_config, collisions, artifacts
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_classify(monkeypatch):
    """Monkeypatch Classifier.classify (experiment.py constructs real
    Classifier instances, so a duck-typed fake object isn't wireable here)."""
    log = {"calls": 0}

    def _classify(self, text):
        log["calls"] += 1
        fields = set(self.classification_model.model_fields.keys())
        if "category_description" in fields:
            # Induction role (spec 2's positional description_i fields, one
            # per label position -- variable count, no fixed field set).
            return {f: "d" for f in fields}
        if fields == {"challenges", "proposed_label", "argument"}:
            return {"challenges": False, "proposed_label": None, "argument": "no challenge"}
        if fields == {"labels", "reasoning"}:
            return {"labels": ["positive"], "reasoning": "r"}
        cat_name = next(iter(fields))
        return {cat_name: ["positive"]}

    monkeypatch.setattr(Classifier, "classify", _classify)
    return log


def _write_csv(path, rows, columns):
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path, index=False)
    return path


def _write_categories(path, name="sentiment"):
    path.write_text(
        json.dumps(
            {
                "categories": [
                    {
                        "name": name,
                        "description": "d",
                        "labels": [
                            {"value": "positive", "description": "p"},
                            {"value": "negative", "description": "n"},
                        ],
                    }
                ]
            }
        )
    )
    return path


def _run_main(argv):
    sys.argv = ["experiment.py"] + argv
    try:
        main()
        return 0
    except SystemExit as e:
        return e.code


def test_run_subcommand_has_induction_flags():
    parser = build_parser()
    ns = parser.parse_args(
        [
            "run", "--train-file", "t.csv", "--test-file", "e.csv",
            "--text-column", "text", "--label-column", "label",
            "--category-name", "cat", "--run-dir", "d",
            "--seed", "5", "--examples-per-label", "3", "--induction-model", "m",
        ]
    )
    assert ns.seed == 5
    assert ns.examples_per_label == 3
    assert ns.induction_model == "m"


def test_induce_alone_resolves_induction_model_fallback():
    parser = build_parser()
    ns = parser.parse_args(
        ["induce", "--train-file", "t.csv", "--text-column", "text",
         "--label-column", "label", "--category-name", "cat", "--run-dir", "d",
         "--model", "base-model"]
    )
    assert ns.model == ["base-model"]  # nargs="+"; resolved to a single value downstream
    assert ns.induction_model is None  # resolved to --model inside _construct_classifiers


def test_run_end_to_end(tmp_path, fake_classify):
    train = _write_csv(
        tmp_path / "train.csv",
        [("great product", "positive"), ("bad product", "negative")] * 3,
        ["text", "label"],
    )
    test = _write_csv(
        tmp_path / "test.csv",
        [("fantastic", "positive"), ("awful", "negative")],
        ["text", "label"],
    )
    run_dir = tmp_path / "run1"
    code = _run_main(
        [
            "run", "--train-file", str(train), "--test-file", str(test),
            "--text-column", "text", "--label-column", "label",
            "--category-name", "sentiment", "--run-dir", str(run_dir),
        ]
    )
    assert code in (0, None)
    for name in ("categories.json", "train.csv", "test.csv", "test_classified.csv",
                 "induction_prompt.txt", "sampled_examples.json", "run_config.json"):
        assert (run_dir / name).exists(), name
    config = json.loads((run_dir / "run_config.json").read_text())
    assert config["status"] == "completed"
    assert config["categories_source"] == "induced"
    assert config["label_values"] == ["negative", "positive"]
    # Default invocation (neither sizing flag given) must still work and
    # record the resolved literal default -- proves _resolve_induction_sizing
    # doesn't leave the ordinary default path passing (None, None) into
    # induce()'s exactly-one-non-None check.
    assert config["examples_per_label"] == 20
    assert config["induction_examples"] is None


def test_induce_subcommand_writes_categories_only(tmp_path, fake_classify):
    train = _write_csv(tmp_path / "train.csv", [("a", "positive"), ("b", "negative")], ["text", "label"])
    run_dir = tmp_path / "runind"
    code = _run_main(
        ["induce", "--train-file", str(train), "--text-column", "text",
         "--label-column", "label", "--category-name", "sentiment", "--run-dir", str(run_dir)]
    )
    assert code in (0, None)
    assert (run_dir / "categories.json").exists()
    assert not (run_dir / "test_classified.csv").exists()
    assert fake_classify["calls"] == 1  # only the induction call, zero classification calls


def test_induce_budget_mode_end_to_end_writes_valid_categories(tmp_path, fake_classify):
    train = _write_csv(
        tmp_path / "train.csv",
        [("a1", "positive"), ("a2", "positive"), ("b1", "negative"), ("b2", "negative")],
        ["text", "label"],
    )
    run_dir = tmp_path / "runind_budget"
    code = _run_main(
        ["induce", "--train-file", str(train), "--text-column", "text",
         "--label-column", "label", "--category-name", "sentiment", "--run-dir", str(run_dir),
         "--induction-examples", "4"]
    )
    assert code in (0, None)
    from query_classification.categories import load_categories

    cats = load_categories(run_dir / "categories.json")
    assert {lbl.value for lbl in cats[0].labels} == {"positive", "negative"}
    config = json.loads((run_dir / "run_config.json").read_text())
    assert config["examples_per_label"] is None
    assert config["induction_examples"] == 4


def test_induce_examples_flags_mutually_exclusive(tmp_path, fake_classify):
    train = _write_csv(tmp_path / "train.csv", [("a", "positive"), ("b", "negative")], ["text", "label"])
    code = _run_main(
        ["induce", "--train-file", str(train), "--text-column", "text",
         "--label-column", "label", "--category-name", "sentiment", "--run-dir", str(tmp_path / "r"),
         "--examples-per-label", "5", "--induction-examples", "10"]
    )
    assert code == 1
    assert fake_classify["calls"] == 0


def test_induce_budget_below_label_count_rejected(tmp_path, fake_classify):
    train = _write_csv(
        tmp_path / "train.csv",
        [("a", "positive"), ("b", "negative"), ("c", "neutral")],
        ["text", "label"],
    )
    code = _run_main(
        ["induce", "--train-file", str(train), "--text-column", "text",
         "--label-column", "label", "--category-name", "sentiment", "--run-dir", str(tmp_path / "r"),
         "--induction-examples", "2"]
    )
    assert code == 1
    assert fake_classify["calls"] == 0


def test_induce_all_rows_filtered_reports_fr_2_7_error_not_arity_error(tmp_path, fake_classify, capsys):
    """Regression guard: computing n_labels ahead of build_induction_model
    must not let a generic "n_labels must be >= 1" error preempt FR-2.7's own
    actionable "train_df is empty after filtering" message for an
    all-rows-filtered train split."""
    train = _write_csv(tmp_path / "train.csv", [("", "positive"), ("   ", "negative")], ["text", "label"])
    code = _run_main(
        ["induce", "--train-file", str(train), "--text-column", "text",
         "--label-column", "label", "--category-name", "sentiment", "--run-dir", str(tmp_path / "r")]
    )
    assert code == 1
    assert fake_classify["calls"] == 0
    out = capsys.readouterr().out
    assert "empty" in out
    assert "n_labels" not in out


def test_classify_rejects_multi_category_file(tmp_path, fake_classify):
    cats_path = tmp_path / "cats.json"
    cats_path.write_text(
        json.dumps(
            {
                "categories": [
                    {"name": "a", "description": "d", "labels": [{"value": "x", "description": "d"}]},
                    {"name": "b", "description": "d", "labels": [{"value": "y", "description": "d"}]},
                ]
            }
        )
    )
    test = _write_csv(tmp_path / "test.csv", [("hi", "x")], ["text", "label"])
    code = _run_main(
        ["classify", "--test-file", str(test), "--text-column", "text",
         "--label-column", "label", "--categories", str(cats_path), "--run-dir", str(tmp_path / "r")]
    )
    assert code == 1


def test_classify_rejects_supplied_sentinel(tmp_path):
    cats_path = tmp_path / "cats.json"
    cats_path.write_text(
        json.dumps(
            {"categories": [{"name": "a", "description": "d",
                             "labels": [{"value": "none", "description": "d"}]}]}
        )
    )
    test = _write_csv(tmp_path / "test.csv", [("hi", "x")], ["text", "label"])
    code = _run_main(
        ["classify", "--test-file", str(test), "--text-column", "text",
         "--label-column", "label", "--categories", str(cats_path), "--run-dir", str(tmp_path / "r")]
    )
    assert code == 1


def test_missing_column_errors_list_available(tmp_path):
    test = _write_csv(tmp_path / "test.csv", [("hi", "x")], ["text", "label"])
    cats_path = _write_categories(tmp_path / "cats.json")
    code = _run_main(
        ["classify", "--test-file", str(test), "--text-column", "nope",
         "--label-column", "label", "--categories", str(cats_path), "--run-dir", str(tmp_path / "r")]
    )
    assert code == 1


def test_text_label_column_must_differ(tmp_path):
    test = _write_csv(tmp_path / "test.csv", [("hi", "x")], ["text", "label"])
    cats_path = _write_categories(tmp_path / "cats.json")
    code = _run_main(
        ["classify", "--test-file", str(test), "--text-column", "text",
         "--label-column", "text", "--categories", str(cats_path), "--run-dir", str(tmp_path / "r")]
    )
    assert code == 1


@pytest.mark.parametrize(
    "kwargs,expect_error",
    [
        ({"subcommand": "classify", "has_train": False, "has_test": False, "has_categories": True}, True),
        ({"subcommand": "run", "has_train": False, "has_test": True, "has_categories": False}, True),
        ({"subcommand": "run", "has_train": True, "has_test": True, "has_categories": True}, True),
    ],
)
def test_split_availability_error_cases(tmp_path, kwargs, expect_error):
    args = ["run" if kwargs["subcommand"] == "run" else "classify"]
    cats_path = _write_categories(tmp_path / "cats.json")
    if kwargs["has_test"]:
        test = _write_csv(tmp_path / "test.csv", [("hi", "x")], ["text", "label"])
        args += ["--test-file", str(test)]
    if kwargs["has_train"]:
        train = _write_csv(tmp_path / "train.csv", [("hi", "x")], ["text", "label"])
        args += ["--train-file", str(train)]
    if kwargs["has_categories"]:
        args += ["--categories", str(cats_path)]
    args += ["--text-column", "text", "--label-column", "label", "--run-dir", str(tmp_path / "r")]
    if kwargs["subcommand"] == "run" and kwargs["has_train"] and not kwargs["has_categories"]:
        args += ["--category-name", "sentiment"]
    code = _run_main(args)
    assert (code == 1) == expect_error


def test_gold_column_rename_and_collision(tmp_path, fake_classify):
    test = _write_csv(tmp_path / "test.csv", [("hi", "positive")], ["label", "label"])  # placeholder, fixed below
    # Build explicit CSV to control the exact column named 'sentiment' (== label_column value name clash test)
    (tmp_path / "test.csv").write_text("text,label\nhi,positive\n")
    cats_path = _write_categories(tmp_path / "cats.json", name="sentiment")
    run_dir = tmp_path / "r"
    code = _run_main(
        ["classify", "--test-file", str(tmp_path / "test.csv"), "--text-column", "text",
         "--label-column", "label", "--categories", str(cats_path), "--run-dir", str(run_dir)]
    )
    assert code in (0, None)
    df = pd.read_csv(run_dir / "test_classified.csv")
    assert "label_gold" in df.columns
    assert "sentiment" in df.columns

    # Now the collision case: a pre-existing label_gold column.
    (tmp_path / "test2.csv").write_text("text,label,label_gold\nhi,positive,bogus\n")
    run_dir2 = tmp_path / "r2"
    code2 = _run_main(
        ["classify", "--test-file", str(tmp_path / "test2.csv"), "--text-column", "text",
         "--label-column", "label", "--categories", str(cats_path), "--run-dir", str(run_dir2)]
    )
    assert code2 == 1
    assert not (run_dir2 / "test_classified.csv").exists()


def test_collision_against_projected_away_column(tmp_path, fake_classify):
    # 'sentiment' (the category name) exists as an EXTRA source column that
    # AR-1.3's projection would otherwise silently drop before Task 4 ever
    # sees it — the collision must still be detected via source_column_names.
    (tmp_path / "test.csv").write_text("text,label,sentiment\nhi,positive,bogus\n")
    cats_path = _write_categories(tmp_path / "cats.json", name="sentiment")
    code = _run_main(
        ["classify", "--test-file", str(tmp_path / "test.csv"), "--text-column", "text",
         "--label-column", "label", "--categories", str(cats_path), "--run-dir", str(tmp_path / "r")]
    )
    assert code == 1


def test_critics_audit_column_collision(tmp_path, fake_classify):
    (tmp_path / "test.csv").write_text("text,label,sentiment_votes\nhi,positive,bogus\n")
    cats_path = _write_categories(tmp_path / "cats.json", name="sentiment")
    code = _run_main(
        ["classify", "--test-file", str(tmp_path / "test.csv"), "--text-column", "text",
         "--label-column", "label", "--categories", str(cats_path), "--run-dir", str(tmp_path / "r"),
         "--critics"]
    )
    assert code == 1


def test_critics_end_to_end_audit_columns(tmp_path, fake_classify):
    test = _write_csv(tmp_path / "test.csv", [("good", "positive")], ["text", "label"])
    cats_path = _write_categories(tmp_path / "cats.json")
    run_dir = tmp_path / "r"
    code = _run_main(
        ["classify", "--test-file", str(test), "--text-column", "text", "--label-column", "label",
         "--categories", str(cats_path), "--run-dir", str(run_dir), "--critics",
         "--sampling-runs", "3", "--consensus-threshold", "2"]
    )
    assert code in (0, None)
    df = pd.read_csv(run_dir / "test_classified.csv")
    for suffix in ("_initial", "_votes", "_challenged", "_reconciled"):
        assert f"sentiment{suffix}" in df.columns


def test_plain_mode_closed_vocabulary(tmp_path, fake_classify):
    test = _write_csv(tmp_path / "test.csv", [("good", "positive")], ["text", "label"])
    cats_path = _write_categories(tmp_path / "cats.json")
    run_dir = tmp_path / "r"
    _run_main(
        ["classify", "--test-file", str(test), "--text-column", "text", "--label-column", "label",
         "--categories", str(cats_path), "--run-dir", str(run_dir)]
    )
    # allow_new_labels defaults False in BOTH modes here (unlike classify.py).
    from query_classification.experiment import _construct_classifiers
    import argparse

    cats = load_categories(cats_path)
    args = argparse.Namespace(
        model=["m"], allow_new_labels=False, system_prompt=None, task_description=None,
        extra_prompt=None, critics=False, sampling_temperature=0.7, retries=3, api_base=None,
        critic_model=None, reconciler_model=None,
    )
    classifiers, _ = _construct_classifiers(args, cats, will_induce=False, will_classify=True)
    assert "none - " not in classifiers["classification"].system_prompt


def test_limit_truncates_before_classifying(tmp_path, fake_classify):
    rows = [(f"row{i}", "positive") for i in range(10)]
    test = _write_csv(tmp_path / "test.csv", rows, ["text", "label"])
    cats_path = _write_categories(tmp_path / "cats.json")
    run_dir = tmp_path / "r"
    code = _run_main(
        ["classify", "--test-file", str(test), "--text-column", "text", "--label-column", "label",
         "--categories", str(cats_path), "--run-dir", str(run_dir), "--test-limit", "3"]
    )
    assert code in (0, None)
    df = pd.read_csv(run_dir / "test_classified.csv")
    assert len(df) == 3
    assert df["sentiment"].isna().sum() == 0


def test_run_end_to_end_withheld_test_labels(tmp_path, fake_classify):
    test = _write_csv(tmp_path / "test.csv", [("a", ""), ("b", "")], ["text", "label"])
    cats_path = _write_categories(tmp_path / "cats.json")
    run_dir = tmp_path / "r"
    code = _run_main(
        ["classify", "--test-file", str(test), "--text-column", "text", "--label-column", "label",
         "--categories", str(cats_path), "--run-dir", str(run_dir)]
    )
    assert code in (0, None)
    config = json.loads((run_dir / "run_config.json").read_text())
    assert config["test_labels"] == "withheld"
    df = pd.read_csv(run_dir / "test_classified.csv")
    assert "label_gold" not in df.columns


def test_unseen_test_labels_recorded(tmp_path, fake_classify):
    test = _write_csv(
        tmp_path / "test.csv", [("a", "positive"), ("b", "neutral")], ["text", "label"]
    )
    cats_path = _write_categories(tmp_path / "cats.json")
    run_dir = tmp_path / "r"
    _run_main(
        ["classify", "--test-file", str(test), "--text-column", "text", "--label-column", "label",
         "--categories", str(cats_path), "--run-dir", str(run_dir)]
    )
    config = json.loads((run_dir / "run_config.json").read_text())
    assert config["unseen_test_labels"] == ["neutral"]


def test_run_dir_refuses_nonempty_without_overwrite(tmp_path, fake_classify):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    (run_dir / "notes.txt").write_text("keep me")
    test = _write_csv(tmp_path / "test.csv", [("a", "positive")], ["text", "label"])
    cats_path = _write_categories(tmp_path / "cats.json")
    code = _run_main(
        ["classify", "--test-file", str(test), "--text-column", "text", "--label-column", "label",
         "--categories", str(cats_path), "--run-dir", str(run_dir)]
    )
    assert code == 1
    assert (run_dir / "notes.txt").exists()


def test_run_dir_overwrite_preserves_unrelated_files(tmp_path, fake_classify):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    (run_dir / "notes.txt").write_text("keep me")
    (run_dir / "run_config.json").write_text('{"status": "completed"}')
    test = _write_csv(tmp_path / "test.csv", [("a", "positive")], ["text", "label"])
    cats_path = _write_categories(tmp_path / "cats.json")
    code = _run_main(
        ["classify", "--test-file", str(test), "--text-column", "text", "--label-column", "label",
         "--categories", str(cats_path), "--run-dir", str(run_dir), "--overwrite"]
    )
    assert code in (0, None)
    assert (run_dir / "notes.txt").read_text() == "keep me"


def test_overwrite_same_path_categories_noop(tmp_path, fake_classify):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    cats_path = _write_categories(run_dir / "categories.json")
    before = cats_path.read_bytes()
    test = _write_csv(tmp_path / "test.csv", [("a", "positive")], ["text", "label"])
    code = _run_main(
        ["classify", "--test-file", str(test), "--text-column", "text", "--label-column", "label",
         "--categories", str(cats_path), "--run-dir", str(run_dir), "--overwrite"]
    )
    assert code in (0, None)
    assert cats_path.read_bytes() == before


def test_supplied_categories_byte_verbatim_copy(tmp_path, fake_classify):
    cats_path = tmp_path / "cats.json"
    cats_path.write_text(  # deliberately unusual formatting, to prove no re-serialization
        '{"categories":   [{"name":"sentiment","description":"d",'
        '"labels":[{"value":"positive","description":"p"},'
        '{"value":"negative","description":"n"}]}]}'
    )
    test = _write_csv(tmp_path / "test.csv", [("a", "positive")], ["text", "label"])
    run_dir = tmp_path / "r"
    _run_main(
        ["classify", "--test-file", str(test), "--text-column", "text", "--label-column", "label",
         "--categories", str(cats_path), "--run-dir", str(run_dir)]
    )
    assert (run_dir / "categories.json").read_bytes() == cats_path.read_bytes()


def test_completeness_check_partial_failure(tmp_path, monkeypatch):
    calls = {"n": 0}

    def _flaky_classify(self, text):
        calls["n"] += 1
        fields = set(self.classification_model.model_fields.keys())
        cat_name = next(iter(fields))
        if calls["n"] <= 2:
            raise RuntimeError("boom")
        return {cat_name: ["positive"]}

    monkeypatch.setattr(Classifier, "classify", _flaky_classify)
    rows = [(f"row{i}", "positive") for i in range(5)]
    test = _write_csv(tmp_path / "test.csv", rows, ["text", "label"])
    cats_path = _write_categories(tmp_path / "cats.json")
    run_dir = tmp_path / "r"
    code = _run_main(
        ["classify", "--test-file", str(test), "--text-column", "text", "--label-column", "label",
         "--categories", str(cats_path), "--run-dir", str(run_dir), "--retries", "1"]
    )
    assert code == 1
    config = json.loads((run_dir / "run_config.json").read_text())
    assert config["status"] == "completed_with_failures"
    assert config["unclassified_rows"]["count"] == 2


def test_completeness_check_allow_partial(tmp_path, monkeypatch):
    calls = {"n": 0}

    def _flaky_classify(self, text):
        calls["n"] += 1
        fields = set(self.classification_model.model_fields.keys())
        cat_name = next(iter(fields))
        if calls["n"] <= 2:
            raise RuntimeError("boom")
        return {cat_name: ["positive"]}

    monkeypatch.setattr(Classifier, "classify", _flaky_classify)
    rows = [(f"row{i}", "positive") for i in range(5)]
    test = _write_csv(tmp_path / "test.csv", rows, ["text", "label"])
    cats_path = _write_categories(tmp_path / "cats.json")
    run_dir = tmp_path / "r"
    code = _run_main(
        ["classify", "--test-file", str(test), "--text-column", "text", "--label-column", "label",
         "--categories", str(cats_path), "--run-dir", str(run_dir), "--retries", "1",
         "--allow-partial"]
    )
    assert code in (0, None)


def test_failed_run_writes_status_failed(tmp_path):
    # No test file, no hf-dataset -> _resolve_mode raises before run-dir is
    # touched: exercised separately. Here we force a failure AFTER the run
    # directory has begun receiving artifacts (a bad --categories file
    # content is caught early; instead, force via a train file with a
    # column that doesn't exist to fail during loading, after preflight).
    (tmp_path / "test.csv").write_text("text,label\nhi,positive\n")
    (tmp_path / "train.csv").write_text("wrongcol\nvalue\n")
    run_dir = tmp_path / "r"
    code = _run_main(
        ["run", "--train-file", str(tmp_path / "train.csv"), "--test-file", str(tmp_path / "test.csv"),
         "--text-column", "text", "--label-column", "label", "--category-name", "sentiment",
         "--run-dir", str(run_dir)]
    )
    assert code == 1
    config = json.loads((run_dir / "run_config.json").read_text())
    assert config["status"] == "failed"
    assert "failure" in config


# ---------------------------------------------------------------------------
# experiment.py — --models (multi-model voting) mode [spec 4]
# ---------------------------------------------------------------------------


def test_models_and_critics_mutually_exclusive(tmp_path, fake_classify):
    test = _write_csv(tmp_path / "test.csv", [("hi", "positive")], ["text", "label"])
    cats_path = _write_categories(tmp_path / "cats.json")
    code = _run_main(
        ["classify", "--test-file", str(test), "--text-column", "text",
         "--label-column", "label", "--categories", str(cats_path),
         "--run-dir", str(tmp_path / "r"), "--models", "a", "b", "--critics"]
    )
    assert code == 1
    assert fake_classify["calls"] == 0


def test_models_single_value_overrides_model_silently(tmp_path, fake_classify):
    """spec 4 FR-1.1 points 3/9: a single-value --models silently overrides
    --model and resolves to ordinary single-classifier mode -- it must not
    require a second value (this used to be an error; the redesign made
    plain `--models a` behave exactly like `--model a`)."""
    test = _write_csv(tmp_path / "test.csv", [("hi", "positive")], ["text", "label"])
    cats_path = _write_categories(tmp_path / "cats.json")
    run_dir = tmp_path / "r"
    code = _run_main(
        ["classify", "--test-file", str(test), "--text-column", "text",
         "--label-column", "label", "--categories", str(cats_path),
         "--run-dir", str(run_dir), "--models", "only-one"]
    )
    assert code in (0, None)
    config = json.loads((run_dir / "run_config.json").read_text())
    assert config["models"]["model"] == "only-one"
    assert config["models"]["classification_models"] is None
    df = pd.read_csv(run_dir / "test_classified.csv")
    assert "sentiment_by_model" not in df.columns


def test_models_duplicate_value_uses_occurrence_suffixed_keys(tmp_path, fake_classify):
    """spec 4 AR-1.3: --models given 2+ values allows duplicates -- this used
    to be an error; the redesign preserves duplicates and disambiguates
    repeated model ids with occurrence-suffixed classifier keys ("a#1",
    "a#2"), while the once-only id keeps its bare key."""
    test = _write_csv(tmp_path / "test.csv", [("hi", "positive")], ["text", "label"])
    cats_path = _write_categories(tmp_path / "cats.json")
    run_dir = tmp_path / "r"
    code = _run_main(
        ["classify", "--test-file", str(test), "--text-column", "text",
         "--label-column", "label", "--categories", str(cats_path),
         "--run-dir", str(run_dir), "--models", "a", "a", "b"]
    )
    assert code in (0, None)
    config = json.loads((run_dir / "run_config.json").read_text())
    assert config["models"]["model"] is None
    assert config["models"]["classification_models"] == ["a", "a", "b"]
    assert set(config["models"]["model_failure_counts"].keys()) == {"a#1", "a#2", "b"}


def test_models_rejects_empty_value(tmp_path, fake_classify):
    test = _write_csv(tmp_path / "test.csv", [("hi", "positive")], ["text", "label"])
    cats_path = _write_categories(tmp_path / "cats.json")
    code = _run_main(
        ["classify", "--test-file", str(test), "--text-column", "text",
         "--label-column", "label", "--categories", str(cats_path),
         "--run-dir", str(tmp_path / "r"), "--models", "a", ""]
    )
    assert code == 1
    assert fake_classify["calls"] == 0


def test_models_rejects_cerebus_azure_mode(tmp_path, fake_classify, monkeypatch):
    monkeypatch.setenv("CEREBUS_MODE", "azure")
    test = _write_csv(tmp_path / "test.csv", [("hi", "positive")], ["text", "label"])
    cats_path = _write_categories(tmp_path / "cats.json")
    code = _run_main(
        ["classify", "--test-file", str(test), "--text-column", "text",
         "--label-column", "label", "--categories", str(cats_path),
         "--run-dir", str(tmp_path / "r"), "--models", "a", "b"]
    )
    assert code == 1
    assert fake_classify["calls"] == 0


def test_induce_help_does_not_list_models(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["induce", "--help"])
    # --model's own help text legitimately cross-references "--models" in
    # prose (it's a shared `common`-group flag, so this text is visible even
    # under `induce`) -- what must NOT appear is the flag's own definition
    # line, which argparse always renders as "--models MODEL" (its metavar).
    assert "--models MODEL" not in capsys.readouterr().out


def test_models_end_to_end_run_config(tmp_path, fake_classify):
    test = _write_csv(tmp_path / "test.csv", [("hi", "positive")], ["text", "label"])
    cats_path = _write_categories(tmp_path / "cats.json")
    run_dir = tmp_path / "r"
    code = _run_main(
        ["classify", "--test-file", str(test), "--text-column", "text",
         "--label-column", "label", "--categories", str(cats_path),
         "--run-dir", str(run_dir), "--models", "a", "b"]
    )
    assert code in (0, None)
    config = json.loads((run_dir / "run_config.json").read_text())
    assert config["schema_version"] == 4
    assert config["models"]["classification_models"] == ["a", "b"]
    assert config["models"]["model"] is None
    assert set(config["models"]["model_failure_counts"].keys()) == {"a", "b"}
    df = pd.read_csv(run_dir / "test_classified.csv")
    for suffix in ("_votes", "_by_model", "_model_errors"):
        assert f"sentiment{suffix}" in df.columns


def test_models_non_models_run_has_null_keys(tmp_path, fake_classify):
    # A plain (non --models) run must still show classification_models and
    # model_failure_counts as present-but-null, never omitted.
    test = _write_csv(tmp_path / "test.csv", [("hi", "positive")], ["text", "label"])
    cats_path = _write_categories(tmp_path / "cats.json")
    run_dir = tmp_path / "r"
    code = _run_main(
        ["classify", "--test-file", str(test), "--text-column", "text",
         "--label-column", "label", "--categories", str(cats_path),
         "--run-dir", str(run_dir)]
    )
    assert code in (0, None)
    config = json.loads((run_dir / "run_config.json").read_text())
    models_cfg = config["models"]
    assert "classification_models" in models_cfg and models_cfg["classification_models"] is None
    assert "model_failure_counts" in models_cfg and models_cfg["model_failure_counts"] is None
    assert models_cfg["model"] is not None
    # `classify` has neither induction flag at all (not a parent of that
    # subparser) -- both sizing keys must be null, not the per-label default.
    assert config["examples_per_label"] is None
    assert config["induction_examples"] is None


def test_models_run_with_induction_merges_all_fields(tmp_path, fake_classify):
    # A `run` invocation that both induces AND uses --models must not
    # silently keep the induction call's --model value in models_final
    # ("model" must become null), and classification_models/
    # model_failure_counts must be carried across from the classification-
    # role _construct_classifiers call, not dropped.
    train = _write_csv(
        tmp_path / "train.csv",
        [("great product", "positive"), ("bad product", "negative")] * 3,
        ["text", "label"],
    )
    test = _write_csv(
        tmp_path / "test.csv",
        [("fantastic", "positive"), ("awful", "negative")],
        ["text", "label"],
    )
    run_dir = tmp_path / "run_models"
    code = _run_main(
        ["run", "--train-file", str(train), "--test-file", str(test),
         "--text-column", "text", "--label-column", "label",
         "--category-name", "sentiment", "--run-dir", str(run_dir),
         "--induction-model", "induction-model-id", "--models", "a", "b"]
    )
    assert code in (0, None)
    config = json.loads((run_dir / "run_config.json").read_text())
    assert config["models"]["model"] is None
    assert config["models"]["classification_models"] == ["a", "b"]
    assert "model_failure_counts" in config["models"]


def test_models_source_column_collision(tmp_path, fake_classify):
    (tmp_path / "test.csv").write_text("text,label,sentiment_by_model\nhi,positive,bogus\n")
    cats_path = _write_categories(tmp_path / "cats.json", name="sentiment")
    code = _run_main(
        ["classify", "--test-file", str(tmp_path / "test.csv"), "--text-column", "text",
         "--label-column", "label", "--categories", str(cats_path), "--run-dir", str(tmp_path / "r"),
         "--models", "a", "b"]
    )
    assert code == 1
    assert fake_classify["calls"] == 0


@pytest.mark.parametrize("suffix", ["_votes", "_by_model", "_model_errors"])
def test_models_source_column_collision_all_suffixes(tmp_path, fake_classify, suffix):
    (tmp_path / "test.csv").write_text(f"text,label,sentiment{suffix}\nhi,positive,bogus\n")
    cats_path = _write_categories(tmp_path / "cats.json", name="sentiment")
    code = _run_main(
        ["classify", "--test-file", str(tmp_path / "test.csv"), "--text-column", "text",
         "--label-column", "label", "--categories", str(cats_path), "--run-dir", str(tmp_path / "r"),
         "--models", "a", "b"]
    )
    assert code == 1
    assert fake_classify["calls"] == 0


def test_models_failure_counts_per_model(tmp_path, monkeypatch):
    # Force direct-provider mode regardless of the local .env's
    # DEFAULT_LLM_PROVIDER setting, so self.model_id is exactly the raw
    # --models value ("bad"/"good"), not a Cerebus-prefixed variant
    # (cerebus_model_id would turn "bad" into "openai/bad").
    monkeypatch.setattr("query_classification.experiment.cerebus_enabled_via_env", lambda: False)

    def _classify(self, text):
        if self.model_id == "bad":
            raise RuntimeError("boom")
        return {"sentiment": ["positive"]}

    monkeypatch.setattr(Classifier, "classify", _classify)
    test = _write_csv(tmp_path / "test.csv", [("hi", "positive")], ["text", "label"])
    cats_path = _write_categories(tmp_path / "cats.json")
    run_dir = tmp_path / "r"
    code = _run_main(
        ["classify", "--test-file", str(test), "--text-column", "text",
         "--label-column", "label", "--categories", str(cats_path),
         "--run-dir", str(run_dir), "--models", "bad", "good"]
    )
    assert code in (0, None)
    config = json.loads((run_dir / "run_config.json").read_text())
    counts = config["models"]["model_failure_counts"]
    assert counts["bad"] > 0
    assert counts["good"] == 0  # explicit zero entry, not an omitted key


def test_n_classifiers_replication_produces_multi_classifier_run_config(tmp_path, fake_classify):
    """spec 4: --model x --n-classifiers 3 (no --models) must engage the same
    downstream sites --models does -- the pre-projection collision check,
    the post-run completeness check, and failure-count aggregation -- since
    all three used to gate on raw args.models truthiness (args.models is
    None here) rather than the resolved classifier list's length. This
    directly targets the exact regression an earlier draft of this task
    would have shipped."""
    test = _write_csv(tmp_path / "test.csv", [("hi", "positive")], ["text", "label"])
    cats_path = _write_categories(tmp_path / "cats.json")
    run_dir = tmp_path / "r"
    code = _run_main(
        ["classify", "--test-file", str(test), "--text-column", "text",
         "--label-column", "label", "--categories", str(cats_path),
         "--run-dir", str(run_dir), "--model", "x", "--n-classifiers", "3"]
    )
    assert code in (0, None)
    df = pd.read_csv(run_dir / "test_classified.csv")
    for suffix in ("_votes", "_by_model", "_model_errors"):
        assert f"sentiment{suffix}" in df.columns
    config = json.loads((run_dir / "run_config.json").read_text())
    assert config["unclassified_rows"]["count"] == 0
    assert config["models"]["classification_models"] == ["x", "x", "x"]
    counts = config["models"]["model_failure_counts"]
    assert set(counts.keys()) == {"x#1", "x#2", "x#3"}
    assert all(v == 0 for v in counts.values())


def test_n_classifiers_replication_source_column_collision(tmp_path, fake_classify):
    """spec 4 AR-1.5: the pre-projection source-column collision check must
    also gate on the resolved classifier list's length, not raw args.models
    truthiness -- a --model+--n-classifiers replication (args.models is None)
    must still detect a pre-existing audit-column-shaped source column."""
    (tmp_path / "test.csv").write_text("text,label,sentiment_by_model\nhi,positive,bogus\n")
    cats_path = _write_categories(tmp_path / "cats.json", name="sentiment")
    code = _run_main(
        ["classify", "--test-file", str(tmp_path / "test.csv"), "--text-column", "text",
         "--label-column", "label", "--categories", str(cats_path), "--run-dir", str(tmp_path / "r"),
         "--model", "x", "--n-classifiers", "3"]
    )
    assert code == 1
    assert fake_classify["calls"] == 0


def test_n_classifiers_repeated_instance_failure_count_keeps_keys_distinct(tmp_path, monkeypatch):
    """spec 4 FR-1.9's pre-seed fix: when one occurrence of a repeated model
    fails, its own occurrence-suffixed key must show a nonzero count while
    the other occurrence of the SAME model shows an explicit 0 -- proving
    the pre-seed step uses the actual constructed classifier keys, not a raw
    --models/--model dict-comprehension that would collapse both into one
    bare "same-model" entry (already covered for distinct models by
    test_models_failure_counts_per_model above)."""
    monkeypatch.setattr("query_classification.experiment.cerebus_enabled_via_env", lambda: False)

    construction_order = []
    original_init = Classifier.__init__

    def recording_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        construction_order.append(id(self))

    def fake_classify_method(self, text):
        idx = construction_order.index(id(self))
        if idx == 0:
            raise RuntimeError("boom")
        return {"sentiment": ["positive"]}

    monkeypatch.setattr(Classifier, "__init__", recording_init)
    monkeypatch.setattr(Classifier, "classify", fake_classify_method)

    test = _write_csv(tmp_path / "test.csv", [("hi", "positive")], ["text", "label"])
    cats_path = _write_categories(tmp_path / "cats.json")
    run_dir = tmp_path / "r"
    code = _run_main(
        ["classify", "--test-file", str(test), "--text-column", "text",
         "--label-column", "label", "--categories", str(cats_path),
         "--run-dir", str(run_dir), "--model", "same-model", "--n-classifiers", "2"]
    )
    assert code in (0, None)
    config = json.loads((run_dir / "run_config.json").read_text())
    counts = config["models"]["model_failure_counts"]
    assert counts["same-model#1"] > 0
    assert counts["same-model#2"] == 0


# --- _resolve_classifier_models: Shared Resolution Truth Table (spec 4) ---


def _resolve(monkeypatch, argv, cerebus_mode=None):
    from query_classification.experiment import _resolve_classifier_models

    if cerebus_mode is None:
        monkeypatch.delenv("CEREBUS_MODE", raising=False)
    else:
        monkeypatch.setenv("CEREBUS_MODE", cerebus_mode)
    ns = build_parser().parse_args(["classify", "--text-column", "text", "--run-dir", "d", *argv])
    return _resolve_classifier_models(ns)


def test_experiment_resolve_models_alone_ignores_default_n_classifiers(monkeypatch):
    """The regression guard: `--models a b c` alone (no --n-classifiers) must
    resolve to exactly [a, b, c]."""
    assert _resolve(monkeypatch, ["--models", "a", "b", "c"]) == ["a", "b", "c"]


def test_experiment_resolve_models_ignores_n_classifiers_even_when_explicitly_conflicting(monkeypatch):
    resolved = _resolve(monkeypatch, ["--models", "a", "b", "c", "--n-classifiers", "99"])
    assert resolved == ["a", "b", "c"]


def test_experiment_resolve_n_classifiers_zero_always_errors_even_with_models(monkeypatch):
    with pytest.raises(ValueError, match="--n-classifiers"):
        _resolve(monkeypatch, ["--models", "a", "b", "c", "--n-classifiers", "0"])


def test_experiment_resolve_model_replication_with_cerebus_azure_mode_succeeds_for_same_model(monkeypatch):
    """CEREBUS_MODE=azure only blocks 2+ DISTINCT resolved models -- a single
    model replicated via --n-classifiers must still be allowed."""
    resolved = _resolve(monkeypatch, ["--model", "a", "--n-classifiers", "3"], cerebus_mode="azure")
    assert resolved == ["a", "a", "a"]


def test_experiment_resolve_models_distinct_with_cerebus_azure_mode_errors(monkeypatch):
    with pytest.raises(ValueError, match="CEREBUS_MODE=azure"):
        _resolve(monkeypatch, ["--models", "a", "b"], cerebus_mode="azure")


def test_experiment_resolve_induce_ignores_models_attribute_entirely(monkeypatch):
    """induce's args Namespace has no --models/--n-classifiers attributes at
    all (only --model lives in the `common` group) -- _resolve_classifier_
    models must short-circuit via hasattr, not crash with AttributeError."""
    from query_classification.experiment import _resolve_classifier_models

    monkeypatch.delenv("CEREBUS_MODE", raising=False)
    ns = build_parser().parse_args(
        ["induce", "--train-file", "t.csv", "--text-column", "text",
         "--label-column", "label", "--category-name", "cat", "--run-dir", "d",
         "--model", "base-model"]
    )
    assert _resolve_classifier_models(ns) == ["base-model"]


def test_experiment_build_classifier_dict_collision_guard_rejects_pathological_model_id():
    """AR-1.3's collision guard: if a supplied model id already looks like an
    occurrence-suffixed key another position would independently produce,
    raise a clear error instead of silently letting one classifier overwrite
    the other."""
    from query_classification.experiment import _build_classifier_dict

    with pytest.raises(ValueError, match="collision"):
        _build_classifier_dict(["a", "a", "a#1"], lambda m: m)


def test_module_invocation_help():
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "query_classification.experiment", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "induce" in result.stdout and "classify" in result.stdout and "run" in result.stdout
