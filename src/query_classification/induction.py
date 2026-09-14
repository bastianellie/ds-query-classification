"""Category induction: sample examples per label from a train split, make one
LLM call, and build a ``Category`` from the response.

This module is a pure function with respect to the filesystem — it never
reads or writes anything beyond the ``train_df`` it is given. A later task
(``experiment.py``) is the sole writer of every run-directory artifact
(``categories.json``, ``induction_prompt.txt``, ``sampled_examples.json``);
this module only returns the data those artifacts are built from.

The induction call's response is **positional**: it carries one description
per label, matched by position against this module's own sorted label list
(``schema.build_induction_model``'s ``description_1..description_N`` fields)
— never label names. Label identity therefore never round-trips through the
model's response, so a missing or invented label is structurally impossible
here, not something to detect after the fact.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

import pandas as pd

from query_classification.categories import Category, Label
from query_classification.classifier import Classifier

_TRUNCATION_MARKER = "…[truncated]"
_DATA_START = "<<<DATA>>>"
_DATA_END = "<<<END DATA>>>"


@dataclass
class InductionOutcome:
    """The result of one induction call, not yet written to disk.

    Attributes:
        category: The constructed (not-yet-persisted) ``categories.py`` object.
        rendered_prompt: The exact user message text sent to the induction
            classifier, for a later task to snapshot verbatim.
        sampled_examples: A per-label manifest — for each label, the sampled
            row positions and their truncated texts — for a later task to
            snapshot verbatim.
    """

    category: Category
    rendered_prompt: str
    sampled_examples: dict


def _truncate(text: str, max_example_chars: int) -> str:
    """Truncate ``text`` to ``max_example_chars`` Unicode code points, with the
    fixed marker appended *outside* that budget when truncation occurred."""
    if len(text) <= max_example_chars:
        return text
    return text[:max_example_chars] + _TRUNCATION_MARKER


def _resolve_quotas(
    distinct_labels: list[Any],
    row_counts: dict[Any, int],
    examples_per_label: int | None,
    induction_examples: int | None,
) -> dict[Any, int]:
    """Resolve each label's example quota under exactly one of two sizing
    modes (FR-2.1). ``distinct_labels`` must already be sorted — sort order
    is this function's tie-break rule throughout.

    - ``examples_per_label`` (per-label cap): each label's quota is
      ``min(row_counts[label], examples_per_label)`` — unchanged from the
      original single-mode behavior.
    - ``induction_examples`` (total budget): quotas start at
      ``induction_examples // n_labels``, with the
      ``induction_examples % n_labels`` leftover going one each to the first
      that many labels in sorted order (equivalent to "largest fractional
      part, ties by sorted label value" — every label's fractional part is
      identical here, since the budget is split equally rather than
      proportionally to row counts, so sorted order is the entire rule). Any
      label whose quota exceeds its row count is clamped to its row count,
      and the freed shortfall is re-allocated as increments among labels that
      still have spare capacity, by the same sorted-order rule, repeating
      until the budget is met or no spare capacity remains. Quotas always sum
      to ``min(induction_examples, sum(row_counts.values()))``.
    """
    if induction_examples is None:
        cap = examples_per_label
        return {label: min(row_counts[label], cap) for label in distinct_labels}

    n_labels = len(distinct_labels)
    base, remainder = divmod(induction_examples, n_labels)
    quotas = {
        label: base + (1 if i < remainder else 0)
        for i, label in enumerate(distinct_labels)
    }
    while True:
        shortfall = sum(
            quotas[label] - row_counts[label]
            for label in distinct_labels
            if quotas[label] > row_counts[label]
        )
        for label in distinct_labels:
            quotas[label] = min(quotas[label], row_counts[label])
        if shortfall == 0:
            break
        spare = [label for label in distinct_labels if quotas[label] < row_counts[label]]
        if not spare:
            break
        extra_base, extra_remainder = divmod(shortfall, len(spare))
        for i, label in enumerate(spare):
            quotas[label] += extra_base + (1 if i < extra_remainder else 0)
    return quotas


def induce(
    train_df: pd.DataFrame,
    text_column: str,
    label_column: str,
    category_name: str,
    induction_classifier: Classifier,
    *,
    seed: int,
    examples_per_label: int | None = None,
    induction_examples: int | None = None,
    max_example_chars: int,
    max_prompt_chars: int,
) -> InductionOutcome:
    """Sample examples from ``train_df`` under exactly one sizing mode
    (FR-2.1), induce a ``Category`` via one call to ``induction_classifier``,
    and build it directly from this module's own sorted label list — the
    response supplies descriptions only, matched by position.

    Exactly one of ``examples_per_label``/``induction_examples`` must be
    given; the caller (``experiment.py``) resolves the CLI's default before
    calling this function, so both being ``None`` is a caller bug, not a user
    input to validate gracefully — it is still checked explicitly, since this
    function is also called directly by tests.

    Never touches the filesystem. Raises ``ValueError`` for any degenerate
    input, sizing-mode misuse, reserved-label use, an internal arity
    mismatch, or oversized prompt (all checked before any classifier call);
    raises ``RuntimeError`` (sanitized) if the classifier call itself fails
    (already retried internally by ``Classifier.classify`` up to its own
    ``max_retries`` — not retried again here, per AR-2.3).
    """
    if (examples_per_label is None) == (induction_examples is None):
        raise ValueError(
            "exactly one of examples_per_label/induction_examples must be given, got "
            f"examples_per_label={examples_per_label!r}, "
            f"induction_examples={induction_examples!r}"
        )
    if train_df.empty:
        raise ValueError("train_df is empty after filtering; nothing to induce from")

    distinct_labels = sorted(train_df[label_column].unique().tolist())
    if len(distinct_labels) == 0:
        raise ValueError(
            f"train_df[{label_column!r}] has zero distinct label values; nothing to induce from"
        )

    for label in distinct_labels:
        normalized = str(label).strip().lower()
        if normalized == "none" or normalized.startswith("none - "):
            raise ValueError(
                f"label value {label!r} collides with the reserved 'none'/'none - <label>' "
                "convention and cannot be induced as a real label"
            )

    if induction_examples is not None and induction_examples < len(distinct_labels):
        raise ValueError(
            f"--induction-examples must be >= the number of distinct labels "
            f"({len(distinct_labels)}), got {induction_examples}"
        )

    # Group row positions per label, preserving each label's original
    # (stable) relative row order for the at-or-below-quota, no-RNG-consumed
    # case.
    positions_by_label: dict[Any, list[int]] = {label: [] for label in distinct_labels}
    for position, label in enumerate(train_df[label_column].tolist()):
        positions_by_label[label].append(position)

    quotas = _resolve_quotas(
        distinct_labels,
        {label: len(positions_by_label[label]) for label in distinct_labels},
        examples_per_label,
        induction_examples,
    )

    rng = random.Random(seed)  # single shared generator, consumed sequentially below
    sampled_examples: dict[str, Any] = {}
    label_entries: list[dict[str, Any]] = []

    for position_1_indexed, label in enumerate(distinct_labels, start=1):
        stable_positions = positions_by_label[label]
        quota = quotas[label]
        if len(stable_positions) > quota:
            chosen_positions = rng.sample(stable_positions, quota)
        else:
            chosen_positions = list(stable_positions)

        label_key = str(label)
        entries = []
        texts = []
        for position in chosen_positions:
            raw_text = str(train_df[text_column].iloc[position])
            truncated = _truncate(raw_text, max_example_chars)
            entries.append({"position": position, "text": truncated})
            texts.append(truncated)
        sampled_examples[label_key] = entries
        label_entries.append(
            {"position": position_1_indexed, "label": label_key, "examples": texts}
        )

    payload = {"category_name": category_name, "labels": label_entries}
    serialized_user_message = (
        f"{_DATA_START}\n{json.dumps(payload, sort_keys=False)}\n{_DATA_END}"
    )

    # Prompt-size preflight, before any classifier call. The induction
    # Classifier already has its system prompt baked in at construction time
    # and .classify() only takes the user-message text, so there is no way to
    # cleanly recover the exact system-prompt string from a Classifier
    # instance without reaching into its private construction details. We
    # deliberately measure only the serialized user message's length against
    # max_prompt_chars: the budget exists to catch runaway example/label
    # counts, and the user message (which scales with the sizing mode in
    # effect and max_example_chars) dominates that risk. This is a
    # conservative simplification, not an attempt to model the full request
    # size.
    total_chars = len(serialized_user_message)
    if total_chars > max_prompt_chars:
        sizing_flag = (
            "--induction-examples" if induction_examples is not None else "--examples-per-label"
        )
        raise ValueError(
            f"induction user message is {total_chars} characters, exceeding "
            f"max_prompt_chars={max_prompt_chars}; lower {sizing_flag} "
            "or --max-example-chars"
        )

    # Arity guard (AR-2.1's positional contract, defensive/internal): induction.py
    # cannot import schema.py (INV-1), so it reads the model off the classifier
    # it was handed rather than rebuilding it, and compares the exact expected
    # field set -- not merely a count, since a count alone would accept
    # malformed names like "description_0"/"description_x" and defer failure to
    # an opaque KeyError below. This is the only guard against a classifier
    # built for a different label count than the one just sampled; it is not
    # reachable via the CLI, which always builds the model from this same
    # distinct_labels count (spec 2 Spec Deviations #2).
    expected_fields = {"category_description"} | {
        f"description_{i}" for i in range(1, len(distinct_labels) + 1)
    }
    actual_fields = set(induction_classifier.classification_model.model_fields)
    if actual_fields != expected_fields:
        raise ValueError(
            f"induction classifier's schema has fields {sorted(actual_fields)}, expected "
            f"{sorted(expected_fields)} for {len(distinct_labels)} distinct label(s) — "
            "internal inconsistency between the sampled label count and the model it "
            "was built with"
        )

    try:
        result = induction_classifier.classify(serialized_user_message)
    except Exception as e:  # noqa: BLE001 - sanitized and re-raised as the induction failure
        raise RuntimeError(f"{type(e).__name__}: induction call failed") from e

    category = Category(
        name=category_name,
        description=result["category_description"],
        labels=[
            Label(value=str(label), description=result[f"description_{i}"])
            for i, label in enumerate(distinct_labels, start=1)
        ],
    )

    return InductionOutcome(
        category=category,
        rendered_prompt=serialized_user_message,
        sampled_examples=sampled_examples,
    )
