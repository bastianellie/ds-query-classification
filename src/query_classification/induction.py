"""Category induction: sample examples per label from a train split, make one
LLM call, and reconcile the response into a ``Category``.

This module is a pure function with respect to the filesystem — it never
reads or writes anything beyond the ``train_df`` it is given. A later task
(``experiment.py``) is the sole writer of every run-directory artifact
(``categories.json``, ``induction_prompt.txt``, ``sampled_examples.json``);
this module only returns the data those artifacts are built from.
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


def induce(
    train_df: pd.DataFrame,
    text_column: str,
    label_column: str,
    category_name: str,
    induction_classifier: Classifier,
    *,
    seed: int,
    examples_per_label: int,
    max_example_chars: int,
    max_prompt_chars: int,
    max_reconciliation_retries: int = 3,
) -> InductionOutcome:
    """Sample examples per label from ``train_df``, induce a ``Category`` via
    one or more calls to ``induction_classifier``, and reconcile the response
    against the dataset's actual label set.

    Never touches the filesystem. Raises ``ValueError`` for any degenerate
    input, reserved-label use, or oversized prompt (checked once, before any
    classifier call); raises ``RuntimeError`` (sanitized) if the classifier
    call itself fails (already retried internally by ``Classifier.classify``
    up to its own ``max_retries``, so not retried again here).

    The classifier's output schema (``schema.build_induction_model``) is
    intentionally an open-ended ``list`` of label entries, not a fixed
    structure with one slot per known label (dataset label values routinely
    aren't valid Python/Pydantic identifiers) — so nothing at the schema
    level stops the model from silently omitting or inventing a label. A
    reconciliation mismatch is therefore a plausible one-off LLM failure, not
    necessarily a systemic one (more likely with a larger
    ``examples_per_label``/label-count prompt), so it gets its own bounded
    retry here: up to ``max_reconciliation_retries`` attempts, each a fresh
    call with the identical prompt (the induction classifier's own
    provider-default, non-zero temperature is what gives each retry a chance
    at a different, hopefully complete, response). Only a reconciliation
    mismatch is retried this way — the prompt is only ever built once.
    """
    if max_reconciliation_retries < 1:
        raise ValueError(
            f"max_reconciliation_retries must be >= 1, got {max_reconciliation_retries}"
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

    # Group row positions per label, preserving each label's original
    # (stable) relative row order for the at-or-below-cap, no-RNG-consumed case.
    positions_by_label: dict[Any, list[int]] = {label: [] for label in distinct_labels}
    for position, label in enumerate(train_df[label_column].tolist()):
        positions_by_label[label].append(position)

    rng = random.Random(seed)  # single shared generator, consumed sequentially below
    sampled_examples: dict[str, Any] = {}
    labels_payload: dict[str, list[str]] = {}

    for label in distinct_labels:
        stable_positions = positions_by_label[label]
        if len(stable_positions) > examples_per_label:
            chosen_positions = rng.sample(stable_positions, examples_per_label)
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
        labels_payload[label_key] = texts

    payload = {"category_name": category_name, "labels": labels_payload}
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
    # counts, and the user message (which scales with examples_per_label and
    # max_example_chars) dominates that risk. This is a conservative
    # simplification, not an attempt to model the full request size.
    total_chars = len(serialized_user_message)
    if total_chars > max_prompt_chars:
        raise ValueError(
            f"induction user message is {total_chars} characters, exceeding "
            f"max_prompt_chars={max_prompt_chars}; lower --examples-per-label "
            "or --max-example-chars"
        )

    expected = set(distinct_labels)
    reconciliation_error: ValueError | None = None
    category: Category | None = None

    for attempt in range(1, max_reconciliation_retries + 1):
        try:
            result = induction_classifier.classify(serialized_user_message)
        except Exception as e:  # noqa: BLE001 - sanitized and re-raised as the induction failure
            raise RuntimeError(f"{type(e).__name__}: induction call failed") from e

        returned = [entry["label"] for entry in result["labels"]]
        missing = expected - set(returned)
        unexpected = set(returned) - expected

        if not missing and not unexpected and len(returned) == len(expected):
            category = Category(
                name=category_name,
                description=result["category_description"],
                labels=[
                    Label(value=entry["label"], description=entry["description"])
                    for entry in result["labels"]
                ],
            )
            reconciliation_error = None
            break

        if missing or unexpected:
            reconciliation_error = ValueError(
                "induction response label set does not match the dataset's labels "
                f"(missing: {sorted(missing)}, unexpected/invented: {sorted(unexpected)})"
            )
        else:
            reconciliation_error = ValueError(
                f"induction response returned {len(returned)} label entries but expected "
                f"{len(expected)} distinct labels; response likely contains a duplicate "
                f"({sorted(returned)})"
            )

    if reconciliation_error is not None:
        raise reconciliation_error
    assert category is not None  # guaranteed by the break above whenever no error survives

    return InductionOutcome(
        category=category,
        rendered_prompt=serialized_user_message,
        sampled_examples=sampled_examples,
    )
