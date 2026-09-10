"""Per-row multi-model classification and plurality-vote merging for the
``--models`` mode.

N distinct models each classify the same row once, independently (no
sampling), and their answers are merged per category by plurality vote on
each model's *top* label — the same tally/tie-break/none-bucketing
convention ``debate.py``'s ``--critics`` consensus step already uses
(``_vote_bucket``, reused here rather than duplicated), just voting across
distinct models instead of samples of one model. There is no consensus
threshold and no Critic/Reconciler escalation.

Unlike ``debate.run_debate`` (which raises when every sample fails, and the
row is silently discarded), ``run_multi_model`` never raises for ordinary
model-call failures, including the all-fail case — a systemic failure (bad
credential, quota exhaustion, provider outage) still produces a result, with
null categories and a full per-model error map, so it stays auditable
instead of vanishing.
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from query_classification.categories import Category
from query_classification.classifier import Classifier
from query_classification.debate import _vote_bucket

AUDIT_COLUMN_SUFFIXES = (
    "_votes",
    "_by_model",
    "_model_errors",
)


def _sanitize_error(exc: Exception) -> str:
    """A safe-to-persist failure note: exception type + a fixed stage
    message, never any part of the raw exception text (which could contain
    secrets, credentials, or internal endpoint URLs) — mirrors debate.py's
    own ``_sanitize_error`` convention; this feature's one and only stage is
    "Classification"."""
    return f"{type(exc).__name__}: Classification call failed"


def run_multi_model(
    text: str,
    categories: list[Category],
    classifiers: dict[str, Classifier],
    *,
    allow_new_labels: bool,
) -> dict[str, Any]:
    """Classify ``text`` under ``--models`` mode: call every classifier in
    ``classifiers`` (keyed by model id) once, concurrently, and merge the
    results per category by plurality vote. Never raises.

    Results are collected into pre-allocated, index-ordered lists (indexed
    by each model's position in ``classifiers``) before any tallying
    happens — never tallied in ``as_completed()`` arrival order — so the
    plurality tie-break (``Counter.most_common()``'s first-insertion-order
    tie-break) reflects ``--models`` list order regardless of which model's
    call actually finishes first.
    """
    model_ids = list(classifiers.keys())
    n = len(model_ids)
    samples: list[dict[str, Any] | None] = [None] * n
    errors: list[Exception | None] = [None] * n

    with ThreadPoolExecutor(max_workers=max(1, n)) as executor:
        future_to_idx = {
            executor.submit(classifiers[model_ids[i]].classify, text): i for i in range(n)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                samples[idx] = future.result()
            except Exception as e:  # noqa: BLE001 - one bad model shouldn't abort the row; never raise so the row's audit trail survives even total failure
                errors[idx] = e

    successful = [(i, s) for i, s in enumerate(samples) if s is not None]
    model_errors = {
        model_ids[i]: _sanitize_error(errors[i]) for i in range(n) if errors[i] is not None
    }
    model_errors_json = json.dumps(model_errors)

    result: dict[str, Any] = {}
    for cat in categories:
        tally: Counter[str] = Counter()
        by_model: dict[str, list[str]] = {}
        for i, sample in successful:
            labels = sample[cat.name]
            by_model[model_ids[i]] = labels
            tally[_vote_bucket(labels[0], allow_new_labels)] += 1

        result[f"{cat.name}_votes"] = json.dumps(dict(tally))
        result[f"{cat.name}_by_model"] = json.dumps(by_model)
        result[f"{cat.name}_model_errors"] = model_errors_json

        if tally:
            leading_bucket, _ = tally.most_common()[0]
            result[cat.name] = next(
                sample[cat.name]
                for i, sample in successful
                if _vote_bucket(sample[cat.name][0], allow_new_labels) == leading_bucket
            )
        else:
            result[cat.name] = None

    return result
