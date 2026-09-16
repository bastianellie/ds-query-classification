"""Batch classification of a CSV column.

Reads a CSV, classifies one text column (optionally across several worker
threads), writes one output column per category, and saves incrementally so a
long run can be resumed (``restore``) or truncated for debugging (``limit``).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from query_classification import cost, debate, multi_model
from query_classification.batching import BatchRunner
from query_classification.categories import Category
from query_classification.classifier import Classifier


def _audit_columns(category_name: str, audit_suffixes: tuple[str, ...]) -> set[str]:
    """The audit-trail columns a category generates under ``--critics``/
    ``--models`` (not including the category's own base column)."""
    return {f"{category_name}{suffix}" for suffix in audit_suffixes}


def _classify_tracked(classifier: Classifier, text: str):
    """Wraps ``classifier.classify(text)`` (unchanged arguments) in
    ``track_usage()``, entirely on the worker thread (INV-6: only the
    read-only call happens there). On success, returns ``(result, usage)``.
    On failure, attaches the usage observed before the failure (which may be
    ``None`` -- no billable attempt at all) to the exception as
    ``_tracked_usage`` and re-raises, per FR-1.2's retain-on-failure design."""
    with classifier.track_usage() as sink:
        try:
            result = classifier.classify(text)
        except Exception as exc:
            exc._tracked_usage = sink.usage
            raise
        return result, sink.usage


def classify_csv(
    input_path: str | Path,
    column: str,
    classifier: Classifier | None,
    categories: list[Category],
    output_path: str | Path | None = None,
    restore: bool = False,
    limit: int | None = None,
    workers: int = 8,
    save_every: int = 20,
    critics: bool = False,
    critic_classifiers: dict[str, Classifier] | None = None,
    reconciler_classifiers: dict[str, Classifier] | None = None,
    sampling_runs: int = 5,
    consensus_threshold: int = 4,
    allow_new_labels: bool = False,
    models: dict[str, Classifier] | None = None,
    batch_runner: BatchRunner | None = None,
    cost_collector: "cost.QueryCostCollector | None" = None,
    cost_per_token: "cost.CostPerToken | None" = None,
) -> Path:
    """Classify ``column`` of the input CSV and write the augmented CSV.

    Rows are classified concurrently across ``workers`` threads (LLM calls are
    I/O-bound). Results are written back by row index, so the output preserves
    the original input order regardless of completion order. The CSV is flushed
    to disk every ``save_every`` completed rows (and once at the end) so a
    crashed run can be resumed with ``restore``.

    Exactly one of ``classifier`` or ``models`` must be given. When
    ``critics=True``, ``classifier`` is used as the *sampling* classifier
    (already constructed with the desired temperature by the caller) and each
    row is classified via ``debate.run_debate`` instead of a single
    ``classifier.classify`` call, using ``critic_classifiers``/
    ``reconciler_classifiers`` (one ``Classifier`` per category, keyed by
    category name). ``sampling_temperature`` itself is not a parameter here —
    it's only meaningful at the point ``classifier`` is constructed. When
    ``models`` is given instead (keyed by model id, each value a distinct
    ``Classifier``), each row is classified via
    ``multi_model.run_multi_model`` — every model classifies the row once,
    independently, merged by plurality vote; mutually exclusive with
    ``critics``.

    Returns the path the result was written to. Defaults to overwriting the
    input file when ``output_path`` is not given.
    """
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path
    category_names = [cat.name for cat in categories]

    if column in category_names:
        raise ValueError(
            f"Category name '{column}' collides with the text column being "
            f"classified. Rename the category or choose a different --column; "
            f"otherwise the source text would be overwritten before classification."
        )

    have_classifier = classifier is not None
    have_models = models is not None
    have_batch = batch_runner is not None
    if have_classifier and have_models:
        raise ValueError("classifier and models are mutually exclusive — provide exactly one")
    if not have_classifier and not have_models and not have_batch:
        raise ValueError("exactly one of classifier or models must be provided")
    if have_batch and (have_classifier or have_models):
        raise ValueError("batch_runner is mutually exclusive with classifier and models")
    if critics and have_models:
        raise ValueError("critics and models are mutually exclusive")
    if have_batch and critics:
        raise ValueError("batch_runner and critics are mutually exclusive")
    if have_batch and have_models:
        raise ValueError("batch_runner and models are mutually exclusive")
    if have_models:
        if len(models) < 2:
            raise ValueError(f"models must have at least 2 entries, got {len(models)}")
        empty_keys = [repr(k) for k in models if not k.strip()]
        if empty_keys:
            raise ValueError(f"models keys must be non-empty, got {', '.join(empty_keys)}")

    audit_suffixes: tuple[str, ...] = (
        debate.AUDIT_COLUMN_SUFFIXES
        if critics
        else multi_model.AUDIT_COLUMN_SUFFIXES
        if have_models
        else ()
    )

    if critics:
        if sampling_runs < 1:
            raise ValueError(f"sampling_runs must be >= 1, got {sampling_runs}")
        if not (1 <= consensus_threshold <= sampling_runs):
            raise ValueError(
                f"consensus_threshold must be between 1 and sampling_runs "
                f"({sampling_runs}) inclusive, got {consensus_threshold}"
            )
        if critic_classifiers is None or reconciler_classifiers is None:
            raise ValueError(
                "critic_classifiers and reconciler_classifiers are required when "
                "critics=True"
            )
        missing_critics = [n for n in category_names if n not in critic_classifiers]
        missing_reconcilers = [n for n in category_names if n not in reconciler_classifiers]
        if missing_critics or missing_reconcilers:
            raise ValueError(
                f"critic_classifiers/reconciler_classifiers missing entries for "
                f"category name(s) — critics missing {missing_critics}, "
                f"reconcilers missing {missing_reconcilers}."
            )

    if audit_suffixes:
        # Collision check extended to every generated audit column: none may
        # equal the text column, and no two categories' generated column sets
        # may overlap (this also catches e.g. category "sentiment_votes"
        # colliding with category "sentiment"'s own generated audit column).
        # Applies to both --critics and --models, whichever is active.
        owner: dict[str, str] = {}
        for cat_name in category_names:
            generated = {cat_name, *_audit_columns(cat_name, audit_suffixes)}
            for col in generated:
                if col == column:
                    raise ValueError(
                        f"Category '{cat_name}' generates column '{col}', which "
                        f"collides with the text column being classified "
                        f"('{column}'). Rename the category or choose a "
                        f"different --column."
                    )
                if col in owner and owner[col] != cat_name:
                    raise ValueError(
                        f"Categories '{owner[col]}' and '{cat_name}' both "
                        f"generate column '{col}'. Rename one of the categories "
                        f"to avoid collision."
                    )
                owner[col] = cat_name

    df = pd.read_csv(input_path)
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found. Available columns: {list(df.columns)}"
        )

    # Add category columns (and, under --critics/--models, their audit
    # columns) if not present.
    for cat_name in category_names:
        generated = {cat_name, *_audit_columns(cat_name, audit_suffixes)}
        for col in generated:
            if col not in df.columns:
                df[col] = None

    # When resuming into a separate --output file, seed already-classified
    # category columns (and audit columns, under --critics/--models) from the
    # prior output before deciding what's left to do.
    if restore and output_path.exists() and output_path.resolve() != input_path.resolve():
        prior = pd.read_csv(output_path)
        if len(prior) != len(df):
            raise ValueError(
                f"Cannot restore from '{output_path}': it has {len(prior)} rows, "
                f"but '{input_path}' has {len(df)} rows."
            )
        for cat_name in category_names:
            generated = {cat_name, *_audit_columns(cat_name, audit_suffixes)}
            for col in generated:
                if col in prior.columns:
                    df[col] = prior[col]

    work_idx = df.index

    if restore:
        completeness_cols = list(category_names)
        if critics:
            completeness_cols += [f"{name}_votes" for name in category_names]
        elif have_models:
            # All 3 multi-model audit columns must be non-null, not just one —
            # a row's answer isn't trustworthy without its full audit trail.
            completeness_cols += [
                col for name in category_names for col in _audit_columns(name, audit_suffixes)
            ]
        already_done = df.loc[work_idx, completeness_cols].notna().all(axis=1)
        work_idx = work_idx[~already_done]
    else:
        reset_cols = list(category_names)
        reset_cols += [
            col for name in category_names for col in _audit_columns(name, audit_suffixes)
        ]
        # Force object dtype before resetting: a column read back from a prior
        # run's CSV (e.g. a bool-valued `_challenged`/`_reconciled` column with
        # no missing values yet) can be inferred as a non-nullable dtype that
        # can't hold None/NaN.
        for col in reset_cols:
            if df[col].dtype != object:
                df[col] = df[col].astype(object)
        df.loc[work_idx, reset_cols] = None

    if limit is not None:
        work_idx = work_idx[:limit]

    total = len(work_idx)
    if cost_collector is not None:
        cost_collector.set_rows_attempted(total)
    classified = 0
    failed = 0
    completed = 0

    progress = tqdm(total=total, unit="row", desc="Classifying")
    # Worker threads only run the (read-only) LLM call and return data; all
    # mutation of `df` happens here on the main thread as futures complete, so
    # no locking is needed and output order stays tied to the row index.
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        if have_batch:
            rows_with_index = [(idx, str(df.at[idx, column])) for idx in work_idx]
            text_by_idx = dict(rows_with_index)
            batches = list(batch_runner.iter_batches(rows_with_index))
            future_to_batch_idx = {
                executor.submit(batch_runner.run, [text_by_idx[i] for i in batch_idx]): batch_idx
                for batch_idx in batches
            }
            for future in as_completed(future_to_batch_idx):
                batch_idx = future_to_batch_idx[future]
                try:
                    outcomes = future.result()
                except Exception as exc:  # noqa: BLE001 - the batching layer's own code
                    # failed (not a per-row LLM failure -- BatchRunner.run already
                    # converts those to per-row Exception objects and never raises
                    # itself) -- convert to one failure per covered row so
                    # classified + failed == completed == total still holds.
                    outcomes = [exc] * len(batch_idx)
                for idx, outcome in zip(batch_idx, outcomes):
                    if isinstance(outcome, Exception):
                        failed += 1
                    else:
                        for cat, val in outcome.items():
                            df.at[idx, cat] = val
                        classified += 1
                    completed += 1
                    progress.update(1)
                    progress.set_postfix(ok=classified, failed=failed)
                    if completed % save_every == 0:
                        df.to_csv(output_path, index=False)
            df.to_csv(output_path, index=False)
            progress.close()
            return output_path

        if critics:
            future_to_idx = {
                executor.submit(
                    debate.run_debate,
                    str(df.at[idx, column]),
                    categories,
                    classifier,
                    critic_classifiers,
                    reconciler_classifiers,
                    sampling_runs=sampling_runs,
                    consensus_threshold=consensus_threshold,
                    allow_new_labels=allow_new_labels,
                ): idx
                for idx in work_idx
            }
        elif have_models:
            future_to_idx = {
                executor.submit(
                    multi_model.run_multi_model,
                    str(df.at[idx, column]),
                    categories,
                    models,
                    allow_new_labels=allow_new_labels,
                ): idx
                for idx in work_idx
            }
        elif cost_collector is not None:
            future_to_idx = {
                executor.submit(_classify_tracked, classifier, str(df.at[idx, column])): idx
                for idx in work_idx
            }
        else:
            future_to_idx = {
                executor.submit(classifier.classify, str(df.at[idx, column])): idx
                for idx in work_idx
            }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                if cost_collector is not None and not critics and not have_models:
                    classification, usage = result
                else:
                    classification = result
                for cat, val in classification.items():
                    df.at[idx, cat] = val
                classified += 1
                if cost_collector is not None and not critics and not have_models:
                    if usage is not None and cost_per_token is not None:
                        row_cost = cost.compute_cost(*usage, cost_per_token)
                        cost_collector.record(row_cost)
            except Exception as e:  # noqa: BLE001 - one bad row shouldn't abort the run.
                # Under --models, multi_model.run_multi_model never raises for
                # ordinary model failures (even if every model failed), so this
                # branch is reached only by a genuinely unexpected orchestration
                # error — most --models rows increment `classified` above
                # regardless of per-model outcomes. See {category}_model_errors/
                # model_failure_counts for per-model health, not this counter.
                failed += 1
                if cost_collector is not None and not critics and not have_models:
                    usage = getattr(e, "_tracked_usage", None)
                    if usage is not None and cost_per_token is not None:
                        row_cost = cost.compute_cost(*usage, cost_per_token)
                        cost_collector.record(row_cost)
            completed += 1
            progress.update(1)
            progress.set_postfix(ok=classified, failed=failed)
            if completed % save_every == 0:
                df.to_csv(output_path, index=False)

    df.to_csv(output_path, index=False)
    progress.close()
    return output_path
