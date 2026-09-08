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

from query_classification import debate
from query_classification.categories import Category
from query_classification.classifier import Classifier


def _audit_columns(category_name: str) -> set[str]:
    """The audit-trail columns a category generates under ``--critics`` (not
    including the category's own base column)."""
    return {f"{category_name}{suffix}" for suffix in debate.AUDIT_COLUMN_SUFFIXES}


def classify_csv(
    input_path: str | Path,
    column: str,
    classifier: Classifier,
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
) -> Path:
    """Classify ``column`` of the input CSV and write the augmented CSV.

    Rows are classified concurrently across ``workers`` threads (LLM calls are
    I/O-bound). Results are written back by row index, so the output preserves
    the original input order regardless of completion order. The CSV is flushed
    to disk every ``save_every`` completed rows (and once at the end) so a
    crashed run can be resumed with ``restore``.

    When ``critics=True``, ``classifier`` is used as the *sampling* classifier
    (already constructed with the desired temperature by the caller) and each
    row is classified via ``debate.run_debate`` instead of a single
    ``classifier.classify`` call, using ``critic_classifiers``/
    ``reconciler_classifiers`` (one ``Classifier`` per category, keyed by
    category name). ``sampling_temperature`` itself is not a parameter here —
    it's only meaningful at the point ``classifier`` is constructed.

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

        # Collision check extended to every generated audit column: none may
        # equal the text column, and no two categories' generated column sets
        # may overlap (this also catches e.g. category "sentiment_votes"
        # colliding with category "sentiment"'s own generated audit column).
        owner: dict[str, str] = {}
        for cat_name in category_names:
            generated = {cat_name, *_audit_columns(cat_name)}
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

    # Add category columns (and, under --critics, their audit columns) if not present.
    for cat_name in category_names:
        generated = {cat_name, *_audit_columns(cat_name)} if critics else {cat_name}
        for col in generated:
            if col not in df.columns:
                df[col] = None

    # When resuming into a separate --output file, seed already-classified
    # category columns (and audit columns, under --critics) from the prior
    # output before deciding what's left to do.
    if restore and output_path.exists() and output_path.resolve() != input_path.resolve():
        prior = pd.read_csv(output_path)
        if len(prior) != len(df):
            raise ValueError(
                f"Cannot restore from '{output_path}': it has {len(prior)} rows, "
                f"but '{input_path}' has {len(df)} rows."
            )
        for cat_name in category_names:
            generated = {cat_name, *_audit_columns(cat_name)} if critics else {cat_name}
            for col in generated:
                if col in prior.columns:
                    df[col] = prior[col]

    work_idx = df.index

    if restore:
        completeness_cols = list(category_names)
        if critics:
            completeness_cols += [f"{name}_votes" for name in category_names]
        already_done = df.loc[work_idx, completeness_cols].notna().all(axis=1)
        work_idx = work_idx[~already_done]
    else:
        reset_cols = list(category_names)
        if critics:
            reset_cols += [col for name in category_names for col in _audit_columns(name)]
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
    classified = 0
    failed = 0
    completed = 0

    progress = tqdm(total=total, unit="row", desc="Classifying")
    # Worker threads only run the (read-only) LLM call and return data; all
    # mutation of `df` happens here on the main thread as futures complete, so
    # no locking is needed and output order stays tied to the row index.
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
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
        else:
            future_to_idx = {
                executor.submit(classifier.classify, str(df.at[idx, column])): idx
                for idx in work_idx
            }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                classification: dict[str, Any] = future.result()
                for cat, val in classification.items():
                    df.at[idx, cat] = val
                classified += 1
            except Exception:  # noqa: BLE001 - one bad row shouldn't abort the run
                failed += 1
            completed += 1
            progress.update(1)
            progress.set_postfix(ok=classified, failed=failed)
            if completed % save_every == 0:
                df.to_csv(output_path, index=False)

    df.to_csv(output_path, index=False)
    progress.close()
    return output_path
