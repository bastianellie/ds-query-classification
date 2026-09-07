"""Batch classification of a CSV column.

Reads a CSV, classifies one text column (optionally across several worker
threads), writes one output column per category, and saves incrementally so a
long run can be resumed (``restore``) or truncated for debugging (``limit``).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from query_classification.categories import Category
from query_classification.classifier import Classifier


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
) -> Path:
    """Classify ``column`` of the input CSV and write the augmented CSV.

    Rows are classified concurrently across ``workers`` threads (LLM calls are
    I/O-bound). Results are written back by row index, so the output preserves
    the original input order regardless of completion order. The CSV is flushed
    to disk every ``save_every`` completed rows (and once at the end) so a
    crashed run can be resumed with ``restore``.

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

    df = pd.read_csv(input_path)
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found. Available columns: {list(df.columns)}"
        )

    # Add category columns if not present.
    for col in category_names:
        if col not in df.columns:
            df[col] = None

    # When resuming into a separate --output file, seed already-classified
    # category columns from the prior output before deciding what's left to do.
    if restore and output_path.exists() and output_path.resolve() != input_path.resolve():
        prior = pd.read_csv(output_path)
        if len(prior) != len(df):
            raise ValueError(
                f"Cannot restore from '{output_path}': it has {len(prior)} rows, "
                f"but '{input_path}' has {len(df)} rows."
            )
        for col in category_names:
            if col in prior.columns:
                df[col] = prior[col]

    work_idx = df.index

    if restore:
        already_done = df.loc[work_idx, category_names].notna().all(axis=1)
        work_idx = work_idx[~already_done]
    else:
        df.loc[work_idx, category_names] = None

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
        future_to_idx = {
            executor.submit(classifier.classify, str(df.at[idx, column])): idx
            for idx in work_idx
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                classification = future.result()
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
