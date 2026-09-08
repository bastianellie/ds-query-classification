"""Loading and normalizing train/test dataset splits.

Supports two sources: a local CSV pair, or a HuggingFace Hub dataset id. Both
are normalized down to plain ``pandas`` DataFrames with string labels (or
``None`` for a withheld/missing label), leaving row-level filtering to
:func:`project_and_filter` so both sources share one filtering rule.

The ``datasets`` package (needed only for the HF path) is an optional,
lazily-imported dependency — see the ``hf`` extra in ``pyproject.toml``.
``huggingface_hub`` (used for Hub-id validation and best-effort revision
resolution) is a normal, always-available import: it is already a
transitive dependency of ``datasets>=4`` and is installed regardless of
whether the ``hf`` extra itself is installed.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from huggingface_hub import HfApi
from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id

# Packaged `datasets` builder names: syntactically valid single-segment Hub
# ids (they pass `validate_repo_id` cleanly), but they resolve to a local
# file-format builder, not a Hub dataset — accepting them here would
# silently mislabel the resulting data as `source: "hf"`.
_PACKAGED_BUILDER_NAMES = frozenset(
    {
        "csv",
        "json",
        "parquet",
        "arrow",
        "text",
        "xml",
        "webdataset",
        "imagefolder",
        "audiofolder",
        "videofolder",
    }
)


def validate_hf_id(hf_id: str) -> None:
    """Validate ``hf_id`` as a real HuggingFace Hub dataset id.

    Raises ``ValueError`` if the id fails Hub syntax validation (wrong
    length/characters, a `.git` suffix, `--`/`..`, not `owner/name` or a
    canonical single-segment name, etc. — see
    ``huggingface_hub.utils.validate_repo_id``) or if it names a packaged
    `datasets` builder (e.g. ``"csv"``) rather than an actual Hub dataset.
    """
    try:
        validate_repo_id(hf_id)
    except HFValidationError as e:
        raise ValueError(f"'{hf_id}' is not a valid HuggingFace Hub dataset id: {e}") from e

    if hf_id in _PACKAGED_BUILDER_NAMES:
        raise ValueError(
            f"'{hf_id}' is a packaged `datasets` builder name (a local "
            f"file-format loader), not a HuggingFace Hub dataset id. Use a "
            f"local dataset source instead, or choose a real Hub dataset id."
        )


def resolve_hf_labels(dataset: Any, label_column: str) -> list[str | None]:
    """Resolve ``label_column`` to a plain list of strings (or ``None``).

    Reads ``dataset.features[label_column]`` and the raw per-row values via
    item access (``dataset[label_column]``) — independent of whatever
    ``to_pandas()`` does with a ``ClassLabel`` column internally.

    * ``datasets.ClassLabel``: each value maps through ``.int2str(v)`` for
      ``v >= 0``; negative indices (the withheld-label sentinel) map to
      ``None`` instead, since ``int2str`` raises ``ValueError`` on them.
    * ``datasets.Value`` with a string dtype: raw values pass through
      verbatim.
    * anything else: raises, naming the feature's actual type and dtype.

    A mixed split (some rows withheld, some not) needs no special handling
    here — this is already row-level, so it just produces a list with some
    ``None`` entries and some strings, in the same order as ``dataset``.
    """
    import datasets  # local: only reachable once a caller has already imported it

    feature = dataset.features[label_column]
    raw_values = dataset[label_column]

    if isinstance(feature, datasets.ClassLabel):
        return [feature.int2str(v) if v >= 0 else None for v in raw_values]

    if isinstance(feature, datasets.Value) and feature.dtype in ("string", "large_string"):
        return list(raw_values)

    raise ValueError(
        f"Label column '{label_column}' has an unsupported feature type "
        f"{type(feature).__name__} (dtype={getattr(feature, 'dtype', None)!r}); "
        f"expected datasets.ClassLabel or a string-typed datasets.Value."
    )


def load_hf_splits(
    hf_id: str,
    config: str | None,
    revision: str | None,
    train_split: str | None,
    test_split: str | None,
    text_column: str,
    label_column: str | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict[str, Any]]:
    """Load and normalize the requested splits of a HuggingFace Hub dataset.

    Returns ``(train_df, test_df, metadata)`` — either frame is ``None`` if
    the corresponding ``*_split`` argument is ``None`` (not requested).
    ``metadata`` carries ``resolved_revision`` (the Hub commit SHA if
    resolvable, else the requested ``revision`` verbatim — including
    ``None`` if none was given) and ``source_column_names`` (a
    ``{"train": [...] | None, "test": [...] | None}`` dict of each fetched
    split's full column list, captured before projection, for a later
    collision check).
    """
    try:
        import datasets
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Loading a HuggingFace dataset requires the optional 'datasets' "
            "dependency, which is not installed. Install it with: "
            "pip install '.[hf]'"
        ) from e

    try:
        # `split=` is intentionally omitted: this returns one DatasetDict, so
        # split-membership can be checked directly below rather than issuing
        # a second `get_dataset_split_names` call. Known, accepted cost:
        # this materializes every split in the dataset regardless of which
        # is actually requested.
        dd = datasets.load_dataset(hf_id, name=config, revision=revision)
    except Exception as e:  # noqa: BLE001 - HF load/auth failures may carry tokens or URLs in their message
        raise RuntimeError(f"{type(e).__name__}: HF dataset load failed") from e

    available_splits = list(dd.keys())
    for role, split_name in (("train", train_split), ("test", test_split)):
        if split_name is not None and split_name not in dd:
            raise ValueError(
                f"Requested {role} split '{split_name}' not found in dataset "
                f"'{hf_id}' (config={config!r}, revision={revision!r}). "
                f"Available splits: {available_splits}"
            )

    try:
        # Best-effort only: any failure (network, auth, unresolvable
        # revision) falls back to the requested revision string, never
        # raises — this metadata call must never fail the run.
        resolved_revision = HfApi().dataset_info(hf_id, revision=revision).sha
    except Exception:  # noqa: BLE001 - resolved-revision lookup is best-effort metadata only
        resolved_revision = revision

    source_column_names: dict[str, list[str] | None] = {"train": None, "test": None}
    frames: dict[str, pd.DataFrame | None] = {"train": None, "test": None}
    keep_columns = {text_column} | ({label_column} if label_column is not None else set())

    for role, split_name in (("train", train_split), ("test", test_split)):
        if split_name is None:
            continue
        dataset = dd[split_name]

        # HF assembly order (the one place to_pandas()'s ClassLabel
        # representation would matter, so the order below makes it not
        # matter):
        # (1) resolve labels first, while `dataset.features` is available.
        labels = resolve_hf_labels(dataset, label_column) if label_column is not None else None

        # (2) capture the full column list before projection.
        source_column_names[role] = list(dataset.column_names)

        # (3)-(4) project to just the needed columns, then materialize.
        drop_columns = [c for c in dataset.column_names if c not in keep_columns]
        frame = dataset.remove_columns(drop_columns).to_pandas()

        # (5) overwrite the label column with step (1)'s precomputed list —
        # guaranteed same row order/length as `frame`, since no filtering
        # has happened yet. Built as an explicit object-dtype Series so a
        # withheld-label `None` survives as-is, rather than pandas silently
        # normalizing it into a dtype-specific NA marker (e.g. its newer
        # string-dtype inference would otherwise swallow the distinction).
        if labels is not None:
            frame[label_column] = pd.Series(labels, index=frame.index, dtype=object)

        frames[role] = frame

    metadata: dict[str, Any] = {
        "resolved_revision": resolved_revision,
        "source_column_names": source_column_names,
    }
    return frames["train"], frames["test"], metadata


def load_local_split(
    path: str, text_column: str, label_column: str | None
) -> tuple[pd.DataFrame, list[str]]:
    """Read a local CSV split, whole, with all NA-sniffing disabled.

    Every value is read as a raw string (``dtype=str, keep_default_na=False,
    na_values=[]``), so values like ``"001"``, ``"NA"``, or ``"None"``
    survive unchanged in any column, with no per-column special-casing.
    Returns the frame and its full header (``source_column_names``, before
    any column is dropped downstream).
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
    source_column_names = list(df.columns)

    # Not spelled out in the plan text, but text_column/label_column are
    # otherwise-unused parameters without this: fail fast with a clear
    # message rather than a confusing KeyError several steps later.
    missing = [c for c in (text_column, label_column) if c is not None and c not in df.columns]
    if missing:
        raise ValueError(
            f"Column(s) {missing} not found in '{path}'. Available columns: "
            f"{source_column_names}"
        )

    return df, source_column_names


def _missing_mask(series: pd.Series) -> pd.Series:
    """True where a value is null (``None``/``NaN``/``pd.NA``) or its
    stripped string form is empty — the one missing-value predicate used
    everywhere in this module."""
    return series.isna() | series.fillna("").astype(str).str.strip().eq("")


def project_and_filter(
    df: pd.DataFrame,
    text_column: str,
    label_column: str | None,
    *,
    is_train_role: bool,
) -> tuple[pd.DataFrame, list[int]]:
    """Drop rows with a missing text value, matching a count-once-per-row
    rule (a single mask unioned across columns, not two separate drops).

    When ``label_column`` is given and ``is_train_role`` is true, rows with
    a missing label are dropped too, folded into the same mask. Operates on
    an already-normalized frame (post the HF assembly order in
    :func:`load_hf_splits`, or straight from :func:`load_local_split` for
    the local case).

    Returns the filtered frame and the 0-based positions in the input frame
    that were dropped (``df.index[mask].tolist()``, before any reset — valid
    because both dataset-loading paths start from a fresh 0-based
    ``RangeIndex``).
    """
    mask = _missing_mask(df[text_column])
    if label_column is not None and is_train_role:
        mask = mask | _missing_mask(df[label_column])

    dropped_positions = df.index[mask].tolist()
    filtered = df.loc[~mask]
    return filtered, dropped_positions
