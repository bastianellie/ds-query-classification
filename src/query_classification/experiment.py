"""Experiment runner: induce a ``categories.json`` from a labeled train split,
classify a test split with any classifier mode the repo offers, and write a
self-contained, auditable run directory.

Three subcommands:

* ``induce``   — load a train split, induce ``categories.json``, stop.
* ``classify`` — load a test split, classify it against a supplied
  ``categories.json``, stop.
* ``run``      — both, in sequence, unless ``--categories`` is supplied (then
  induction is skipped and ``run`` behaves like ``classify``).

This module constructs every classifier role itself (AR-3.1/AR-3.4):
``pipeline.classify_csv`` is only the per-row loop, not the classifier/prompt
construction that ``cli.py`` normally provides, and this spec deliberately
does not modify ``cli.py`` or ``pipeline.py`` to share that code.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import math
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import litellm
import pandas as pd
from dotenv import load_dotenv

from query_classification import debate
from query_classification.categories import Category, Label, load_categories
from query_classification.classifier import (
    Classifier,
    build_cerebus_completion_kwargs,
    cerebus_enabled_via_env,
    cerebus_model_id,
    reject_insecure_cerebus_endpoint,
)
from query_classification.dataset_io import (
    load_hf_splits,
    load_local_split,
    project_and_filter,
    validate_hf_id,
)
from query_classification.induction import induce as run_induction
from query_classification.pipeline import classify_csv
from query_classification.prompts import (
    build_critic_prompt,
    build_induction_prompt,
    build_reconciler_prompt,
    build_system_prompt,
)
from query_classification.resources import (
    DEFAULT_SYSTEM_PROMPT_FILE,
    check_default_resources_available,
)
from query_classification.schema import (
    build_classification_model,
    build_critic_model,
    build_induction_model,
    build_reconciler_model,
)

_ARTIFACT_FILENAMES = (
    "run_config.json",
    "categories.json",
    "train.csv",
    "test.csv",
    "test_classified.csv",
    "induction_prompt.txt",
    "sampled_examples.json",
)

_RESERVED_SENTINEL_PREFIX = "none - "
_RESERVED_SENTINEL = "none"

_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the three-subcommand parser (``induce``/``classify``/``run``).

    Three explicit flag groups, not two: ``run`` performs induction *and*
    classification, so it needs both groups, not just the classification one.
    ``--model`` lives in the common group since every subcommand's own role
    (`--induction-model`, `--critic-model`, ...) defaults to it.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--train-file", metavar="FILE", help="Local train split CSV")
    common.add_argument("--test-file", metavar="FILE", help="Local test split CSV")
    common.add_argument(
        "--hf-dataset", metavar="ID", help="HuggingFace Hub dataset id (owner/name)"
    )
    common.add_argument(
        "--hf-config", metavar="NAME", help="HuggingFace dataset configuration name"
    )
    common.add_argument("--hf-revision", metavar="REV", help="HuggingFace dataset revision")
    common.add_argument(
        "--hf-train-split", default="train", metavar="NAME", help="Train split name (default: train)"
    )
    common.add_argument(
        "--hf-test-split", default="test", metavar="NAME", help="Test split name (default: test)"
    )
    common.add_argument("--text-column", required=True, metavar="COL", help="Text column name")
    common.add_argument(
        "--label-column",
        metavar="COL",
        help="Gold label column name (required when inducing; optional otherwise "
        "— carried through to test_classified.csv as '{label-column}_gold' when given)",
    )
    common.add_argument(
        "--category-name",
        metavar="NAME",
        help="Name for the induced category (required when inducing)",
    )
    common.add_argument(
        "--categories", metavar="FILE", help="Path to a categories.json (single category only)"
    )
    common.add_argument("--run-dir", required=True, metavar="DIR", help="Run directory (required)")
    common.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace known artifact filenames in a non-empty --run-dir (never the "
        "directory itself or unrelated files)",
    )
    common.add_argument(
        "--model",
        default="azure/gpt-5-chat",
        help="Base LiteLLM model id; every role below defaults to this",
    )
    common.add_argument(
        "--cerebus",
        action="store_true",
        help="Route every LLM call (classification, critic/reconciler, induction) "
        "through the Cerebus/Portkey gateway instead of a direct provider. "
        "Equivalent to setting DEFAULT_LLM_PROVIDER=cerebus in .env (either one "
        "turns it on). Minimal setup: DEFAULT_LLM_PROVIDER=cerebus + optional "
        "CEREBUS_API_KEY in .env — see .env.example for advanced overrides "
        "(CEREBUS_MODE/CEREBUS_GATEWAY_*_URL/CEREBUS_CONFIG_ID). --model and "
        "friends still name the underlying model/slug; --api-base overrides the "
        "gateway URL if explicitly given.",
    )

    induction = argparse.ArgumentParser(add_help=False)
    induction.add_argument(
        "--seed", type=int, default=0, metavar="N", help="Induction sampling seed (default: 0)"
    )
    induction.add_argument(
        "--examples-per-label",
        type=int,
        default=20,
        metavar="N",
        help="Max sampled examples per label for induction (default: 20)",
    )
    induction.add_argument(
        "--max-example-chars",
        type=int,
        default=500,
        metavar="N",
        help="Truncate each sampled example to this many characters (default: 500)",
    )
    induction.add_argument(
        "--max-prompt-chars",
        type=int,
        default=100_000,
        metavar="N",
        help="Reject induction if the serialized prompt exceeds this many characters "
        "(default: 100000)",
    )
    induction.add_argument(
        "--induction-model", metavar="MODEL", help="LiteLLM model id for induction; defaults to --model"
    )

    classification = argparse.ArgumentParser(add_help=False)
    classification.add_argument("--api-base", metavar="URL", help="Override the API endpoint/base URL")
    classification.add_argument(
        "--retries", type=int, default=3, metavar="N", help="Retry attempts per row (default: 3)"
    )
    classification.add_argument(
        "--workers", type=int, default=8, metavar="N", help="Concurrent worker threads (default: 8)"
    )
    classification.add_argument(
        "--test-limit",
        type=int,
        metavar="N",
        help="Classify only the first N rows of the test split (omit to classify all rows)",
    )
    classification.add_argument("--system-prompt", metavar="FILE", help="Custom system prompt template")
    classification.add_argument(
        "--task-description", metavar="FILE", help="Fills the {task_description} template slot"
    )
    classification.add_argument("--extra-prompt", metavar="FILE", help="Additional prompt instructions")
    classification.add_argument(
        "--critics", action="store_true", help="Enable self-consistency sampling + Critic/Reconciler debate"
    )
    classification.add_argument("--sampling-runs", type=int, default=5, metavar="N")
    classification.add_argument("--sampling-temperature", type=float, default=0.7, metavar="T")
    classification.add_argument("--consensus-threshold", type=int, default=4, metavar="N")
    classification.add_argument(
        "--allow-new-labels",
        action="store_true",
        help="Allow label invention in BOTH classifier modes (off by default — a "
        "deliberate divergence from classify.py, which forces this on in plain mode; "
        "see README)",
    )
    classification.add_argument("--critic-model", metavar="MODEL")
    classification.add_argument("--reconciler-model", metavar="MODEL")
    classification.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exit zero even if some rows failed to classify (still recorded in run_config.json)",
    )

    parser = argparse.ArgumentParser(
        description="Run dataset-loading, label-induction, and classification experiments."
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    subparsers.add_parser("induce", parents=[common, induction])
    subparsers.add_parser("classify", parents=[common, classification])
    subparsers.add_parser("run", parents=[common, induction, classification])
    return parser


def _quiet_logging() -> None:
    """Suppress litellm/library chatter — duplicated from cli.py's tiny body
    rather than imported, per this spec's deliberate small-duplication call."""
    litellm.suppress_debug_info = True
    litellm.set_verbose = False
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _log_phase(phase: str) -> None:
    print(f"[{phase}]", file=sys.stderr)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _check_reserved_sentinel(label_values: list[str]) -> None:
    """FR-2.5: reject a label value colliding with the 'none'/'none - X' sentinel."""
    for value in label_values:
        normalized = str(value).strip().lower()
        if normalized == _RESERVED_SENTINEL or normalized.startswith(_RESERVED_SENTINEL_PREFIX):
            raise ValueError(
                f"label value {value!r} collides with the reserved 'none'/'none - <label>' "
                "convention and cannot be used"
            )


def _validate_args(args: argparse.Namespace) -> None:
    if args.text_column == args.label_column:
        raise ValueError("--text-column and --label-column must differ")

    has_hf = args.hf_dataset is not None
    has_local = args.train_file is not None or args.test_file is not None
    if has_hf and has_local:
        raise ValueError(
            "--hf-dataset and --train-file/--test-file are mutually exclusive"
        )
    if has_hf:
        validate_hf_id(args.hf_dataset)

    has_categories = args.categories is not None
    if args.train_file is not None and has_categories:
        raise ValueError(
            "--train-file and --categories are mutually exclusive: a supplied train "
            "split alongside a supplied categories.json is ambiguous — drop one"
        )

    if args.subcommand == "induce" and has_categories:
        raise ValueError(
            "induce produces categories.json; passing --categories is not applicable "
            "here (did you mean 'classify' or 'run'?)"
        )

    for name, value in (
        ("--examples-per-label", getattr(args, "examples_per_label", None)),
        ("--max-example-chars", getattr(args, "max_example_chars", None)),
        ("--max-prompt-chars", getattr(args, "max_prompt_chars", None)),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")
    if getattr(args, "seed", None) is not None and args.seed < 0:
        raise ValueError(f"--seed must be >= 0, got {args.seed}")
    if getattr(args, "test_limit", None) is not None and args.test_limit < 0:
        raise ValueError(f"--test-limit must be >= 0, got {args.test_limit}")

    if getattr(args, "critics", False):
        if not math.isfinite(args.sampling_temperature):
            raise ValueError(
                f"--sampling-temperature must be a finite number, got {args.sampling_temperature}"
            )
        if args.sampling_temperature < 0:
            raise ValueError(f"--sampling-temperature must be >= 0, got {args.sampling_temperature}")
        if args.sampling_runs < 1:
            raise ValueError(f"--sampling-runs must be >= 1, got {args.sampling_runs}")
        if not (1 <= args.consensus_threshold <= args.sampling_runs):
            raise ValueError(
                f"--consensus-threshold must be between 1 and --sampling-runs "
                f"({args.sampling_runs}) inclusive, got {args.consensus_threshold}"
            )


def _resolve_mode(args: argparse.Namespace) -> tuple[bool, bool]:
    """Return ``(will_induce, will_classify)``, enforcing FR-1.6's five
    split-availability cases. Raises ``ValueError`` naming what's missing for
    every error case, before any load/LLM call is attempted."""
    has_hf = args.hf_dataset is not None
    has_train = args.train_file is not None or has_hf
    has_test = args.test_file is not None or has_hf
    has_categories = args.categories is not None

    if args.subcommand == "induce":
        if not has_train:
            raise ValueError("induce requires a train split: --train-file or --hf-dataset")
        if not args.label_column:
            raise ValueError("--label-column is required when inducing")
        if not args.category_name:
            raise ValueError("--category-name is required when inducing")
        return True, False

    if args.subcommand == "classify":
        if not has_categories:
            raise ValueError("classify requires --categories")
        if not has_test:
            raise ValueError("classify requires a test split: --test-file or --hf-dataset")
        return False, True

    # run
    if not has_test:
        raise ValueError("run requires a test split: --test-file or --hf-dataset")
    if not has_train and not has_categories:
        raise ValueError(
            "run requires either a train split (to induce from) or --categories "
            "(to classify with) — neither was given"
        )
    will_induce = has_train and not has_categories
    if will_induce:
        if not args.label_column:
            raise ValueError("--label-column is required when inducing")
        if not args.category_name:
            raise ValueError("--category-name is required when inducing")
    return will_induce, True


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_split_frame(
    args: argparse.Namespace, role: str, want_train: bool, want_test: bool
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict[str, Any]]:
    """Load whichever of train/test this invocation needs, from local files
    or a HuggingFace dataset. Returns (train_df, test_df, dataset_meta) where
    dataset_meta carries the ``dataset`` run_config field plus internal-only
    ``source_column_names`` for the collision check."""
    if args.hf_dataset is not None:
        train_split = args.hf_train_split if want_train else None
        test_split = args.hf_test_split if want_test else None
        train_df, test_df, hf_meta = load_hf_splits(
            args.hf_dataset,
            args.hf_config,
            args.hf_revision,
            train_split,
            test_split,
            args.text_column,
            args.label_column,
        )
        dataset_meta = {
            "dataset": {
                "source": "hf",
                "id": args.hf_dataset,
                "config": args.hf_config,
                "revision_requested": args.hf_revision,
                "revision_resolved": hf_meta["resolved_revision"],
                "train_split": train_split,
                "test_split": test_split,
            },
            "source_column_names": hf_meta["source_column_names"],
        }
        return train_df, test_df, dataset_meta

    train_df = train_src_cols = None
    test_df = test_src_cols = None
    if want_train:
        train_df, train_src_cols = load_local_split(args.train_file, args.text_column, args.label_column)
    if want_test:
        test_df, test_src_cols = load_local_split(args.test_file, args.text_column, args.label_column)
    dataset_meta = {
        "dataset": {
            "source": "local",
            "train_file": args.train_file if want_train else None,
            "test_file": args.test_file if want_test else None,
        },
        "source_column_names": {"train": train_src_cols, "test": test_src_cols},
    }
    return train_df, test_df, dataset_meta


def _missing(series: pd.Series) -> pd.Series:
    return series.isna() | series.fillna("").astype(str).str.strip().eq("")


# ---------------------------------------------------------------------------
# Classifier construction
# ---------------------------------------------------------------------------


def _resolve_gateway_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve the gateway kwargs (api_base/api_key/extra_headers) once, so
    every classifier role — across however many `_construct_classifiers`
    calls one invocation makes (e.g. `run` calls it twice: once for
    induction, once for classification) — shares the identical values rather
    than each call independently re-reading the environment and re-resolving
    the key. --api-base still wins if the user gave it explicitly, but must
    be HTTPS under --cerebus (the gateway key/headers get attached to it
    either way)."""
    api_base = getattr(args, "api_base", None)
    if not getattr(args, "cerebus", False):
        return {"api_base": api_base, "api_key": None, "extra_headers": None}
    if api_base:
        reject_insecure_cerebus_endpoint(api_base)
    resolved = build_cerebus_completion_kwargs()
    return {
        "api_base": api_base or resolved["api_base"],
        "api_key": resolved["api_key"],
        "extra_headers": resolved["extra_headers"],
    }


def _construct_classifiers(
    args: argparse.Namespace,
    categories: list[Category] | None,
    will_induce: bool,
    will_classify: bool,
    gateway_kwargs: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, str | None]]:
    """Construct every classifier role this invocation needs. Returns
    (classifiers, resolved_model_ids). ``gateway_kwargs`` should be resolved
    once per invocation via `_resolve_gateway_kwargs` and passed in by the
    caller; if omitted (e.g. direct unit testing of this function), it's
    resolved locally instead."""
    classifiers: dict[str, Any] = {}
    models: dict[str, str | None] = {
        "model": args.model,
        "induction_model": None,
        "critic_model": None,
        "reconciler_model": None,
    }

    use_cerebus = getattr(args, "cerebus", False)
    if gateway_kwargs is None:
        gateway_kwargs = _resolve_gateway_kwargs(args)

    def _model_id(raw: str) -> str:
        return cerebus_model_id(raw) if use_cerebus else raw

    if will_induce:
        models["induction_model"] = getattr(args, "induction_model", None) or args.model
        classifiers["induction"] = Classifier(
            model_id=_model_id(models["induction_model"]),
            system_prompt=build_induction_prompt(),
            classification_model=build_induction_model(),
            max_retries=getattr(args, "retries", 3),
            **gateway_kwargs,
        )

    if will_classify:
        assert categories is not None
        allow_new_labels = args.allow_new_labels
        classification_model = build_classification_model(categories, allow_new_labels=allow_new_labels)
        system_prompt_file = Path(args.system_prompt) if args.system_prompt else DEFAULT_SYSTEM_PROMPT_FILE
        task_description = Path(args.task_description).read_text() if args.task_description else None
        extra_prompt = Path(args.extra_prompt).read_text() if args.extra_prompt else None
        system_prompt = build_system_prompt(
            classification_model,
            system_prompt_file,
            task_description=task_description,
            extra_prompt=extra_prompt,
            allow_new_labels=allow_new_labels,
        )
        classifiers["classification"] = Classifier(
            model_id=_model_id(args.model),
            system_prompt=system_prompt,
            classification_model=classification_model,
            max_retries=args.retries,
            temperature=args.sampling_temperature if args.critics else None,
            **gateway_kwargs,
        )

        if args.critics:
            models["critic_model"] = args.critic_model or args.model
            models["reconciler_model"] = args.reconciler_model or args.model
            critic_classifiers = {}
            reconciler_classifiers = {}
            for cat in categories:
                label_options = "; ".join(f'"{lbl.value}": {lbl.description}' for lbl in cat.labels)
                critic_classifiers[cat.name] = Classifier(
                    model_id=_model_id(models["critic_model"]),
                    system_prompt=build_critic_prompt(cat.name, cat.description, label_options),
                    classification_model=build_critic_model(cat),
                    max_retries=args.retries,
                    **gateway_kwargs,
                )
                reconciler_classifiers[cat.name] = Classifier(
                    model_id=_model_id(models["reconciler_model"]),
                    system_prompt=build_reconciler_prompt(cat.name, cat.description, label_options),
                    classification_model=build_reconciler_model(cat),
                    max_retries=args.retries,
                    **gateway_kwargs,
                )
            classifiers["critics"] = critic_classifiers
            classifiers["reconcilers"] = reconciler_classifiers

    return classifiers, models


# ---------------------------------------------------------------------------
# Run-directory helpers
# ---------------------------------------------------------------------------


def _write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON via a temp file + os.replace so an interruption never
    leaves a malformed control file on disk."""
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(data, f, indent=2)
        Path(tmp_name).replace(path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _preflight_run_dir(run_dir: Path, overwrite: bool) -> None:
    if run_dir.exists() and run_dir.is_symlink():
        raise ValueError(f"--run-dir '{run_dir}' is a symlink; refusing to write through it")
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in run_dir.iterdir()]
    if existing and not overwrite:
        raise ValueError(
            f"--run-dir '{run_dir}' is not empty (use --overwrite to replace known "
            f"artifact files only, never the directory or unrelated files)"
        )


def _cleanup_known_artifacts(run_dir: Path, *, skip: set[str]) -> None:
    for name in _ARTIFACT_FILENAMES:
        if name in skip:
            continue
        p = run_dir / name
        if p.is_symlink():
            p.unlink()
        elif p.exists():
            p.unlink()


def _versions_dict() -> dict[str, str | None]:
    def _v(pkg: str) -> str | None:
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "query_classification": _v("ds-query-classification"),
        "litellm": _v("litellm"),
        "pandas": _v("pandas"),
        "pydantic": _v("pydantic"),
        "datasets": _v("datasets"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    load_dotenv(override=True)
    _quiet_logging()
    args = build_parser().parse_args()
    # DEFAULT_LLM_PROVIDER=cerebus is an alternative to --cerebus, for a
    # set-it-in-.env-and-forget-it setup; either one turns gateway routing on.
    args.cerebus = args.cerebus or cerebus_enabled_via_env()

    try:
        _log_phase("validating")
        _validate_args(args)
        will_induce, will_classify = _resolve_mode(args)

        uses_bundled_defaults = (
            getattr(args, "system_prompt", None) is None
            or getattr(args, "critics", False)
            or will_induce
        )
        if uses_bundled_defaults:
            check_default_resources_available()

        # Resolved once here, before any run-dir work, and reused by every
        # _construct_classifiers call below — not re-resolved per role.
        gateway_kwargs = _resolve_gateway_kwargs(args)

        run_dir = Path(args.run_dir)

        # Resolve and read/validate every supplied input path BEFORE any
        # --overwrite cleanup deletes anything — a supplied --categories that
        # resolves to <run-dir>/categories.json must not be deleted before
        # it's read.
        supplied_category: Category | None = None
        supplied_bytes: bytes | None = None
        same_path_categories = False
        if args.categories is not None:
            supplied_path = Path(args.categories).resolve()
            cats = load_categories(supplied_path)
            if len(cats) != 1:
                raise ValueError(
                    f"--categories must contain exactly one category, got {len(cats)}"
                )
            supplied_category = cats[0]
            _check_reserved_sentinel([lbl.value for lbl in supplied_category.labels])
            same_path_categories = supplied_path == (run_dir / "categories.json").resolve()
            supplied_bytes = supplied_path.read_bytes()

        # Preflight the empty/--overwrite check before any file is modified —
        # this belongs with the other early validation, not the
        # failure-must-produce-a-run_config.json path below, since nothing
        # has been written yet at this point.
        _preflight_run_dir(run_dir, args.overwrite)
    except (ValueError, FileNotFoundError, ModuleNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    stage = "starting"
    config: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "subcommand": args.subcommand,
        "text_column": args.text_column,
        "label_column": args.label_column,
        "category_name": args.category_name,
        "seed": getattr(args, "seed", None),
        "examples_per_label": getattr(args, "examples_per_label", None),
        "max_example_chars": getattr(args, "max_example_chars", None),
        "max_prompt_chars": getattr(args, "max_prompt_chars", None),
        "classifier_config": (
            {
                "critics": args.critics,
                "sampling_runs": args.sampling_runs,
                "sampling_temperature": args.sampling_temperature,
                "consensus_threshold": args.consensus_threshold,
                "allow_new_labels": args.allow_new_labels,
                "retries": args.retries,
                "workers": args.workers,
                "test_limit": args.test_limit,
            }
            if will_classify
            else None
        ),
        "prompt_inputs": (
            {
                "system_prompt": getattr(args, "system_prompt", None),
                "task_description": getattr(args, "task_description", None),
                "extra_prompt": getattr(args, "extra_prompt", None),
            }
            if will_classify
            else None
        ),
        "versions": _versions_dict(),
        # Enabled/mode only — never the gateway URL, config ID, or key,
        # matching the existing rule that api_base's value is never recorded.
        "cerebus": {
            "enabled": getattr(args, "cerebus", False),
            "mode": os.getenv("CEREBUS_MODE") if getattr(args, "cerebus", False) else None,
        },
    }

    try:
        skip_cleanup = {"categories.json"} if same_path_categories else set()
        _cleanup_known_artifacts(run_dir, skip=skip_cleanup)

        stage = "loading"
        _log_phase("loading")
        train_df, test_df, dataset_meta = _load_split_frame(
            args, args.subcommand, want_train=will_induce, want_test=will_classify
        )
        config["dataset"] = dataset_meta["dataset"]
        source_column_names = dataset_meta["source_column_names"]

        for df, role in ((train_df, "train"), (test_df, "test")):
            if df is None:
                continue
            missing_cols = [
                c for c in (args.text_column, args.label_column) if c is not None and c not in df.columns
            ]
            if missing_cols:
                raise ValueError(
                    f"Column(s) {missing_cols} not found in {role} split. Available "
                    f"columns: {list(df.columns)}"
                )

        # AR-1.3: reduce to only the needed columns. The HF path already
        # projects before materializing (dataset_io.load_hf_splits); the
        # local-CSV path reads the whole file as-is, so the projection needs
        # to happen here too — this keeps the collision check's use of
        # source_column_names (the pre-projection header) meaningful for
        # both sources alike.
        keep_columns = [c for c in (args.text_column, args.label_column) if c is not None]
        if train_df is not None:
            train_df = train_df[keep_columns]
        if test_df is not None:
            test_df = test_df[keep_columns]

        excluded_rows: dict[str, Any] = {"train": None, "test": None}
        if train_df is not None:
            train_df, dropped = project_and_filter(
                train_df, args.text_column, args.label_column, is_train_role=True
            )
            excluded_rows["train"] = {"count": len(dropped), "positions": dropped}
        if test_df is not None:
            test_df, dropped = project_and_filter(
                test_df, args.text_column, args.label_column, is_train_role=False
            )
            excluded_rows["test"] = {"count": len(dropped), "positions": dropped}
        config["excluded_rows"] = excluded_rows

        outcome = None
        if will_induce:
            stage = "inducing"
            _log_phase("inducing")
            classifiers, models = _construct_classifiers(
                args, None, will_induce=True, will_classify=False, gateway_kwargs=gateway_kwargs
            )
            outcome = run_induction(
                train_df,
                args.text_column,
                args.label_column,
                args.category_name,
                classifiers["induction"],
                seed=args.seed,
                examples_per_label=args.examples_per_label,
                max_example_chars=args.max_example_chars,
                max_prompt_chars=args.max_prompt_chars,
            )
            (run_dir / "train.csv").write_text(train_df.to_csv(index=False))
            (run_dir / "induction_prompt.txt").write_text(outcome.rendered_prompt)
            _write_json_atomic(run_dir / "sampled_examples.json", outcome.sampled_examples)
            categories_path = run_dir / "categories.json"
            _write_json_atomic(
                categories_path,
                {"categories": [outcome.category.model_dump(mode="json")]},
            )
            load_categories(categories_path)  # round-trip validation of what's on disk
            categories = [outcome.category]
            label_values = sorted(lbl.value for lbl in outcome.category.labels)
            categories_source = "induced"
            models_final = models
        else:
            categories = None
            label_values = None
            categories_source = "supplied"
            models_final = {"model": args.model, "induction_model": None, "critic_model": None, "reconciler_model": None}

        if will_classify:
            stage = "classifying"
            _log_phase("classifying")
            if categories is None:
                assert supplied_category is not None
                categories = [supplied_category]
                label_values = sorted(lbl.value for lbl in supplied_category.labels)
                if not same_path_categories:
                    (run_dir / "categories.json").write_bytes(supplied_bytes)

            category_name = categories[0].name

            # test_labels derivation (FR-1.8): "withheld" iff every value is
            # missing/unusable (label column absent, or all-None/empty after
            # Task 1's per-row resolution), else "present". FR-3.6's rename is
            # mandatory only when gold carry-through actually happens.
            test_labels_status = "withheld"
            unseen_test_labels: list[str] = []
            gold_col = f"{args.label_column}_gold" if args.label_column else None
            if args.label_column is not None:
                usable_mask = ~_missing(test_df[args.label_column])
                if usable_mask.any():
                    test_labels_status = "present"
                    test_df = test_df.rename(columns={args.label_column: gold_col})
                    gold_values = set(test_df.loc[usable_mask.values, gold_col].tolist())
                    unseen_test_labels = sorted(gold_values - set(label_values or []))
                else:
                    test_df = test_df.drop(columns=[args.label_column])
                    gold_col = None
            config["test_labels"] = test_labels_status
            config["unseen_test_labels"] = unseen_test_labels
            config["label_values"] = sorted(label_values or [])
            config["categories_source"] = categories_source

            # Collision check (FR-3.6), against pre-projection source column
            # names, not the (already-projected) materialized frame — a
            # source column that AR-1.3 projected away is still a real
            # collision risk for the name we're about to generate.
            test_source_cols = set(source_column_names.get("test") or [])
            if args.label_column is not None:
                other_cols = test_source_cols - {args.label_column}
                if gold_col is not None and f"{args.label_column}_gold" in other_cols:
                    raise ValueError(
                        f"Dataset already has a column named '{args.label_column}_gold', "
                        "which collides with the gold-label column this run would create"
                    )
            generated = {category_name}
            if args.critics:
                generated |= {f"{category_name}{suffix}" for suffix in debate.AUDIT_COLUMN_SUFFIXES}
            collide = generated & (test_source_cols - ({args.label_column} if args.label_column else set()))
            if collide:
                raise ValueError(
                    f"Generated column(s) {sorted(collide)} collide with pre-existing "
                    f"source column(s) of the same name"
                )

            if args.test_limit is not None:
                test_df = test_df.iloc[: args.test_limit].reset_index(drop=True)

            test_csv_path = run_dir / "test.csv"
            test_df.to_csv(test_csv_path, index=False)

            classifiers, models = _construct_classifiers(
                args, categories, will_induce=False, will_classify=True, gateway_kwargs=gateway_kwargs
            )
            if will_induce:
                models_final["critic_model"] = models.get("critic_model")
                models_final["reconciler_model"] = models.get("reconciler_model")
            else:
                models_final = models

            classified_path = classify_csv(
                input_path=test_csv_path,
                column=args.text_column,
                classifier=classifiers["classification"],
                categories=categories,
                output_path=run_dir / "test_classified.csv",
                limit=None,
                workers=args.workers,
                critics=args.critics,
                critic_classifiers=classifiers.get("critics"),
                reconciler_classifiers=classifiers.get("reconcilers"),
                sampling_runs=args.sampling_runs,
                consensus_threshold=args.consensus_threshold,
                allow_new_labels=args.allow_new_labels,
            )

            stage = "verifying"
            _log_phase("verifying")
            result_df = pd.read_csv(classified_path)
            check_cols = [category_name]
            if args.critics:
                check_cols.append(f"{category_name}_votes")
            incomplete_mask = result_df[check_cols].isna().any(axis=1)
            unclassified_positions = result_df.index[incomplete_mask].tolist()
            config["unclassified_rows"] = {
                "count": len(unclassified_positions),
                "positions": unclassified_positions,
            }
            status = "completed_with_failures" if unclassified_positions else "completed"
        else:
            config["unclassified_rows"] = None
            config["test_labels"] = None
            config["unseen_test_labels"] = None
            config["label_values"] = label_values
            config["categories_source"] = categories_source
            status = "completed"

        config["models"] = models_final
        config["status"] = status
        stage = "finalizing"
        _log_phase("finalizing")
        _write_json_atomic(run_dir / "run_config.json", config)

    except Exception as e:  # noqa: BLE001 - any phase failure must still produce an auditable run_config.json
        for key in (
            "dataset",
            "excluded_rows",
            "test_labels",
            "unseen_test_labels",
            "label_values",
            "categories_source",
            "unclassified_rows",
            "models",
        ):
            config.setdefault(key, None)
        config["status"] = "failed"
        config["failure"] = f"{type(e).__name__}: failed during {stage}"
        try:
            _write_json_atomic(run_dir / "run_config.json", config)
        except OSError:
            pass
        print(f"Error: {e}")
        sys.exit(1)

    if will_classify and status == "completed_with_failures" and not args.allow_partial:
        sys.exit(1)


if __name__ == "__main__":
    main()
