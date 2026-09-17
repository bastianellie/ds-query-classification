"""Extract a single ``Category`` taxonomy from a free-text instruction, via one
LLM call (spec 8). Unlike ``induction.py``, this needs no labeled training
data at all: no ``train_df``, no label column, no sampled examples.

Per INV-1, this module may import ``categories.py``, ``classifier.py``,
``schema.py``, ``prompts.py``, ``resources.py``, and ``cost.py`` -- it must
not import ``pipeline.py``, ``batching.py``, ``debate.py``, ``multi_model.py``,
``cli.py``, or ``experiment.py``. ``extract_category_from_prompt`` itself is a
pure function with respect to the filesystem, using only ``categories.py``/
``classifier.py`` -- mirroring ``induction.py``'s own contract; ``main()``
(added in a later task) is the sole reader/writer of any file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import litellm
from dotenv import load_dotenv

from query_classification import cost
from query_classification.categories import Category, Label, load_categories
from query_classification.classifier import (
    Classifier,
    build_cerebus_completion_kwargs,
    cerebus_enabled_via_env,
    cerebus_model_id,
    reject_insecure_cerebus_endpoint,
)
from query_classification import resources as resources_module
from query_classification.prompts import build_category_extraction_prompt
from query_classification.resources import (
    DEFAULT_CATEGORY_EXTRACTION_PROMPT_FILE,
    check_default_resources_available,
)
from query_classification.schema import build_category_extraction_model

_DEFAULT_MODEL = "azure/gpt-5-chat"

# Duplicated from experiment.py's _RESERVED_SENTINEL/_RESERVED_SENTINEL_PREFIX/
# _check_reserved_sentinel, not imported -- this module must not import
# experiment.py (AR-1.5/INV-1). A test asserts this stays behaviorally
# identical to experiment.py's own function, mirroring spec 5's own
# _DATA_START/_DATA_END duplicate-equality precedent between batching.py and
# induction.py.
_RESERVED_SENTINEL_PREFIX = "none - "
_RESERVED_SENTINEL = "none"


def _check_reserved_sentinel(label_values: list[str]) -> None:
    """Reject a label value colliding with the 'none'/'none - X' sentinel."""
    for value in label_values:
        normalized = str(value).strip().lower()
        if normalized == _RESERVED_SENTINEL or normalized.startswith(_RESERVED_SENTINEL_PREFIX):
            raise ValueError(
                f"label value {value!r} collides with the reserved 'none'/'none - <label>' "
                "convention and cannot be used"
            )


def extract_category_from_prompt(
    instruction_text: str, category_name: str, classifier: Classifier
) -> Category:
    """Extract one ``Category`` from ``instruction_text`` via one
    ``classifier.classify()`` call. ``classify()``'s own signature/contract is
    completely unchanged.

    ``category_name`` is used as-is, never validated here -- FR-1.2's
    downstream-safety validation is CLI-level (``main()``, a later task),
    checked before this function is ever called. This function trusts its
    caller, exactly as ``induction.py``'s own pure function trusts
    ``experiment.py`` to have validated ``--category-name`` upstream.
    """
    result = classifier.classify(instruction_text)

    labels_raw = result["labels"]
    if not labels_raw:
        raise ValueError("at least one label is required, got zero")

    values = [str(label["value"]) for label in labels_raw]
    duplicates = [v for v in values if values.count(v) > 1]
    if duplicates:
        raise ValueError(f"duplicate label value(s): {sorted(set(duplicates))}")

    _check_reserved_sentinel(values)

    category_description = str(result["category_description"])
    if not category_description.strip():
        raise ValueError("category_description must not be blank")

    labels: list[Label] = []
    for label in labels_raw:
        value = str(label["value"])
        description = str(label["description"])
        if not value.strip():
            raise ValueError("label value must not be blank")
        if not description.strip():
            raise ValueError(f"label {value!r}'s description must not be blank")
        labels.append(Label(value=value, description=description))

    return Category(name=category_name, description=category_description, labels=labels)


def _validate_category_name(name: str) -> None:
    """FR-1.2: reject a --category-name that would crash
    build_classification_model downstream (verified live: '__root__' and any
    'model_'-prefixed name). Checked immediately after argument parsing,
    before any LLM call, so the message can name which specific rule failed."""
    if not name.isidentifier():
        raise ValueError(f"--category-name {name!r} must be a valid Python identifier")
    if name == "__root__":
        raise ValueError("--category-name must not be '__root__'")
    if name.startswith("model_"):
        raise ValueError(f"--category-name {name!r} must not start with 'model_'")


def _quiet_logging() -> None:
    """Suppress litellm/library chatter, matching cli.py's own convention."""
    litellm.suppress_debug_info = True
    litellm.set_verbose = False
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract a single categories.json Category from a free-text "
        "instruction file, via one LLM call. No labeled training data needed.",
    )
    parser.add_argument(
        "--prompt-file", required=True, metavar="FILE",
        help="Path to the plain-text instruction file describing the taxonomy to extract.",
    )
    parser.add_argument(
        "--category-name", required=True, metavar="NAME",
        help="Output Category.name (must be a valid Python identifier, not '__root__' or "
        "'model_'-prefixed). Never invented by the LLM.",
    )
    parser.add_argument(
        "--output", default="categories.json", metavar="FILE",
        help="Output categories.json path (default: categories.json in the current directory).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow replacing an existing file at --output.",
    )
    parser.add_argument(
        "--extraction-prompt", default=None, metavar="FILE",
        help="Path to the system prompt template file (default: bundled template).",
    )
    parser.add_argument(
        "--model", default=_DEFAULT_MODEL, metavar="MODEL",
        help=f"LiteLLM model identifier (default: {_DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--retries", type=int, default=3, metavar="N",
        help="Max retries for the classify call (default: 3).",
    )
    parser.add_argument(
        "--cerebus", action="store_true",
        help="Route the classify() call through the Cerebus/Portkey gateway instead of a "
        "direct provider. Equivalent to setting DEFAULT_LLM_PROVIDER=cerebus in .env.",
    )
    parser.add_argument(
        "--api-base", default=None, metavar="URL",
        help="Override the API endpoint/base URL.",
    )
    return parser


def main() -> None:
    load_dotenv(override=True)
    _quiet_logging()
    args = build_parser().parse_args()
    args.cerebus = args.cerebus or cerebus_enabled_via_env()

    try:
        _validate_category_name(args.category_name)

        output_path = Path(args.output)
        # Check the lexical, as-given path for a symlink BEFORE any resolve() --
        # resolve() follows the final symlink, so checking is_symlink() after
        # resolving would inspect the symlink's target, not the caller-supplied
        # link itself, and could silently miss this refusal (FR-1.7).
        if output_path.is_symlink():
            raise ValueError(f"--output '{output_path}' is a symlink; refusing to write through it")
        if output_path.is_dir():
            raise ValueError(f"--output '{output_path}' is a directory, not a file")
        if not output_path.parent.is_dir():
            raise FileNotFoundError(f"--output parent directory does not exist: {output_path.parent}")
        if output_path.exists() and not args.overwrite:
            raise ValueError(
                f"--output '{output_path}' already exists; pass --overwrite to replace it"
            )

        instruction_text = Path(args.prompt_file).read_text(encoding="utf-8")
        if not instruction_text.strip():
            raise ValueError(f"--prompt-file '{args.prompt_file}' is empty or whitespace-only")

        if args.extraction_prompt is None:
            try:
                check_default_resources_available()
            except FileNotFoundError as e:  # noqa: BLE001 - re-raised with this
                # tool's own remediation text below; the shared helper's message
                # names --categories/--system-prompt/--task-description, none of
                # which are flags on this entry point.
                raise FileNotFoundError(
                    f"Bundled resources directory not found: {resources_module.RESOURCES_DIR}. "
                    "Default prompt templates are only available from a source checkout or an "
                    "editable install (`pip install -e .`) - a plain wheel install does not "
                    "include them. Pass an explicit --extraction-prompt path instead."
                ) from e
        extraction_prompt_file = args.extraction_prompt or DEFAULT_CATEGORY_EXTRACTION_PROMPT_FILE
        system_prompt = build_category_extraction_prompt(extraction_prompt_file)

        # Cerebus is opt-in and resolved once, mirroring cli.py's own block
        # (cli.py:614-629) -- --api-base still wins if given explicitly, but
        # must be HTTPS, since the gateway key/headers get attached to it
        # either way.
        gateway_kwargs: dict = {"api_base": args.api_base, "api_key": None, "extra_headers": None}
        if args.cerebus:
            if args.api_base:
                reject_insecure_cerebus_endpoint(args.api_base)
            resolved = build_cerebus_completion_kwargs()
            gateway_kwargs["api_base"] = args.api_base or resolved["api_base"]
            gateway_kwargs["api_key"] = resolved["api_key"]
            gateway_kwargs["extra_headers"] = resolved["extra_headers"]

        model_id = cerebus_model_id(args.model) if args.cerebus else args.model
        classifier = Classifier(
            model_id=model_id,
            system_prompt=system_prompt,
            classification_model=build_category_extraction_model(),
            max_retries=args.retries,
            **gateway_kwargs,
        )

        category = extract_category_from_prompt(instruction_text, args.category_name, classifier)

        cost.write_json_atomic(output_path, {"categories": [category.model_dump(mode="json")]})
        load_categories(output_path)  # round-trip validation of what's on disk

        print(f"Wrote {output_path}")
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
