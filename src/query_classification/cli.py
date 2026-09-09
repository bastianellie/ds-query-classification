"""Command-line entry point: ``python classify.py`` / ``python -m query_classification``."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import litellm
from dotenv import load_dotenv

from query_classification.categories import load_categories
from query_classification.classifier import (
    Classifier,
    build_cerebus_completion_kwargs,
    cerebus_model_id,
    reject_insecure_cerebus_endpoint,
)
from query_classification.pipeline import classify_csv
from query_classification.prompts import (
    build_critic_prompt,
    build_reconciler_prompt,
    build_system_prompt,
)
from query_classification.resources import (
    DEFAULT_CATEGORIES_FILE,
    DEFAULT_SYSTEM_PROMPT_FILE,
    check_default_resources_available,
)
from query_classification.schema import (
    build_classification_model,
    build_critic_model,
    build_reconciler_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify text from a CSV column against user-defined categories using an LLM.",
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input CSV file")
    parser.add_argument("-c", "--column", required=True, help="Column name to classify")
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Path to output CSV file (default: overwrite input)",
    )
    parser.add_argument(
        "--categories",
        default=str(DEFAULT_CATEGORIES_FILE),
        help="Path to categories JSON file (default: bundled example categories)",
    )
    parser.add_argument(
        "--model",
        default="azure/gpt-5-chat",
        help="LiteLLM model identifier, e.g. gpt-4o-mini, claude-3-5-haiku-20241022",
    )
    parser.add_argument(
        "--api-base",
        metavar="URL",
        help="Override the API endpoint/base URL. Defaults to the first of "
        "LITELLM_API_BASE, AZURE_API_BASE, AZURE_OPENAI_ENDPOINT, OPENAI_BASE_URL, "
        "OPENAI_API_BASE that is set; otherwise litellm's own resolution is used.",
    )
    parser.add_argument(
        "--system-prompt",
        metavar="FILE",
        help="Path to the system prompt template file (default: bundled template)",
    )
    parser.add_argument(
        "--task-description",
        metavar="FILE",
        help="Path to a plain-text file describing the classification task/domain "
        "(fills the {task_description} slot in the template)",
    )
    parser.add_argument(
        "--extra-prompt",
        metavar="FILE",
        help="Path to a plain-text file with additional prompt instructions",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="Only classify the first N rows (useful for debugging)",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Skip rows that already have classification values; resume from a previous run",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        metavar="N",
        help="Number of retry attempts per row on LLM error (default: 3)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        metavar="N",
        help="Number of concurrent worker threads for LLM calls (default: 8). "
        "Use 1 to limit row-level concurrency to one row at a time (with "
        "--critics, that row's own sampling calls still run concurrently with "
        "each other, so this no longer means zero concurrency overall).",
    )
    parser.add_argument(
        "--critics",
        action="store_true",
        help="Enable self-consistency sampling + Critic/Reconciler debate mode: "
        "each row is sampled --sampling-runs times and voted on; categories "
        "without consensus are argued out by a Critic and, if challenged, "
        "settled by a Reconciler. Adds a full audit trail to the output CSV. "
        "Roughly multiplies LLM call volume by --sampling-runs per row.",
    )
    parser.add_argument(
        "--sampling-runs",
        type=int,
        default=5,
        metavar="N",
        help="Number of independent classification samples per row under "
        "--critics (default: 5).",
    )
    parser.add_argument(
        "--sampling-temperature",
        type=float,
        default=0.7,
        metavar="T",
        help="Sampling temperature for the --critics self-consistency samples "
        "(default: 0.7). Ignored without --critics.",
    )
    parser.add_argument(
        "--consensus-threshold",
        type=int,
        default=4,
        metavar="N",
        help="Minimum vote count (out of --sampling-runs) for a category to "
        "bypass debate under --critics (default: 4).",
    )
    parser.add_argument(
        "--allow-new-labels",
        action="store_true",
        help="Under --critics, allow the sampling classifier to suggest "
        '"none - <new label>" instead of only the predefined labels or plain '
        '"none". Off by default so votes converge on a small, fixed set of '
        "options.",
    )
    parser.add_argument(
        "--critic-model",
        metavar="MODEL",
        help="LiteLLM model id for the --critics Critic role. Defaults to "
        "--model. May route the same row text to a different provider than "
        "--model.",
    )
    parser.add_argument(
        "--reconciler-model",
        metavar="MODEL",
        help="LiteLLM model id for the --critics Reconciler role. Defaults to "
        "--model. May route the same row text to a different provider than "
        "--model.",
    )
    parser.add_argument(
        "--cerebus",
        action="store_true",
        help="Route every LLM call through the Cerebus/Portkey gateway instead "
        "of a direct provider. Configure via CEREBUS_MODE/CEREBUS_GATEWAY_*_URL/"
        "CEREBUS_CONFIG_ID/CEREBUS_API_KEY in .env (see .env.example). --model "
        "and friends still name the underlying model/slug; --api-base overrides "
        "the gateway URL if explicitly given.",
    )
    return parser


def _quiet_logging() -> None:
    """Suppress litellm/library chatter so only the tqdm bar shows."""
    litellm.suppress_debug_info = True
    litellm.set_verbose = False
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def main() -> None:
    load_dotenv(override=True)
    _quiet_logging()
    args = build_parser().parse_args()

    try:
        uses_bundled_defaults = (
            args.categories == str(DEFAULT_CATEGORIES_FILE)
            or args.system_prompt is None
            or args.critics  # Critic/Reconciler role prompts are always bundled.
        )
        if uses_bundled_defaults:
            check_default_resources_available()

        if args.critics and not math.isfinite(args.sampling_temperature):
            raise ValueError(
                f"--sampling-temperature must be a finite number, got "
                f"{args.sampling_temperature}"
            )
        if args.critics and args.sampling_temperature < 0:
            raise ValueError(
                f"--sampling-temperature must be >= 0, got {args.sampling_temperature}"
            )

        categories = load_categories(args.categories)
        # Under --critics, the sampling classifier's schema/prompt must both be
        # built with the same allow_new_labels value for the constraint to
        # actually apply (a schema-only or prompt-only fix is incomplete).
        allow_new_labels = args.allow_new_labels if args.critics else True
        classification_model = build_classification_model(
            categories, allow_new_labels=allow_new_labels
        )

        system_prompt_file = (
            Path(args.system_prompt) if args.system_prompt else DEFAULT_SYSTEM_PROMPT_FILE
        )
        task_description = (
            Path(args.task_description).read_text() if args.task_description else None
        )
        extra_prompt = Path(args.extra_prompt).read_text() if args.extra_prompt else None
        system_prompt = build_system_prompt(
            classification_model,
            system_prompt_file,
            task_description=task_description,
            extra_prompt=extra_prompt,
            allow_new_labels=allow_new_labels,
        )

        if not args.output:
            print(
                "\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                "  WARNING: no --output specified. The input file will be\n"
                f"  OVERWRITTEN in place: {args.input}\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            )

        # Cerebus is opt-in and resolved once: every Classifier below shares
        # the same gateway credentials/headers. --api-base still wins if the
        # user gave it explicitly (an override, not a gateway bypass) — but
        # it must be HTTPS, since the gateway key/headers still get attached
        # to it either way.
        gateway_kwargs: dict = {"api_base": args.api_base, "api_key": None, "extra_headers": None}
        if args.cerebus:
            if args.api_base:
                reject_insecure_cerebus_endpoint(args.api_base)
            resolved = build_cerebus_completion_kwargs()
            gateway_kwargs["api_base"] = args.api_base or resolved["api_base"]
            gateway_kwargs["api_key"] = resolved["api_key"]
            gateway_kwargs["extra_headers"] = resolved["extra_headers"]

        def _model_id(raw: str) -> str:
            return cerebus_model_id(raw) if args.cerebus else raw

        classifier = Classifier(
            model_id=_model_id(args.model),
            system_prompt=system_prompt,
            classification_model=classification_model,
            max_retries=args.retries,
            temperature=args.sampling_temperature if args.critics else None,
            **gateway_kwargs,
        )

        critic_classifiers = None
        reconciler_classifiers = None
        if args.critics:
            critic_classifiers = {
                cat.name: Classifier(
                    model_id=_model_id(args.critic_model or args.model),
                    system_prompt=build_critic_prompt(
                        cat.name,
                        cat.description,
                        "; ".join(f'"{lbl.value}": {lbl.description}' for lbl in cat.labels),
                    ),
                    classification_model=build_critic_model(cat),
                    max_retries=args.retries,
                    **gateway_kwargs,
                )
                for cat in categories
            }
            reconciler_classifiers = {
                cat.name: Classifier(
                    model_id=_model_id(args.reconciler_model or args.model),
                    system_prompt=build_reconciler_prompt(
                        cat.name,
                        cat.description,
                        "; ".join(f'"{lbl.value}": {lbl.description}' for lbl in cat.labels),
                    ),
                    classification_model=build_reconciler_model(cat),
                    max_retries=args.retries,
                    **gateway_kwargs,
                )
                for cat in categories
            }

        classify_csv(
            input_path=args.input,
            column=args.column,
            classifier=classifier,
            categories=categories,
            output_path=args.output,
            restore=args.restore,
            limit=args.limit,
            workers=args.workers,
            critics=args.critics,
            critic_classifiers=critic_classifiers,
            reconciler_classifiers=reconciler_classifiers,
            sampling_runs=args.sampling_runs,
            consensus_threshold=args.consensus_threshold,
            allow_new_labels=allow_new_labels,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
