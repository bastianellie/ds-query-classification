"""Command-line entry point: ``python classify.py`` / ``python -m query_classification``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import litellm
from dotenv import load_dotenv

from query_classification.categories import load_categories
from query_classification.classifier import Classifier
from query_classification.pipeline import classify_csv
from query_classification.prompts import build_system_prompt
from query_classification.resources import (
    DEFAULT_CATEGORIES_FILE,
    DEFAULT_SYSTEM_PROMPT_FILE,
    check_default_resources_available,
)
from query_classification.schema import build_classification_model


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
        "Use 1 to disable parallelism.",
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
            args.categories == str(DEFAULT_CATEGORIES_FILE) or args.system_prompt is None
        )
        if uses_bundled_defaults:
            check_default_resources_available()

        categories = load_categories(args.categories)
        classification_model = build_classification_model(categories)

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
        )

        if not args.output:
            print(
                "\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                "  WARNING: no --output specified. The input file will be\n"
                f"  OVERWRITTEN in place: {args.input}\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            )

        classifier = Classifier(
            model_id=args.model,
            system_prompt=system_prompt,
            classification_model=classification_model,
            max_retries=args.retries,
            api_base=args.api_base,
        )
        classify_csv(
            input_path=args.input,
            column=args.column,
            classifier=classifier,
            categories=categories,
            output_path=args.output,
            restore=args.restore,
            limit=args.limit,
            workers=args.workers,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
