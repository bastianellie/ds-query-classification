"""Command-line entry point: ``python classify.py`` / ``python -m query_classification``."""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from collections import Counter
from pathlib import Path

import litellm
from dotenv import load_dotenv

from query_classification.categories import load_categories
from query_classification.classifier import (
    Classifier,
    build_cerebus_completion_kwargs,
    cerebus_enabled_via_env,
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
        nargs="+",
        default=None,
        metavar="MODEL",
        help="LiteLLM model identifier, e.g. gpt-4o-mini, claude-3-5-haiku-20241022 "
        "(default: azure/gpt-5-chat). Accepts exactly one value — use --models for "
        "multiple, or --n-classifiers to replicate this one model into several "
        "independent classifier instances.",
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
        "--models",
        nargs="+",
        metavar="MODEL",
        help="Enable multi-classifier voting mode: an alternative to --critics. "
        "2 or more space-separated LiteLLM model ids (e.g. azure/gpt-5-chat "
        "gpt-4o-mini gemini/gemini-2.5-pro) — duplicates allowed, e.g. the same "
        "model repeated for a plurality vote of several independent samples; "
        "a single value is equivalent to --model. Each classifies the row "
        "independently once (no sampling); results are merged per category by "
        "plurality vote on each instance's top label. Adds a vote/by-model/error "
        "audit trail to the output CSV. Mutually exclusive with --critics when "
        "given 2+ values. Roughly multiplies LLM call volume (and concurrent "
        "in-flight requests) by the number of classifier instances — size "
        "--workers down accordingly.",
    )
    parser.add_argument(
        "--n-classifiers",
        type=int,
        default=1,
        metavar="N",
        help="Number of classifier instances to construct (default: 1). Only "
        "meaningful when replicating a single resolved model (--model, or "
        "--models with exactly one value) — has no effect when --models is "
        "given with 2+ values, where the classifier count is simply "
        "len(--models).",
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
        "of a direct provider. Equivalent to setting DEFAULT_LLM_PROVIDER=cerebus "
        "in .env (either one turns it on). Minimal setup: DEFAULT_LLM_PROVIDER=cerebus "
        "+ optional CEREBUS_API_KEY in .env — see .env.example for advanced overrides "
        "(CEREBUS_MODE/CEREBUS_GATEWAY_*_URL/CEREBUS_CONFIG_ID). --model and friends "
        "still name the underlying model/slug; --api-base overrides the gateway URL "
        "if explicitly given.",
    )
    return parser


def _quiet_logging() -> None:
    """Suppress litellm/library chatter so only the tqdm bar shows."""
    litellm.suppress_debug_info = True
    litellm.set_verbose = False
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)


_DEFAULT_MODEL = "azure/gpt-5-chat"


def _resolve_classifier_models(args: argparse.Namespace) -> list[str]:
    """Resolve --model/--models/--n-classifiers into the final, ordered list of
    model ids to construct one classifier instance per (spec 4 FR-1.1). Raises
    ValueError for any invalid combination, before any LLM call. Duplicates are
    preserved when --models supplies them directly.

    A resolved list of length 1 is today's existing plain/--critics-classifier
    behavior, unchanged; length 2+ engages multi-classifier voting mode.
    """
    if args.model is not None and len(args.model) > 1:
        raise ValueError(
            f"--model only accepts a single model id, got {len(args.model)}: "
            f"{args.model!r}. Use --models for multiple models."
        )
    if args.n_classifiers < 1:
        raise ValueError(f"--n-classifiers must be >= 1, got {args.n_classifiers}")

    models_multi = args.models is not None and len(args.models) > 1

    if args.model is not None and args.models is not None and len(args.models) != 1:
        raise ValueError(
            "--model and --models cannot both be given, unless --models has "
            "exactly one value (which silently overrides --model)"
        )

    if models_multi:
        if args.critics:
            raise ValueError("--models with 2+ values and --critics are mutually exclusive")
        if any(not m.strip() for m in args.models):
            raise ValueError(f"--models values must be non-empty, got {args.models!r}")
        resolved = list(args.models)
    else:
        if args.models is not None:
            single = args.models[0]  # single-value --models overrides --model
        elif args.model is not None:
            single = args.model[0]
        else:
            single = _DEFAULT_MODEL
        if not single.strip():
            raise ValueError("model id must not be empty/whitespace-only")
        if args.n_classifiers > 1 and args.critics:
            raise ValueError("--n-classifiers > 1 and --critics are mutually exclusive")
        resolved = [single] * args.n_classifiers

    if len(set(resolved)) > 1 and os.getenv("CEREBUS_MODE") == "azure":
        raise ValueError(
            "--models/--n-classifiers with 2+ distinct models cannot be combined "
            "with CEREBUS_MODE=azure: CEREBUS_CONFIG_ID is a single, "
            "workspace/model-specific value and cannot be applied to multiple "
            "distinct models"
        )
    return resolved


def _build_classifier_dict(resolved_models: list[str], build_one) -> dict[str, Classifier]:
    """Build {key: Classifier} for a resolved multi-classifier list (spec 4
    AR-1.3): keyed by bare model id, unless that id occurs more than once in
    ``resolved_models``, in which case every occurrence gets an
    occurrence-suffixed key ("<id>#N", 1-indexed by position) so repeated
    models remain individually addressable in the audit trail. ``build_one``
    constructs a Classifier from a raw model id."""
    counts = Counter(resolved_models)
    seen: dict[str, int] = {}
    result: dict[str, Classifier] = {}
    for m in resolved_models:
        if counts[m] > 1:
            seen[m] = seen.get(m, 0) + 1
            key = f"{m}#{seen[m]}"
        else:
            key = m
        if key in result:
            raise ValueError(
                f"Model id collision building classifier keys: '{key}' would be "
                f"assigned to two different classifier instances. This can happen "
                f"if a supplied model id already looks like '<other-id>#N'. Rename "
                f"the conflicting model id."
            )
        result[key] = build_one(m)
    return result


def main() -> None:
    load_dotenv(override=True)
    _quiet_logging()
    args = build_parser().parse_args()
    # DEFAULT_LLM_PROVIDER=cerebus is an alternative to --cerebus, for a
    # set-it-in-.env-and-forget-it setup; either one turns gateway routing on.
    args.cerebus = args.cerebus or cerebus_enabled_via_env()

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

        # --model/--models/--n-classifiers resolution (spec 4, FR-1.1) — raised
        # before any LLM call, mirroring the --sampling-temperature checks above.
        resolved_models = _resolve_classifier_models(args)

        categories = load_categories(args.categories)
        # Under --critics or multi-classifier mode, the classifier's schema/
        # prompt must both be built with the same allow_new_labels value for the
        # constraint to actually apply (a schema-only or prompt-only fix is
        # incomplete). Multi-classifier mode is a --critics sibling, not a
        # plain-mode variant, so it shares --critics' carve-out here instead of
        # plain mode's forced-invention default. NOTE: `args.critics` must stay
        # in this condition — a resolved single classifier (ordinary --critics
        # usage) must not be forced into allow_new_labels=True.
        allow_new_labels = (
            args.allow_new_labels if (args.critics or len(resolved_models) > 1) else True
        )
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

        classifier = None
        models_classifiers = None
        if len(resolved_models) > 1:
            # Multi-classifier mode is a --critics sibling: one Classifier per
            # resolved model instance, sharing the identical system prompt/
            # schema/temperature=None (each model's own provider default, not
            # --critics' sampling temperature) and the identical gateway/api
            # config — mirrors the critic_classifiers/reconciler_classifiers
            # dict-comprehension pattern below, but keyed by (occurrence-
            # suffixed) model id instead of category name.
            models_classifiers = _build_classifier_dict(
                resolved_models,
                lambda model_id: Classifier(
                    model_id=_model_id(model_id),
                    system_prompt=system_prompt,
                    classification_model=classification_model,
                    max_retries=args.retries,
                    temperature=None,
                    **gateway_kwargs,
                ),
            )
        else:
            classifier = Classifier(
                model_id=_model_id(resolved_models[0]),
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
                    model_id=_model_id(args.critic_model or resolved_models[0]),
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
                    model_id=_model_id(args.reconciler_model or resolved_models[0]),
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
            models=models_classifiers,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
