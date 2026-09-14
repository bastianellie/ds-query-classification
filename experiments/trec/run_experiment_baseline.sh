#!/usr/bin/env bash
# Experiment: TREC question/intent classification, baseline — a single
# `classify` step (no induction, no --critics) against an already-extracted
# categories.json, for comparison against run_experiment.sh's induce ->
# critics loop.
#
# Purpose: classify the HuggingFace Hub test split directly with a supplied
# categories.json. Classifies the full test split by default; set TEST_LIMIT
# for a small, cheap smoke test first.
#
# TREC (Text REtrieval Conference) question classification: 5.45k train /
# 500 test questions, 6 coarse intent classes (ABBR, ENTY, DESC, HUM, LOC,
# NUM). Columns: `text` (string), `label_coarse_text` (string). Uses the
# SetFit re-upload rather than the original `CogComp/trec` repo, since the
# latter still ships a legacy loading script that `datasets>=4` refuses to
# run ("Dataset scripts are no longer supported").
# https://huggingface.co/datasets/SetFit/TREC-QC
#
# Usage:
#   experiments/trec/run_experiment_baseline.sh --categories FILE [--suffix NAME] [extra experiment.py flags...]
#
# Examples:
#   experiments/trec/run_experiment_baseline.sh \
#     --categories experiments/trec/runs/20260909-163215/categories.json
#   experiments/trec/run_experiment_baseline.sh \
#     --categories experiments/trec/runs/20260909-163215/categories.json --suffix baseline
#   experiments/trec/run_experiment_baseline.sh \
#     --categories experiments/trec/runs/20260909-163215/categories.json --suffix baseline --model gpt-4o-mini
#
# Env overrides:
#   TEST_LIMIT (unset by default -> classifies the entire 500-row test split;
#   set it to a small number for a cheap smoke test, e.g. TEST_LIMIT=40)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXPERIMENT_DIR="$SCRIPT_DIR"

CATEGORIES_FILE=""
NAME_SUFFIX=""
# Collected separately from "$@" so recognized flags can be pulled out from
# anywhere in the argument list while everything else still passes through
# to experiment.py, in order.
PASSTHROUGH_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --categories)
      CATEGORIES_FILE="$2"
      shift 2
      ;;
    --categories=*)
      CATEGORIES_FILE="${1#*=}"
      shift
      ;;
    --suffix)
      NAME_SUFFIX="$2"
      shift 2
      ;;
    --suffix=*)
      NAME_SUFFIX="${1#*=}"
      shift
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ -z "$CATEGORIES_FILE" ]; then
  echo "Usage: $(basename "$0") --categories FILE [--suffix NAME] [extra experiment.py flags...]" >&2
  echo "--categories is required: path to a previously extracted categories.json" >&2
  exit 1
fi

# Named the run so its directory is easy to tell apart at a glance, e.g.
# --suffix baseline yields runs/20260910-153000-baseline.
RUN_DIR="$EXPERIMENT_DIR/runs/$(date +%Y%m%d-%H%M%S)${NAME_SUFFIX:+-$NAME_SUFFIX}"

TEST_LIMIT="${TEST_LIMIT:-}"

cd "$REPO_ROOT"

if ! .venv/bin/python -c "import datasets" >/dev/null 2>&1; then
  echo "The 'datasets' package is required for --hf-dataset. Install it with:" >&2
  echo "  .venv/bin/pip install -e '.[hf]'" >&2
  exit 1
fi

echo "Run directory: $RUN_DIR"
echo "Categories:    $CATEGORIES_FILE"

# Built as one array (never empty, since it always carries the base flags
# below) rather than a separately-conditional array: macOS's default
# /bin/bash is 3.2, which raises "unbound variable" under `set -u` when
# expanding an *empty* array via "${ARR[@]}" (fixed upstream in bash 4.4,
# but this repo can't assume a newer bash is installed/first-in-PATH). Same
# reason PASSTHROUGH_ARGS is only expanded below when non-empty.
ARGS=(
  --hf-dataset SetFit/TREC-QC
  --text-column text
  --label-column label_coarse_text
  --categories "$CATEGORIES_FILE"
  --model @sciencedirect-global-openai/gpt-5.4-mini-2026-03-17
  --run-dir "$RUN_DIR"
)
if [ -n "$TEST_LIMIT" ]; then
  ARGS+=(--test-limit "$TEST_LIMIT")
fi

if [ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]; then
  .venv/bin/python experiment.py classify "${ARGS[@]}" "${PASSTHROUGH_ARGS[@]}"
else
  .venv/bin/python experiment.py classify "${ARGS[@]}"
fi

echo
echo "Done. Artifacts written to: $RUN_DIR"
echo "  cat $RUN_DIR/run_config.json"
