#!/usr/bin/env bash
# Experiment: PubMed RCT sentence-role classification, via the experiment
# runner's induce -> classify loop, against the HuggingFace Hub dataset
# directly.
#
# Purpose: exercise the induce/classify/run train-test loop end-to-end on a
# real dataset (not a synthetic fixture). Classifies the full test split by
# default; set TEST_LIMIT for a small, cheap smoke test first -- this
# dataset's test split is ~29.6k rows, far larger than ag-news/TREC's.
#
# PubMed 20k RCT: 176.6k train / 29.6k test sentences drawn from randomized
# controlled trial abstracts, 5 structural-role classes (background,
# objective, methods, results, conclusions). Columns: `text` (string),
# `label` (string). Also has a `validation` split (unused here).
# https://huggingface.co/datasets/armanc/pubmed-rct20k
#
# Usage:
#   experiments/pubmed-rct/run_experiment.sh [extra experiment.py flags...]
#
# Examples:
#   experiments/pubmed-rct/run_experiment.sh
#   experiments/pubmed-rct/run_experiment.sh --model gpt-4o-mini
#   experiments/pubmed-rct/run_experiment.sh --critics --sampling-runs 3
#
# Env overrides (all optional):
#   SEED (default 42), EXAMPLES_PER_LABEL (default 20)
#   TEST_LIMIT (unset by default -> classifies the entire ~29.6k-row test
#   split; set it to a small number for a cheap smoke test, e.g.
#   TEST_LIMIT=40 -- strongly recommended here given the test split's size)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXPERIMENT_DIR="$SCRIPT_DIR"
RUN_DIR="$EXPERIMENT_DIR/runs/$(date +%Y%m%d-%H%M%S)"

SEED="${SEED:-42}"
EXAMPLES_PER_LABEL="${EXAMPLES_PER_LABEL:-20}"
TEST_LIMIT="${TEST_LIMIT:-}"

cd "$REPO_ROOT"

if ! .venv/bin/python -c "import datasets" >/dev/null 2>&1; then
  echo "The 'datasets' package is required for --hf-dataset. Install it with:" >&2
  echo "  .venv/bin/pip install -e '.[hf]'" >&2
  exit 1
fi

echo "Run directory: $RUN_DIR"

# Built as one array (never empty, since it always carries the base flags
# below) rather than a separately-conditional array: macOS's default
# /bin/bash is 3.2, which raises "unbound variable" under `set -u` when
# expanding an *empty* array via "${ARR[@]}" (fixed upstream in bash 4.4,
# but this repo can't assume a newer bash is installed/first-in-PATH).
ARGS=(
  --hf-dataset armanc/pubmed-rct20k
  --text-column text
  --label-column label
  --category-name sentence_role
  --model @sciencedirect-global-openai/gpt-5.4-mini-2026-03-17
  --induction-model @sciencedirect-global-openai/gpt-5.4-mini-2026-03-17
  --critics --critic-model @sciencedirect-global-openai/gpt-5.4-2026-03-05 --reconciler-model @sciencedirect-global-openai/gpt-5.4-2026-03-05
  --run-dir "$RUN_DIR"
  --seed "$SEED"
  --examples-per-label "$EXAMPLES_PER_LABEL"
)
if [ -n "$TEST_LIMIT" ]; then
  ARGS+=(--test-limit "$TEST_LIMIT")
fi

.venv/bin/python experiment.py run "${ARGS[@]}" "$@"

echo
echo "Done. Artifacts written to: $RUN_DIR"
echo "  cat $RUN_DIR/run_config.json"
echo "  cat $RUN_DIR/categories.json"
