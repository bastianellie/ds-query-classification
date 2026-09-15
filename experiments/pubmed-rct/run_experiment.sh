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
#   SEED (default 42)
#   EXAMPLES_PER_LABEL (default 20) -- per-label sampling cap. Mutually
#   exclusive with INDUCTION_EXAMPLES (and with a trailing --induction-examples
#   flag); set it to an empty string (EXAMPLES_PER_LABEL=) to suppress the
#   default when passing --induction-examples as a trailing flag instead.
#   INDUCTION_EXAMPLES (unset by default) -- total example budget across all
#   labels instead of a per-label cap, e.g. INDUCTION_EXAMPLES=60. Overrides
#   EXAMPLES_PER_LABEL when set.
#   TEST_LIMIT (unset by default -> classifies the entire ~29.6k-row test
#   split; set it to a small number for a cheap smoke test, e.g.
#   TEST_LIMIT=40 -- strongly recommended here given the test split's size)
#   DRY_RUN=1 -- print the assembled experiment.py argv instead of running it
#   (no provider call, no `datasets` package required); useful for checking
#   which sizing mode would actually be sent.
#   CRITICS (default 1/on) -- set CRITICS=0 to disable --critics/--critic-model/
#   --reconciler-model (required before enabling BATCH, since --batch and
#   --critics are mutually exclusive at the experiment.py level).
#   BATCH (unset by default) -- 'dynamic' or a positive integer to enable
#   --batch; requires CRITICS=0.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXPERIMENT_DIR="$SCRIPT_DIR"
RUN_DIR="$EXPERIMENT_DIR/runs/$(date +%Y%m%d-%H%M%S)"

SEED="${SEED:-42}"
# No-colon form: an EXPLICITLY empty value (EXAMPLES_PER_LABEL=) must stay
# empty, not collapse back to 20 -- that's how a caller suppresses the
# default when switching to --induction-examples/INDUCTION_EXAMPLES instead.
EXAMPLES_PER_LABEL="${EXAMPLES_PER_LABEL-20}"
INDUCTION_EXAMPLES="${INDUCTION_EXAMPLES:-}"
TEST_LIMIT="${TEST_LIMIT:-}"
DRY_RUN="${DRY_RUN:-}"
CRITICS="${CRITICS:-1}"
BATCH="${BATCH:-}"

cd "$REPO_ROOT"

if [ -z "$DRY_RUN" ] && ! .venv/bin/python -c "import datasets" >/dev/null 2>&1; then
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

# --batch and --critics are mutually exclusive (experiment.py's FR-3.3) --
# setting BATCH requires CRITICS=0 explicitly, rather than silently dropping
# --critics: these scripts produce the accuracy numbers compared across
# configurations, and silently switching the experimental condition would be
# exactly the class of error the comparability caveat exists to prevent.
if [ -n "$BATCH" ] && [ "$CRITICS" != "0" ]; then
  echo "BATCH and CRITICS are mutually exclusive: set CRITICS=0 to enable BATCH." >&2
  exit 1
fi

# A literal --critics passed through in "$@" would silently re-enable the
# critics trio CRITICS=0 just disabled, the same passthrough-bypass risk
# induction_examples_in_passthrough (below) guards against for
# --induction-examples.
critics_in_passthrough=0
for arg in "$@"; do
  if [ "$arg" = "--critics" ]; then
    critics_in_passthrough=1
    break
  fi
done
if [ -n "$BATCH" ] && [ "$critics_in_passthrough" -eq 1 ]; then
  echo "BATCH and --critics (passed through) are mutually exclusive." >&2
  exit 1
fi

ARGS=(
  --hf-dataset armanc/pubmed-rct20k
  --text-column text
  --label-column label
  --category-name sentence_role
  --model @sciencedirect-global-openai/gpt-5.4-mini-2026-03-17
  --induction-model @sciencedirect-global-openai/gpt-5.4-mini-2026-03-17
)
if [ "$CRITICS" != "0" ]; then
  ARGS+=(--critics --critic-model "@sciencedirect-global-openai/gpt-5.4-2026-03-05" --reconciler-model "@sciencedirect-global-openai/gpt-5.4-2026-03-05")
fi
ARGS+=(
  --run-dir "$RUN_DIR"
  --seed "$SEED"
)
# --examples-per-label/--induction-examples are mutually exclusive
# (experiment.py's FR-2.1) -- only add the default per-label cap when no
# budget mode is already in effect, either via INDUCTION_EXAMPLES or a
# trailing --induction-examples flag passed directly in "$@".
induction_examples_in_passthrough=0
for arg in "$@"; do
  if [ "$arg" = "--induction-examples" ]; then
    induction_examples_in_passthrough=1
    break
  fi
done

if [ -n "$INDUCTION_EXAMPLES" ]; then
  ARGS+=(--induction-examples "$INDUCTION_EXAMPLES")
elif [ "$induction_examples_in_passthrough" -eq 0 ] && [ -n "$EXAMPLES_PER_LABEL" ]; then
  ARGS+=(--examples-per-label "$EXAMPLES_PER_LABEL")
fi

if [ -n "$TEST_LIMIT" ]; then
  ARGS+=(--test-limit "$TEST_LIMIT")
fi

if [ -n "$BATCH" ]; then
  ARGS+=(--batch "$BATCH")
fi

if [ -n "$DRY_RUN" ]; then
  echo "Would run: experiment.py run $(printf '%q ' "${ARGS[@]}" "$@")"
  exit 0
fi

.venv/bin/python experiment.py run "${ARGS[@]}" "$@"

echo
echo "Done. Artifacts written to: $RUN_DIR"
echo "  cat $RUN_DIR/run_config.json"
echo "  cat $RUN_DIR/categories.json"
