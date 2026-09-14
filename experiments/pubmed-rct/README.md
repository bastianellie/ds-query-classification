# Experiment: PubMed RCT

Exercises the experiment runner's `induce` -> `classify` loop end-to-end
against a real HuggingFace Hub dataset: [`armanc/pubmed-rct20k`][pubmed-rct],
a 5-class sentence-role classification corpus drawn from randomized
controlled trial abstracts (background, objective, methods, results,
conclusions; 176.6k train / 29.6k test rows, columns `text`/`label`). This is
domain-specific, scientific-register text — a useful contrast to
ag-news/TREC's general-domain text.

[pubmed-rct]: https://huggingface.co/datasets/armanc/pubmed-rct20k

## Prerequisites

From the repo root:

```bash
.venv/bin/pip install -e '.[hf]'   # installs datasets>=4
cp .env.example .env               # if not already done, then fill in credentials
```

## Run

```bash
experiments/pubmed-rct/run_experiment.sh
```

This induces `sentence_role` categories from a small seeded sample of the
train split, then classifies the **entire** ~29.6k-row test split by
default — that's ~60x larger than ag-news's test split, so set `TEST_LIMIT`
(see below) unless you actually want the full-cost run.

Any extra flags are forwarded to `experiment.py run`, e.g.:

```bash
experiments/pubmed-rct/run_experiment.sh --model gpt-4o-mini
experiments/pubmed-rct/run_experiment.sh --critics --sampling-runs 3
```

Override scale via env vars: `SEED`, `EXAMPLES_PER_LABEL`, `TEST_LIMIT`. Set
`TEST_LIMIT` (e.g. `TEST_LIMIT=40 experiments/pubmed-rct/run_experiment.sh`)
for a cheap smoke test of the loop instead of a full, costlier run —
especially relevant with `--critics`, which multiplies LLM call volume per
row, and especially important here given the test split's size.

By default induction samples up to `EXAMPLES_PER_LABEL` (20) examples per
label. For a total example *budget* across all labels instead (mutually
exclusive with `EXAMPLES_PER_LABEL` — set `EXAMPLES_PER_LABEL=` to an empty
string, or pass `--induction-examples` as a trailing flag, which
automatically suppresses the default cap), use `INDUCTION_EXAMPLES`, e.g.
`INDUCTION_EXAMPLES=60 experiments/pubmed-rct/run_experiment.sh`. Set
`DRY_RUN=1` to print the assembled `experiment.py` invocation instead of
running it — no provider call, no `datasets` package required — to check
which sizing mode would actually be sent.

## Output

Each run writes a fresh, timestamped directory under `runs/`
(`runs/<YYYYMMDD-HHMMSS>/`), so repeated runs never collide or overwrite each
other:

- `categories.json` — the induced `sentence_role` category (labels + descriptions)
- `train.csv` / `test.csv` — the materialized, filtered splits actually used
- `test_classified.csv` — test set with predictions and gold labels
  (`label_gold`) side by side
- `induction_prompt.txt`, `sampled_examples.json` — induction replay artifacts
- `run_config.json` — full run provenance (dataset revision, models, flags,
  label values, excluded/unclassified row positions, package versions)

`runs/` is gitignored — only this script and README are checked in.

## Analysis

`analyze_run.ipynb` compares `test.csv` against `test_classified.csv` for a
given run (defaults to the most recent one under `runs/`) and reports
accuracy, per-class precision/recall/F1, and a confusion matrix. It needs a
few extra packages not in the base install:

```bash
.venv/bin/pip install scikit-learn matplotlib seaborn ipykernel
```
