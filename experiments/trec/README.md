# Experiment: TREC

Exercises the experiment runner's `induce` -> `classify` loop end-to-end
against a real HuggingFace Hub dataset: [`SetFit/TREC-QC`][trec-qc], a
6-class question/intent classification corpus (ABBR, ENTY, DESC, HUM, LOC,
NUM; 5.45k train / 500 test rows, columns `text`/`label_coarse_text`).

This is the SetFit re-upload of the original TREC (Text REtrieval Conference)
question classification dataset — the original [`CogComp/trec`][cogcomp-trec]
repo still ships a legacy `datasets` loading script that `datasets>=4`
refuses to execute ("Dataset scripts are no longer supported"), so it isn't
usable with this repo's `hf` extra as-is.

[trec-qc]: https://huggingface.co/datasets/SetFit/TREC-QC
[cogcomp-trec]: https://huggingface.co/datasets/CogComp/trec

## Prerequisites

From the repo root:

```bash
.venv/bin/pip install -e '.[hf]'   # installs datasets>=4
cp .env.example .env               # if not already done, then fill in credentials
```

## Run

```bash
experiments/trec/run_experiment.sh
```

This induces `question_type` categories from a small seeded sample of the
train split, then classifies the **entire** 500-row test split by default.
Any extra flags are forwarded to `experiment.py run`, e.g.:

```bash
experiments/trec/run_experiment.sh --model gpt-4o-mini
experiments/trec/run_experiment.sh --critics --sampling-runs 3
```

Override scale via env vars: `SEED`, `EXAMPLES_PER_LABEL`, `TEST_LIMIT`. Set
`TEST_LIMIT` (e.g. `TEST_LIMIT=40 experiments/trec/run_experiment.sh`) for
a cheap smoke test of the loop instead of a full, costlier run — especially
relevant with `--critics`, which multiplies LLM call volume per row.

## Output

Each run writes a fresh, timestamped directory under `runs/`
(`runs/<YYYYMMDD-HHMMSS>/`), so repeated runs never collide or overwrite each
other:

- `categories.json` — the induced `question_type` category (labels + descriptions)
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
