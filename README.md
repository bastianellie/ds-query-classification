# ds-query-classification

Classify arbitrary text queries against arbitrary, user-defined categories using
an LLM. Point it at a CSV, tell it which column holds the text and which
categories to use, and it writes one column of labels per category.

It is fully domain-agnostic: the output schema and the prompt are built at
runtime from your categories file and an optional task description, so the same
machinery can classify game-theory reasoning traces, support tickets, survey
responses, or anything else.

## How it works

1. **Categories** (`resources/categories/*.json`) define the fields to classify
   and the candidate labels for each.
2. A Pydantic output model is **built dynamically** from those categories.
3. The **system prompt** is rendered from a template
   (`resources/prompts/system_prompt.txt`) with two slots:
   - `{task_description}` — domain framing (what is being classified and how to
     read the labels), supplied via `--task-description`.
   - `{schema_description}` — auto-generated from the categories.
4. Each row's text is sent to the model via [LiteLLM] using structured output
   (with a graceful fallback to JSON mode), with bounded retries.
5. Results are written back to the CSV **incrementally**, so runs can be resumed
   (`--restore`) or truncated for debugging (`--limit`).

[LiteLLM]: https://docs.litellm.ai/

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # add ".[dev]" for the tests
cp .env.example .env        # then fill in your provider credentials
```

## Usage

```bash
python classify.py \
  --input data/queries.csv \
  --column text \
  --output output/queries_classified.csv \
  --categories resources/categories/example_support_tickets.json \
  --model azure/gpt-5-chat
```

Equivalent to `python -m query_classification ...`.

### Critics mode: self-consistency sampling + Critic/Reconciler debate

`--critics` samples each row multiple times, votes on the result per category,
and escalates categories without consensus to a Critic (devil's advocate) and,
if it raises a real challenge, a Reconciler — adding a full audit trail to the
output CSV:

```bash
python classify.py \
  --input data/queries.csv \
  --column text \
  --output output/queries_classified.csv \
  --categories resources/categories/example_support_tickets.json \
  --critics \
  --sampling-runs 5 \
  --consensus-threshold 4
```

This roughly multiplies LLM call volume by `--sampling-runs` per row (plus one
Critic call, and one Reconciler call, per category that doesn't reach
consensus) — size `--workers` down accordingly for large inputs. `--critic-
model`/`--reconciler-model` can each point at a different LiteLLM model than
`--model`, which means the same row text is routed to whichever provider each
model resolves to; make sure that's intentional before pointing them at a
different provider than your main classification calls.

### Multi-classifier mode: N classifier instances, merged by plurality vote

`--models`/`--n-classifiers` is a standalone alternative to `--critics`
(mutually exclusive with it when 2+ distinct models or `--n-classifiers > 1`
are in play): N classifier instances each classify a row once, independently —
no sampling, no Critic/Reconciler debate — and the results are merged per
category by plurality vote on each instance's top label. Ties are broken in
favor of whichever tied label the earliest-listed/constructed instance picked.
Adds a per-instance/vote-tally/failure audit trail to the output CSV.

Two ways to get there:

```bash
# 2+ distinct models, one classifier each (duplicates allowed, e.g. to give
# one model extra weight in the vote):
python classify.py \
  --input data/queries.csv \
  --column text \
  --output output/queries_classified.csv \
  --categories resources/categories/example_support_tickets.json \
  --models azure/gpt-5-chat gpt-4o-mini gemini/gemini-2.5-pro

# One model, replicated into N independent classifier instances (a plurality
# vote of N samples from the same model, without --critics' sampling/debate
# machinery):
python classify.py \
  --input data/queries.csv \
  --column text \
  --output output/queries_classified.csv \
  --categories resources/categories/example_support_tickets.json \
  --model azure/gpt-5-chat \
  --n-classifiers 5
```

A single-value `--models` is equivalent to `--model` (silently overrides it if
both are given). `--n-classifiers` only matters when replicating one resolved
model — it has no effect when `--models` is given 2+ values, where the
classifier count is simply the number of values given. In the audit trail,
each classifier instance is keyed by its bare model id, unless that id repeats
in the resolved list, in which case every occurrence gets an occurrence-
suffixed key (e.g. `azure/gpt-5-chat#1`, `azure/gpt-5-chat#2`) so repeated
instances stay individually addressable.

This multiplies per-row LLM call volume by the number of classifier instances
(before any internal retries/JSON-mode fallback), and concurrent in-flight
requests are roughly `--workers × <number of instances>` — size `--workers`
down accordingly for large inputs, same as `--critics`. Every instance gets
the identical system prompt/schema/`--allow-new-labels` setting and its own
provider default temperature (not `--critics`' `--sampling-temperature`). As
with `--critic-model`/`--reconciler-model`, the same row text is sent to every
provider each listed model resolves to — this is a bigger data-egress surface
than a single `--model`, since it's potentially N providers instead of one;
make sure that's intentional for sensitive input before pointing `--models` at
providers outside your usual one. 2+ *distinct* resolved models cannot be
combined with `CEREBUS_MODE=azure` (`CEREBUS_CONFIG_ID` is a single,
workspace/model-specific value that can't apply to several distinct models) —
replicating a single model via `--n-classifiers` is unaffected; it works
normally with Cerebus direct mode or with direct providers.

### Reproducing the original decision-classification setup

```bash
python classify.py \
  --input data/reasoning_traces.csv \
  --column reasoning \
  --output output/classified.csv \
  --categories resources/categories/example_categories.json \
  --task-description resources/prompts/example_task_description.txt
```

### Options

| Flag | Description |
| --- | --- |
| `-i, --input` | Input CSV file (required). |
| `-c, --column` | Name of the text column to classify (required). |
| `-o, --output` | Output CSV file. **Omitting it overwrites the input in place.** |
| `--categories` | Categories JSON file (default: bundled example). |
| `--model` | LiteLLM model id (default: `azure/gpt-5-chat`). Accepts exactly one value — use `--models` for multiple, or `--n-classifiers` to replicate this one model into several independent classifier instances. |
| `--api-base` | Override the API endpoint/base URL (default: resolved from the first of `LITELLM_API_BASE`, `AZURE_API_BASE`, `AZURE_OPENAI_ENDPOINT`, `OPENAI_BASE_URL`, `OPENAI_API_BASE` that is set). |
| `--system-prompt` | Override the system prompt template. |
| `--task-description` | Text file describing the task/domain (`{task_description}` slot). |
| `--extra-prompt` | Text file with extra ad-hoc instructions, appended to the prompt. |
| `--limit N` | Only classify the first N rows. |
| `--restore` | Skip rows that already have labels; resume a previous run. |
| `--retries N` | Retry attempts per row on LLM error (default: 3). |
| `--workers N` | Concurrent worker threads for LLM calls (default: 8). |
| `--critics` | Enable self-consistency sampling + Critic/Reconciler debate mode (see below). |
| `--sampling-runs N` | Independent samples per row under `--critics` (default: 5). |
| `--sampling-temperature T` | Sampling temperature for `--critics` (default: 0.7). |
| `--consensus-threshold N` | Vote count (out of `--sampling-runs`) needed to bypass debate under `--critics` (default: 4). |
| `--allow-new-labels` | Under `--critics`, allow suggesting `"none - <new label>"` instead of only predefined labels/`"none"`. |
| `--critic-model` | LiteLLM model id for the `--critics` Critic role (default: `--model`). |
| `--reconciler-model` | LiteLLM model id for the `--critics` Reconciler role (default: `--model`). |
| `--models` | 2+ model ids (duplicates allowed) for multi-classifier voting mode (see above); a single value is equivalent to `--model`. Mutually exclusive with `--critics` when given 2+ values. |
| `--n-classifiers` | Number of classifier instances to construct (default: 1); replicates a single resolved model. No effect when `--models` has 2+ values. Mutually exclusive with `--critics` when > 1. |
| `--cerebus` | Route every LLM call through the Cerebus/Portkey gateway instead of a direct provider (see below). |

## Cerebus / Portkey gateway

Both entry points (`classify.py` and `experiment.py`) can route every LLM call through
Cerebus — an internal gateway (built on [Portkey]) that fronts Azure OpenAI and
direct-provider (OpenAI, Gemini, ...) models behind one OpenAI-compatible endpoint —
instead of a direct provider. `--model`/`--critic-model`/`--reconciler-model`/
`--induction-model` still name the underlying model or workspace slug; Cerebus only
changes how the call is authenticated and where it's sent.

[Portkey]: https://portkey.ai/

**Minimal setup** (matches the convention used elsewhere internally) — in `.env`:

```bash
DEFAULT_LLM_PROVIDER=cerebus
CEREBUS_API_KEY=...   # optional — falls back to AWS Secrets Manager if unset
```

Nothing else is required: the gateway mode defaults to `"direct"` (no per-model
configuration needed) and the gateway URL defaults to this org's shared nonprod Cerebus
endpoint. `--cerebus` on either entry point is an equivalent, per-invocation alternative
to setting `DEFAULT_LLM_PROVIDER` — either one turns gateway routing on.

The API key resolves from `CEREBUS_API_KEY` if set, otherwise from AWS Secrets Manager
(requires `pip install -e '.[cerebus]'` for the `boto3` dependency, and an active AWS SSO
session: `aws sso login --profile kd-nonprod`) — a misconfiguration in either path fails
with a clear, actionable error rather than a raw traceback.

**Advanced overrides** (see `.env.example`), only needed to deviate from the defaults:

```bash
CEREBUS_MODE=azure                  # opt into Azure-config-mode routing (default: "direct")
CEREBUS_GATEWAY_AZURE_URL=...       # override the default Azure gateway URL
CEREBUS_GATEWAY_DIRECT_URL=...      # override the default direct gateway URL
CEREBUS_CONFIG_ID=...               # required when CEREBUS_MODE=azure (no default — workspace/model-specific)
```

`--api-base` still overrides the resolved gateway URL if given explicitly (must be
`https://` when Cerebus is active).

## Experiment runner

`experiment.py` (a separate entry point from `classify.py`) runs a full
train-to-classified-test-set experiment against a local CSV pair or a
HuggingFace Hub dataset, with three subcommands:

```bash
# Induce categories.json from a labeled train split
python experiment.py induce \
  --train-file data/train.csv --text-column text --label-column label \
  --category-name sentiment --run-dir runs/1

# Classify a test split with an existing categories.json
python experiment.py classify \
  --test-file data/test.csv --text-column text --label-column label \
  --categories runs/1/categories.json --run-dir runs/2

# Both, in one invocation
python experiment.py run \
  --train-file data/train.csv --test-file data/test.csv \
  --text-column text --label-column label --category-name sentiment \
  --run-dir runs/3
```

Equivalent to `python -m query_classification.experiment ...`. All of
`classify.py`'s classifier-configuration flags are available on
`classify`/`run` (`--model`, `--critics`, `--models`, `--n-classifiers`,
`--sampling-runs`, ...) — not `induce`, which has no classification role (only
`--model` lives in the group shared with `induce`, for its induction-model
fallback); `induce`/`run` additionally take `--seed`, `--examples-per-label`
(default 20, per-label sampling cap), `--induction-examples` (a total example
budget across all labels instead, split as evenly as possible — mutually
exclusive with `--examples-per-label`, and requires at least one example per
distinct label; makes prompt size independent of the label count),
`--max-example-chars`, `--max-prompt-chars`, and `--induction-model`.
`--cerebus` is available on all three subcommands (it applies to the
induction call too). One exception:
`classify.py`'s `--limit` is called `--test-limit` here, to make clear it only
bounds the test split being classified (not the train split used for
induction); omitting it classifies the entire test split. `--restore` has no
equivalent here at all — every `experiment.py` invocation classifies its test
split fresh (this only matters for `classify.py`'s own `--restore`, not
`experiment.py`).

For a HuggingFace dataset instead of local files, install the optional
`hf` extra (`pip install '.[hf]'`, requires `datasets>=4`) and pass
`--hf-dataset owner/name` (plus `--hf-config`/`--hf-revision`/
`--hf-train-split`/`--hf-test-split` as needed) in place of
`--train-file`/`--test-file`.

**Every run writes a self-contained run directory** (`--run-dir`, required):
`categories.json` (induced, or a verbatim copy of a supplied one),
`train.csv`/`test.csv` (materialized, filtered, needed columns only),
`test_classified.csv`, `induction_prompt.txt` and `sampled_examples.json`
(when inducing), and `run_config.json` — a full reproducibility/audit record
(dataset provenance, resolved models, flags, label values, excluded/
unclassified row positions, package versions), written last with a terminal
`status`. The runner refuses to write into a non-empty `--run-dir` unless
`--overwrite` is passed, and `--overwrite` only ever replaces these known
artifact filenames — never the directory itself or any other file in it.

**Deliberate divergence from `classify.py`:** in plain (non-`--critics`) mode,
`classify.py` always allows label invention (`cli.py`'s
`allow_new_labels = args.allow_new_labels if args.critics else True`). The
experiment runner instead honors `--allow-new-labels` (default **off**) in
**both** classifier modes, so an experiment's plain-mode output stays within
the induced/supplied closed vocabulary — comparing induced-vs-hand-written
categories, or plain-vs-`--critics` mode, requires that gold labels can
actually match a prediction.

**Data egress note:** the `induce` subcommand sends sampled **train** text to
whichever model `--induction-model` (or `--model`) resolves to; `classify`/
`run` send **test** text to the resolved classifier model(s) (and, under
`--critics`, to `--critic-model`/`--reconciler-model` too; under multi-
classifier mode with 2+ *distinct* resolved models, to every one of them — a
bigger egress surface than any single-model mode, since it's N providers
instead of one; replicating one model via `--n-classifiers` stays a single
provider). Point any of these at a different provider than your default only
if you intend that split's text to go there. As with `classify.py`, output CSV
cells (including multi-classifier mode's `_votes`/`_by_model`/`_model_errors`
audit columns) are not sanitized against spreadsheet-formula injection if
opened in a spreadsheet application — treat `test_classified.csv` as you would
any other untrusted CSV before opening it that way.

**Analysis notebook compatibility:** `experiments/ag-news/analyze_run.ipynb`
reads `--critics`-specific columns (`_initial`/`_votes`/`_challenged`/
`_reconciled`) and cannot analyze a multi-classifier run's output as-is.

## Defining categories

```json
{
  "categories": [
    {
      "name": "sentiment",
      "description": "The overall sentiment expressed in the text.",
      "labels": [
        {"value": "positive", "description": "Satisfied, appreciative, or friendly."},
        {"value": "negative", "description": "Frustrated, angry, or disappointed."}
      ]
    }
  ]
}
```

Each category becomes one output column holding a JSON list of 1–3 labels,
ordered by strength of evidence. If no label fits, the model returns `"none"`
or `"none - <suggested new label>"`.

## Project layout

```
src/query_classification/
  categories.py   # load + validate the categories JSON
  schema.py       # build the dynamic Pydantic output model + schema text
  prompts.py      # render the system prompt
  classifier.py   # single-text LLM classification (Classifier)
  pipeline.py     # CSV batch loop (restore / limit / incremental save)
  debate.py       # --critics: sampling, consensus voting, Critic/Reconciler debate
  multi_model.py  # --models/--n-classifiers: N-classifier plurality-vote classification/audit trail
  cli.py          # argparse entry point (classify.py)
  dataset_io.py   # local/HF dataset loading, label normalization, row filtering
  induction.py    # category induction: sampling, one LLM call, reconciliation
  experiment.py   # argparse entry point (experiment.py) — induce/classify/run
resources/
  categories/     # example category definitions
  prompts/        # system/critic/reconciler/induction prompt templates + example task descriptions
tests/            # unit tests for the offline building blocks
```

Use it as a library, too:

```python
from query_classification import (
    load_categories, build_classification_model,
    build_system_prompt, Classifier, classify_csv,
)
```
