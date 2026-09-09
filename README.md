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
| `--model` | LiteLLM model id (default: `azure/gpt-5-chat`). |
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
| `--cerebus` | Route every LLM call through the Cerebus/Portkey gateway instead of a direct provider (see below). |

## Cerebus / Portkey gateway

Both entry points (`classify.py` and `experiment.py`) accept `--cerebus` to route every
LLM call through Cerebus — an internal gateway (built on [Portkey]) that fronts Azure
OpenAI and direct-provider (OpenAI, Gemini, ...) models behind one OpenAI-compatible
endpoint. `--model`/`--critic-model`/`--reconciler-model`/`--induction-model` still name
the underlying model or workspace slug; `--cerebus` only changes how the call is
authenticated and where it's sent.

[Portkey]: https://portkey.ai/

Configure it via `.env` (see `.env.example`):

```bash
CEREBUS_MODE=azure                  # or "direct" — see .env.example for the difference
CEREBUS_GATEWAY_AZURE_URL=...       # required when CEREBUS_MODE=azure
CEREBUS_GATEWAY_DIRECT_URL=...      # required when CEREBUS_MODE=direct
CEREBUS_CONFIG_ID=...               # required when CEREBUS_MODE=azure
```

The API key resolves from `CEREBUS_API_KEY` if set, otherwise from AWS Secrets Manager
(requires `pip install -e '.[cerebus]'` for the `boto3` dependency, and an active AWS SSO
session) — a misconfiguration in either path fails with a clear, actionable error rather
than a raw traceback. `--api-base` still overrides the gateway URL if given explicitly.

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
`classify`/`run` (`--model`, `--critics`, `--sampling-runs`, ...); `induce`/
`run` additionally take `--seed`, `--examples-per-label`,
`--max-example-chars`, `--max-prompt-chars`, and `--induction-model`.
`--cerebus` is available on all three subcommands (it applies to the
induction call too).

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
`run` send **test** text to `--model` (and, under `--critics`, to
`--critic-model`/`--reconciler-model` too). Point any of these at a different
provider than your default only if you intend that split's text to go there.
As with `classify.py`, output CSV cells are not sanitized against
spreadsheet-formula injection if opened in a spreadsheet application — treat
`test_classified.csv` as you would any other untrusted CSV before opening it
that way.

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
