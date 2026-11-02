# ds-query-classification

Classify arbitrary text queries against arbitrary, user-defined categories using
an LLM. Point it at a CSV (or Excel) file, tell it which column holds the text
and which categories to use, and it writes one column of labels per category.

It is fully domain-agnostic: the output schema and the prompt are built at
runtime from your categories file and an optional task description, so the same
machinery can classify game-theory reasoning traces, support tickets, survey
responses, or anything else.

## Contents

- [Which tool do I want?](#which-tool-do-i-want)
- [Quick start](#quick-start)
- [Install and configure](#install-and-configure)
- [Defining categories](#defining-categories)
- [`classify.py`: classify a file](#classifypy-classify-a-file)
  - [Flag reference](#flag-reference)
  - [Choosing a classification mode](#choosing-a-classification-mode)
  - [Reading the output](#reading-the-output)
  - [Resuming, partial failures and large inputs](#resuming-partial-failures-and-large-inputs)
  - [Excel input and output](#excel-input-and-output)
  - [Customising the prompt](#customising-the-prompt)
  - [Reproducing the original decision-classification setup](#reproducing-the-original-decision-classification-setup)
- [`extract_categories.py`: write a categories file from a prompt](#extract_categoriespy-write-a-categories-file-from-a-prompt)
- [`experiment.py`: induce categories and evaluate against gold labels](#experimentpy-induce-categories-and-evaluate-against-gold-labels)
- [Cost and timing](#cost-and-timing)
- [Cerebus / Portkey gateway](#cerebus--portkey-gateway)
- [Troubleshooting](#troubleshooting)
- [Project layout and library use](#project-layout-and-library-use)

## Which tool do I want?

There are three separate entry points. They do not call each other.

| I want to... | Use | You provide | You get |
| --- | --- | --- | --- |
| Label every row of a CSV/Excel file with categories I already have | [`classify.py`](#classifypy-classify-a-file) | input file, text column, `categories.json` | the same table plus one label column per category |
| Turn a plain-English description of a taxonomy into a `categories.json` | [`extract_categories.py`](#extract_categoriespy-write-a-categories-file-from-a-prompt) | a text file of instructions, a category name | a `categories.json` with LLM-written label descriptions |
| Learn a `categories.json` from a *labeled* dataset and/or measure accuracy against gold labels | [`experiment.py`](#experimentpy-induce-categories-and-evaluate-against-gold-labels) | a train and/or test split with a gold label column | a self-contained run directory (categories, classified test set, config, cost) |

A typical workflow is: write or generate a `categories.json`, try it on a few rows with
`classify.py --limit`, then run it on the full file. Use `experiment.py` when you have gold
labels and want numbers, not just output.

## How it works

1. **Categories** (`resources/categories/*.json`) define the fields to classify
   and the candidate labels for each.
2. A Pydantic output model is **built dynamically** from those categories.
3. The **system prompt** is rendered from a template
   (`resources/prompts/system_prompt.txt`) with three slots (see
   [Customising the prompt](#customising-the-prompt)): your task description, an auto-generated
   description of the categories, and a policy for labels that fit nothing.
4. Each row's text is sent to the model via [LiteLLM] using structured output
   (with a graceful fallback to JSON mode), with bounded retries.
5. Results are written back to the CSV **incrementally**, so runs can be resumed
   (`--restore`) or truncated for debugging (`--limit`).

[LiteLLM]: https://docs.litellm.ai/

## Quick start

```bash
# 1. Install (details below)
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Configure credentials: copy the template, then EDIT IT (see the warning below)
cp .env.example .env

# 3. Create the output directory. classify.py never creates it; if it is missing
#    the run stops immediately with an error, before any LLM call is made.
mkdir -p output

# 4. Classify the bundled example file (5 rows)
python classify.py \
  --input data/example_queries.csv \
  --column text \
  --categories resources/categories/example_support_tickets.json \
  --output output/example_classified.csv
```

> **Read this before step 2.** `.env.example` ships with `DEFAULT_LLM_PROVIDER=cerebus`
> *uncommented*, which routes every call through the internal Cerebus gateway. If you are using
> a direct provider (Azure OpenAI, OpenAI, Gemini, ...), delete or comment out that line and set
> your provider's credentials instead, otherwise the first call fails trying to resolve a
> Cerebus key. See [Install and configure](#install-and-configure) and
> [Cerebus / Portkey gateway](#cerebus--portkey-gateway).

The run prints a progress bar (`ok=` and `failed=` counters), then a line like
`Cost: $0.0021 | Time: 4.8s`. It writes two files:

- `output/example_classified.csv`: your input columns plus one new column per category (here
  `intent`, `sentiment`, `urgency`), e.g. `['bug_report']`. See [Reading the output](#reading-the-output).
- `output/example_classified.cost_report.json`: estimated spend and wall-clock time.

Once that works, point `--input`/`--column` at your own file. On a large file, add `--limit 20` first
to check the output and the cost before committing to the full run.

## Install and configure

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # add ".[dev]" for the tests
cp .env.example .env        # then fill in your provider credentials
```

`./install.sh` is an alternative that builds a Python 3.12 `.venv` from `requirements.txt`; it
does **not** install the package itself, so run `pip install -e .` afterwards if you want
`python -m query_classification` or library imports. `python classify.py` works without the
editable install.

Optional extras: `pip install -e '.[cerebus]'` (the `boto3` dependency for the AWS Secrets Manager
key fallback), `pip install -e '.[hf]'` (the `datasets` package, needed for `experiment.py
--hf-dataset`), `pip install -e '.[dev]'` (pytest).

Python 3.10+ is required (developed and tested on 3.12). Run the tests with `pytest`; they never
make live provider calls.

### How `.env` is read

All three entry points call `load_dotenv(override=True)` on startup. Two consequences:

- A value in `.env` **wins over** a variable you exported in your shell. To use a different value
  for one run, edit `.env`; exporting it is not enough.
- `.env` is found by walking up from the package's own location, not from your current directory,
  so it is the repo's `.env` that applies wherever you run from.

### Model ids

`--model` takes any [LiteLLM model id](https://docs.litellm.ai/docs/providers), for example
`azure/gpt-5-chat` (the default), `gpt-4o-mini`, or `gemini/gemini-2.5-pro`. Provider credentials
come from the environment variables LiteLLM expects for that provider (see `.env.example`). With
the Cerebus gateway, the model id is the gateway's workspace slug, e.g.
`@sciencedirect-global-openai/gpt-5.4-mini-2026-03-17`.

## Defining categories

A categories file is JSON with a top-level `categories` list. `classify.py` accepts any number of
categories (one output column each); `experiment.py` requires exactly one.

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

- **`name`** becomes the output column name and the model's field name. It must be unique across
  categories and must **not** equal the `--column` you are classifying (that would overwrite your
  source text, so the run is refused). Because it is used as a Python field name, avoid names that
  are not valid identifiers or that Pydantic reserves (`__root__`, anything starting with
  `model_`). `extract_categories.py` enforces this for the names it writes; hand-written files are
  your responsibility.
- **`description`** and every label's **`description`** are sent to the model on *every row*, so
  they are the main lever for quality: say what distinguishes a label from its nearest neighbour,
  not just what it is. They also cost tokens on every call, so keep them tight.
- **Labels** are matched as exact strings. The model returns between one and three per category,
  ordered by strength of evidence.
- **`none`** is reserved. If no label fits, the model may return `"none"`, or
  `"none - <suggested new label>"` when label invention is allowed (see `--allow-new-labels`). Do
  not use `none` or a `none - ...` prefix as one of your own label values: `experiment.py` rejects
  such files, and `extract_categories.py` rejects them from the model's output.

Bundled examples live in `resources/categories/`: `example_support_tickets.json` (three
categories: `intent`, `sentiment`, `urgency`; used throughout this README) and
`example_categories.json` (the default when `--categories` is omitted).

Don't want to write the JSON by hand? Use
[`extract_categories.py`](#extract_categoriespy-write-a-categories-file-from-a-prompt) (from a
description) or [`experiment.py induce`](#experimentpy-induce-categories-and-evaluate-against-gold-labels)
(from labeled data).

## `classify.py`: classify a file

```bash
python classify.py \
  --input data/queries.csv \
  --column text \
  --output output/queries_classified.csv \
  --categories resources/categories/example_support_tickets.json \
  --model azure/gpt-5-chat
```

Equivalent to `python -m query_classification ...` (needs the editable install). Only `--input`
and `--column` are required.

> **`--output`'s parent directory must already exist.** `classify.py` never creates it. It checks
> `--output` before any LLM call (and before the model client is built), so a missing parent
> directory, or an `--output` that is an existing directory, stops the run immediately with
> `Error: --output parent directory does not exist: <dir>` (or
> `Error: --output '<path>' is a directory, not a file`) and exit code 1: no traceback, no cost
> incurred. Run `mkdir -p output` first. This also applies to Excel input, which always needs an
> explicit `--output`.

> **Omitting `--output` overwrites your input CSV in place**, after printing a warning. Pass
> `--output` unless that is what you want. (It is required for Excel input.)

### Flag reference

Flags are grouped by what they control. `classify.py --help` lists everything.

#### Input and output

| Flag | Description |
| --- | --- |
| `-i, --input` | Input CSV or Excel (`.xlsx`/`.xlsm`) file. **Required.** |
| `-c, --column` | Name of the text column to classify. **Required.** |
| `-o, --output` | Output CSV path. Always a CSV, even with `--output-format xlsx`. Omitting it overwrites a CSV input in place; **required for Excel input.** The parent directory must already exist. |
| `--output-format` | `csv` (default) or `xlsx`. `xlsx` additionally writes `<output stem>.xlsx` once the run finishes (see [Excel](#excel-input-and-output)). |
| `--sheet` | Excel only: worksheet to read, by exact name (checked first) or zero-based index. Default: the first sheet. |
| `--keep-columns` | Comma-separated input columns to carry into the output. The `--column` text column is always kept. Default: all columns. Unknown names fail before any LLM call; names containing commas cannot be selected. |

#### Categories and prompt

| Flag | Description |
| --- | --- |
| `--categories` | Categories JSON file (default: the bundled `example_categories.json`; you almost always want to set this). |
| `--task-description` | Text file describing the task/domain; fills the `{task_description}` slot of the system prompt. |
| `--extra-prompt` | Text file of extra ad-hoc instructions appended to the system prompt. |
| `--system-prompt` | Replace the whole system prompt template. See [Customising the prompt](#customising-the-prompt) for the rules. |

#### Model and connection

| Flag | Description |
| --- | --- |
| `--model` | LiteLLM model id (default `azure/gpt-5-chat`). Exactly one value; use `--models` for several. |
| `--api-base` | Override the endpoint URL. Default: the first set of `LITELLM_API_BASE`, `AZURE_API_BASE`, `AZURE_OPENAI_ENDPOINT`, `OPENAI_BASE_URL`, `OPENAI_API_BASE`. Must be `https://` under Cerebus. |
| `--cerebus` | Route every call through the Cerebus/Portkey gateway. Equivalent to `DEFAULT_LLM_PROVIDER=cerebus` in `.env`; either one turns it on. See [Cerebus](#cerebus--portkey-gateway). |
| `--retries N` | Retry attempts per row on a retryable LLM error (default 3). Non-retryable errors (bad credentials, policy refusals) fail immediately. |

#### Run control

| Flag | Description |
| --- | --- |
| `--limit N` | Classify only the first N rows still to do. Counts only classifiable rows (empty-text rows are excluded first). `--limit 0` classifies nothing. Useful for trying a setup cheaply. |
| `--restore` | Skip rows that already have labels and resume a previous run. Without it, **every row's labels are discarded and redone.** See [Resuming](#resuming-partial-failures-and-large-inputs). |
| `--workers N` | Concurrent worker threads (default 8). Rows run concurrently; under `--critics`/`--models`/`--batch` the real request concurrency is higher (see [Choosing a mode](#choosing-a-classification-mode)). |
| `--save-every N` | Rewrite the output CSV every N completed rows (default 20, must be >= 1). Each save rewrites the whole file, so use a larger value (e.g. 5000) on big inputs. |

#### Classification mode

These choose *how* each row is classified. The next section compares them.

| Flag | Description |
| --- | --- |
| `--critics` | Self-consistency sampling plus a Critic/Reconciler debate on disagreement. |
| `--sampling-runs N` | Samples per row under `--critics` (default 5). |
| `--sampling-temperature T` | Temperature for those samples (default 0.7). Ignored without `--critics`. |
| `--consensus-threshold N` | Votes (out of `--sampling-runs`) needed to skip the debate (default 4). Must be between 1 and `--sampling-runs`. |
| `--critic-model`, `--reconciler-model` | Models for the Critic / Reconciler roles under `--critics` (default: `--model`). |
| `--allow-new-labels` | Let the model answer `"none - <new label>"` instead of only a known label or plain `"none"`. **Only honored under `--critics` and multi-classifier mode; in plain and `--batch` mode label invention is always on.** Off by default where honored. |
| `--models M1 M2 ...` | Multi-classifier plurality vote across 2+ models (duplicates allowed). A single value equals `--model`. |
| `--n-classifiers N` | Replicate one model into N independent classifiers and vote. No effect when `--models` has 2+ values. |
| `--batch dynamic\|N` | Put several queries in one call (see below). |
| `--batch-max-size N` | Hard cap on rows per batch (default 50). Requires `--batch`. |
| `--batch-max-input-tokens N` | Override the input token budget used to size batches. Requires `--batch`. |
| `--batch-max-output-tokens N` | Override the output token budget used as a runaway-response guard. Requires `--batch`. |

### Choosing a classification mode

Four ways to classify a row. They trade cost against robustness, and most are mutually exclusive.

| Mode | Turn on with | LLM calls per row | Adds audit columns | Best for |
| --- | --- | --- | --- | --- |
| **Plain** (default) | nothing | 1 | no | most work; cheapest and simplest |
| **Critics** | `--critics` | `--sampling-runs` (default 5), plus a Critic (and Reconciler) call per category that lacks consensus | yes | catching genuinely ambiguous rows |
| **Multi-classifier** | `--models A B ...` or `--n-classifiers N` | one per classifier | yes | independent votes, optionally across providers |
| **Batched** | `--batch dynamic` or `--batch N` | roughly 1 / batch size | no | large inputs where call volume dominates cost |

**Mutual exclusions** (each fails before any LLM call): `--critics` with `--batch`;
`--critics` with `--models` (2+) or `--n-classifiers` > 1; `--batch` with `--models` (2+) or
`--n-classifiers` > 1. The `--batch-max-*` flags require `--batch`. 2+ *distinct* `--models`
cannot be combined with `CEREBUS_MODE=azure`.

**Set expectations honestly before reaching for `--critics`.** It only spends its extra calls on
rows where the 5 samples *disagree*, and models are very consistent with themselves: in the
benchmark runs under `experiments/`, roughly 91-94% of rows were unanimous and never reached a critic.
Votes cannot fix a label the model is confidently and consistently wrong about (typically a
fuzzy category boundary), so for most tasks, sharpening the label descriptions pays off far more
than a heavier mode. Also note that `--sampling-temperature` is sent to the provider, and some
models (notably reasoning models) reject it. When that happens it is dropped automatically and a
one-time warning (`provider rejected 'temperature' as unsupported`) is logged. If that warning
appears, your 5 "independent" samples are not diversified by temperature.

#### Critics: self-consistency sampling plus debate

`--critics` samples each row `--sampling-runs` times and votes on the top label per category. A
category whose leading label gets at least `--consensus-threshold` votes is accepted as is. Any
other category is escalated to a Critic (a devil's advocate) and, if the Critic raises a real
challenge, a Reconciler that settles it. The result is a full audit trail in the output.

```bash
python classify.py \
  --input data/queries.csv --column text \
  --output output/queries_classified.csv \
  --categories resources/categories/example_support_tickets.json \
  --critics --sampling-runs 5 --consensus-threshold 4
```

With the defaults (5 samples, threshold 4), a category is escalated only when no label reaches 4
votes. `--consensus-threshold 5` escalates on any single dissenting sample. This multiplies call
volume by `--sampling-runs` per row, so size `--workers` down for large inputs.
`--critic-model`/`--reconciler-model` may point at a different (often stronger) model than
`--model`; that sends the same row text to whichever provider each model resolves to, so make
sure that is intentional.

#### Multi-classifier: N instances, plurality vote

`--models`/`--n-classifiers` is a standalone alternative to `--critics`: N classifier instances
each classify the row once, independently (no sampling, no debate), and results are merged per
category by plurality vote on each instance's top label. Ties go to the earliest-listed instance.

```bash
# 2+ distinct models, one classifier each (duplicates allowed, e.g. to give one model extra weight):
python classify.py \
  --input data/queries.csv --column text \
  --output output/queries_classified.csv \
  --categories resources/categories/example_support_tickets.json \
  --models azure/gpt-5-chat gpt-4o-mini gemini/gemini-2.5-pro

# One model replicated into N independent classifiers:
python classify.py \
  --input data/queries.csv --column text \
  --output output/queries_classified.csv \
  --categories resources/categories/example_support_tickets.json \
  --model azure/gpt-5-chat --n-classifiers 5
```

In the audit trail each classifier is keyed by its bare model id, or by an occurrence-suffixed
key (`azure/gpt-5-chat#1`, `#2`, ...) when an id repeats. Concurrent in-flight requests are
roughly `--workers x number of classifiers`. Every instance uses the same prompt and schema and
its own provider-default temperature (not `--sampling-temperature`). Listing several distinct
models sends each row to *every* provider involved, a bigger data-egress surface than a single
`--model`; replicating one model stays on one provider.

#### Batched: several queries per call

`--batch` places several queries into one classification call, cutting call volume roughly by
the batch size.

```bash
# dynamic: size every batch from the model's input token budget
python classify.py -i data/queries.csv -c text -o output/queries_classified.csv \
  --categories resources/categories/example_support_tickets.json --batch dynamic

# fixed: exactly N rows per call (trimmed down only if a batch would not fit the budget)
python classify.py -i data/queries.csv -c text -o output/queries_classified.csv \
  --categories resources/categories/example_support_tickets.json --batch 8
```

- `--batch dynamic` needs a known input token budget for the model. If LiteLLM cannot resolve
  one, the run fails *before any LLM call* with an error naming the model; fix it with
  `--batch-max-input-tokens N` or switch to a fixed `--batch N`. An unresolvable *output* budget
  is never fatal; the request just carries no output cap.
- A batch failure is handled per cause: a validation or truncation error splits the batch in half
  and recurses (down to single rows); a rate limit or transient error retries the whole batch a
  bounded number of times; a credential or policy error fails the whole batch immediately.
- **Batched results are not directly comparable to an unbatched run.** A warning prints once per
  run. A row's neighbours can influence its labels even though the prompt says to classify each
  independently, and after a `--restore` the batch composition itself changes.
- `--workers` counts concurrent in-flight *batches*, so token throughput scales with
  `--workers x batch size`.

### Reading the output

The output CSV contains your input columns (or just those you kept) plus, per category:

| Column | Present in | Format | Meaning |
| --- | --- | --- | --- |
| `<name>` | always | **Python list literal**, e.g. `['billing']` or `['bug_report', 'billing']` | 1-3 labels, strongest evidence first |
| `<name>_initial` | `--critics` | list literal | the leading label *before* any debate |
| `<name>_votes` | `--critics`, `--models` | **JSON**, e.g. `{"billing": 4, "other": 1}` | vote tally |
| `<name>_challenged` | `--critics` | boolean | did the Critic raise a challenge |
| `<name>_critic_argument` | `--critics` | text | the Critic's argument (`Proposed: X. ...`) |
| `<name>_reconciled` | `--critics` | boolean | did a Reconciler settle a challenge |
| `<name>_reconciler_reasoning` | `--critics` | text | the Reconciler's verdict reasoning |
| `<name>_debate_error` | `--critics` | text | non-empty if the Critic/Reconciler failed (the answer then degrades to the leading candidate) |
| `<name>_by_model` | `--models` | JSON | each classifier's answer, keyed by model id |
| `<name>_model_errors` | `--models` | JSON | which classifiers failed on this row |

The two formats differ, and the difference bites when you load the file:

```python
import ast, json, pandas as pd

df = pd.read_csv("output/queries_classified.csv")
df["intent_labels"] = df["intent"].map(ast.literal_eval)            # category columns: Python literal
top = df["intent_labels"].map(lambda x: x[0] if x else None)        # strongest label
votes = df["intent_votes"].map(json.loads)                          # audit columns: JSON (critics/models)
```

Cells are written by pandas as-is and **not** sanitised against spreadsheet-formula injection:
treat the file like any untrusted CSV before opening it in a spreadsheet application.

**A blank category cell means one of two things**, and the run will not tell you which beyond the
progress bar:

- the row had **empty text** (null, empty, or whitespace-only; also for CSV input). It is never
  sent to the model and stays in the output, in place, with empty labels. A line
  `Skipped N rows with empty text` is printed.
- the row **failed** (after `--retries`). Failures are counted in the progress bar
  (`failed=N`) but **the run still exits 0**. Re-run the same command with `--restore` to retry
  just the failed rows.

### Resuming, partial failures and large inputs

**`--restore`** skips rows that already have labels, so a crashed or partly-failed run continues
instead of starting over. How it finds prior results depends on `--output`:

- `--output` omitted (in place), or the same path as `--input`: labels already in the file are used.
- `--output` is a *different* file that exists: its label columns are read back and matched to the
  input **by row position**; it must have the same row count as the input, otherwise the run stops
  with `Cannot restore from ...: it has N rows, but ... has M rows`.
- `--output` does not exist yet: there is nothing to restore and every row is classified.

A row counts as done when its label cell (and, under `--critics`/`--models`, its audit cells) is
non-empty, so failed rows are retried and finished rows are not. `--limit N` combined with
`--restore` means "the next N rows still to do". **Forgetting `--restore` on a re-run discards
every previous label** and redoes the whole file.

**Large inputs.** The output CSV is rewritten *in full* every `--save-every` completed rows
(default 20). That is fine for small files, but for hundreds of thousands of rows each rewrite
takes seconds (about 2 s for a 629k-row, 210 MB file in one local measurement), so use something
like `--save-every 5000`.

- A crash loses the rows completed since the last save: up to N, or up to N plus one batch with
  `--batch` (saves happen at batch boundaries). The writes are not atomic, so a crash *during* a
  write can truncate the file and make it unusable for `--restore`. Keep a copy of long runs'
  output.
- A cloud-synced output folder (OneDrive, Dropbox) makes each save slower and can stall on sync.
  Write to a local folder and copy the finished file afterwards.
- Everything is held in memory, and concurrency multiplies under `--critics`/`--models`/`--batch`.
  Size `--workers` down accordingly.

### Excel input and output

`--input` accepts `.xlsx`/`.xlsm` workbooks directly (legacy `.xls` is not supported; re-save as
`.xlsx`). Only the parsed cell values of the selected sheet are used: macros, formulas (their
cached values are read), formatting, hidden sheets, and workbook metadata are discarded. Cells are
read without type coercion, so ID-like strings such as `007` stay `007`.

```bash
python classify.py \
  --input logs/august.xlsx --sheet Sheet1 --column GENAI_INPUT \
  --keep-columns CONVERSATION_ID,TIMESTAMP \
  --output output/august_classified.csv \
  --categories resources/categories/example_support_tickets.json
```

- **`--output` is required for Excel input** and must name a CSV, so the workbook is never
  overwritten. The CSV is the working format: it is rewritten incrementally and is what
  `--restore` reads.
- **`--output-format xlsx`** also writes `<output stem>.xlsx` (one sheet named `Classified`) once,
  after the final CSV, never during the run. The write is atomic (temp file, then rename). If it
  fails, the CSV is kept, the cost report is still written, and the command exits with code 1.
  Strings are stored as literal text (a query like `=1+1` is not turned into a formula),
  characters Excel cannot store are dropped, and cells over 32,767 characters are truncated (a
  one-line summary is printed). The CSV is always written verbatim, so a CSV opened in a
  spreadsheet application can still interpret values starting with `=`, `+`, `-` or `@`. A sheet
  holds at most 1,048,575 data rows; larger inputs are rejected up front.
- **`--restore` re-reads the whole workbook on every run** and matches prior labels to rows purely
  by position (plus a row-count check). Resuming with a different `--sheet`, `--keep-columns`, or
  an edited input of the same length silently pairs old labels with the current rows, and carried
  columns always come from the current input. Keep those fixed across a resumed run.
- **Large workbooks are slow to read** (about a minute for a 145 MB / 629k-row log with
  `--keep-columns`) and the whole table is held in memory; `--output-format xlsx` additionally
  needs a temporary file of about the workbook's size on the output volume.
- Duplicate header cells are disambiguated by pandas (`a`, `a.1`), and `--keep-columns a` selects
  only the first.

### Customising the prompt

Three levers, from lightest to heaviest:

1. **`--task-description FILE`** (usually all you need): free text describing the domain and how
   to read the labels. It fills the `{task_description}` slot. This is where domain context goes
   (what the texts are, what a label means *in this dataset*).
2. **`--extra-prompt FILE`**: extra instructions appended to the end of the finished prompt
   ("Additional instructions: ..."), without touching the template.
3. **`--system-prompt FILE`**: replace the template entirely. The template is rendered with
   Python's `str.format`, so it may use these slots, and only these:
   `{task_description}`, `{schema_description}` (the auto-generated category/label summary), and
   `{invention_policy}` (the rule for labels that fit nothing). **Any other `{...}` in your
   template raises a `KeyError`** at render time, so write literal braces as `{{` and `}}`. You are
   not required to use every slot.

Under `--critics` the Critic and Reconciler roles always use the bundled templates
(`critic_prompt.txt`, `reconciler_prompt.txt`); only the main classification prompt is
customisable from the command line.

### Reproducing the original decision-classification setup

```bash
python classify.py \
  --input data/reasoning_traces.csv \
  --column reasoning \
  --output output/classified.csv \
  --categories resources/categories/example_categories.json \
  --task-description resources/prompts/example_task_description.txt
```

## `extract_categories.py`: write a categories file from a prompt

If you know in plain English what a category's labels should be but have no labeled dataset (so
`experiment.py induce` is not an option), `extract_categories.py` turns a short instruction file
into a `categories.json` with one LLM call:

```bash
python extract_categories.py \
  --prompt-file instructions.txt \
  --category-name sentiment \
  --output categories.json
```

`instructions.txt` might read: *"Classify the overall sentiment of a message as positive,
negative, or neutral."* The model invents the label values and descriptions; you supply the
category name, which the model never chooses. The output is exactly one category.

| Flag | Description |
| --- | --- |
| `--prompt-file` | Plain-text instruction file. **Required.** Empty or whitespace-only files are rejected. |
| `--category-name` | Becomes the output `name`. **Required.** Must be a valid Python identifier, not `__root__`, and not start with `model_`. |
| `--output` | Where to write (default `categories.json` in the current directory). The parent directory must exist, and the path must not be a symlink or a directory. |
| `--overwrite` | Allow replacing an existing `--output` file. Without it, an existing file is an error, checked before any LLM call. |
| `--extraction-prompt` | Replace the bundled system prompt for this call. |
| `--model`, `--retries`, `--cerebus`, `--api-base` | As on `classify.py` (`--model` takes a single value here). |

Every check runs before the LLM call where it can, and the output is validated afterwards: it
must contain at least one label, no duplicate label values, no blank values or descriptions, and
no `none`/`none - ...` values, otherwise nothing is written. **Read the result before trusting
it**: label descriptions written by a model from a short prompt are a draft, and they drive every
later classification. The instruction text is sent to the model, so do not put anything in it you
would not send to your provider.

This is a standalone tool. It does not read from or write into `experiment.py`'s run directory,
and neither entry point calls it.

## `experiment.py`: induce categories and evaluate against gold labels

`experiment.py` runs a full train-to-classified-test-set experiment against a local CSV pair or a
HuggingFace Hub dataset, and writes a self-contained run directory. It has three subcommands:

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

Equivalent to `python -m query_classification.experiment ...`.

### What each subcommand needs

| | `induce` | `classify` | `run` |
| --- | --- | --- | --- |
| Train split | **required** | not used | required unless `--categories` is given |
| Test split | not used | **required** | **required** |
| `--categories` | not allowed | **required** | optional: if given, induction is skipped and `run` behaves like `classify` |
| `--label-column` | **required** | optional | required when inducing, otherwise optional |
| `--category-name` | **required** | not used | required when inducing |
| Classification flags (`--critics`, `--batch`, ...) | no | yes | yes |
| Induction flags (`--seed`, `--examples-per-label`, ...) | yes | no | yes |

A "split" is either local files (`--train-file`/`--test-file`) or a HuggingFace dataset
(`--hf-dataset`); the two are mutually exclusive. `--train-file` together with `--categories` is
rejected as ambiguous. `--text-column` and `--label-column` must differ. `--label-column`, when
given while classifying, carries the gold labels through to the output as `<label-column>_gold`,
which is what you compare predictions against. If a test split has no usable labels, the run still
works and records `test_labels: "withheld"`.

### `experiment.py` flag reference

**Data**

| Flag | Description |
| --- | --- |
| `--train-file`, `--test-file` | Local CSV splits. |
| `--hf-dataset`, `--hf-config`, `--hf-revision` | A HuggingFace dataset (`owner/name`), config, and revision. Needs `pip install -e '.[hf]'`. |
| `--hf-train-split`, `--hf-test-split` | Split names (defaults `train` / `test`). |
| `--text-column` | Text column. **Required.** |
| `--label-column` | Gold label column (see above). |
| `--category-name` | Name of the induced category. |
| `--categories` | An existing `categories.json` with **exactly one** category. |
| `--run-dir` | Output directory. **Required.** Refused if non-empty unless `--overwrite`. |
| `--overwrite` | Replace this tool's known artifact filenames in a non-empty `--run-dir`. Never touches the directory itself or other files. |
| `--test-limit N` | Classify only the first N test rows (this is `classify.py`'s `--limit`, renamed because it bounds only the test split). Omit to classify the whole split. |

**Induction** (`induce`, `run`)

| Flag | Description |
| --- | --- |
| `--seed N` | Sampling seed (default 0). |
| `--examples-per-label N` | Max sampled examples per label (default 20). Mutually exclusive with `--induction-examples`. |
| `--induction-examples N` | Total example budget across all labels, split as evenly as possible (a label with too few rows gives all it has and the slack goes to other labels). Must be >= the number of distinct labels. Makes prompt size independent of how many labels there are. |
| `--max-example-chars N` | Truncate each sampled example (default 500). |
| `--max-prompt-chars N` | Refuse induction if the serialized prompt exceeds this (default 100000). |
| `--induction-model` | Model for induction (default: `--model`). |

**Classification** (`classify`, `run`): the same flags as `classify.py`'s
[mode](#classification-mode) and [model](#model-and-connection) groups (`--model`, `--api-base`,
`--retries`, `--workers`, `--critics`, `--sampling-runs`, `--sampling-temperature`,
`--consensus-threshold`, `--critic-model`, `--reconciler-model`, `--models`, `--n-classifiers`,
`--batch` and its `--batch-max-*` siblings, `--allow-new-labels`), plus `--system-prompt`,
`--task-description`, `--extra-prompt`, and:

| Flag | Description |
| --- | --- |
| `--allow-partial` | Exit 0 even if some rows could not be classified (they are still recorded in `run_config.json`). Without it, any unclassified row makes the command exit 1. |

`--model` and `--cerebus` are available on all three subcommands (`--cerebus` also covers the
induction call).

### What is different from `classify.py`

- **Label invention honors `--allow-new-labels` in every mode, default off.** `classify.py` always
  allows it in plain mode. The experiment runner deliberately does not, so plain-mode output stays
  inside the induced or supplied label vocabulary; otherwise gold labels could never match a
  prediction, and plain-vs-`--critics` comparisons would not be like for like.
- **No `--restore`, no `--save-every`, no Excel input, no `--keep-columns`.** Every invocation
  classifies its test split fresh.
- `--limit` is `--test-limit`.
- **Exits 1 if any row is unclassified** (unless `--allow-partial`); `classify.py` exits 0.
- **One category only** in `--categories`.

### The run directory

Every run writes `--run-dir`, containing:

| File | When | Contents |
| --- | --- | --- |
| `categories.json` | always | induced, or a verbatim copy of the one you supplied |
| `train.csv`, `test.csv` | when used | materialized, filtered, needed columns only |
| `test_classified.csv` | classifying | predictions (plus `<label-column>_gold` and any audit columns) |
| `induction_prompt.txt`, `sampled_examples.json` | inducing | exactly what the model was shown |
| `run_config.json` | always, written last | the audit record: dataset provenance (with resolved HF revision), resolved models, flags, label values, excluded/unclassified row positions, package versions, and a terminal `status` (`completed`, `completed_with_failures`, or `failed`) |
| `cost_report.json` | always | see [Cost and timing](#cost-and-timing) |

Rows with missing text are dropped when the splits are loaded (and, in the *train* split only, rows
with a missing label; a test row without a label is kept and classified). Dropped positions are
recorded under `excluded_rows` in `run_config.json`. A run that fails partway still writes a
`run_config.json` with `status: "failed"`.

**Data egress:** `induce` sends sampled *train* text to `--induction-model` (or `--model`);
`classify`/`run` send *test* text to the classifier model(s) and, under `--critics`, to the
critic/reconciler models too. Multi-classifier mode with 2+ distinct models sends it to all of
them. Only point these at a provider other than your usual one if you mean that split's text to
go there.

### Running the bundled benchmarks

`experiments/` holds ready-made scripts for three public datasets (`ag-news`, `trec`,
`pubmed-rct`), each with its own README, a `run_experiment.sh` (induce then classify, with
`--critics` on by default) and a `run_experiment_baseline.sh` (classify only, with a supplied
`categories.json`, for comparison; it requires `--categories FILE` and takes an optional `--suffix NAME`
to label the run directory). Both pass extra `experiment.py` flags straight through. `run_experiment.sh`
reads these environment variables (the baseline script reads only `TEST_LIMIT`):

| Variable | Effect |
| --- | --- |
| `TEST_LIMIT` | Classify only the first N test rows. **Set this for a first run**: the pubmed test split alone is ~29.6k rows. |
| `CRITICS=0` | Turn off `--critics` (required before `BATCH`, since they are mutually exclusive). |
| `BATCH` | `dynamic` or an integer to enable `--batch` (needs `CRITICS=0`). |
| `SEED`, `EXAMPLES_PER_LABEL`, `INDUCTION_EXAMPLES` | Induction sampling. |
| `DRY_RUN=1` | Print the assembled command instead of running it. |

```bash
TEST_LIMIT=40 DRY_RUN=1 experiments/pubmed-rct/run_experiment.sh   # inspect the command
TEST_LIMIT=40 experiments/pubmed-rct/run_experiment.sh --workers 16 # cheap smoke test
```

Each experiment directory has an `analyze_run.ipynb` notebook. The `--critics`-specific analysis
reads the `_initial`/`_votes`/`_challenged`/`_reconciled` columns and cannot analyse a
multi-classifier run's output as is.

## Cost and timing

Every run writes a `cost_report.json`: `<output stem>.cost_report.json` next to the output for
`classify.py`, `<run-dir>/cost_report.json` for `experiment.py`. `classify.py` writes it only on
success; `experiment.py` also writes it (best effort) when a run fails. A one-line
`Cost: $X | Time: Ns` summary is printed at the end.

```json
{
  "schema_version": 1,
  "total_cost_usd": 0.682,
  "induction_cost_usd": 0.0,
  "query_cost": {"mean_usd": 0.00068, "stddev_usd": 0.000015, "count": 1000, "estimated": false},
  "batch_cost": null,
  "pricing": {"resolved_input_model": "gpt-5.4-mini-2026-03-17", "resolved_output_model": "gpt-5.4-mini-2026-03-17"},
  "wall_clock_seconds": 148.5
}
```

- Figures are **estimates** from LiteLLM's bundled pricing table and the token usage the provider
  reports, not billing records. If a model's price cannot be resolved, the monetary fields are
  `null` rather than guessed; `pricing` shows which model id each side was priced from.
- In plain mode, `query_cost` is the real per-row distribution (`estimated: false`). Under
  `--critics` and `--models`, per-row costs are not attributed individually: `query_cost` is the
  phase total divided by rows attempted, with `stddev_usd: 0`, and `estimated: true`.
- Under `--batch`, `batch_cost` is the real per-batch distribution (first attempt of each formed
  batch only; retries and bisection show up in `total_cost_usd`), and `query_cost` is derived from
  it (`estimated: true`).
- `pricing` is `null` when 2+ distinct models were priced in one run.
- `induction_cost_usd` is always `null` for `classify.py`.
- A model that rejects a request parameter (typically `temperature` or `max_tokens`) has it
  dropped automatically and the call retried, which is logged once per classifier.

## Cerebus / Portkey gateway

All three entry points can route every LLM call through Cerebus, an internal gateway (built on
[Portkey]) that fronts Azure OpenAI and direct-provider (OpenAI, Gemini, ...) models behind one
OpenAI-compatible endpoint, instead of a direct provider. `--model`/`--critic-model`/
`--reconciler-model`/`--induction-model` still name the underlying model or workspace slug;
Cerebus only changes how the call is authenticated and where it is sent.

[Portkey]: https://portkey.ai/

**Minimal setup**, in `.env`:

```bash
DEFAULT_LLM_PROVIDER=cerebus
CEREBUS_API_KEY=...   # optional: falls back to AWS Secrets Manager if unset
```

Nothing else is required: the mode defaults to `direct` and the gateway URL to this org's shared
nonprod endpoint. `--cerebus` on the command line is an equivalent per-invocation alternative;
either one turns routing on. Model ids are sent with an `openai/` prefix added automatically.

The key resolves from `CEREBUS_API_KEY` if set, otherwise from AWS Secrets Manager. That needs
`pip install -e '.[cerebus]'` (for `boto3`) and an active AWS SSO session
(`aws sso login --profile kd-nonprod`). A misconfiguration in either path fails with an
actionable error rather than a raw traceback.

**Advanced overrides** (see `.env.example`), only to deviate from the defaults:

```bash
CEREBUS_MODE=azure                  # Azure-config-mode routing (default: "direct")
CEREBUS_GATEWAY_AZURE_URL=...       # override the Azure gateway URL
CEREBUS_GATEWAY_DIRECT_URL=...      # override the direct gateway URL
CEREBUS_CONFIG_ID=...               # required when CEREBUS_MODE=azure (workspace/model-specific)
```

`--api-base` still overrides the resolved gateway URL if given, and must be `https://` when
Cerebus is active. `CEREBUS_MODE=azure` takes one config id, so it cannot be combined with 2+
distinct models in `--models`.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `No Cerebus API key is available ... could not be fetched from AWS Secrets Manager` on a run where you did not ask for Cerebus | Your `.env` still has `DEFAULT_LLM_PROVIDER=cerebus` (it ships that way). Comment it out to use a direct provider, or set `CEREBUS_API_KEY` / run `aws sso login --profile kd-nonprod`. |
| Exported a variable in your shell but it has no effect | `.env` overrides the shell environment. Edit `.env`. |
| `Error: --output parent directory does not exist: ...` right at startup | `classify.py` does not create `--output`'s parent directory. `mkdir -p` it and re-run. Nothing was classified and no LLM call was made. |
| `Error: --output '...' is a directory, not a file` | `--output` must name a CSV file, not an existing directory. Append a filename, e.g. `--output output/result.csv`. |
| Input file silently changed | You omitted `--output`, so the input CSV is overwritten in place. |
| All previous labels gone after re-running | Re-running without `--restore` redoes every row. |
| `Cannot restore from ...: it has N rows, but ... has M rows` | `--output` exists but is not the same length as `--input`; point `--output` at a fresh path or the matching file. |
| Some rows have empty labels but the run "succeeded" | Those rows failed or had empty text (see [Reading the output](#reading-the-output)); re-run with `--restore`. |
| `Category name '<x>' collides with the text column being classified` | A category `name` equals `--column`. Rename the category. |
| `Impossible to dynamically calculate number of queries: no max token value known for <model>` | `--batch dynamic` could not resolve the model's input budget. Use `--batch N` or `--batch-max-input-tokens N`. |
| `--batch and --critics are mutually exclusive` (or the `--models` variant) | See the [mutual exclusions](#choosing-a-classification-mode). |
| `--batch-max-size has no effect without --batch` | The `--batch-max-*` flags require `--batch`. |
| `--run-dir ... is not empty` | `experiment.py` refuses to write into a used directory. Use a fresh one, or `--overwrite` to replace its known artifact files. |
| `classify requires --categories` / `--train-file and --categories are mutually exclusive` | See [what each subcommand needs](#what-each-subcommand-needs). |
| `--categories must contain exactly one category` | `experiment.py` supports a single category; `classify.py` does not have this limit. |
| `KeyError` when using `--system-prompt` | Your template contains a `{...}` that is not one of the three slots. Double the braces: `{{`/`}}`. |
| `provider rejected 'temperature' as unsupported` in the log | The model does not accept a temperature, so it was dropped. `--critics` samples are then not temperature-diversified. |
| `experiment.py` exits 1 but produced output | Some rows were unclassified; check `unclassified_rows` in `run_config.json`, or pass `--allow-partial`. |

## Project layout and library use

```
classify.py, experiment.py, extract_categories.py   # script entry points (work without installing)
src/query_classification/
  categories.py          # load + validate the categories JSON
  schema.py              # build the dynamic Pydantic output models + schema text
  prompts.py             # render the system prompt
  classifier.py          # single-text LLM classification (Classifier), retries, fallback, usage capture
  table_io.py            # CSV/Excel table reading, --keep-columns, optional .xlsx export
  pipeline.py            # file batch loop (restore / limit / incremental save)
  debate.py              # --critics: sampling, consensus voting, Critic/Reconciler debate
  multi_model.py         # --models/--n-classifiers: plurality-vote classification and audit trail
  batching.py            # --batch: token-budget resolution, batch sizing, payload assembly, bisection
  cost.py                # pricing, cost/timing statistics, atomic JSON writes
  cli.py                 # argparse entry point (classify.py)
  dataset_io.py          # local/HF dataset loading, label normalization, row filtering
  induction.py           # category induction: sampling, one LLM call, reconciliation
  experiment.py          # argparse entry point (experiment.py): induce / classify / run
  category_extraction.py # argparse entry point (extract_categories.py)
resources/
  categories/            # example category definitions
  prompts/               # system/critic/reconciler/induction/extraction prompt templates + examples
experiments/             # benchmark scripts, run outputs and analysis notebooks
spec/                    # specs, plans and ARCHITECTURE.md (module boundaries and invariants)
tests/                   # offline unit tests (no live provider calls)
```

Use it as a library, too:

```python
from query_classification import (
    load_categories, build_classification_model,
    build_system_prompt, Classifier, classify_csv,
)
```

Features here are spec-driven (`spec/<N>-<name>/spec.md`); read `spec/ARCHITECTURE.md` before any
structural change.
