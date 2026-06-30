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
| `--system-prompt` | Override the system prompt template. |
| `--task-description` | Text file describing the task/domain (`{task_description}` slot). |
| `--extra-prompt` | Text file with extra ad-hoc instructions, appended to the prompt. |
| `--limit N` | Only classify the first N rows. |
| `--restore` | Skip rows that already have labels; resume a previous run. |
| `--retries N` | Retry attempts per row on LLM error (default: 3). |

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
  cli.py          # argparse entry point
resources/
  categories/     # example category definitions
  prompts/        # system prompt template + example task descriptions
tests/            # unit tests for the offline building blocks
```

Use it as a library, too:

```python
from query_classification import (
    load_categories, build_classification_model,
    build_system_prompt, Classifier, classify_csv,
)
```
