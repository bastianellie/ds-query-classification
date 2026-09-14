# Spec 2: Experiment Runner (Dataset Loading, Label Induction, Test Classification)

> **Status: CLOSED** — Implemented and verified on 2026-09-08.
> Implementation summary: `spec/2-experiment-runner/implementation-summary.md`
> Merged from worktree branch `spec/2-experiment-runner` on 2026-09-08.
> **Reopened and re-closed 2026-09-14** for the induction pipeline redesign (positional
> description contract + `--induction-examples` total-budget sampling) — see the Change
> Log's final entry and `implementation-summary.md`'s appended 2026-09-14 section.
> Re-merged from worktree branch `spec/2-experiment-runner` on 2026-09-14.

## Overview

Adds an experiment runner that takes a labeled dataset — a local file pair or a
HuggingFace dataset — and runs a two-phase pipeline over it: a "training" phase where an
LLM reads sampled train-split examples per label and induces an exhaustive description
for each, producing a `categories.json`; and a classification phase where the test split
is classified with either of the repo's existing classifier modes (plain single-shot, or
Spec 1's `--critics` debate mode), producing a classified test set. Every run writes a
self-contained run directory so a result stays auditable and replayable later.

Scoring is deliberately **not** part of this spec (see Out of Scope) — the runner carries
the dataset's gold labels through to the output CSV so they can be scored externally.

## Goals

- Turn a labeled dataset into a working `categories.json` without hand-writing label
  descriptions, using the train split as the evidence base.
- Let the same dataset be classified by any classifier mode the repo offers, so the
  effect of a change (induced vs hand-written categories, plain vs `--critics`) can be
  compared on identical data.
- Make each run **auditable and replayable**: the exact rows, sampled examples, rendered
  prompt, seed, models, flags, and package versions behind a result are recoverable from
  disk. (Not *bitwise reproducible* — LLM responses are non-deterministic regardless of
  seed, which controls row sampling only.)
- Add HuggingFace support without making the base install heavier for people who only
  ever use local CSVs.

---

## Feature 1: Dataset Loading and Split Resolution

**Who & why:** Someone evaluating this classifier on a benchmark has the data either as
local CSVs or as a HuggingFace dataset id, and today has to hand-convert it into the
shape `classify_csv` wants before they can run anything. They need the runner to take the
dataset as-is, figure out the train/test splits, and normalize the label column into
human-readable label names — including the common HuggingFace cases where labels are
stored as `ClassLabel` integers, or withheld entirely on the test split.

### Functional Requirements

#### FR-1.1: Local dataset input via explicit train/test file paths
`--train-file` and `--test-file` each accept a path to a local CSV. Either may be omitted:
train-only is valid for the `induce` subcommand, test-only is valid for the `classify`
subcommand (which requires an explicit `--categories`). Passing neither, for a subcommand
that needs data, is an error naming which file was missing.
**Verify:** `induce --train-file t.csv` loads `t.csv`'s rows as the train split;
`classify --test-file e.csv --categories c.json` loads `e.csv` as the test split; running
`classify` with no `--test-file` and no `--hf-dataset` exits with an error naming the
missing input.

#### FR-1.2: HuggingFace dataset input
`--hf-dataset <id>` loads from the HuggingFace Hub via `datasets.load_dataset`, with
`--hf-config` (the dataset's configuration name, e.g. `sst2` for `nyu-mll/glue`, passed as
`load_dataset`'s `name=`), `--hf-revision`, `--hf-train-split` (default `train`), and
`--hf-test-split` (default `test`). Specifically:
- **The id must be a Hub repository id** (`owner/name`, or a canonical single-segment
  name), validated before loading. `load_dataset(path=...)` also resolves local
  directories and packaged builder names (`csv`, `parquet`, …), so an unvalidated value
  could read local files while the run recorded `source: "hf"` — a silent provenance lie.
  Local data goes through FR-1.1 instead.
- `--hf-dataset` and `--train-file`/`--test-file` are mutually exclusive; passing both is
  an error.
- Split names are resolved from a **single** load where possible (loading without `split=`
  returns a `DatasetDict` whose keys are the split names), rather than issuing a second
  remote `get_dataset_split_names` call after a failure — which would duplicate remote work
  and can mask the original error with a network/auth error. If a requested split is
  missing, the error lists the available names. Where a separate lookup is unavoidable,
  `get_dataset_split_names` must receive `config_name=<--hf-config>`; omitting it fails or
  resolves the wrong config for exactly the multi-config datasets `--hf-config` exists for.
- The **resolved immutable commit SHA** is recorded in `run_config.json` on a best-effort
  basis. A `--hf-revision` of `main` (or an omitted revision) is a mutable pointer, so
  recording the requested value alone would not pin anything; the materialized CSVs remain
  the actual row-level guarantee.
**Verify:** `--hf-dataset ./some/local/dir` exits with an id-validation error;
`--hf-dataset foo/bar --hf-test-split nonexistent` exits with an error listing the
dataset's actual split names; passing both `--hf-dataset` and `--train-file` exits with a
mutual-exclusion error.

#### FR-1.3: Label column normalized to verbatim, human-readable label names
The label column is resolved to strings before induction or output:
- A HuggingFace `ClassLabel` column (integer-encoded) is mapped to its names via the
  feature's `int2str` (equivalently `names[i]`).
- A string label column is used verbatim.
- **Local CSVs are read with the label column forced to string and NA-coercion disabled**
  for that column. `pandas.read_csv`'s defaults infer dtypes and missing values, so a
  legitimate taxonomy of `001`/`01` would silently become `1`, and labels like `NA`/`None`
  would become missing — either corrupting the label set or pushing it into the
  integer-label error path below. Forcing string dtype is what makes "verbatim" true.
- An integer label column with **no** `ClassLabel` feature is an error: the label names
  can't be recovered, and inducing descriptions for labels called `0`/`1` would produce a
  useless `categories.json`.
- Any other non-string label feature (bool, float, categorical, sequence, nested) is an
  error naming the actual feature type, rather than being coerced.
**Verify:** a fake HF dataset whose label column is `ClassLabel(names=["neg","pos"])` with
values `[0,1,1]` yields train labels `["neg","pos","pos"]`; a local CSV whose label column
is `001,002` yields labels `"001","002"` (not `1,2`); a bare integer label column with no
feature typing, and a float label column, each exit with an actionable error naming the
type.

#### FR-1.4: Text and label column selection, validated against the data
`--text-column` (required) names the column holding the text to classify;
`--label-column` (required for induction) names the gold label column. If either is
absent from the loaded split, the error lists the available column names — matching the
existing `classify_csv` behavior for a missing text column
(`pipeline.py`'s `Column '{column}' not found. Available columns: [...]`).
`--text-column` and `--label-column` must differ (FR-3.6 explains why).
**Verify:** `--text-column nope` on a dataset with columns `[text, label]` exits with an
error listing `['text', 'label']`; `--text-column x --label-column x` exits with an error.

#### FR-1.5: `datasets` is an optional extra, imported lazily
`datasets` is declared as an optional dependency (`pip install '.[hf]'`) with a
**`datasets>=4` version floor** — not merely `datasets` — rather than in the base
`requirements.txt`/`dependencies`, and is imported **inside** the HF-loading function,
never at module import time. The floor is load-bearing, not hygiene: `datasets` 4.0.0
removed dataset loading scripts and the `trust_remote_code` parameter, so only on 4.x is
"loading a Hub dataset executes no Hub dataset loading script" actually true (see Security
in the checklist). A `datasets<4` install could execute a dataset repo's loading script.
A local-file experiment requires no new dependency at all. If `--hf-dataset` is used
without the extra installed, the runner exits with an actionable message naming the install
command — not a bare `ImportError` traceback.
**Verify:** importing the experiment module succeeds with `datasets` absent from the
environment; invoking `--hf-dataset foo/bar` with it absent exits with an error containing
the `.[hf]` install hint.

#### FR-1.6: Split-availability rules
- **Train split present** (and `--categories` not given): induction runs.
- **Train split absent, `--categories` given**: induction is skipped; the supplied
  `categories.json` is used to classify the test split.
- **Train split absent, `--categories` absent**, for a command that needs categories:
  error — there is nothing to induce from and nothing supplied.
- **Test split absent**, for a command that classifies: error naming the missing test
  input.
- **Train split present *and* `--categories` given**: error. Silently ignoring a supplied
  train split would leave a `train.csv`-shaped question mark over the run — a later reader
  could not tell whether it influenced the result. The user must drop one.
**Verify:** each of the five cases above produces the stated outcome, with the three error
cases exiting non-zero before any LLM call is attempted.

#### FR-1.7: Rows unusable for their phase are excluded, and the exclusion is recorded
A row whose text is unusable, or (for the train split) whose label is unusable, cannot
serve its purpose: an empty train example is not evidence for any label, and an empty test
text would be stringified into a literal `"nan"` by `classify_csv`'s
`str(df.at[idx, column])` and sent to the LLM, spending a call to produce a meaningless
prediction that would silently pollute downstream scoring.

"Unusable" is one predicate: the value is null (`None`/`NaN`/`pd.NA`) **or** its string
form is empty/whitespace-only after stripping. A row failing on both text and label counts
once, not twice. Such rows are excluded from the materialized split; the excluded count
and the **0-based row positions in the source split as loaded** (not a CSV index column,
which may or may not exist) are recorded in the run config.
**Verify:** a train split with 2 empty-text rows and a test split with 1 null-text row
produce materialized CSVs shorter by exactly those rows, and a run config recording the
counts `{train: 2, test: 1}` and the dropped positions; a train row missing both text and
label increments the count by exactly 1.

#### FR-1.8: Withheld or unseen test gold labels are handled, never fabricated
Benchmark test splits frequently ship without usable labels, in two forms, and both must be
handled rather than crashing or fabricating ground truth:
- **Label column absent from the test split** — classification proceeds; the gold
  carry-through (FR-3.6) is skipped.
- **Label column present but values outside the `ClassLabel` range** (the `-1` sentinel
  used by GLUE and similar benchmarks) — treated as unlabeled and **not** passed through
  FR-1.3's normalization. `ClassLabel.int2str` raises on a negative index, and any
  workaround that produced a name anyway would write a fabricated label into `test.csv`
  looking exactly like real ground truth.

In both cases `run_config.json` records `test_labels: "withheld"`. Additionally, when test
gold labels *are* present, any value **not** in the induced train label set is recorded in
`run_config.json` under `unseen_test_labels` — the classifier can never predict such a
label, so an external scorer would count those rows wrong for a structural reason worth
surfacing. A withheld **train** label column remains an error (FR-1.6).
**Verify:** a fake test split with no label column produces `test_classified.csv` with
predictions and no `*_gold` column, plus `test_labels: "withheld"`; a fake test split whose
`ClassLabel`-typed label column is all `-1` produces the same outcome without raising; a
test split containing a gold label absent from train records it in `unseen_test_labels`.

### Architectural Requirements

#### AR-1.1: New `dataset_io.py` module — must NOT be named `datasets.py`
Dataset loading/split-resolution/label-normalization lives in
`src/query_classification/dataset_io.py`. **The module must not be named `datasets.py`**:
`src/` is on `sys.path` (via `pyproject.toml`'s `pythonpath = ["src"]` and `classify.py`'s
injection), so a sibling module named `datasets.py` would shadow the real `datasets`
package and break the very import this feature depends on. This is a real hazard given
the module's subject matter, so the name is specified rather than left to the implementer.

#### AR-1.2: Architecture update required for the new modules
`spec/ARCHITECTURE.md` INV-1 fixes the allowed import graph and its Module Boundary Map
has no row for the modules this spec adds (`dataset_io.py`, `induction.py`,
`experiment.py`). As with Spec 1's AR-1.6, implementing this spec requires an ADR
(`spec/2-experiment-runner/ADR.md`) recording the revised graph — new rows plus amended
INV-1 wording — which `/spec-close` folds into `spec/ARCHITECTURE.md`. The same ADR
covers the INV-7 test-import exception for these modules (they are internal, not added to
`__init__.py`'s public re-exports, but are unit-tested directly).

#### AR-1.3: Splits are materialized to CSV with only the needed columns
A loaded split is reduced to the columns the run actually needs — the text column, the
label column (when present), and nothing else — **before** `Dataset.to_pandas()`, then
written to a CSV inside the run directory, which is what the classification phase reads.
Two reasons, both material:
- **Privacy/footprint:** benchmark datasets carry extra columns (ids, annotator metadata,
  URLs, free-text notes) that may be sensitive. "Self-contained run directory" must not
  quietly mean "a second copy of every source column."
- **Memory:** HF data already exists as an Arrow-backed `Dataset`, is converted wholesale
  to pandas, written to CSV, and read back into another DataFrame by `classify_csv` —
  projecting columns first is the cheapest available cut on that amplification.

`streaming=True` is not used: per-label sampling (FR-2.1) and `classify_csv` both need a
materialized, indexable split.

---

## Feature 2: Label-Description Induction ("training")

**Who & why:** Writing a good `categories.json` by hand is the hardest and most
error-prone part of using this tool — label descriptions have to be exhaustive enough for
the LLM to classify against, and discriminative enough to separate neighbouring labels. A
labeled train split already contains that knowledge implicitly. This phase extracts it:
an LLM reads sampled examples for every label side by side and writes descriptions that
say what each label means *and* what distinguishes it from the others.

### Functional Requirements

#### FR-2.1: Seeded example sampling, sized by either a per-label cap or a total budget
Examples are grouped by their (normalized, FR-1.3) label value, with labels processed in
**sorted** order, and selected by a **single named, seeded generator**
(`random.Random(seed)`, `--seed` default 0) over each label's stable post-filter row
positions — pinned because `random`, NumPy, and `DataFrame.sample` select different rows
for the same seed, and an unpinned choice would make the recorded seed meaningless across
implementations or dependency bumps.

The sample size is set by exactly one of two **mutually exclusive** flags. Supplying both
is an error naming both, raised before any LLM call; supplying neither means
`--examples-per-label`'s default. The mutual-exclusion check must distinguish *explicitly
passed* from *left at its default* — a per-label flag that defaults to `20` outright would
collide with every `--induction-examples` invocation — so `--examples-per-label`'s argparse
default becomes `None` and the literal `20` is applied during resolution, the same mechanic
spec 4's AR-1.8 required of `--model`:

- **`--examples-per-label N`** (default 20) — each label contributes up to `N` examples;
  a label with fewer than `N` usable rows contributes all of them, in source order. Total
  prompt volume scales with the label count (`labels × N`).
- **`--induction-examples N`** (no default) — `N` examples **in total**, spread as evenly
  as possible across labels, so prompt volume is bounded by `N` no matter how many labels
  the dataset has. Quotas are assigned by **largest remainder**: every label gets
  `N // n_labels`, then the `N % n_labels` leftover slots go one each to the labels with
  the largest fractional part, ties broken by sorted label value. Quotas therefore always
  sum to exactly `N` and are fully determined by `(N, sorted label values)` — e.g. `N=100`
  over 3 labels yields 34/33/33. A label with fewer usable rows than its quota contributes
  all of them, and the freed slots are re-divided by the same rule among labels that still
  have unused rows, repeating until either `N` is reached or no label has spare rows — so
  the budget is met whenever the data allows.

`--induction-examples N` must be at least the number of distinct labels, otherwise some
label's quota would be 0 and that label would have no examples to describe from. A smaller
`N` is an error naming both `N` and the label count. This is a data-dependent check, so it
is raised once the label set is known but still before any LLM call, alongside FR-2.7's
related zero-usable-examples checks.

**Reproducibility:** in both modes, the same `--seed`, train split, and flags select an
identical example set. The modes differ in one guarantee, and only the per-label cap keeps
it: under `--examples-per-label`, a label's selection is independent of every other
label's row count (a label at or below the cap consumes no generator state at all); under
`--induction-examples`, quotas depend on the other labels' row counts by construction, so
adding rows to one label can change another label's sample. `--seed` and the **effective**
sizing values are recorded in the run config — the mode that actually ran, so on an
inducing run exactly one of `examples_per_label`/`induction_examples` is non-null there
(per-label mode records the resolved cap, including the literal default, rather than the
`None` argparse saw). A `classify`-only run records both as null, since the induction flag
group does not exist on that subcommand.
**Verify:** two induction runs over the same train split with the same `--seed` select an
identical example set, in both modes (assert on the examples embedded in the prompt sent to
a fake classifier); for a label with **more** examples than the cap, changing `--seed`
selects a different subset (a label at or below the cap contributes all its examples
regardless of seed, so seed-sensitivity must be asserted only on an over-cap label); a
label with 3 examples and `--examples-per-label 20` contributes exactly 3; under
`--examples-per-label`, growing one label's pool leaves another label's sample unchanged;
`--induction-examples 100` over 3 labels selects 34/33/33 summing to exactly 100;
`--induction-examples 10` where one label has only 2 usable rows still selects 10 in total,
that label contributing exactly 2; `--induction-examples 2` over 3 labels, and
`--examples-per-label` together with `--induction-examples`, each exit with an error naming
the offending flags/counts and issue zero LLM calls.

#### FR-2.2: Single induction call sees all labels together
One LLM call receives the sampled examples for **every** label at once, grouped by label
and presented in a stable numbered order, and returns one description per label in that
same order (AR-2.1). Seeing the labels side by side is what lets the model write
descriptions that discriminate between them rather than describing each in isolation.
**Verify:** inducing over a 3-label train split issues exactly one `.classify()` call on
the induction classifier, and the prompt sent contains examples grouped under all 3 label
values in sorted order with their positions stated.

#### FR-2.3: Example texts are truncated, and total prompt size is bounded up front
Each example's text is truncated to `--max-example-chars` (default 500) Unicode code
points before being placed in the prompt. The truncation marker (a fixed `…[truncated]`
suffix) is appended **outside** the cap, so the cap applies to retained source text only.

Per-example caps alone do not bound the prompt under `--examples-per-label`: the **number**
of labels is unbounded there, so `labels × examples_per_label × max_example_chars` can
still exceed the model's context window, and FR-2.2 forbids chunking as a fallback. The
runner therefore performs a **total-size preflight** — estimated characters across all
labels' sampled, truncated examples against a configurable `--max-prompt-chars` budget
(default 100000) — and fails with an actionable error naming the offending totals and the
flags to lower, **before** the provider call. A context-window rejection mid-call would
otherwise waste the call and surface as an opaque provider error.

`--induction-examples` (FR-2.1) bounds the example count directly, so it makes the prompt
size a function of `N × max_example_chars` alone rather than of the label count — the
preflight is far less likely to fire in that mode. It remains a required backstop in both
modes regardless, since `--examples-per-label` is still label-count-unbounded and even a
fixed `N` can exceed the budget at a large `--max-example-chars`.
**Verify:** an example of 5000 characters appears in the induction prompt truncated to 500
characters plus the marker; a 400-label train split at default caps exits with the
preflight error naming the sizing flag in effect and `--max-example-chars`, and issues zero
LLM calls.

#### FR-2.4: Induced output is a valid `categories.json` with exactly the dataset's labels
Induction writes a `categories.json` conforming to the existing format
(`{"categories": [{"name", "description", "labels": [{"value", "description"}]}]}`,
defined by `Category`/`Label` in `categories.py`), containing exactly **one** category:
- `name` — from `--category-name` (default: the `--label-column` value).
- `description` — induced: what the category as a whole captures.
- `labels` — one entry per distinct label value in the train split. Each `value` is the
  label string **verbatim** from the dataset (FR-1.3); each `description` is induced.

The written file is loaded back through `load_categories` and must validate. Every `value`
comes from the runner's **own** sorted label set, never from the model's response: the
response carries descriptions only, mapped back onto labels **by position** (AR-2.1). A
missing or invented label value is therefore structurally impossible, rather than a
discrepancy to detect after the fact — there is no label-set reconciliation step, and no
induction-specific retry layer for one (AR-2.3).

The only residual failure mode is a wrong **number** of descriptions, and AR-2.1 closes
that at the schema level: each description is its own required field, so a short response
is rejected either by the provider's strict structured-output mode or — if that mode is
unavailable and `Classifier` falls back to JSON-object mode — by Pydantic validation inside
`Classifier.classify`, which retries it under the existing `--retries` budget. Either way
induction receives exactly one description per label or raises.
**Verify:** inducing over a train split with labels `{neg, pos}` writes a file that
`load_categories` parses into one `Category` with exactly two `Label`s valued `neg` and
`pos`, carrying the response's 1st and 2nd descriptions in that order; the induction model
built for that split has exactly the required fields
`category_description`/`description_1`/`description_2` and rejects a response omitting
`description_2`; no response a fake classifier can return causes a label value outside
`{neg, pos}` to be written.

#### FR-2.5: Induced label values must not collide with the reserved `none` sentinel
If a dataset's distinct label values include `none`, or any value beginning with
`none - ` (compared case-insensitively after stripping surrounding whitespace), induction
fails with a clear error. Those strings are the classifier's reserved "no label applies"
sentinel (`schema.py`'s field description and `system_prompt.txt`), so a gold label of
`none` would be indistinguishable from the classifier declining to label the row. The same
check is applied to a **supplied** `categories.json` (FR-3.2), since `load_categories`
itself accepts `none` as an ordinary label value.
**Verify:** a train split containing a label literally valued `none` exits with an error
naming the reserved sentinel, before any LLM call; so does `NONE`, and so does a supplied
`categories.json` containing a `none` label.

#### FR-2.6: Induction failure surfaces, it does not produce a partial file
If the induction call fails after its own retries (reusing `Classifier`'s existing bounded
retry/fallback), the command exits non-zero and writes **no** `categories.json`. A
half-written or absent-descriptions categories file would silently degrade every downstream
classification run that used it.

The **persisted and CLI-surfaced** error is sanitized to an exception type plus a fixed
stage message, never raw provider text (which can carry endpoint details or credentials).
This requirement is scoped to the persisted/CLI error deliberately: `Classifier` logs raw
exception text on every failed attempt (`classifier.py:120,129`) and AR-2.3 keeps it
unmodified, so claiming the raw text never appears *anywhere* would be false.
**Verify:** a fake induction classifier that always raises causes a non-zero exit, no
`categories.json` on disk, and a CLI error containing the exception type but not a fake
secret embedded in the exception message.

#### FR-2.7: Degenerate train splits are rejected before the induction call
Induction requires something to induce from. Each of the following is an error naming the
condition, raised before any LLM call: the filtered train split is empty (every row was
excluded by FR-1.7); the train split has zero distinct usable label values; a label has
zero usable examples. A single-label train split is **allowed** (a degenerate but coherent
taxonomy) and proceeds normally.
**Verify:** a train split whose every row is filtered out exits with an error and issues
zero LLM calls; a single-label train split induces successfully.

### Architectural Requirements

#### AR-2.1: Induction schema builder in `schema.py` — one required description field per label position
`schema.py` gains a builder returning the induction call's Pydantic output model, alongside
the existing `build_classification_model`/`build_critic_model`/`build_reconciler_model`. It
takes the label count and returns a model whose fields are **positional**, carrying no
label names at all:

- `category_description: str` — what the category as a whole captures.
- `description_1 … description_N: str` — one **required** field per label, in FR-2.1's
  sorted label order, where `N` is the number of distinct labels.

It is built with `create_model`, the same mechanism `build_classification_model` already
uses. Field names are **positional indices, not label values**: label values are
dataset-chosen strings that routinely contain spaces or punctuation (`"very negative"`,
`"class 1"`, `"none-ish"`) and are therefore not valid Python/Pydantic field names, so a
field-*per-label-value* model would break for exactly the datasets this feature targets.
Indices are always valid identifiers, which is what makes the positional form possible
where the label-named form is not.

Arity is pinned by **required fields**, not by a length-bounded list. This is a verified
constraint, not a preference: litellm serializes a Pydantic `response_format` with
`"strict": true`, and strict structured output does not support `minItems`/`maxItems`, so a
`list[...]` with `min_length == max_length == N` would be rejected by the provider — and
`Classifier._attempt_completion`'s `except litellm.BadRequestError` would silently degrade
**every** induction call to JSON-object mode. Required fields plus strict mode's
`additionalProperties: false` express exactly-`N` in a form the strict path accepts.

What this buys, and its limit: because the model never emits a label name, label identity
cannot be lost or invented in transit (FR-2.4), and a short response is rejected at
validation rather than discovered during reconciliation. Per INV-11 the schema still
enforces shape and cardinality **only** — the descriptions' content is never validated, and
a description that is confidently wrong about its label remains undetectable here.

#### AR-2.2: Induction prompt — static system prompt, examples in the user message
The induction system prompt lives in `resources/prompts/induction_prompt.txt` with its
path exported as a constant from `resources.py` (per INV-2: bundled resource paths are
computed only there), rendered by a new builder in `prompts.py` following the existing
`build_critic_prompt`/`build_reconciler_prompt` plain-string-parameter shape.

The **sampled examples, label values, and category name go in the per-call user message**,
not the static system prompt — matching Spec 1's established split (`cli.py` builds the
system prompt once; `classifier.py:99-102` builds a fresh user message per call, and
`debate.py` assembles per-call untrusted content there). They are serialized in a
delimited, structured form (e.g. JSON between fixed delimiters) rather than interpolated
as loose prose.

Because AR-2.1's response is positional, the payload must make each label's position
**explicit**: labels are carried as an ordered sequence of entries that each state their
own 1-based position alongside the label value and its examples — not as a JSON object
keyed by label value, which would leave the positional contract resting on object-key
ordering. The system prompt states the mapping rule directly (the description for the label
at position `i` goes in `description_i`) and must be updated accordingly: the shipped
`induction_prompt.txt` currently closes by asking for `a "labels" array with one entry per
label, each entry containing that label's exact name`, which is exactly the label-name echo
AR-2.1 removes.

The prompt must instruct the model to treat **all** of that payload — example texts, label
values, **and** the category name — as untrusted data to summarize, never as instructions
to follow. Label names are as dataset-controlled as the example texts, so framing only the
examples would leave the obvious injection vector open.

`check_default_resources_available()` needs **no change**: it is deliberately
directory-level (`RESOURCES_DIR.is_dir()`, `resources.py:31`), so placing the new template
under `RESOURCES_DIR` already puts it behind the existing guard for the wheel-install case
INV-3 describes. A single file deleted from an otherwise-present `resources/` directory
remains outside the guard's scope by design, exactly as for Spec 1's `critic_prompt.txt`.

#### AR-2.3: Induction reuses `Classifier` as-is
The induction call is made through an ordinary `Classifier` instance built with the
induction prompt (AR-2.2) and induction schema (AR-2.1) — reusing its existing structured-
output/JSON-fallback/bounded-retry logic unchanged, exactly as Spec 1's AR-2.3 does for
the Critic and Reconciler. No new retry or LLM-call code is written. `--induction-model`
selects its model, defaulting to `--model`.

**`--induction-retries` is removed by this update.** It was shipped on 2026-09-14 as an
interim response to a real production failure (a PubMed RCT run with 100 examples per label
whose induction response repeatedly omitted one label), adding an
`induce()`-level `max_reconciliation_retries` loop (default 3) that re-sent the identical
prompt whenever the old label-set reconciliation failed. That loop was itself a violation of
this AR's own "no new retry or LLM-call code" rule — a second, induction-specific retry
layer sitting above `Classifier`'s. AR-2.1 removes the failure mode it existed for: label
identity no longer round-trips through the response, and a wrong description count is
rejected at validation and retried by `Classifier`'s existing `--retries` budget. Keeping
both would leave two overlapping retry mechanisms where one suffices, so retry
responsibility returns wholly to `Classifier`, and this AR's rule holds again without
exception.

---

## Feature 3: Experiment Orchestration and Run Artifacts

**Who & why:** Comparing "induced vs hand-written categories" or "plain vs `--critics`"
only means something if both runs used identical data and the difference between them is
recorded. Someone running these experiments needs a single command per run, a place to
stop and read the induced descriptions before spending classification calls, and enough
on disk afterwards to reconstruct what produced a given result — including whether every
row actually got classified.

### Functional Requirements

#### FR-3.1: `induce` subcommand
`induce` loads the train split (Feature 1), runs induction (Feature 2), and writes
`categories.json` into the run directory. It makes no classification calls.
**Verify:** `induce` over a fake dataset writes a valid `categories.json` and issues zero
calls on the classification classifier.

#### FR-3.2: `classify` subcommand
`classify` requires `--categories` (a hand-written or previously-induced file), loads the
test split, and classifies it. The supplied file must contain **exactly one** category —
`load_categories` accepts any number, but this runner's gold-comparison and collision model
(FR-3.6) is built around a single category, so a multi-category file is an error rather
than a silently half-supported case. Its label values are checked against FR-2.5's
sentinel rule. It makes no induction calls.
**Verify:** `classify --categories c.json` produces a classified test CSV and issues zero
calls on the induction classifier; a two-category `c.json` exits with an error.

#### FR-3.3: `run` subcommand — induce then classify end-to-end
`run` performs `induce` followed by `classify` in one invocation, using the freshly
induced `categories.json`. Passing `--categories` to `run` alongside a train input is an
error (FR-1.6); `run --categories` with only a test input behaves as `classify`.
**Verify:** `run` over a fake dataset with both splits writes both a `categories.json` and
a classified test CSV, and the classification uses the induced file's category name and
label values.

#### FR-3.4: Every run writes a self-contained, auditable run directory
Each invocation writes to `--run-dir` (required). Contents are listed in Data
Requirements; the rules are:
- **Overwrite policy:** the runner refuses to write into a non-empty existing directory
  unless `--overwrite` is passed. `--overwrite` replaces **only the known artifact
  filenames** this spec defines — it never deletes the directory or unrelated files, so it
  cannot destroy a user's notes or a previous run's renamed artifacts. The
  empty/overwrite check is a preflight completing **before** any file is modified.
- **`run_config.json` is written last**, carrying a terminal `status`
  (`completed` | `completed_with_failures` | `failed`) plus the failure summary from
  FR-3.8. Writing it first would leave a `status: completed` record behind a crashed run.
- **Replay snapshots:** the rendered induction prompt and a sampled-example manifest are
  written (Data Requirements), plus package versions in `run_config.json`. `seed` + cap
  alone cannot reconstruct the LLM request if the bundled prompt text, CSV parsing, or a
  dependency's behavior changes between runs — which is precisely what a months-later
  audit needs.
- If `--categories` resolves to the run directory's own `categories.json`, the copy is a
  no-op rather than an error or a self-truncating copy.
**Verify:** a completed `run` leaves every Data-Requirements artifact in `--run-dir` with
`status: completed`; re-running into the same non-empty directory without `--overwrite`
exits with an error having modified nothing; with `--overwrite`, an unrelated file placed
in the directory beforehand still exists afterwards.

#### FR-3.5: Classifier configuration, and the closed-vocabulary divergence
The classification phase exposes: `--model`, `--api-base`, `--retries`, `--workers`,
`--test-limit`, the prompt-shaping inputs `--system-prompt`, `--task-description` and
`--extra-prompt` (these materially change predictions, so an experiment tool must be able
to vary and record them), and Spec 1's `--critics`, `--sampling-runs`,
`--sampling-temperature`, `--consensus-threshold`, `--allow-new-labels`, `--critic-model`,
`--reconciler-model`. `--restore` is deliberately **not** exposed: run directories are
always fresh (FR-3.4), so there is no prior in-place run to resume.

Two behaviors differ from `classify.py` and are deliberate:
- **Closed vocabulary in both modes.** `cli.py:201` currently does
  `allow_new_labels = args.allow_new_labels if args.critics else True` — forcing label
  invention **on** whenever `--critics` is off, with `classify_csv`'s parameter inert
  outside the critics path (`pipeline.py:194`). The experiment runner instead honors
  `--allow-new-labels` (default off) in **both** modes, building the classification schema
  *and* system prompt with it. Without this, plain-mode experiment output could contain
  invented labels that no gold label can ever match — silently corrupting the comparison
  the runner exists to enable.
- **`--test-limit` truncates the materialized test split; omitting it classifies the
  whole thing.** `classify_csv` creates prediction columns for every row and classifies
  only `work_idx[:limit]` (`pipeline.py:171`), leaving a blank tail. The runner therefore
  truncates `test.csv` to the limit **before** classifying, so `test.csv` and
  `test_classified.csv` both contain exactly the evaluated rows and an external scorer
  cannot mistake an unclassified tail for wrong predictions. The flag has no default —
  when it is not passed, no truncation happens and the entire test split is classified.
  It is named `--test-limit`, not `--limit`, to make explicit that it bounds only the
  test/classification split; it has no effect on the train split or induction sampling
  (those are controlled independently by `--examples-per-label`/`--induction-examples`/
  `--seed`), and the name
  avoids collision with `classify.py`'s own unrelated `--limit` flag on a different entry
  point.
**Verify:** `run --critics --sampling-runs 3 --consensus-threshold 2` produces a classified
CSV carrying Spec 1's audit columns, and the same run without `--critics` produces one with
no audit columns; a plain-mode run with `--allow-new-labels` omitted builds its
classification schema and system prompt with invention disabled (assert no `none - ` text
in the rendered prompt); `--test-limit 5` over a 100-row test split produces a `test.csv`
and `test_classified.csv` of exactly 5 rows with no null prediction rows; omitting
`--test-limit` over the same 100-row test split produces a `test.csv` and
`test_classified.csv` of all 100 rows.

#### FR-3.6: Generated columns must not collide with any source column
The materialized `test.csv` (and therefore the classified output) retains the dataset's
gold labels in a column named `{label_column}_gold` (skipped when labels are withheld,
FR-1.8). The rename is mandatory, and the mechanism matters so it isn't mistaken for
cosmetics: `classify_csv` writes each category's predictions into a column named after the
category, and the natural category name **is** the label column's name (FR-2.4's default).
Worse than an overwrite, on a non-restore run the reset step does
`df.loc[work_idx, reset_cols] = None` with `reset_cols` starting as `list(category_names)`
(`pipeline.py:157-169`), so a same-named gold column is **nulled before any classification
happens** — ground truth destroyed, not merely replaced.

The check generalizes beyond the gold column: the runner precomputes the **full generated
column set** for the active mode (the category name, plus Spec 1's
`debate.AUDIT_COLUMN_SUFFIXES`-derived audit columns when `--critics` is on) and rejects
any intersection with the materialized input's columns, other than the gold column it
deliberately renamed. Nothing upstream protects these: Spec 1's collision check compares
generated columns against the text column and other categories' generated sets only, never
against arbitrary pre-existing columns (its AR-1.5). This is also why FR-1.4 requires
`--text-column != --label-column`: renaming a shared column to `_gold` would remove the
classification input, and keeping it would conflate text with ground truth.
**Verify:** a run whose category is named `label` over a dataset with a `label` column
produces a `test_classified.csv` containing both a `label` column (predictions) and a
`label_gold` column (unmodified ground truth); a dataset that already has a `label_gold`
column exits with a collision error; a `--critics` run over a dataset with a pre-existing
`label_votes` column exits with a collision error.

#### FR-3.7: Ship unit tests
The implementation ships a `pytest`-runnable suite (in `tests/`, no live network or
network-backed HF calls — HF loading is exercised with a fake injected `datasets` module
whose function signatures match the real ones, so a signature-loose fake can't let a wrong
call pass; LLM calls with fake `Classifier`-shaped objects exposing
`.classify(text) -> dict`), covering every FR-1.x/FR-2.x/FR-3.x **Verify:** condition above
including the error and edge cases (FR-1.2's id validation and missing split, FR-1.3's
untyped-integer/float labels and the local `001` case, FR-1.5's missing extra, FR-1.6's
five split cases, FR-1.7's dropped rows and count-once rule, FR-1.8's two withheld forms
and unseen labels, FR-2.1's over-cap seed sensitivity plus both sizing modes' quota,
exclusivity and budget-floor rules, FR-2.3's preflight, FR-2.4's positional label mapping
and required-field arity, FR-2.5's sentinel including a supplied file, FR-2.6's induction
failure, FR-2.7's degenerate splits, FR-3.2's multi-category rejection, FR-3.4's overwrite
preservation, FR-3.5's plain-mode closed vocabulary and `--test-limit` truncation, FR-3.6's
three collision cases, FR-3.8's partial-failure status).
**Verify:** `pytest` passes and includes test functions exercising every FR's Verify
condition above without making a real LLM or Hub call.

#### FR-3.8: Incomplete classification is detected and reported, not silent
`classify_csv` catches every per-row exception, increments an in-memory `failed` counter,
and returns normally — so a run can finish "successfully" with some or all rows
unclassified and nothing on disk recording it (a documented Known Gap in
`spec/ARCHITECTURE.md`). For an artifact whose entire purpose is to be scored later, that
makes "completed run" undefined.

Since AR-3.1 forbids modifying `classify_csv`, the runner performs an **external
completeness check** on the returned CSV after classification: a row is complete when its
category prediction column is non-null (and, under `--critics`, its `{cat}_votes` column
is non-null too, matching Spec 1's own completeness signal). It records
`unclassified_rows: {count, positions}` and sets `run_config.json`'s `status` to
`completed_with_failures` when the count is non-zero, exiting **non-zero** in that case
unless `--allow-partial` is passed.
**Verify:** a fake classifier that raises on 2 of 10 rows yields
`status: completed_with_failures`, `unclassified_rows.count == 2`, and a non-zero exit; the
same run with `--allow-partial` exits zero while still recording the count.

### Architectural Requirements

#### AR-3.1: Classification delegates the row loop to `classify_csv`, unmodified
The classification phase calls the existing
`pipeline.classify_csv(input_path, column, classifier, categories, output_path=..., ...)`
with its current signature (including Spec 1's `critics`/`critic_classifiers`/
`reconciler_classifiers`/`sampling_runs`/`consensus_threshold`/`allow_new_labels`
parameters), and does not modify it.

What is shared is precisely **the row loop**: threading, per-row dispatch, incremental
save, restore/limit mechanics, and the critics routing. What is **not** shared — because it
lives in `cli.py`, not `classify_csv` — is classifier construction, prompt rendering,
schema building, resource checking, model fallback resolution, and numeric validation; see
AR-3.4. The earlier framing of "same behavior by delegating to `classify_csv`" was
inaccurate and is corrected here, since it is what concealed FR-3.5's plain-mode
vocabulary contradiction.

#### AR-3.2: New `experiment.py` root script mirrors `classify.py`
A root-level `experiment.py` provides `python experiment.py <subcommand> ...`, injecting
`src/` onto `sys.path` exactly as `classify.py:13` does so it works with or without an
editable install. The subcommand parser and orchestration live in
`src/query_classification/experiment.py`, with `python -m query_classification.experiment`
as the module-form entry point. Today's `classify.py`/`cli.py` and their flat flag list are
left untouched.

#### AR-3.3: No console-script entry point is added
Consistent with the current `pyproject.toml` (which declares no `[project.scripts]`), this
spec adds no console script — `python experiment.py` and
`python -m query_classification.experiment` are the two supported invocations, matching how
`classify.py` is invoked today.

#### AR-3.4: The runner owns classifier construction and numeric validation
Because `classify_csv` is only the row loop (AR-3.1), the experiment runner is responsible
for the setup `cli.py` performs today, applying the same rules:
- `load_dotenv(override=True)` and the litellm log quieting, as `cli.py` does.
- `check_default_resources_available()` before relying on any bundled default — and, per
  Spec 1's FR-1.7 precedent, unconditionally when the run needs the bundled induction or
  role prompts.
- Building the classification model + system prompt (with FR-3.5's `allow_new_labels`), and
  constructing the sampling/critic/reconciler `Classifier` instances per category exactly
  as Spec 1's `cli.py` wiring does.
- Resolving per-role model fallbacks (`--induction-model`/`--critic-model`/
  `--reconciler-model` each defaulting to `--model`).
- Numeric validation, extending `cli.py`'s existing rules (`sampling_runs >= 1`,
  `1 <= consensus_threshold <= sampling_runs`, finite non-negative `sampling_temperature`,
  `max_retries >= 1`) with this spec's: `--examples-per-label >= 1`,
  `--induction-examples >= 1` and mutually exclusive with `--examples-per-label` (FR-2.1),
  `--max-example-chars >= 1`, `--max-prompt-chars >= 1`, `--seed` a non-negative integer,
  `--test-limit >= 0`. All validation runs **before** any directory write, Hub load, or LLM
  call. FR-2.1's `--induction-examples >= n_labels` rule is the one exception to that
  ordering: it depends on the loaded label set, so it runs as soon as the labels are known
  and still before any LLM call.

---

## Data Requirements

**Run directory layout** (`--run-dir`):

| File | Written by | Contents |
|---|---|---|
| `run_config.json` | all subcommands (**last**) | Reproducibility/audit record (below), including terminal `status` |
| `categories.json` | all subcommands | Induced categories; or a verbatim copy of `--categories` |
| `train.csv` | `induce`, `run` (when inducing) | Materialized train split actually sampled from, needed columns only, post-FR-1.7 filtering |
| `test.csv` | `classify`, `run` | Materialized test split actually classified, needed columns only, post-FR-1.7 filtering and post-`--test-limit` truncation (whole split when omitted), gold column renamed per FR-3.6 |
| `test_classified.csv` | `classify`, `run` | `test.csv` plus prediction columns (and Spec 1 audit columns under `--critics`) |
| `induction_prompt.txt` | `induce`, `run` (when inducing) | The **rendered** system prompt + user message actually sent, for replay (FR-3.4) |
| `sampled_examples.json` | `induce`, `run` (when inducing) | Per-label manifest of the sampled source row positions and their (truncated) texts |

**`run_config.json` fields:**

| Field | Meaning |
|---|---|
| `schema_version` | Integer version of this record's shape, so later readers can migrate. Bumped to **4** by FR-2.1's sampling redesign: `induction_examples` is new, `examples_per_label` becomes nullable (null whenever the total-budget mode ran), and `induction_retries` — recorded only by the interim flag AR-2.3 removes — disappears. That is a real change to the persisted shape, not an additive one |
| `status` | `completed` \| `completed_with_failures` \| `failed` (FR-3.4, FR-3.8) |
| `subcommand` | `induce` \| `classify` \| `run` |
| `dataset` | Either `{source: "local", train_file, test_file}` or `{source: "hf", id, config, revision_requested, revision_resolved, train_split, test_split}` |
| `text_column`, `label_column`, `category_name` | Column/category selection |
| `seed`, `examples_per_label`, `induction_examples`, `max_example_chars`, `max_prompt_chars` | Induction sampling/bounding parameters (FR-2.1, FR-2.3). **When induction ran**, exactly one of `examples_per_label`/`induction_examples` is non-null, recording which sizing mode was used. For a `classify`-only run **both are null**, along with the other induction parameters — the whole induction flag group is absent from that subcommand, so there is no sizing mode to record |
| `models` | `{model, induction_model, critic_model, reconciler_model}` as resolved (after defaulting to `--model`) |
| `classifier_config` | `{critics, sampling_runs, sampling_temperature, consensus_threshold, allow_new_labels, retries, workers, test_limit}` |
| `prompt_inputs` | `{system_prompt, task_description, extra_prompt}` — the paths given, or null |
| `label_values` | The distinct gold label values found, **sorted** — pinned to sorted rather than discovery order so two identical runs produce byte-identical configs; the same sorted order groups labels in the induction prompt |
| `test_labels` | `"present"` \| `"withheld"` (FR-1.8) |
| `unseen_test_labels` | Test gold values absent from the induced train label set (FR-1.8) |
| `excluded_rows` | `{train: {count, positions}, test: {count, positions}}` per FR-1.7 |
| `unclassified_rows` | `{count, positions}` per FR-3.8 |
| `categories_source` | `"induced"` or `"supplied"` |
| `versions` | `{python, query_classification, litellm, pandas, pydantic, datasets}` as installed (FR-3.4) |

`run_config.json` records resolved model **ids** and flag values only — never API keys,
tokens, or the `api_base` value (see Constraints).

## Integration Points

- **New** `src/query_classification/dataset_io.py` — local/HF loading, split resolution,
  label normalization, row filtering, column projection, materialization (AR-1.1, AR-1.3).
- **New** `src/query_classification/induction.py` — sampling, prompt-size preflight,
  induction call, label-set reconciliation, `categories.json` writing (Feature 2).
- **New** `src/query_classification/experiment.py` — subcommand parser, classifier
  construction, validation, run-directory orchestration, completeness check (Feature 3).
- **New** root `experiment.py` — script entry point (AR-3.2).
- **New** `resources/prompts/induction_prompt.txt` (AR-2.2).
- `src/query_classification/schema.py` — gains the induction schema builder (AR-2.1).
- `src/query_classification/prompts.py` — gains the induction prompt builder (AR-2.2).
- `src/query_classification/resources.py` — gains the induction prompt path constant
  (AR-2.2); `check_default_resources_available()` itself is unchanged.
- `src/query_classification/pipeline.py` — **called, not modified** (AR-3.1).
- `pyproject.toml` — new `[project.optional-dependencies] hf = ["datasets>=4"]` (FR-1.5;
  the `>=4` floor is required, see that FR).
- `README.md` — experiment-runner usage, the `.[hf]` extra, run-directory layout, the
  plain-mode closed-vocabulary divergence, and the data-egress note.
- `spec/2-experiment-runner/ADR.md` — INV-1/INV-7 amendments (AR-1.2).

## Related Specs

| Spec | Relationship | Affected Requirements |
|------|-------------|---------------------|
| Spec 1: Multi-Agent Debate Classification with Critics | **Depends on** — the classification phase selects between plain and `--critics` mode and passes Spec 1's parameters through to `classify_csv`; Spec 1 must be implemented first (it is: CLOSED) | FR-3.5, FR-3.6, FR-3.8, AR-3.1, AR-3.4 |
| Spec 1: Multi-Agent Debate Classification with Critics | **Modifies** — the experiment runner deliberately diverges from the plain-mode `allow_new_labels` behavior Spec 1 established in `cli.py:201`, honoring the flag in both modes rather than forcing invention on outside critics mode | FR-3.5 |
| Spec 1: Multi-Agent Debate Classification with Critics | **References** — reuses its patterns: `Classifier`-as-is for a new LLM role (its AR-2.3), plain-string prompt builders to avoid new `prompts.py` import edges, per-call untrusted data in the user message, per-role model flags defaulting to `--model`, `AUDIT_COLUMN_SUFFIXES` for collision computation, and the ADR-for-invariant-change process (its AR-1.6/AR-1.7) | AR-1.2, AR-2.1, AR-2.2, AR-2.3, FR-3.6 |

## Constraints

- **No scoring.** The runner produces a classified test set with gold labels carried
  through; it computes no accuracy/F1/confusion metrics. Scoring is done externally against
  `test_classified.csv`.
- **Single label column, single category.** One `--label-column` produces one category, and
  a supplied `categories.json` must contain exactly one (FR-3.2).
- **Plain mode diverges from `classify.py`.** Per FR-3.5, the experiment runner honors
  `--allow-new-labels` in both classifier modes, whereas `classify.py` forces invention on
  outside `--critics`. This is intentional — comparability requires a closed vocabulary —
  and must be documented in `--help` and README so results are not assumed identical to a
  plain `classify.py` run.
- **No automatic train/test splitting.** A dataset without a train split can only be
  classified with a supplied `--categories` (FR-1.6); the runner never invents a split.
- **HF streaming is not supported** (AR-1.3) — splits are always materialized.
- **Data egress.** Sampled **train** text goes to the induction model; **test** text goes
  to the classification model, and additionally to the critic/reconciler models under
  `--critics`. With `--induction-model`/`--critic-model`/`--reconciler-model` these may be
  different providers, so a single run can send dataset content to several third parties.
  This must be stated in `--help`/README; the run config records which model id served each
  role so the routing is auditable after the fact.
- **Gated/private HF datasets** authenticate through the standard `datasets` mechanism —
  an `HF_TOKEN` environment variable (deprecated alias `HUGGING_FACE_HUB_TOKEN`) or the
  local Hub login cache. No token flag is added; no token value is ever written to
  `run_config.json` or logged. The runner does not attempt to determine whether a user is
  entitled to a gated dataset — a Hub authorization failure is surfaced sanitized.
- **`datasets>=4` prevents Hub *loading-script* execution, not all third-party code.**
  Data-file parsers, decompressors, and feature decoders still process
  attacker-controlled bytes; the floor removes the arbitrary-Python-from-a-dataset-repo
  vector specifically (FR-1.5).
- **Prediction-vs-gold comparability.** The classifier may still emit the `none` sentinel,
  which never appears in a gold label set; such rows simply won't match when scored
  externally. FR-2.5 prevents the inverse (a gold label named `none`).
- **`--critics` cost multiplier is inherited.** Per-row LLM volume is roughly
  `sampling_runs + contested_categories + challenged_categories` logical calls (Spec 1's
  Constraints) on top of `workers` rows in flight, so instantaneous concurrency approaches
  `workers × sampling_runs` — significant on a full benchmark. `--test-limit` is exposed
  (FR-3.5) for a cheap canary run first; the flag has no default, so a full run needs no
  special invocation, only the omission of `--test-limit`.
- **Non-atomic CSV writes are inherited, not fixed.** `classify_csv`'s incremental writes
  are not atomic (a documented Known Gap); a crash mid-write can truncate
  `test_classified.csv`. Run directories are always fresh (FR-3.4), so unlike an in-place
  `classify.py` run this can never damage an input dataset.
- **Induction quality is not guaranteed by this spec.** Whether induced descriptions beat
  hand-written ones is the empirical question this runner exists to let the user answer —
  not a requirement the implementation can be tested against.

## Out of Scope

- **Metrics/scoring of any kind** (accuracy, per-label P/R/F1, confusion matrices,
  significance testing) — explicitly deferred; a natural follow-up spec once the runner
  exists.
- **Multi-label or multi-category induction** (several label columns, or per-row label
  sets).
- **Automatic/seeded train-test splitting** of a single unsplit dataset.
- **Validation-split-driven iteration** — no prompt/threshold tuning loop, no
  early-stopping.
- **Streaming HF datasets** (`streaming=True`) and datasets too large to materialize; no
  disk-space or split-size preflight beyond the induction prompt budget (FR-2.3).
- **Non-CSV local inputs** (JSONL, parquet, arrow) — local files are CSV, matching what
  `classify_csv` already reads. HF datasets of any backing format are supported, since
  `datasets` materializes them.
- **A global concurrency cap or rate-limit backoff** across the nested sampling/row thread
  pools. Real, but explicitly Out of Scope in Spec 1 for the same reason: it is a
  cross-cutting change to the shared pipeline, not something an experiment wrapper should
  introduce unilaterally. The multiplier stays documented in Constraints.
- **CSV formula-injection sanitization** of the new artifacts. Pre-existing and tool-wide
  (Spec 1's Out of Scope); this spec restates the consumption warning — opening artifacts
  in Excel/Sheets can evaluate formula-like cells — but does not change CSV-writing
  behavior for one entry point only.
- **Recording an `api_base` fingerprint** for provider auditability. Deliberately still
  excluded: the value can carry credentials in userinfo/query, and a fingerprint adds a
  leak surface for marginal gain on a local tool. The egress warning and per-role model ids
  cover the auditability need.
- **Chunk-and-merge induction over all examples**, and **per-label or refinement-pass
  induction prompting** — bounded seeded sampling with one all-labels call only (FR-2.1,
  FR-2.2); both alternatives were considered and rejected during the spec interview.
- **Custom induction prompt template override** (no `--induction-prompt` flag), matching
  Spec 1's decision to ship bundled role templates only.
- **Cross-run comparison tooling** — no leaderboard, no run-diffing, no aggregation. Each
  run stands alone on disk.
- **Enumerated filesystem failure handling** (malformed CSV, disk full, permissions) —
  left to normal exception propagation.
- **Console-script entry point** (AR-3.3).

## Spec Completeness Checklist

- [x] **Scope & acceptance criteria** — FR-1.1 through FR-3.8 each carry a testable
  **Verify:** line, including the previously-undefined cases surfaced in critique
  (withheld/`-1` labels, `--test-limit` truncation, partial classification, overwrite semantics,
  classify-only artifact copying, degenerate splits). Out of Scope names thirteen
  considered-and-rejected directions, including the three deferred with explicit reasons
  (global concurrency cap, CSV sanitization, `api_base` fingerprinting).
- [x] **Testing strategy** — FR-3.7 requires a `pytest` suite covering every FR's Verify
  condition with the two external dependencies faked (a **signature-accurate** injected
  `datasets` module, and `Classifier`-shaped fakes), matching `AGENTS.md`'s binding "no live
  network/provider calls in tests" rule, and explicitly enumerating the error/edge cases so
  "every Verify condition" can't be satisfied while skipping them.
- [x] **Existing patterns** — AR-2.3 reuses `Classifier` as-is (Spec 1's precedent); AR-2.2
  follows the plain-string prompt-builder shape and the system-prompt/user-message split
  Spec 1 established; AR-3.1 calls `classify_csv` unmodified while AR-3.4 states explicitly
  what `classify_csv` does *not* provide (the correction that exposed FR-3.5's plain-mode
  contradiction); AR-2.1 sits alongside the three existing schema builders and now uses
  `build_classification_model`'s own `create_model` mechanism, differing only in that its
  dynamic field names are positional indices rather than caller-chosen names (dataset label
  values aren't valid identifiers); FR-3.6 reuses `debate.AUDIT_COLUMN_SUFFIXES`.
- [x] **Dependencies** — one new dependency (`datasets>=4`), justified by the
  user-requested HF support, scoped to an optional `[hf]` extra with lazy import (FR-1.5).
  Every API referenced (`load_dataset` with `name=`, `get_dataset_split_names` with
  `config_name=`, `Dataset.to_pandas`, `.features`, `ClassLabel.int2str`/`.names`,
  and `int2str`'s rejection of negative indices) was verified against the current
  `datasets` 4.8.4 documentation during spec-write, not assumed. Installed versions are
  captured per-run (FR-3.4).
- [x] **Architecture & interfaces** — Integration Points enumerates every touched and new
  file; AR-1.2 requires the ADR for the INV-1/INV-7 changes the three new modules force;
  AR-1.1 pins the `dataset_io.py` name to avoid shadowing the `datasets` package; AR-3.4
  assigns the construction/validation responsibilities that are not `classify_csv`'s.
- [x] **Error handling & failure modes** — FR-1.2 (id validation, missing split, mutual
  exclusion), FR-1.3 (unnameable/non-string labels, CSV coercion), FR-1.4 (missing columns,
  text==label), FR-1.5 (missing extra), FR-1.6 (five split-availability cases), FR-1.7
  (unusable rows, counted once), FR-1.8 (withheld and unseen labels), FR-2.3 (prompt-size
  preflight), FR-2.4 (missing/invented labels), FR-2.5 (reserved sentinel, incl. supplied
  files), FR-2.6 (induction failure writes nothing, sanitization scoped honestly), FR-2.7
  (degenerate splits), FR-3.2 (multi-category rejection), FR-3.4 (overwrite preflight,
  status written last), FR-3.6 (full collision set), FR-3.8 (partial classification
  detected, non-zero exit) each specify the behavior.
- [x] **Security review** — data egress across up to four model roles is disclosed and the
  per-role routing recorded (Constraints); HF tokens are never flagged, logged, or
  persisted, and the variable name is the documented `HF_TOKEN`; the remote-code claim is
  narrowed to Hub *loading scripts* with `datasets>=4` as the enforcing mechanism; the
  induction prompt must frame example texts **and** dataset-controlled label/category names
  as untrusted data with delimited serialization (AR-2.2); run directories carry only the
  needed columns rather than copying potentially-sensitive source fields (AR-1.3); FR-2.6's
  sanitization is scoped to persisted/CLI errors because `Classifier` logs raw text on
  retry (`classifier.py:120,129`) and AR-2.3 keeps it unmodified. CSV formula-injection
  exposure is restated as inherited, with the deferral reason given in Out of Scope.
- [x] **Performance impact** — induction is one bounded LLM call whose prompt is capped on
  all three axes (FR-2.1 count, FR-2.3 per-example chars **and** a total-size preflight),
  and FR-2.1's `--induction-examples` mode tightens the first axis from
  `labels × examples_per_label` to a flat `N` independent of the label count;
  classification cost is `classify_csv`'s existing profile with Spec 1's `--critics`
  multiplier and the `workers × sampling_runs` instantaneous-concurrency figure stated in
  Constraints, and `--test-limit` exposed for canary runs; AR-1.3's column projection before
  `to_pandas()` cuts the Arrow→pandas→CSV→pandas amplification. Whole-split materialization
  remains bounded by the no-streaming line in Out of Scope rather than by a mechanism, and
  a global concurrency cap is explicitly deferred there.
- [x] **Rollout & migration** — the original feature was purely additive: new entry point,
  new modules, new optional extra; `classify.py`/`cli.py`/`pipeline.py` behavior untouched
  (AR-3.1, AR-3.2), nothing to migrate. The one behavioral difference is the experiment
  runner's own plain-mode closed vocabulary (FR-3.5), scoped to the new entry point and
  documented in Constraints/`--help`/README. Run directories are always fresh and
  `--overwrite` touches only known artifact names (FR-3.4), so no prior run or unrelated
  file is at risk. The 2026-09-14 induction redesign is **not** purely additive and its
  three breaking edges are each bounded: `--induction-retries` is removed (AR-2.3) — it
  shipped the same day and was never released, so no documented invocation relies on it;
  `run_config.json`'s shape changes, handled by the `schema_version` bump to 4 (Data
  Requirements) rather than silently; and `induction_prompt.txt` plus the induction response
  schema change, so `induction_prompt.txt` replay artifacts from earlier runs describe a
  contract the current code no longer speaks. Previously written `categories.json` files are
  unaffected — that format is unchanged (FR-2.4) — so existing `classify`/`run`
  `--categories` invocations keep working.
- [x] **Assumptions & risks** — Constraints states the assumptions (single label column,
  materializable datasets, gold labels never `none`, `none`-prediction comparability) and
  the inherited risks (Spec 1 cost multiplier, non-atomic CSV writes). Critique-surfaced
  risks are now explicit too: withheld/mutable-revision datasets (FR-1.2, FR-1.8), CSV
  parsing lossiness (FR-1.3), incomplete classification (FR-3.8), and sensitive-column
  replication (AR-1.3). The headline risk is stated plainly: induction quality is the
  empirical question the runner exists to answer, and cannot be asserted as a requirement.

---

## Change Log

### Update from manual update (2026-09-09)

**Applied:**
- Renamed the classification-phase flag `--limit` to `--test-limit` (FR-3.5, AR-3.4),
  and the corresponding `run_config.json` field `classifier_config.limit` to
  `classifier_config.test_limit` — verified already implemented in
  `src/query_classification/experiment.py`, `tests/test_experiment.py`, and `README.md`.
  The rename is cosmetic only: the flag never had a default, and omitting it already
  classified the entire test split before and after this change. FR-3.5 and the Data
  Requirements table now state the no-default behavior explicitly (it was previously
  true but not called out), and explain the rename's motivation — disambiguating from
  `classify.py`'s own separate, pre-existing `--limit` flag on a different entry point,
  and making clear the flag bounds only the test/classification split, not induction
  sampling (which FR-2.1 controls independently via `--examples-per-label`/`--seed`).
- Updated every other `--limit` reference in this spec (FR-3.7, AR-3.4, Data
  Requirements, Constraints, Spec Completeness Checklist) to `--test-limit` for
  consistency.

**Rejected:** None — this update is a straightforward rename with no open design
questions.

**Reorganized:** None.

### Update: induction re-engineering — positional descriptions + total-budget sampling (2026-09-14, post-close)

Prompted by a real production failure: `experiments/pubmed-rct` runs (5 labels, 100
examples per label) repeatedly died with `induction response label set does not match the
dataset's labels (missing: ['conclusions'], unexpected/invented: [])`. Diagnosis, verified
against the code: the old AR-2.1 schema made the model **echo back label names** in an
open-ended `labels` list with no arity bound, so a dropped label passed Pydantic and then
hard-failed FR-2.4's post-hoc set comparison — even though the runner already knew the
exact label set before the call and had put it in the prompt itself.

**Applied:**
- **AR-2.1 rewritten** — the induction model is now built with `create_model` as
  `category_description` plus one **required** `description_1 … description_N` field per
  label position, carrying no label names. Label identity never round-trips through the
  response, so a missing or invented label became structurally impossible rather than
  something to detect afterwards.
- **FR-2.4 rewritten** — every `Label.value` now comes from the runner's own sorted label
  set, with the response mapped on **by position**. The label-set reconciliation step, and
  its Verify conditions (`a fake response omitting pos`, `a fake response adding an
  invented neutral`), are gone: neither response is expressible under the new schema. They
  were replaced with required-field-arity and positional-mapping conditions.
- **FR-2.1 rewritten** — added `--induction-examples N` (total budget across all labels,
  spread by largest-remainder so quotas sum to exactly `N`, with freed slots from
  row-scarce labels re-divided among labels that still have spare rows), **mutually
  exclusive** with `--examples-per-label`. Prompt volume is now boundable independently of
  the label count. Also stated: the `--induction-examples >= n_labels` floor (a smaller
  budget would give some label a zero quota and nothing to describe from, colliding with
  FR-2.7), that `--examples-per-label`'s argparse default must become `None` so
  "explicitly passed" is distinguishable from "left at default" (a default of `20` would
  otherwise collide with every `--induction-examples` invocation — the same mechanic spec
  4's AR-1.8 required of `--model`), and which reproducibility guarantee each mode keeps:
  both are seed-deterministic, but only `--examples-per-label` keeps cross-label
  independence, which `--induction-examples` gives up by construction since quotas depend
  on other labels' row counts.
- **AR-2.2 extended** — the user-message payload must carry labels as an ordered sequence
  whose entries state their own 1-based position, not a JSON object keyed by label value,
  so the positional contract doesn't rest on object-key ordering; and
  `induction_prompt.txt` must be updated, since its shipped text still closes by asking for
  `each entry containing that label's exact name`.
- **AR-2.3** — documented `--induction-retries` as shipped behavior (added earlier the same
  day as an interim fix: an `induce()`-level `max_reconciliation_retries` loop, default 3,
  re-sending the identical prompt on a reconciliation mismatch; validated `>= 1`, recorded
  in `run_config.json`, 5 tests), then **removed** it. It was itself a violation of
  AR-2.3's own "no new retry or LLM-call code is written" rule, and AR-2.1 eliminates the
  failure mode it existed for; keeping it would leave two overlapping retry layers where
  `Classifier`'s `--retries` now suffices.
- **`schema_version` bumped to 4** — `induction_examples` is new, `examples_per_label`
  becomes nullable, and `induction_retries` disappears. Recorded as a real persisted-shape
  change in Data Requirements, and the Rollout checklist item no longer claims this spec is
  "purely additive" — its three breaking edges are enumerated and bounded there instead.

**Rejected:**
- **Bounding the description list with `min_length == max_length == N`** (the mechanism
  originally proposed for Change A). Verified against the installed stack rather than
  assumed: `litellm.utils.type_to_response_format_param` serializes a Pydantic
  `response_format` with `"strict": true`, and strict structured output does not support
  `minItems`/`maxItems` — the provider would reject the schema and
  `Classifier._attempt_completion`'s `except litellm.BadRequestError` would silently
  degrade **every** induction call to JSON-object mode, wasting a round-trip and losing
  strict-mode guarantees permanently. Required fields express exactly-`N` in a form the
  strict path accepts, which is why AR-2.1 uses them.
- **Splitting FR-2.1 into separate per-label and total-budget requirements.** The two are
  mutually exclusive answers to one question (how many examples each label contributes), so
  one requirement is the "one truth per requirement" reading; splitting would also have
  renumbered FR-2.2–FR-2.7 and orphaned the FR-2.2 cross-reference inside FR-2.3, for no
  gain in testability.

**Also corrected (spec/code drift found while verifying the above, unrelated to the
redesign):**
- FR-2.1 claimed `--examples-per-label` defaults to **50**; the implemented default is
  **20**. Fixed, including the Verify clause that used `--examples-per-label 50`.
- FR-2.3 claimed `--max-example-chars` defaults to **1000**; the implemented default is
  **500**. Fixed, including the Verify clause's truncation figure. (This second drift was
  not in the update request — found by checking every stated default against
  `build_parser`.)
- FR-2.3 never stated `--max-prompt-chars`'s default; it is **100000**. Added.

**Reorganized:** None — every change landed in the requirement that already owned the
behavior (FR-2.1 sampling, FR-2.4 output/reconciliation, AR-2.1 schema, AR-2.2 prompt,
AR-2.3 retry policy). No FR/AR was added, removed, or renumbered.
