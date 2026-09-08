# Implementation Plan: 2-experiment-runner

**Status:** Ready
**Date:** 2026-09-08
**Spec:** spec/2-experiment-runner/spec.md
**Plan Critique:** spec/2-experiment-runner/plan-critique-consolidated-v-1.md

## Overview

Build bottom-up in four independent streams — dataset loading, schema/prompt builders,
induction orchestration, and the architecture ADR — then integrate everything in a new
`experiment.py` orchestrator (the only task that ties the streams together and therefore
the only place classifier construction, run-directory management, and the plain-mode
closed-vocabulary fix live), then ship the test suite last.

## Readiness

- Checklist: all 10 items closed `[x]` in the spec — no open gaps.
- Open questions: none blocking. Two implementation-level defaults the spec deliberately
  left unpinned are fixed here rather than re-opening the spec: `--max-prompt-chars`
  defaults to `100000` (roughly 25k tokens at ~4 chars/token — generous headroom below a
  128k-token context window while still catching runaway label counts), and the induction
  user-message wire format is a JSON object between fixed delimiters (Task 3).
- External API/library verification: carried over from spec-write — every `datasets` API
  referenced (`load_dataset(path, name=, revision=, split=)`, `Dataset.to_pandas()`,
  `.features`, `.column_names`, `ClassLabel.int2str`/`.names`) was confirmed against the
  `datasets` 4.8.4 docs, and the `datasets>=4` floor's security rationale (dataset loading
  scripts removed in 4.0.0) was confirmed via search. No new API surface was introduced
  during critique revision, so no re-verification needed.
- Repository state: clean except two pre-existing, unrelated untracked files
  (`data/example_queries_classified.csv`, `spec/initial-classification-with-critics-spec.md`)
  and the `spec/2-experiment-runner/` directory itself — confirmed via `git status`
  immediately before writing this plan. Leave both untouched.

## Code Impact

- **Modules:** `src/query_classification/` gains three new modules
  (`dataset_io.py`, `induction.py`, `experiment.py`); `schema.py`, `prompts.py`,
  `resources.py` extended; `pipeline.py`, `classifier.py`, `categories.py`, `cli.py`,
  `debate.py` **read, not modified** (AR-3.1 and its siblings).
- **Database/schema:** none. New on-disk artifact: the run directory (Data Requirements),
  and one new bundled prompt template.
- **API/interfaces:** two new CLI entry points (`experiment.py` script,
  `python -m query_classification.experiment`) with three subcommands
  (`induce`/`classify`/`run`); no change to `classify.py`/`cli.py`'s existing flags.
- **UI/config/background jobs:** none.
- **Key files:**
  - `src/query_classification/dataset_io.py` (new)
  - `src/query_classification/induction.py` (new)
  - `src/query_classification/experiment.py` (new)
  - `experiment.py` (repo root, new)
  - `src/query_classification/schema.py` (modify — `build_induction_model`)
  - `src/query_classification/prompts.py` (modify — `build_induction_prompt`)
  - `src/query_classification/resources.py` (modify — induction prompt path constant)
  - `resources/prompts/induction_prompt.txt` (new)
  - `pyproject.toml` (modify — `[project.optional-dependencies] hf`)
  - `README.md` (modify)
  - `spec/2-experiment-runner/ADR.md` (new)
  - `tests/test_experiment.py` (new)

## Project Constraints

- **INV-1** (dependency graph) is being *revised*, not violated silently — Task 5 writes
  the ADR before the new modules land; `spec/ARCHITECTURE.md` itself is updated later by
  `/spec-close`, matching Spec 1's precedent. The revision makes `experiment.py` a
  **second** full-access orchestrator alongside `cli.py` (today's INV-1 says `cli.py` is
  the *only* module allowed to import all others) — `experiment.py` needs the same breadth
  because it duplicates `cli.py`'s classifier-construction responsibility (AR-3.4)
  end-to-end for its own three roles plus the new induction role. `cli.py` and
  `experiment.py` do not import each other.
- **INV-2** (bundled resource paths centralized in `resources.py`) — the new induction
  prompt path is a constant there, not computed inline (Task 2).
- **INV-3** (wheel installs lack `resources/`) — `experiment.py` must call
  `check_default_resources_available()` before relying on any bundled default, mirroring
  `cli.py:179-185`'s `uses_bundled_defaults` pattern but covering *four* bundled templates
  (system, critic, reconciler, induction) instead of one, per AR-3.4's "unconditionally
  when the run needs bundled prompts" requirement.
- **INV-4** (broad `except Exception` needs `# noqa: BLE001` + inline reason) — applies to
  `dataset_io.py`'s HF-loading error handling and `induction.py`'s sanitized-failure catch
  (Task 3).
- **INV-7** (test import boundaries) — Task 5's ADR must declare `dataset_io.py`,
  `induction.py`, `experiment.py`, and the two new `schema.py`/`prompts.py` functions as
  accepted direct-import exceptions, extending Spec 1's precedent (`debate.py` +
  `build_critic_model`/etc.) rather than replacing it.
- **INV-11** (schema enforces cardinality/type only, not semantics) — directly motivates
  AR-2.1's fixed-list induction schema and FR-2.4's in-code label-set reconciliation
  (Task 2/3): Pydantic's default extra-field behavior silently drops unknown keys, so a
  dynamic-field or dict-shaped response could hide an invented label.
- **AGENTS.md "Required patterns"** — `from __future__ import annotations` + PEP 604
  unions in every new module.
- **AGENTS.md "Testing policy"** — no live network/provider calls; HF loading is faked via
  an injected `datasets` module with real function signatures (Task 6), never a real Hub
  call.

## Implementation Strategy

- **Size:** large (3 features, 23 FRs, 10 ARs, 3 new modules, 1 architecture change).
- **Execution mode:** 4 parallel streams for Tasks 1/2/3/5, then a sequential chain
  (Task 4 → Task 6).
- **Parallelizable work:** Tasks 1 (`dataset_io.py`), 2 (`schema.py`/`prompts.py`/
  `resources.py`/template), 3 (`induction.py`), and 5 (ADR) touch disjoint files **and**
  have no functional dependency on each other — `induction.py` receives an
  already-constructed `Classifier` and an already-loaded DataFrame as plain arguments
  (mirroring Spec 1's `debate.py`, which needed neither `schema.py` nor `prompts.py`
  directly despite its ADR authorizing both), so Task 3 doesn't need Task 2's code to
  exist, only Task 2's *design* (already fixed by this plan).
- **Sequential blockers:** Task 4 needs all of 1+2+3 (it is the only module that
  constructs classifiers using Task 2's builders, loads data via Task 1, and calls
  Task 3). Task 6 needs Task 4 (exercises the fully-wired CLI-to-everything path).

## Implementation Tasks

### Task 1: `dataset_io.py` — local and HuggingFace dataset loading
**Goal:** Load a train/test split pair from either a local CSV pair or a HF dataset id,
normalized to plain DataFrames with string labels, filtered of unusable rows, and
projected to only the needed columns.
**Files:** `src/query_classification/dataset_io.py` (new), `pyproject.toml`
**Dependencies:** None
**Do:**
- Add `[project.optional-dependencies] hf = ["datasets>=4"]` to `pyproject.toml`
  (FR-1.5), alongside the existing `dev` extra.
- `validate_hf_id(hf_id: str) -> None`: import `huggingface_hub.utils.validate_repo_id`
  (already a transitive dependency of `datasets>=4`, so no new package) and call it,
  translating `huggingface_hub.errors.HFValidationError` into this module's own
  actionable `ValueError`. This is real Hub-ID syntax validation (`owner/name` or a
  canonical single-segment name, length/character/`.git`/`--`/`..` rules) — a
  `Path.exists()` + hardcoded-builder-blacklist check was rejected during plan critique
  as accepting invalid-looking-but-syntactically-valid values (`./missing`, URLs, `foo.git`)
  and drifting across `datasets` releases. **Separately**, still reject the packaged
  builder names (`{"csv","json","parquet","arrow","text","xml","webdataset",
  "imagefolder","audiofolder","videofolder"}`) even though they pass `validate_repo_id` —
  `csv` etc. are syntactically valid single-segment ids but resolve to a local builder,
  not a Hub dataset, silently lying about `source: "hf"` (FR-1.2).
- `load_hf_splits(hf_id, config, revision, train_split, test_split) ->
  tuple[pd.DataFrame | None, pd.DataFrame | None, dict]`: import `datasets` **inside this
  function**, catching `ModuleNotFoundError` narrowly and re-raising with the exact
  `pip install '.[hf]'` hint (FR-1.5) — this must be a concrete bullet here, not just a
  Constraints mention, since the constraint alone left it unimplemented in the prior
  draft. Call `datasets.load_dataset(hf_id, name=config, revision=revision)` **without**
  `split=` so it returns one `DatasetDict` — check membership of `train_split`/
  `test_split` in its keys directly, raising an error listing `list(dd.keys())` if either
  requested-but-present-flag split is missing, rather than a second
  `get_dataset_split_names` call (FR-1.2). **Known, accepted cost:** `load_dataset(...,
  split=None)` materializes every split in the `DatasetDict` regardless of which the
  subcommand needs (verified against the `datasets` 4.8.4 source during critique — the
  prior draft's "only fetch a split if needed" framing was incorrect, since the fetch
  already happened by the time any key is looked up); documented here and in the README
  rather than worked around, since a builder/metadata-only split-name resolution would
  need a second remote call and the spec already prefers "a single load where possible."
  Wrap the load call itself in a broad `except Exception as e: # noqa: BLE001 - HF
  load/auth failures may carry tokens or URLs in their message` and re-raise
  `f"{type(e).__name__}: HF dataset load failed"` only — never `str(e)` — since Hub auth
  failures can embed credentials (Constraints' sanitization requirement, distinct from the
  locally-constructed missing-split error above, which keeps its full, useful message).
  **Resolved revision**, separately: after the load succeeds, best-effort call
  `huggingface_hub.HfApi().dataset_info(hf_id, revision=revision).sha` in its own
  try/except — `load_dataset`'s return value is a `DatasetDict`, not a builder, and
  `DatasetInfo` has no public commit-SHA field, so there is no way to derive this from the
  already-loaded object (the prior draft's "from the builder's info" claim was checked and
  is not a real API). On any failure (network, auth, unresolvable revision) record the
  requested `revision` string (or `null` if none was given) instead of raising — this
  metadata call must never fail the run.
  Also return `source_column_names: list[str] = dataset.column_names` for each fetched
  split, captured **before** the projection step below — needed by Task 4's collision
  check (see Task 4's Do bullets; this is the plan-level resolution to the AR-1.3/FR-3.6
  interaction flagged in plan critique, not a spec change: only column *names*, never the
  projected-out columns' *values*, are carried forward).
- `resolve_hf_labels(dataset: "Dataset", label_column: str) -> list[str | None]`: read
  `dataset.features[label_column]` and `dataset[label_column]` (the raw per-row values, via
  item access — independent of whatever `to_pandas()` does with a `ClassLabel` column
  internally, which was not verified and is deliberately not relied on). If the feature is
  `datasets.ClassLabel`: map each value via `.int2str(v)` for `v >= 0`, else `None` (the
  withheld/`-1` sentinel, FR-1.8) — never call `int2str` on a negative index, since it
  raises. If the feature is `datasets.Value` with a string dtype: use the raw values
  verbatim. Otherwise: raise naming the feature's actual type and dtype (FR-1.3). Returns a
  plain Python list, same length/order as the dataset. **Mixed valid/withheld rows**
  (some `-1`, some real values, in the same split) are not called out as a separate case
  in the spec, but need no separate handling here either — this function is already
  row-level (`list[str | None]`), so a mixed split just produces a list with some `None`
  entries and some strings, exactly as intended; see Task 4 for how the dataset-level
  `test_labels` field is derived from this per-row list.
- **HF assembly order** (this is the one place `to_pandas()`'s `ClassLabel` representation
  would matter, so the order is chosen to make it not matter): (1) call
  `resolve_hf_labels` first, while `dataset.features` is still available; (2) capture
  `source_column_names` from `dataset.column_names`; (3) project columns via
  `dataset.remove_columns([c for c in dataset.column_names if c not in
  {text_column, label_column}])` (AR-1.3); (4) `to_pandas()`; (5) **overwrite** the
  resulting DataFrame's label column with step (1)'s precomputed list (guaranteed same
  row order/length, since no filtering has happened yet) — this makes the final label
  column correct regardless of whether `to_pandas()` would have given raw ints or
  already-mapped names for a `ClassLabel` column.
- `load_local_split(path: str, text_column: str, label_column: str | None) ->
  tuple[pd.DataFrame, list[str]]`: `pd.read_csv(path, dtype=str, keep_default_na=False,
  na_values=[])` — the **whole file**, not just the label column, read with all NA-sniffing
  disabled and every value as a raw string (a prior draft's `keep_default_na=False`/
  `na_values=[""]` were file-wide pandas options being described as if they were
  label-column-specific, which was wrong as written; disabling them file-wide and applying
  FR-1.7's own missing-value predicate uniformly during filtering — see
  `project_and_filter` below — needs no per-column special case and is simpler than the
  prior draft, not just corrected). This is what makes FR-1.3's "verbatim" claim true:
  `"001"`/`"NA"`/`"None"` survive unchanged in any column. Also returns
  `source_column_names = list(df.columns)` (the full header, before any column is
  dropped) for the same Task 4 collision-check reason as the HF path above.
- `project_and_filter(df: pd.DataFrame, text_column: str, label_column: str | None) ->
  tuple[pd.DataFrame, list[int]]`: operates on an **already-normalized** frame (post the
  HF assembly order above, or straight from `load_local_split` for the local case) — no
  separate "gold labels" input. The one missing-value predicate (used everywhere in this
  module, including for local CSVs now that NA-sniffing is off): null (`None`/`NaN`/
  `pd.NA`) **or** the stripped string form is empty. Drops rows where `text_column` fails
  that predicate, **and** (only when `label_column` is given and this is the train role)
  where the label fails it — matching FR-1.7's count-once-per-row rule (a single mask
  unioned across both columns, not two separate drops). Returns the filtered frame and the
  **0-based positions in the input frame** that were dropped (`df.index[mask].tolist()`
  before any reset, since both the local CSV read and the HF assembly above start from a
  fresh 0-based `RangeIndex`).
**Verify:** a fake `datasets` module (see Task 6) proves `load_hf_splits` never calls
`get_dataset_split_names` when the single `DatasetDict` load already contains the answer,
and that a resolved-SHA lookup failure doesn't raise; `validate_hf_id` rejects
`./missing`, an absolute path, a `.git`-suffixed id, and a packaged builder name, while
accepting a valid single-segment and two-segment id; `resolve_hf_labels` on a
`ClassLabel([-1, 0, 1])`-valued column returns `[None, "neg", "pos"]` without raising, and
the same holds when only some rows are `-1` (mixed case); `load_local_split` on a CSV with
label values `001,002` returns the strings `"001"`,`"002"` unchanged, and a text column
containing literal `"NA"` is preserved rather than coerced to null; `project_and_filter` on
a 2-empty-text-row train frame returns a frame 2 rows shorter and the 2 dropped positions;
`source_column_names` includes a column that gets projected away.
**Covers:** FR-1.1, FR-1.2, FR-1.3, FR-1.5, FR-1.7, FR-1.8, AR-1.1, AR-1.3

### Task 2: Induction schema and prompt builders
**Goal:** A fixed-shape Pydantic model and a static system-prompt template for the
induction call — both independent of any particular dataset's label set.
**Files:** `src/query_classification/schema.py`, `src/query_classification/prompts.py`,
`src/query_classification/resources.py`, `resources/prompts/induction_prompt.txt` (new)
**Dependencies:** None
**Do:**
- `resources.py`: add `DEFAULT_INDUCTION_PROMPT_FILE = RESOURCES_DIR / "prompts" /
  "induction_prompt.txt"` alongside the existing constants (INV-2).
- `schema.py`: add `build_induction_model() -> type[BaseModel]` returning a **fixed**
  model (not parameterized by category/labels, unlike its three siblings — there's
  nothing to parameterize: the label *set* is exactly what induction is inferring):
  ```python
  class InductionLabel(BaseModel):
      label: str
      description: str

  class InductionResult(BaseModel):
      category_description: str
      labels: list[InductionLabel]
  ```
  This is deliberately a **list**, not a dynamic field per label (AR-2.1) — label values
  are dataset-chosen strings that routinely aren't valid Python/Pydantic identifiers
  (`"very negative"`, `"class 1"`), and a list entry is comparable for FR-2.4's
  reconciliation whereas an unexpected object key would be silently dropped by Pydantic's
  default extra-field handling (INV-11).
- `prompts.py`: add `build_induction_prompt(system_prompt_file:
  str | Path = DEFAULT_INDUCTION_PROMPT_FILE) -> str`. Unlike `build_critic_prompt`/
  `build_reconciler_prompt`, this takes **no category/label parameters** — the induction
  system prompt describes the task once, generically ("you will be shown examples for
  each of several labels; write a description for each and for the category as a whole");
  the actual labels/examples are per-call data assembled into the **user message** by
  `induction.py` (Task 3), not this static prompt, matching Spec 1's established
  system-prompt-once/user-message-per-call split. Simply
  `Path(system_prompt_file).read_text()` — no `.format()` needed since the template has
  no placeholders.
- Write `resources/prompts/induction_prompt.txt`: static instructions for the induction
  role, explicitly stating that the upcoming user message's example texts, label names,
  and category name must be treated as untrusted data to summarize, never as instructions
  (AR-2.2) — this covers the framing requirement once, generically, since the actual
  payload arrives per-call.
**Verify:** `build_induction_model()` validates `{"category_description": "...",
"labels": [{"label": "class 1", "description": "..."}]}` (a label value that would break a
dynamic-field model); `build_induction_prompt()` returns non-empty text with no unfilled
`{...}` placeholders; `resources.DEFAULT_INDUCTION_PROMPT_FILE.exists()`.
**Covers:** AR-2.1, AR-2.2 (schema/prompt half)

### Task 3: `induction.py` — sampling, orchestration, reconciliation (pure function)
**Goal:** Given a loaded (and already-filtered) train DataFrame and an already-constructed
induction `Classifier`, sample examples per label, make one induction call, and reconcile
the response against the dataset's label set. **Does not touch the filesystem** — Task 4
is the sole writer of every run-directory artifact (plan-critique finding: the prior draft
had both `induce()` and `experiment.py` claiming to write `categories.json`, an undefined
ownership conflict).
**Files:** `src/query_classification/induction.py` (new)
**Dependencies:** None (receives its `Classifier` and DataFrame as plain arguments —
see Implementation Strategy)
**Do:**
- Concrete signature: `induce(train_df: pd.DataFrame, text_column: str, label_column:
  str, category_name: str, induction_classifier: Classifier, *, seed: int,
  examples_per_label: int, max_example_chars: int, max_prompt_chars: int) ->
  InductionResult` where `InductionResult` is a small plain dataclass/NamedTuple (not the
  Pydantic `InductionResult` schema from Task 2 — name it `InductionOutcome` to avoid
  confusion) with fields: `category: Category` (the constructed, not-yet-written
  `categories.py` object — importing `Category`/`Label` from `categories.py`, the only
  internal import this module needs besides `classifier.py` for the type hint, mirroring
  `debate.py`'s minimal-import precedent), `rendered_prompt: str` (the exact system prompt
  + user message text sent, for Task 4 to write verbatim to `induction_prompt.txt`), and
  `sampled_examples: dict` (the per-label manifest of sampled row positions + truncated
  texts, for Task 4 to write verbatim to `sampled_examples.json`). Task 4 owns
  `categories.json`'s writing and the `load_categories` round-trip validation of what
  actually landed on disk.
- **Degenerate-input checks first** (FR-2.7), before any sampling: empty `train_df` after
  filtering, or zero distinct label values → raise naming the condition. A single distinct
  label is allowed.
- **Reserved-sentinel check** (FR-2.5): raise if any distinct label value, stripped and
  lowercased, equals `"none"` or starts with `"none - "`.
- **Sampling** (FR-2.1): sort distinct label values. Instantiate **exactly one**
  `rng = random.Random(seed)` **before** the loop over sorted labels — this is a direct
  fix of a plan-critique blocking finding: the prior draft used
  `random.Random(f"{seed}:{i}")`, a fresh generator per label, which contradicts FR-2.1's
  literal text ("a single named, seeded generator") and would select different rows for
  the same seed than the spec's pinned algorithm. For each label in sorted order: group its
  row positions; if more than `examples_per_label`, call `rng.sample(stable_positions,
  examples_per_label)` (consuming RNG state), else use all positions in stable order
  (consuming no RNG state) — so a label at or below its cap never perturbs a later label's
  draw. Truncate each selected example's text to `max_example_chars` Unicode code points,
  appending a fixed `"…[truncated]"` marker **outside** the cap (FR-2.3).
- **User message assembly** (do this before the preflight below, so the preflight measures
  the real payload): a JSON object between fixed delimiters, e.g.
  ```
  <<<DATA>>>
  {"category_name": "...", "labels": {"<label>": ["<example>", ...], ...}}
  <<<END DATA>>>
  ```
  with labels in sorted order.
- **Prompt-size preflight** (FR-2.3): measure `len()` of the **actual serialized user
  message string** built above (delimiters, JSON escaping, keys, category name, truncation
  markers — everything, not just the raw example/label text) plus the static system
  prompt's length; if the total exceeds `max_prompt_chars`, raise naming the total and
  which of `--examples-per-label`/`--max-example-chars` to lower — before calling the
  classifier. (Plan-critique fix: summing only example+label text undercounts JSON
  overhead and can let a payload exceed budget after the naive check passed.)
- Pass the serialized user message as the sole `text` argument to
  `induction_classifier.classify(...)` (AR-2.3; no new retry/call code, reusing
  `Classifier` exactly as constructed by `experiment.py`, Task 4). **`Classifier.classify()`
  returns a plain `dict[str, Any]`** (its actual signature, `classifier.py:100` —
  validated-then-dumped, not a Pydantic instance), so every access below is by dict key,
  matching `debate.py`'s established pattern (`critic_verdict["challenges"]`, not
  attribute access) — a prior draft of this task incorrectly used attribute access
  (`result.labels`, `entry.label`), caught during this plan-critique pass by re-checking
  the real `Classifier` signature.
- **Reconciliation** (FR-2.4): let `expected = set of distinct label values`,
  `returned = [entry["label"] for entry in result["labels"]]`. Raise if
  `set(returned) != expected` (missing or invented labels, naming which) **or** if
  `len(returned) != len(expected)` (plan-critique fix: set equality alone accepts a
  duplicate like `[neg, neg, pos]` when `pos` and `neg` are the only two expected labels —
  the length check catches this even though the set matches).
- Construct `Category(name=category_name, description=result["category_description"],
  labels=[Label(value=entry["label"], description=entry["description"]) for entry in
  result["labels"]])` and return it as part of `InductionOutcome` — do **not** write or
  round-trip it here; Task 4 does both.
- **Failure handling** (FR-2.6): wrap the `.classify()` call in `except Exception as e:
  # noqa: BLE001 - sanitized and re-raised as the induction failure` (INV-4), re-raising a
  new exception carrying only `f"{type(e).__name__}: induction call failed"` — never the
  raw `str(e)`. Because this function never writes to disk, "no `categories.json` written
  on failure" is automatically true — it's a property of the pure-function boundary, not a
  separate guard to implement.
**Verify:** every FR-2.x Verify condition in the spec, exercised directly against
`induce()` with a fake `Classifier`-shaped object (`.classify(text) -> dict`) — reusing
Spec 1's fake-object pattern, including a fixture with at least two over-cap labels to lock
down the exact single-generator sample sequence, and a duplicate-label response (`[neg,
neg, pos]` against expected `{neg, pos}`) to confirm reconciliation rejects it despite
matching set equality. This is ad-hoc/local verification while writing this task; Task 6
formalizes it into the committed suite.
**Covers:** FR-2.1, FR-2.2, FR-2.3, FR-2.4, FR-2.5, FR-2.6, FR-2.7, AR-2.3

### Task 4: `experiment.py` — orchestration, classifier construction, run directory
**Goal:** The subcommand CLI that ties Tasks 1-3 together: constructs every classifier
role, loads data, runs induction and/or classification, manages the run directory, and
detects incomplete classification.
**Files:** `src/query_classification/experiment.py` (new), `experiment.py` (repo root,
new), `README.md`
**Dependencies:** Task 1, Task 2, Task 3
**Do:**
- Root `experiment.py`: `sys.path.insert(0, str(Path(__file__).resolve().parent /
  "src")); from query_classification.experiment import main; main()` — identical pattern
  to `classify.py:13-15` (AR-3.2). Add `python -m query_classification.experiment` support
  the same way `__main__.py` does for `cli.py` (a two-line `__main__.py`-style module, or a
  guarded `if __name__ == "__main__": main()` at the bottom of `experiment.py` itself if a
  separate `__main__.py` isn't warranted for a package that's really one module — decide
  during implementation; either satisfies AR-3.2, no behavior difference).
- **Argument parsing — three explicit flag groups, not two** (plan-critique fix: the prior
  draft attached the induction group only to `induce`, but `run` performs induction too,
  and put `--model` only on `classify`/`run`, breaking `induce --induction-model`'s
  fallback to it). One top-level parser with `add_subparsers(dest="subcommand",
  required=True)` for `induce`/`classify`/`run`. Three parent parsers via argparse's
  `parents=` mechanism:
  - **common** (all three subcommands): `--train-file`/`--test-file`/`--hf-dataset`/
    `--hf-config`/`--hf-revision`/`--hf-train-split`/`--hf-test-split`/`--text-column`/
    `--label-column`/`--category-name`/`--categories`/`--run-dir`/`--overwrite`/`--model`
    (the shared base model, needed by every subcommand for its own role's fallback).
  - **induction** (`induce` and `run`): `--seed`/`--examples-per-label`/
    `--max-example-chars`/`--max-prompt-chars`/`--induction-model`.
  - **classification** (`classify` and `run`): `--api-base`/`--retries`/`--workers`/
    `--limit`/`--system-prompt`/`--task-description`/`--extra-prompt`/`--critics`/
    `--sampling-runs`/`--sampling-temperature`/`--consensus-threshold`/
    `--allow-new-labels`/`--critic-model`/`--reconciler-model`/`--allow-partial`.
  `induce = parents=[common, induction]`; `classify = parents=[common,
  classification]`; `run = parents=[common, induction, classification]`.
- **Mode/split validation up front** (FR-1.1, FR-1.2, FR-1.6, FR-1.4), before any load:
  `--hf-dataset` XOR (`--train-file` or `--test-file`); the five split-availability cases;
  `--text-column != --label-column`; a supplied `--categories` file has exactly one
  category (FR-3.2) and no label value trips FR-2.5's sentinel check (reuse the check from
  Task 3, factored so both call sites share it rather than duplicating the string
  comparison).
- **Numeric validation** (AR-3.4): extend `cli.py`'s existing checks (`sampling_runs >= 1`,
  threshold range, finite non-negative `sampling_temperature`, `max_retries >= 1`) with
  `examples_per_label >= 1`, `max_example_chars >= 1`, `max_prompt_chars >= 1`, `seed >= 0`,
  `limit >= 0` (when given) — all before any directory write, Hub load, or LLM call.
- **Resource guard**: call `check_default_resources_available()` whenever the run would
  rely on *any* bundled default — the main categories/system-prompt defaults (mirroring
  `cli.py:179-185`'s `uses_bundled_defaults` condition) **or** the bundled critic/
  reconciler/induction prompts (always bundled, no override flags exist) whenever
  `--critics` or induction is active respectively.
- **Classifier/model construction** — the responsibility `classify_csv` doesn't provide
  (AR-3.1's correction, AR-3.4): build the classification model and system prompt with
  `allow_new_labels=args.allow_new_labels` (**not** forced `True` the way
  `cli.py:201`/`pipeline.py`'s plain-mode path does — this is FR-3.5's deliberate
  divergence, applied in *both* classifier modes here, unlike `cli.py`). Construct the
  classification `Classifier` (temperature set only when `--critics`), and, when
  `--critics` is set, the per-category `critic_classifiers`/`reconciler_classifiers` dicts
  — following `cli.py`'s existing construction exactly (same `build_critic_model`/
  `build_critic_prompt`/`build_reconciler_model`/`build_reconciler_prompt` calls, same
  `--critic-model or --model` fallback). Separately, when induction runs, construct one
  induction `Classifier` using `build_induction_model()`/`build_induction_prompt()` (Task
  2) and `--induction-model or --model`. Call `load_dotenv(override=True)` and duplicate
  `cli.py`'s tiny `_quiet_logging()` body directly (4 lines) rather than importing a
  private function across modules — a deliberate, small duplication, not an oversight.
- **`run` / `induce` / `classify` dispatch**: `induce` calls Task 1's loaders for the train
  split only, then Task 3's `induce(...)` (a pure function — Task 4 itself writes the
  returned `InductionOutcome`'s three artifacts: `categories.json`, `induction_prompt.txt`,
  `sampled_examples.json`, per the run-directory-writing steps below). `classify` loads
  the test split only, requires `--categories`, and calls `classify_csv` (AR-3.1) with the
  constructed classifiers. `run` does both in sequence, using the freshly induced
  categories unless `--categories` was given (in which case `run` behaves like `classify`,
  per FR-1.6's train-and-categories-both-given error already ruling out the ambiguous
  case).
- **`--limit` truncation** (FR-3.5): when `--limit N` is given, slice the materialized
  `test.csv` DataFrame to its first `N` rows **before** writing `test.csv` and before
  calling `classify_csv` — do not pass `limit=` through to `classify_csv` itself, since
  its own `limit` slicing (`pipeline.py:171`) operates after column creation and would
  leave a null-prediction tail in the file this spec's `test.csv` is supposed to represent
  exactly.
- **Collision check** (FR-3.6), checked against `source_column_names` (Task 1's
  pre-projection metadata), **not** the materialized test frame's columns — plan-critique
  fix: AR-1.3 projects the test frame down to only `text_column`/`label_column` before
  Task 4 ever sees it, so a source column literally named `{label_column}_gold` or
  `{category_name}_votes` would be invisible to a check that only looks at the
  materialized frame, even with the ordering below already correct. In this exact order
  (reversing it would let a pre-existing `{label_column}_gold` column be silently
  clobbered by the rename instead of rejected):
  1. If `{label_column}_gold` is present in `source_column_names` **excluding**
     `label_column` itself (which is about to be renamed away), error — the "dataset that
     already has a `label_gold` column" collision case.
  2. Only then rename `{label_column}` → `{label_column}_gold` on the materialized frame.
  3. Compute the full generated column set independently — `{category_name}` plus, when
     `--critics`, `{f"{category_name}{suffix}" for suffix in
     debate.AUDIT_COLUMN_SUFFIXES}` (the **public** constant; do not reach into
     `pipeline.py`'s private `_audit_columns` helper) — and reject any intersection with
     `source_column_names` (excluding `label_column`, already handled by the rename).
  4. Reject `--text-column == --label-column` up front, before either check above runs
     (both would otherwise silently reference the same source column).
- **Run directory writing** (FR-3.4) — Task 4 is now the **sole writer** of every
  artifact, since Task 3 was changed to a pure function (no filesystem access) to resolve
  the artifact-ownership conflict a prior plan-critique round found:
  1. Preflight the empty/`--overwrite` check (directory must be empty, or `--overwrite`
     given) before any file is touched.
  2. **Before any `--overwrite` cleanup deletes anything**, resolve and read/validate every
     supplied input path (`--categories`, `--train-file`, `--test-file`): load and check a
     supplied `--categories` file (single category, no reserved-sentinel labels) and, if
     it resolves to the same path as `run_dir / "categories.json"`, remember that as a
     same-path no-op rather than reading it a second time later. Plan-critique fix: the
     prior draft's overwrite step deleted known artifact filenames — including
     `categories.json` — before this validation, so `--categories <run-dir>/categories.json
     --overwrite` would have deleted its own source before it could be read.
  3. Run `--overwrite`'s cleanup: delete only the known artifact filenames (the Data
     Requirements table's list) — never the directory itself or anything else in it.
  4. Write artifacts as each phase produces them: for `induce`/`run`, write `train.csv`
     from the loaded+filtered train frame, then call `induction.induce(...)` (Task 3) and
     write its `InductionOutcome.rendered_prompt` to `induction_prompt.txt`,
     `.sampled_examples` to `sampled_examples.json`, and `.category` to `categories.json`
     via `json.dumps({"categories": [outcome.category.model_dump(mode="json")]},
     indent=2)` — then round-trip that exact file through `load_categories` before
     proceeding, so FR-2.4's "must validate" holds for what's actually on disk. For a
     supplied `--categories` (any subcommand), write it as a **byte-verbatim** copy
     (`shutil.copy` of the raw file, not a parse-then-`model_dump()` round-trip — a prior
     draft's implied re-serialization isn't "verbatim," since it can reorder keys or
     reformat whitespace) unless the same-path no-op from step 2 applies.
  5. Write `test.csv` from the loaded+filtered (+`--limit`-truncated) test frame, then
     call `classify_csv` and write `test_classified.csv`.
  6. Write `run_config.json` **last** (see the field table below), after the completeness
     check, carrying its terminal `status`. Write every JSON artifact via a temp file in
     the same directory plus `Path.replace()` (plan-critique fix: an interruption or
     disk-full error mid-write must not leave a malformed control file on disk).
  7. **Failure path**: wrap the entire phase-dispatch body (steps 4-6) in a top-level
     `try/except Exception as e: # noqa: BLE001 - any phase failure must still produce an
     auditable run_config.json`. On failure, populate every `run_config.json` field that's
     already known (dataset/column/model selections, whatever artifacts were written) with
     `status: "failed"` and a `failure: f"{type(e).__name__}: failed during {stage}"`
     field (`stage` tracked via a simple local variable updated at the start of each
     numbered step above), write it via the same temp-file+replace mechanism, then
     re-raise so the process exits non-zero. Plan-critique fix: `"failed"` was an allowed
     `status` value with no code path that ever wrote it.
- **`run_config.json` field population** (plan-critique fix: the prior draft asserted the
  file without designing any field beyond `status`/`unclassified_rows`) — one row per Data
  Requirements field, source and per-subcommand value:

  | Field | Source | `induce` | `classify` | `run` |
  |---|---|---|---|---|
  | `schema_version` | literal constant, start at `1` | `1` | `1` | `1` |
  | `status` | step 7 above | terminal value | terminal value | terminal value |
  | `subcommand` | `args.subcommand` | `"induce"` | `"classify"` | `"run"` |
  | `dataset` | Task 1's loader return + `source_column_names` is **not** included here (internal only) | local or hf dict, train fields only | local or hf dict, test fields only | both |
  | `text_column`/`label_column`/`category_name` | `args` | present | present | present |
  | `seed`/`examples_per_label`/`max_example_chars`/`max_prompt_chars` | `args` | present | `null` | present |
  | `models` | resolved after `--x-model or --model` defaulting | `{model, induction_model, null, null}` | `{model, null, critic_model, reconciler_model}` (critic/reconciler `null` unless `--critics`) | all four resolved |
  | `classifier_config` | `args`, post-validation | `null` | present | present |
  | `prompt_inputs` | `args` paths or `null` | `null` | present | present |
  | `label_values` | sorted distinct **train** label values (`induce`/`run`) or sorted distinct values from the supplied `categories.json` (`classify`) | from train | from categories.json | from train |
  | `test_labels` | `"withheld"` iff no test row resolves to a usable label (column absent, or every value is `None`/empty after Task 1's per-row resolution — see Task 1's mixed-label note), else `"present"` | `null` (no test split) | computed | computed |
  | `unseen_test_labels` | test gold values (where present) absent from `label_values` | `null` | computed | computed |
  | `excluded_rows` | Task 1's `project_and_filter` dropped-positions return | `{train: {...}, test: null}` | `{train: null, test: {...}}` | both |
  | `unclassified_rows` | completeness check below | `null` | computed | computed |
  | `categories_source` | `"induced"` if this run produced `categories.json` itself, else `"supplied"` | `"induced"` | `"supplied"` | `"induced"` unless `--categories` given |
  | `versions` | `{python: platform.python_version(), query_classification:
    importlib.metadata.version("ds-query-classification"), litellm: ..., pandas: ...,
    pydantic: ..., datasets: importlib.metadata.version("datasets") if installed else
    null}` (plan-critique fix: the package does not export `__version__`; use
    `importlib.metadata.version`, matching how the installed wheel/editable install is
    actually introspectable) | present | present | present |

  `run_config.json` records resolved model **ids** and flag values only — never API keys,
  tokens, or `--api-base`'s value (Constraints).
- **Completeness check** (FR-3.8): after `classify_csv` returns, read the written CSV back
  and check the category column (plus `{category}_votes` under `--critics`) for nulls;
  record `unclassified_rows` in `run_config.json`, set `status` accordingly, and
  `sys.exit(1)` when the count is non-zero unless `--allow-partial` was passed.
- **Run-directory safety**: if `--run-dir` (or any of the known artifact filenames inside
  it, under `--overwrite`) resolves to a symlink, resolve it and validate the real target
  is a directory/regular file as expected; `unlink()` a known-artifact symlink rather than
  following it when replacing, and never recurse into or delete anything not on the known
  artifact list.
- **Phase logging**: emit one concise line per phase (`validating`, `loading`, `inducing`,
  `classifying`, `verifying`, `finalizing`) so a slow HF download or large run doesn't look
  hung — never log raw example text, endpoint URLs, or token/credential values.
- **Suggested internal helpers** (naming seams within this one file, so the task stays a
  single module per the Risks section's reasoning while still being reviewable in pieces):
  `_validate_args`, `_construct_classifiers`, `_write_run_config`, `_check_collisions`,
  `_check_completeness`.
- **README**: document the three subcommands, the `.[hf]` extra, the run-directory
  contents, the plain-mode closed-vocabulary divergence from `classify.py`, the
  data-egress note (which role's model receives train vs. test text), a note that
  `train.csv`/`test.csv`/`sampled_examples.json`/`induction_prompt.txt` contain raw or
  truncated dataset text and should inherit restrictive local file permissions, and the
  inherited CSV formula-injection caution already documented for Spec 1.
**Verify:** `induce`/`classify`/`run` each produce **exactly** the artifacts Data
Requirements lists for that subcommand — assert absent files too, not just present ones;
a plain-mode run with `--allow-new-labels` omitted produces a classification system prompt
containing no `"none - "` text (proving the `cli.py:201` divergence actually took effect);
`--limit 5` over a 100-row test split yields a 5-row `test.csv` and `test_classified.csv`
with zero null predictions; a dataset whose **source** (pre-projection) columns include a
`label_gold` or (under `--critics`) `label_votes` column exits with a collision error
before any LLM call — including the case where that column would have been projected out
by AR-1.3, proving the check runs against `source_column_names` and not the materialized
frame; a supplied `--categories` equal to `<run-dir>/categories.json` under `--overwrite`
leaves the file byte-identical rather than deleted-then-missing; a supplied `--categories`
elsewhere on disk is copied byte-for-byte (not re-serialized); `run --seed 1
--examples-per-label 2 --model ... ` resolves without error (proving the flag-group fix);
`induce` alone (no `--model` typo/omission edge case) resolves `--induction-model`'s
fallback to `--model`; a `run_config.json` schema test per subcommand covering
`completed`/`completed_with_failures`/`failed` populates every field in the table above,
including a forced mid-run exception producing `status: "failed"` with a sanitized
`failure` string; a mixed valid/`-1` test-label column (some rows real, some withheld)
produces `test_labels: "present"` with per-row nulls only where withheld; re-running into a
non-empty `--run-dir` without `--overwrite` modifies nothing (zero Hub/LLM calls, zero
files touched) — this and every other preflight-rejection case must be checked for zero
calls/writes, not just a non-zero exit code.
**Covers:** FR-1.4, FR-1.6, FR-3.1, FR-3.2, FR-3.3, FR-3.4, FR-3.5, FR-3.6, FR-3.8, AR-3.1,
AR-3.2, AR-3.3, AR-3.4

### Task 5: Architecture ADR for the revised dependency graph
**Goal:** Document the three new modules' dependency edges and the new test-import
exceptions, per `spec/ARCHITECTURE.md`'s own stated ADR process — the binding doc itself is
updated later by `/spec-close`.
**Files:** `spec/2-experiment-runner/ADR.md` (new)
**Dependencies:** None — describes a graph already fixed by this plan's design (Tasks
1-4's module boundaries), not by their code being finished.
**Do:**
- State the INV-1 change with explicit before/after wording and new Module Boundary Map
  rows for `dataset_io.py` (may import from: stdlib, pandas, `huggingface_hub` — no
  internal `query_classification` imports, a leaf module like `categories.py`/
  `classifier.py`), `induction.py` (may import from: `categories.py`, `classifier.py`), and
  `experiment.py` (may import from, **enumerated explicitly, not "all of the above"**:
  `dataset_io.py`, `induction.py`, `schema.py`, `prompts.py`, `resources.py`,
  `pipeline.py`, `debate.py`, `categories.py`, `classifier.py` — the same breadth as
  `cli.py`, and explicitly **not** `cli.py` itself). State explicitly that INV-1's
  "cli.py is the only module allowed to import all others" becomes "cli.py and
  experiment.py are the two modules allowed to import all others; neither imports the
  other." Enumerate the root `experiment.py` script's edge the same way `classify.py`'s
  is already documented (a thin entry point, not a package-internal import source).
- State the INV-7 change: `dataset_io.py`, `induction.py`, `experiment.py` (all functions),
  plus `schema.build_induction_model` and `prompts.build_induction_prompt` specifically,
  join Spec 1's `debate.py`/`build_critic_model`/etc. as accepted direct-import exceptions
  for tests — extending, not replacing, that list.
- Reference the FR/AR IDs requiring this change (AR-1.2) and the spec path.
**Verify:** `ADR.md` exists, states both invariant changes precisely enough for
`/spec-close` to fold them into `spec/ARCHITECTURE.md` mechanically.
**Covers:** AR-1.2

### Task 6: Ship unit tests
**Goal:** A `pytest`-runnable suite exercising every FR-1.x/FR-2.x/FR-3.x Verify condition,
with a signature-accurate fake `datasets` module and fake `Classifier`-shaped objects, no
live network or Hub calls.
**Files:** `tests/test_experiment.py` (new)
**Dependencies:** Task 4
**Do:**
- Build a minimal fake `datasets` package/module (injected via `sys.modules` or
  `monkeypatch`) exposing `load_dataset(path, name=None, revision=None, split=None)` →
  returns a fake `DatasetDict`-like mapping of split name → a fake `Dataset` with
  `.features`, `.column_names`, `.to_pandas()`, `.remove_columns()`, and `__getitem__`
  (column access, for Task 1's `resolve_hf_labels` reading `dataset[label_column]` before
  `to_pandas()`) matching the real API's signatures and return shapes closely enough that
  a wrong call (e.g. omitting `config_name` somewhere the real API requires it) would fail
  the test — directly addressing the plan-critique finding that a signature-loose fake
  could hide such a bug. Also fake `huggingface_hub.utils.validate_repo_id` (real
  validation logic — reject invalid syntax) and `huggingface_hub.HfApi().dataset_info(...)`
  (returns an object with `.sha`, and a variant that raises, to exercise the best-effort
  fallback).
  Include a fake `ClassLabel` whose `.int2str()` **always raises `ValueError` on a
  negative index**, matching the real library exactly (plan-critique fix: a prior draft's
  "raises/returns `None` policy" conflated the library's behavior with the runner's own
  guard — Task 1's `resolve_hf_labels`, not the fake, is what catches the raise and
  substitutes `None`).
- One test function (or small parametrized group) per FR's Verify condition — maintain
  this as an explicit traceability table (test name → FR/Verify condition) in the test
  module's docstring or a comment block, not just an implicit correspondence — reusing the
  exact scenarios from the spec and refined during Tasks 1-4: HF id validation (invalid
  syntax, absolute path, `.git` suffix, packaged builder name, valid single/two-segment
  id), missing split listing, `ClassLabel`/string/untyped-integer/float label handling,
  mixed valid/withheld `ClassLabel` rows in one split, local CSV `001`-style verbatim
  labels and literal-`"NA"`-in-text preservation, the `[hf]`-missing error, sanitized HF
  auth-failure error (using a fake secret in the raised exception, asserting it never
  appears in the sanitized message), resolved-SHA best-effort success and failure paths,
  all five split-availability cases, row filtering counted once, both withheld-label forms
  plus unseen-test-label recording, exact single-generator RNG sample sequence with at
  least two over-cap labels (not just a same/different-seed property test), the
  serialized-payload prompt-size preflight, missing/invented/duplicated induction labels,
  the reserved sentinel (including in a supplied file), sanitized induction failure,
  degenerate train splits, multi-category `--categories` rejection, byte-verbatim supplied
  `--categories` copy and the self-path-plus-`--overwrite` case, `--overwrite`
  artifact-only replacement, plain-mode closed vocabulary, `--limit` truncation with zero
  null rows, all collision cases (including a source column projected away by AR-1.3), a
  full field-by-field `run_config.json` schema check per subcommand for
  `completed`/`completed_with_failures`/`failed`, and partial-classification status/
  exit-code behavior for both plain and `--critics` modes.
- Add an explicit CLI-level test (via `build_parser()`-equivalent + `main()` with
  `Classifier` construction and `.classify()` monkeypatched) for at least one invalid-config
  exit path per subcommand, confirming a clean non-zero exit and zero LLM/Hub calls/file
  writes attempted; include a `run`-subcommand parser test asserting the induction flag
  group actually attaches (the flag-group bug this plan revision fixed).
- Add `python -m query_classification.experiment --help` as its own test, not only the
  root-script `--help` (both entry points must work independently).
**Verify:** `pytest` passes with 0 failures, including all new tests; the pre-existing
Spec 1 suite still passes unmodified — checked via `pytest` itself (assert the suite
passes; do not hard-code a specific test count, since it legitimately changes over time).
**Covers:** FR-3.7

## Final Verification

- `.venv/bin/python -m pytest` — full suite passes (do not hard-code a test count).
- `.venv/bin/python experiment.py --help`, `.venv/bin/python experiment.py induce/classify/
  run --help`, and `.venv/bin/python -m query_classification.experiment --help` all show
  the documented flags with correct defaults (plan-critique fix: the module-invocation
  form must be checked too, not only the root script).
- Manual smoke test with fakes wired through the real CLI parsing path for at least one
  full `run` (induce + classify) over a small local CSV pair, confirming every Data
  Requirements artifact is written with the documented shape.
- Confirm `classify.py`/`python -m query_classification`'s existing behavior is
  byte-unchanged (AR-3.1/AR-3.2 touch nothing there) via `git diff --exit-code --
  src/query_classification/pipeline.py src/query_classification/cli.py
  src/query_classification/classifier.py src/query_classification/categories.py
  src/query_classification/debate.py classify.py
  src/query_classification/__init__.py src/query_classification/__main__.py` (plan-critique
  fix: "36 tests still pass" doesn't itself prove zero bytes changed in files this plan
  promises not to touch — a diff check does).
- `git status` shows only the intended new/modified files from Code Impact, plus the
  pre-existing untracked files noted in Readiness (untouched).

## Documentation

- `spec/docs/` does not exist yet — out of scope for this plan; consider `/spec-docs
  --full` as a separate follow-up once this and Spec 1 are both available to document.
- `README.md` — updated in Task 4.
- `AGENTS.md` — no change needed; its existing "Rules" already generically cover the new
  code.
- `spec/ARCHITECTURE.md` — not edited directly by this plan; Task 5's `ADR.md` is folded
  in by `/spec-close`.

## Spec Deviations

None identified. Two implementation-level defaults the spec left unpinned were filled in
(`--max-prompt-chars` default of `100000`; the induction user-message wire format as a
JSON object between fixed delimiters) — neither changes observable acceptance criteria,
both are exactly the kind of detail the spec deliberately left to planning. Plan critique
(v1) proposed three additional items as possibly needing `/spec-update`; each was checked
against the exact spec text and resolved as a plan-level fix or clarification instead —
see `plan-critique-consolidated-v-1.md`'s "Spec Deviations Assessment" for the full
reasoning per item: (1) AR-1.3's column projection vs. FR-3.6's collision check, resolved
by carrying `source_column_names` (names only, no values) through Task 1's return type;
(2) Goals' "rendered prompt" vs. Data Requirements defining only `induction_prompt.txt`,
resolved as a scoping read of the Goals bullet, not a contradiction; (3) mixed valid/
withheld test labels, resolved as a natural extension of the already row-level
`resolve_hf_labels` design.

| # | Spec says | Plan does instead | Reason | Action required |
|---|-----------|-------------------|--------|-----------------|
| — | — | — | — | None identified |

## Risks

- **Argument-parsing complexity.** Three subcommands sharing most but not all flags is
  more surface than Spec 1's flat parser. Mitigation: the `parents=` shared-parser
  approach in Task 4 keeps flag definitions in one place; Task 6's CLI-level test catches
  a misrouted flag before it reaches implementation review.
- **`experiment.py` (Task 4) is the largest single task**, carrying nearly all of Feature
  3 in one file because subcommand parsing, classifier construction, run-directory
  management, and collision/completeness checks are all tightly coupled through shared
  local state (the parsed args, the constructed classifiers, the loaded DataFrames) —
  splitting it across files would mean threading that state through module boundaries for
  no parallelism benefit, since it's one file either way. If it proves unwieldy during
  implementation, splitting into `experiment.py` (parsing + dispatch) and a same-file-group
  `experiment_run.py` (directory/collision/completeness) is a reasonable mid-implementation
  call — but starts as one file per this plan.
- **Rollback/checkpoint guidance:** Tasks 1/2/3/5 are purely additive (no existing file's
  *behavior* changes — `schema.py`/`prompts.py`/`resources.py` only gain new functions).
  Task 4 is the first task that could plausibly need iteration once real classifier
  construction is wired end-to-end; commit after Tasks 1-3+5 land (a clean four-way
  independent checkpoint) before starting Task 4.
- **No existing *behavior* changes for `classify.py`'s current users** (AR-3.1's actual
  guarantee — plan-critique fix: "no existing code is modified" was too broad, since
  `schema.py`/`prompts.py`/`resources.py`/`pyproject.toml`/`README.md` are all modified by
  this plan; what AR-3.1 promises is that `pipeline.py`, `cli.py`, `classifier.py`,
  `categories.py`, and `debate.py` are read-only). The Final Verification's `git diff
  --exit-code` check against exactly those files, plus the pre-existing test suite passing
  unmodified, is the direct, continuous check of this throughout.
