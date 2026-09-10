# Implementation Plan: 4-multi-model-classification

**Status:** Ready
**Date:** 2026-09-10
**Spec:** spec/4-multi-model-classification/spec.md
**Plan Critique:** spec/4-multi-model-classification/plan-critique-consolidated-v-1.md

## Overview

Build `multi_model.py` and its ADR/architecture amendment together (Task 1), extend
`pipeline.classify_csv` to dispatch to it with full end-to-end/restore/public-API test
coverage (Task 2), then wire `--models` through `cli.py` and `experiment.py` in parallel
(Tasks 3-4, disjoint files), and close with a README update (Task 5).

## Readiness

- Checklist: all 10 items closed (`[x]`/`[N/A]`) in `spec.md`; no open items.
- Open questions: none — two rounds of spec critique (v1, v2) and one round of plan
  critique already resolved every blocking item found, including a factual error in
  AR-1.1's own justification (claimed `debate.py` "has no existing dependents" — it
  already has two, `pipeline.py` and `experiment.py`; the *conclusion* was still right,
  just not the stated reason) — corrected directly in `spec.md`.
- External API/library verification: none needed — no new third-party dependency; reuses
  `Classifier`/`ThreadPoolExecutor`/`litellm`/`pydantic`, all already in use.
- Repository state: **dirty**, and this matters for this plan specifically —
  `git status` shows uncommitted changes already in *every file this plan also touches*:
  `cli.py`, `experiment.py`, `README.md`, and all three test files (from prior, unrelated
  work earlier in this session: Cerebus gateway support and the `--test-limit` rename).
  Tasks 2-5 must treat those files' **current on-disk content** as the baseline (read
  before editing; patch additively) rather than assuming a clean/original version — do
  not `git checkout`/revert them to "simplify" the diff.

## Code Impact

- **Modules:** new `src/query_classification/multi_model.py`; modified
  `src/query_classification/pipeline.py`, `cli.py`, `experiment.py`.
- **Database/schema:** none (CSV/JSON files only). `run_config.json`'s `models` dict gains
  two keys, always present (`null` when unused); `_SCHEMA_VERSION` bumps `1` → `2`.
- **API/interfaces:** `classify_csv`'s public signature changes (`classifier` becomes
  `Classifier | None`, same position/required-ness; adds `models: dict[str, Classifier] |
  None = None`) — this is part of the re-exported public API (`__init__.py`), so the
  change must be backward compatible for existing positional callers (verified in spec —
  no reordering, only a type/validation change).
- **UI/config/background jobs:** none.
- **Key files:**
  - `src/query_classification/multi_model.py` (new)
  - `src/query_classification/pipeline.py:22-25,28-220` (`_audit_columns`, `classify_csv`)
  - `src/query_classification/cli.py:40-179` (`build_parser`), `:190-326` (`main`)
  - `src/query_classification/experiment.py:85` (`_SCHEMA_VERSION`), `:187-234`
    (`build_parser`'s `classification` group), `:266-317` (`_validate_args`), `:448-532`
    (`_construct_classifiers`), `:745-780` (`run`'s induction/classification merge),
    `:780` (the no-induction `models_final` fallback literal — easy to miss),
    `:816-836` (source-column collision check), `:869-881` (post-run completeness check,
    including the exact `config["models"] = models_final` / `_write_json_atomic` order at
    `:875-880`)
  - `spec/ARCHITECTURE.md` (INV-1, INV-7), `spec/4-multi-model-classification/ADR.md` (new)
  - `README.md` (Options table, "Experiment runner" section)
  - `tests/test_multi_model.py` (new), `tests/test_debate.py`, `tests/test_experiment.py`

## Project Constraints

- **INV-1** (`spec/ARCHITECTURE.md`): the internal dependency graph is one-directional and
  acyclic. `multi_model.py` may import `categories.py`, `classifier.py`, `schema.py`,
  `prompts.py`, and `debate.py` (a new, legal edge: `debate.py` itself has zero *outbound*
  imports of `pipeline.py`/`cli.py`/`experiment.py`/`multi_model.py` today, even though
  `pipeline.py`/`experiment.py` already import `debate.py` — a new *importer* of
  `debate.py` doesn't create a cycle) — never `pipeline.py`, `cli.py`, or `experiment.py`.
  Authorized via an ADR **written together with the module itself in Task 1**, not
  deferred to a later task — see Task 1's rationale.
- **INV-4**: any broad `except Exception` needs an inline `# noqa: BLE001` plus a one-line
  reason (see `debate.py:95,133` and `pipeline.py:210` for the existing style) —
  `multi_model.py`'s per-model failure handling will need this.
- **INV-7**: `tests/` normally imports only the public API; `multi_model.py` (all
  functions) needs to join the existing named exceptions (`debate.py`, `dataset_io.py`,
  etc.) — same Task 1 ADR as INV-1.
- **INV-8**: category names must stay unique and distinct from `--column`; the new
  `_votes`/`_by_model`/`_model_errors` audit-column suffixes must be checked for collision
  exactly like `debate.AUDIT_COLUMN_SUFFIXES` already is (AR-1.3a/AR-1.4) — including the
  cross-category case (one category named e.g. `sentiment_by_model` colliding with another
  category `sentiment`'s generated column).
- **CLAUDE.md testing policy**: no live network/provider calls in tests — mock
  `Classifier.classify` (see `tests/test_debate.py`'s `SimpleClassifier`/monkeypatch
  pattern and `tests/test_experiment.py`'s `fake_classify` fixture, which dispatches by
  inspecting `self.classification_model.model_fields.keys()`).
- **CLAUDE.md required patterns**: `multi_model.py` starts with
  `from __future__ import annotations` and uses PEP 604 unions (`str | None`, not
  `Optional[...]`), matching every other module.
- **Duplicate-id detection is structurally a list-level (CLI) concern, not a dict-level
  (library) one**: `args.models` (from `nargs="+"`) is a `list[str]`, which *can* contain
  duplicates (`["a", "a", "b"]`) — Tasks 3/4 must check for duplicates on that list, before
  building the `dict[str, Classifier]` that `classify_csv`/`multi_model.py` receive. A
  Python dict literally cannot hold two entries under the same key, so `classify_csv`
  itself (Task 2) can only ever validate count/non-emptiness on the dict it's given, never
  "duplicates" — there's nothing there to detect by the time it arrives.
- **No lint/typecheck/CI configured** — `pytest` passing is the only gate
  (`.venv/bin/python -m pytest`). Use `.venv/bin/python`, not bare `python`, for every
  command below (AGENTS.md: prefix commands this way when venv activation isn't certain).

## Implementation Strategy

- Size: large (10 FRs, 8 ARs, a new module, 3 existing modules touched, an ADR, docs).
- Execution mode: solo Task 1, solo Task 2, then 2 parallel streams (Tasks 3, 4), then
  solo Task 5.
- Parallelizable work: Task 3 (`cli.py` + `tests/test_debate.py`) and Task 4
  (`experiment.py` + `tests/test_experiment.py`) touch entirely disjoint files — zero
  contention, safe to run concurrently once Task 2 lands.
- Sequential blockers: Task 2 needs Task 1's module (imports it, calls its orchestration
  function). Tasks 3/4 need Task 2's `classify_csv` signature. Task 5 (README) is
  sequenced last since it documents the flags' final, implemented behavior.
- **Rollback groups** (reverse-dependency order — see Risks): 5 → {3, 4} → 2 → 1. Never
  revert Task 1 while keeping Task 2 (leaves `pipeline.py` importing a missing module) or
  vice versa in the wrong order; Task 1's ADR/`ARCHITECTURE.md` edit must be reverted
  together with the module it authorizes, not independently.

## Implementation Tasks

### Task 1: `multi_model.py` module + its ADR, together
**Goal:** A new module providing the per-row multi-model classification/voting function
`pipeline.py` will dispatch to — with the architecture decision that authorizes it written
in the same task, not deferred.
**Files:** `src/query_classification/multi_model.py` (new), `tests/test_multi_model.py`
(new, orchestration-level tests only — see Task 2's Notes for why), `spec/ARCHITECTURE.md`,
`spec/4-multi-model-classification/ADR.md` (new)
**Dependencies:** None
**Do:**
- Mirror `debate.py`'s shape: a public orchestration function (e.g. `run_multi_model(text,
  categories, classifiers: dict[str, Classifier], *, allow_new_labels: bool) ->
  dict[str, Any]`), reusing `debate._vote_bucket` for none-label merging (FR-1.3) rather
  than duplicating it.
- Call all N classifiers concurrently via `ThreadPoolExecutor(max_workers=max(1,
  len(classifiers)))`, collecting results into a pre-allocated, index-ordered list keyed
  by each model's position in the `classifiers` dict's iteration order — never tallied in
  `as_completed()` arrival order (AR-1.5; mirror `debate.py:172-185`'s
  `[None] * sampling_runs` pattern).
- Per category: tally each successful model's *top* label via `_vote_bucket`, pick the
  plurality winner (`Counter.most_common()[0]`, index-insertion order gives the
  list-order tie-break for free — FR-1.3), and take that model's full label list as
  `{category}`.
- Build `{category}_votes` (JSON tally), `{category}_by_model` (JSON map of successful
  model id → its full label list), and `{category}_model_errors` (JSON map of failed model
  id → `f"{type(exc).__name__}: Classification call failed"`, per FR-1.7 — literal stage
  text is `"Classification"`; this sanitization guarantee covers only what's persisted to
  CSV/`run_config.json` — `Classifier.classify`'s own existing warning-log call
  (`classifier.py:338-352`) still logs the raw exception on each failed attempt, unchanged
  by this feature and out of scope to alter).
- **Never raise** for ordinary model-call failures, including the all-fail case (AR-1.5):
  when 0 of N succeed, still return a result with `{category}: None` for every category,
  empty `_votes`/`_by_model`, and `_model_errors` covering all N models (FR-1.6). Use
  `# noqa: BLE001` with a one-line reason on the broad per-model `except Exception`,
  matching `debate.py:95,133`'s style (INV-4).
- Write `spec/4-multi-model-classification/ADR.md`, following the format of
  `spec/1-initial-classification-with-critics/ADR.md`/
  `spec/2-experiment-runner/ADR.md`, authorizing the **full** set of import-boundary
  changes this feature needs (not just what this task itself touches): `multi_model.py`'s
  new Module Boundary Map row (allowed: `categories.py`/`classifier.py`/`schema.py`/
  `prompts.py`/`debate.py`; forbidden: `pipeline.py`/`cli.py`/`experiment.py`); the new
  `debate.py → multi_model.py` Mermaid edge; `multi_model.py` added to `pipeline.py`'s and
  `experiment.py`'s own "May import from" cells (consumed by Tasks 2/4, not re-decided by
  them); and an INV-7 addition permitting `tests/` to import `multi_model.py` (all
  functions) directly. Add the standard
  `*(amended by spec/4-multi-model-classification/ADR.md ADR-001, 2026-09-10)*` marker to
  both INV-1 and INV-7.
- Tests (`tests/test_multi_model.py`, calling `run_multi_model` directly — **not** through
  `classify_csv`, which doesn't exist until Task 2): vote-tally/tie-break in isolation
  (including an out-of-order-completion case proving list-order tie-break, per FR-1.3's
  Verify), exact call count (N calls for N models, no repeats), shared config asserted
  (every classifier gets `temperature=None`; identical system prompt/schema across all N),
  audit-column JSON shape/content (FR-1.4), partial-failure (FR-1.6), total-failure
  never-raising with the exact `_model_errors` format (FR-1.6/FR-1.7), the
  malformed-category-fails-whole-row case (FR-1.4), and a bounded-concurrency sanity check
  (N fake classifiers all in flight at once, confirming the executor is sized to `len(
  classifiers)` and shuts down cleanly after a failure).
**Verify:** `.venv/bin/python -m pytest tests/test_multi_model.py -v` passes;
`spec/ARCHITECTURE.md`'s Module Boundary Map table has a `multi_model.py` row with the
imports listed above; its Mermaid graph includes `debate --> multi_model`; INV-1's and
INV-7's prose each mention `multi_model.py` explicitly (a text/table review, not just a
`grep` count, since counting occurrences can't confirm edge direction or which imports are
listed as allowed vs. forbidden).
**Covers:** FR-1.3, FR-1.4, FR-1.6, FR-1.7, FR-1.10 (this module's share), AR-1.1 (module +
ADR together), AR-1.5
**Notes:** Don't import `pipeline.py`/`cli.py`/`experiment.py` from `multi_model.py`
(INV-1). Writing the ADR/`ARCHITECTURE.md` edit in the same task as the module (rather
than after, as an earlier draft of this plan had it) avoids a window where the new module
and its direct test-imports would violate the not-yet-amended INV-1/INV-7 — there's no
tooling that would catch that window today, but there's no reason to open it either.

### Task 2: `pipeline.py` — `classify_csv`'s `models` parameter, validation, dispatch, and its own tests
**Goal:** `classify_csv` accepts a `models` dict, validates it, dispatches to Task 1's
function at every point `critics` already branches, and is itself directly tested (not
only indirectly through Tasks 3/4's CLIs).
**Files:** `src/query_classification/pipeline.py`, `tests/test_multi_model.py` (extends —
adds the end-to-end/restore tests Task 1 couldn't, since `classify_csv` didn't have a
`models` parameter yet), `tests/test_debate.py` (adds public-API validation tests)
**Dependencies:** Task 1
**Do:**
- Change `classifier: Classifier` → `classifier: Classifier | None` (type only — same
  position, still required/non-default, since `categories` right after it has no default
  either; see spec's AR-1.2 note on why giving it a default is a `SyntaxError`). Add
  `models: dict[str, Classifier] | None = None` as a new keyword parameter.
- Add validation before any CSV I/O (mirroring the existing critics-mode block at
  `pipeline.py:75-95`): exactly one of {`classifier is not None`, `models`} configured;
  `critics=True` requires `classifier`; `models` (if given) has ≥2 non-empty-string keys
  (uniqueness is not this layer's concern — see Project Constraints).
- Generalize `_audit_columns` (currently critics-only, `pipeline.py:22-25`) to also
  produce `multi_model`'s 3 suffixes (`_votes`, `_by_model`, `_model_errors`) when `models`
  is active — used by all 5 of the following branch points (AR-1.3 a-e):
  (a) the collision-check block (`:75-118`, including the cross-category case from
  Project Constraints' INV-8 note), (b) initial column creation (`:126-131`), (c)
  restore-seed-from-separate-output (`:136-147`), (d) restore-completeness (`:151-156`),
  (e) non-restore reset/dtype-coercion (`:157-168`).
- Add the `models` dispatch branch alongside the existing `critics`/plain branches in the
  `ThreadPoolExecutor` submission block (`:182-202`), calling Task 1's function. Since that
  function never raises, add a one-line comment at the existing `except Exception:` /
  `failed += 1` site (`:210-211`) noting that for `--models` rows this branch is reached
  only by a genuinely unexpected error, not an ordinary model failure — `ok`/`classified`
  no longer means "at least one model succeeded," only "the orchestration didn't crash";
  `_model_errors`/`model_failure_counts` are the source of truth for per-model health, not
  the progress bar.
- Tests, split from Task 1 by necessity (that `classify_csv` didn't support `models` yet):
  - In `tests/test_multi_model.py`: audit columns end-to-end via `classify_csv` (FR-1.4,
    now actually possible); the two FR-1.5 restore cases, mirroring
    `tests/test_debate.py:364-385`/`:388-443`'s existing critics-mode patterns exactly —
    (1) a partially-populated row (`{category}` filled, `{category}_by_model` empty) is
    reclassified, not skipped, on `--restore`; (2) restoring from a separate `--output`
    file seeds all 4 multi-model columns (`{category}`, `_votes`, `_by_model`,
    `_model_errors`), not just `{category}`; and a fresh-run-clears-stale-values case; the
    pipeline-counter assertion from FR-1.6's Verify line — an all-model-failure row
    increments `classified`, not `failed` (patch `tqdm.tqdm` with a recording fake to
    observe the `ok`/`failed` postfix values, since `classify_csv` returns only a `Path`).
  - In `tests/test_debate.py` (where direct, non-CLI `classify_csv(...)` calls already
    exercise plain/critics validation, e.g. `tests/test_debate.py:347,352`): the
    `classifier`/`models` mutual-exclusion and ≥2/non-empty-keys validation directly
    against `classify_csv`, asserting these six cases fail before any CSV
    read/write — `classifier=None, models=None` (neither); both given; `critics=True,
    classifier=None`; `models={}`; `models` with one entry; `models` with a
    whitespace-only key.
**Verify:** `.venv/bin/python -m pytest tests/test_multi_model.py tests/test_debate.py
tests/test_building_blocks.py -v` passes (no regression to existing plain/critics paths).
**Covers:** FR-1.4 (end-to-end share), FR-1.5, FR-1.6 (counter assertion), AR-1.2, AR-1.3
(a-e), AR-1.5 (integration point), AR-1.8 (INV-8 collision test)

### Task 3: `cli.py` — `--models` flag, validation, and classifier construction
**Goal:** `classify.py` supports `--models` end-to-end.
**Files:** `src/query_classification/cli.py`, `tests/test_debate.py`
**Dependencies:** Task 2
**Do:**
- Add `--models` (`nargs="+"`) to `build_parser()` (`:40-179`), alongside `--critics`.
- Add validation in `main()` before the existing `try:` block's LLM-touching work
  (`:198-216`, alongside the existing `--sampling-temperature` checks) — all 5 of FR-1.1's
  cases, on the `args.models` **list** (duplicate detection belongs here, per Project
  Constraints, before any dict is built): (1) `--models` + `--critics` together; (2) fewer
  than 2 values; (3) a duplicate value; (4) an empty/whitespace-only value; (5) `--models`
  + `CEREBUS_MODE=azure` (read via `os.getenv("CEREBUS_MODE")`; add `import os` if not
  already present).
- Change the `allow_new_labels` line (`:221`, `args.allow_new_labels if args.critics else
  True`) to also honor the flag under `--models` (`if args.critics or args.models else
  True` — FR-1.2's carve-out).
- When `args.models` is set (validated, so now duplicate-free), build
  `models_classifiers = {m: Classifier(model_id=_model_id(m), system_prompt=system_prompt,
  classification_model=classification_model, max_retries=args.retries, **gateway_kwargs)
  for m in args.models}` (mirroring the existing `critic_classifiers`/
  `reconciler_classifiers` dict-comprehension pattern, `:279-306`) instead of the single
  `classifier` (`:267-274`); pass `classifier=None, models=models_classifiers` to
  `classify_csv` (`:308-323`).
- Tests (in `tests/test_debate.py`, where `cli.py`-level validation/end-to-end tests
  already live — there is no `tests/test_cli.py`): all 5 validation-error cases from
  FR-1.1's Verify line (mirroring `test_cli_invalid_sampling_temperature_exits_before_any_
  llm_call`'s pattern at `:557`, asserting exit before any LLM call); an end-to-end
  `--models a b` run asserting the 3 new audit columns exist; the `--allow-new-labels`
  carve-out test from FR-1.2's Verify line; and a direct-provider Cerebus routing test
  confirming every model in `models_classifiers` gets the `_model_id()`-prefixed id and
  the same resolved `api_base`/`api_key`/`extra_headers` (mirroring how `critic_
  classifiers`/`reconciler_classifiers` already get the same `gateway_kwargs` today).
**Verify:** `.venv/bin/python classify.py --help` lists `--models`; `.venv/bin/python -m
pytest tests/test_debate.py -v` passes.
**Covers:** FR-1.1, FR-1.2, FR-1.4 (cli.py surface), FR-1.7 (cli.py surface), FR-1.10
(this task's share), AR-1.6, AR-1.7

### Task 4: `experiment.py` — `--models` on `classify`/`run`, `run_config.json`, checks
**Goal:** `experiment.py classify`/`run` support `--models` end-to-end, including
correct `run_config.json` bookkeeping across both the single-call and induction-then-
classification code paths.
**Files:** `src/query_classification/experiment.py`, `tests/test_experiment.py`
**Dependencies:** Task 2
**Do:**
- Add `--models` (`nargs="+"`) to the `classification` argument group (`:187-225`),
  alongside `--critics` — not the `common`/`induction` groups, so `induce` never sees it
  (FR-1.8).
- Add the same 5 validations as Task 3 (not 4 — see Task 3's list) to `_validate_args`
  (`:266-317`), gated by `getattr(args, "models", None)` the way the existing
  `if getattr(args, "critics", False):` block already is (`:304`); duplicate detection on
  the `args.models` list, same as Task 3.
- `_construct_classifiers` (`:448-532`): initialize the base `models` dict
  (`:461-466`) with `"classification_models": None, "model_failure_counts": None` added
  **unconditionally**, alongside the 4 existing keys — every invocation must return these
  two keys, not only `--models` ones (FR-1.9's "always present" requirement; a
  non-`--models` regression test must assert both keys are present-and-`null`). Then, when
  `args.models` is set, build `classifiers["multi_model"] = {m: Classifier(...) for m in
  args.models}` instead of `classifiers["classification"]`, and overwrite `models["model"]
  = None`, `models["classification_models"] = list(args.models)` in the returned dict.
  Widen the function's return-type annotation (`:454`) since the `models` dict's values
  are no longer always `str | None`.
- Fix the `run`-with-induction merge (`:745-780`): today, when `will_induce` is `True`,
  only `critic_model`/`reconciler_model` are copied from the classification-role
  `_construct_classifiers` call into `models_final` (`models_final["critic_model"] =
  models.get("critic_model")`, etc.) — `model`/`classification_models`/
  `model_failure_counts` are **not** copied, so `models_final["model"]` would silently
  keep the induction call's `args.model` value instead of becoming `None`. Add `model`,
  `classification_models`, and `model_failure_counts` to that same copy list. Also add the
  two new keys to the no-induction fallback literal at `:780` (`null` by default, matching
  the unconditional-init above).
- Pass `classifier=classifiers.get("classification")` (now possibly `None`),
  `models=classifiers.get("multi_model")` to the `classify_csv` call (`:842-856`).
- Post-run completeness check (`:869-881`): today, `check_cols = [category_name]` only
  gains `_votes` `if args.critics`. Change to also branch `elif args.models:
  check_cols += [f"{category_name}_votes", f"{category_name}_by_model",
  f"{category_name}_model_errors"]` (AR-1.3f) — all 3 multi-model audit columns, not just
  2; `--critics`/`--models` are mutually exclusive so `if`/`elif` (not two independent
  `if`s) is correct and clearer.
- Source-column collision check (`:816-836`): add `multi_model`'s 3 suffixes when
  `args.models`, alongside the existing critics suffix check (AR-1.4).
- Compute `model_failure_counts` **into `models_final["model_failure_counts"]` directly**
  (not into `config["models"]`, which doesn't exist as a dict until the single assignment
  `config["models"] = models_final` at `:875` — indexing into it earlier would raise
  `KeyError`), gated on `args.models`, right after `result_df = pd.read_csv(
  classified_path)` is available in the "verifying" stage: initialize
  `{m: 0 for m in args.models}` (FR-1.9 requires every supplied model to have an entry,
  including zero-failure models — a naive tally built only from observed failures would
  omit them), then for each row parse `{category_name}_model_errors` JSON and increment
  each key found. (Only one category ever exists per experiment run —
  `category_name = categories[0].name` — so this is a single-column read, not a
  per-category loop.)
- Bump `_SCHEMA_VERSION` (`:85`) from `1` to `2`.
- Tests (in `tests/test_experiment.py`, where `experiment.py`-level tests already live):
  all 5 validation-error cases (via `_run_main`); a **non**-`--models` regression test
  asserting `run_config.json`'s `models.classification_models`/`model_failure_counts` are
  present and `null`; an end-to-end `--models a b` run on `classify` asserting
  `classification_models`/`model == null`/`schema_version == 2`; a `run` (with induction)
  test specifically asserting `models_final["model"]` ends up `null` (not the induction
  call's `args.model`) — this is the merge bug fixed above; a case with one mocked model
  always failing asserting `model_failure_counts` shows that model's count *and* a
  zero-count entry for the model that didn't fail; and a parameterized source-column
  collision test for each of the 3 new suffixes (AR-1.4/INV-8).
**Verify:** `.venv/bin/python experiment.py classify --help` and `.venv/bin/python
experiment.py run --help` list `--models`; `.venv/bin/python experiment.py induce --help`
does not; `.venv/bin/python -m pytest tests/test_experiment.py -v` passes.
**Covers:** FR-1.1, FR-1.2, FR-1.4 (experiment.py surface), FR-1.5 (confirms N/A here —
no `--restore` flag exists), FR-1.7 (experiment.py surface), FR-1.8, FR-1.9, FR-1.10
(this task's share), AR-1.3f, AR-1.4, AR-1.6, AR-1.7

### Task 5: Documentation — README.md
**Goal:** `README.md` documents `--models` to the same standard as `--critics`.
**Files:** `README.md`
**Dependencies:** Task 3, Task 4
**Do:**
- Add `--models` to the Options table (`:87-111`) with a one-line description.
- Extend the "Experiment runner" section (`:153-222`): mention `--models` as a
  `--critics`-sibling mode; add the precise concurrency note (`workers × len(models)`,
  before retries/fallback — call out likely rate-limit/cost/latency effects, not just the
  formula); extend the existing cross-provider data-egress note (`:214-222`) to cover N
  models; note the existing CSV/spreadsheet-injection caveat applies here too; note
  `experiments/ag-news/analyze_run.ipynb` can't analyze a `--models` run's output as-is (it
  reads critics-specific columns); note that a `--models` run's progress display
  (`classified`/`failed` counts) reflects orchestration success, not per-model success —
  see `model_failure_counts`/`_model_errors` for that.
**Verify:** manual read-through confirms every AR-1.8 item is present; no command to run.
**Covers:** AR-1.8

## Final Verification

- `.venv/bin/python -m pytest` — full suite passes (all existing + new test files); this
  is the actual gate — the items below are supplementary spot checks, not a substitute.
- `.venv/bin/python classify.py --help`, `.venv/bin/python experiment.py classify --help`,
  `.venv/bin/python experiment.py run --help` each list `--models`; `.venv/bin/python
  experiment.py induce --help` does not.
- `spec/ARCHITECTURE.md` and the new ADR are internally consistent (Task 1's Verify
  checklist, re-confirmed once all of Tasks 2-4's imports actually exist).
- No separate "manual dry run" step is prescribed beyond the automated tests above —
  Tasks 2-4's own test suites already exercise `--models a b`/`a b c` end-to-end (audit
  columns, `run_config.json` contents, help text) with mocked `Classifier.classify`, per
  CLAUDE.md's no-live-network-calls policy; a redundant, less-precise manual pass adds
  risk of drifting from what's actually asserted, not confidence beyond it.

## Documentation

- Living docs to update: none under `spec/docs/` — this directory doesn't exist yet in
  this project (per `AGENTS.md`: "This project has no specs yet [as of init]" /
  `spec/docs/` not yet generated). No action needed beyond `README.md` (Task 5) and the
  ADR (Task 1).

## Spec Deviations

None identified. Every implementation detail above (the `_construct_classifiers` branch
structure, the verifying-stage `model_failure_counts` computation and its exact write
target, the `models_final` fallback-literal/merge fixes) is a direct, necessary
translation of the spec's stated observable behavior into this codebase's existing control
flow — none of them change what the spec requires or add/remove scope. (The plan's first
draft had bugs in some of these translations — a `KeyError`-inducing write target, a
merge step that silently dropped `model`, an undercounted validation-case list — caught by
plan critique and fixed above; none of those were spec-level issues, all were plan
defects, now corrected.)

## Risks

- **Risk:** `_construct_classifiers`'s return-type widening (`dict[str, str | None]` →
  something that can also hold a `list[str]`/`dict[str, int]`) could ripple into other
  call sites that assume the narrower type. **Mitigation:** grep all call sites of
  `_construct_classifiers` before changing the annotation (Task 4); the function is
  private (`_`-prefixed) with a small, enumerable call surface inside `experiment.py`
  itself (confirmed: exactly two call sites, `:749` and `:846`).
- **Risk:** every file Tasks 2-5 touch already has uncommitted, unrelated changes on disk
  (see Readiness). **Mitigation:** `git diff` each target file immediately before editing
  it; patch additively against current content; don't assume a clean baseline.
- **Risk:** `Classifier.classify`'s existing warning log (`classifier.py:338-352`) logs
  raw exception text on every failed attempt, unchanged by this feature — the
  sanitization guarantee (FR-1.7) covers only persisted CSV/`run_config.json` cells, not
  process logs. **Mitigation:** none needed for this spec's scope; stated here so nobody
  mistakes "sanitized audit columns" for "sanitized logs" — changing existing log
  behavior would be new scope requiring its own spec decision.
- **Risk:** external tooling that already parses `run_config.json` may assume the old
  4-key `models` shape or `--critics`-only `_votes` semantics. **Mitigation:** none
  in-repo (no consumer exists in this codebase today); flagged for whoever owns any
  external consumer, not actionable within this plan.
- **Rollback/checkpoint guidance:** rollback must follow the reverse-dependency groups
  stated in Implementation Strategy (`5 → {3, 4} → 2 → 1`) — reverting Task 1 alone while
  keeping Task 2 leaves `pipeline.py` importing a missing module; reverting Task 1's
  `ARCHITECTURE.md`/ADR edit alone while keeping the module restores an INV-1/INV-7
  violation. No migration of existing data occurs at any point (purely additive CLI
  flag/output columns/config keys), so no data cleanup is needed at any rollback point —
  only code/doc reversion, in the stated order. Given the dirty working tree (see
  Readiness), revert with targeted `git checkout -- <file>`/patch reversal per task, not a
  blanket reset that could also discard the unrelated pre-existing changes in those same
  files.
