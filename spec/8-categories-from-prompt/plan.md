# Implementation Plan: 8-categories-from-prompt

**Status:** Ready
**Date:** 2026-09-17
**Spec:** spec/8-categories-from-prompt/spec.md
**Plan Critique:** spec/8-categories-from-prompt/plan-critique-consolidated-v-1.md

## Overview

Build bottom-up: `schema.py`'s new fixed, unbounded-list, `extra="forbid"` model first (the
piece every later task's `Classifier` depends on), then the new prompt template/builder, then
`category_extraction.py`'s pure function (`extract_category_from_prompt`, with all of FR-1.1
through FR-1.5's validation), then that same module's `build_parser()`/`main()` (which needs
everything above), then the root script + docs/ADR. Solo, sequential — every task's tests land
in the same brand-new `tests/test_category_extraction.py` file, matching spec 5's/7's own
precedent for sequencing tasks that share one test file.

## Readiness

- **Checklist:** all 10 items `[x]` in the spec.
- **Open questions:** none — the spec's own critique round (`critique-consolidated-v-1.md`)
  already resolved every genuinely blocking design question (the `--category-name` downstream
  crash, reserved-sentinel label values, the module's import-graph/ADR requirement, the `--model`
  flag shape, `extra="forbid"`, the CLI error/exit contract, blank-string validation, output-path
  edge cases).
- **External API/library verification:** re-confirmed live in this session, matching the spec's
  own citations:
  - `create_model`-built model with `category_description: str` + `labels: list[Label]` (no
    `min_length`/`max_length`) serializes via `litellm.utils.type_to_response_format_param` with
    `"strict": true` and no `minItems`/`maxItems` on `labels` — confirmed against the currently
    installed `litellm` (same result as the spec's own Step-3 research).
  - `build_classification_model([Category(name="__root__", ...)])` raises `TypeError`;
    `name="model_config"` raises `TypeError`; `name="model_dump"` raises `ValueError` — all
    reconfirmed live against the installed pydantic, matching FR-1.2's cited crash exactly.
  - `experiment.py._check_reserved_sentinel` (`experiment.py:445-453`) exists exactly as FR-1.4
    describes: normalizes via `.strip().lower()`, rejects `"none"` or a `"none - "` prefix.
  - `schema.build_batch_model`'s `ConfigDict(extra="forbid")` precedent (`schema.py:158-183`)
    confirmed as the pattern AR-1.2 cites.
- **Repository state:** clean except pre-existing, unrelated items already present before this
  plan (`M experiments/ag-news/analyze_run.ipynb`, `M experiments/pubmed-rct/analyze_run.ipynb`,
  untracked `data/example_queries_classified.csv`, untracked
  `experiments/pubmed-rct/.ipynb_checkpoints/`) — none touched by this plan.
- **Test baseline:** re-run fresh for this plan — **34 failed, 396 passed** — every failure the
  same `botocore.exceptions.TokenRetrievalError`/`InvalidGrantException` environmental issue
  documented in specs 5/6/7's own plans (expired local AWS SSO session, `DEFAULT_LLM_PROVIDER=
  cerebus` in this repo's `.env`). This plan's own critique round explicitly re-examined whether
  this should block per `AGENTS.md`'s "`pytest` must pass" rule and FR-2.1's "full existing
  `pytest` suite must still pass" wording: the operative reading, consistent with how specs 5, 6,
  and 7 each interpreted the identical situation and were each still merged successfully, is that
  neither rule is about tolerating a *regression* this plan's own tests never touch (none of the
  34 failures are anywhere near `category_extraction.py`'s own code paths — they are all
  Cerebus/AWS-credential-resolution failures in unrelated `--batch`/`--cerebus` tests). This
  plan's own Final Verification below requires the failure *count* and *identity* to stay
  unchanged (34, the same 34 test names) — not merely "no new failures" in the abstract — so a
  genuine regression would still be caught. Not blocking for this plan, by the same standard
  already applied three times before.

## Code Impact

- **Modules:** new `src/query_classification/category_extraction.py`. Modified: `schema.py`,
  `resources.py`, `prompts.py`. New resource file: `resources/prompts/
  category_extraction_prompt.txt`. New root script: `extract_categories.py`.
- **Database/schema:** none. One brand-new artifact shape (`categories.json`, produced at a
  caller-chosen `--output` path) — reuses `categories.py`'s existing, unmodified on-disk format.
- **API/interfaces:** see Interfaces below.
- **Config/scripts:** `spec/ARCHITECTURE.md`, new `spec/8-categories-from-prompt/ADR.md`.
  `README.md` (routine new-entry-point documentation, matching every prior spec's own precedent
  — not a deviation, since the spec's checklist already expects rollout documentation).
- **Key files:** `schema.py:118-183` (`build_induction_model`/`build_batch_model` — the two
  patterns this plan's new builder combines: fixed-shape like the former, `extra="forbid"` like
  the latter); `resources.py:16-21` (`DEFAULT_*_PROMPT_FILE` constants — this plan's insertion
  point); `prompts.py:102-113` (`build_induction_prompt` — the zero-placeholder pattern this
  plan's new builder copies); `induction.py:1-16` (pure-function-w.r.t.-filesystem docstring —
  the contract `extract_category_from_prompt` follows); `experiment.py:445-453`
  (`_check_reserved_sentinel`, reused directly), `:341` (`_DEFAULT_MODEL`, copied as this
  module's own default); `cli.py:614-629` (the `gateway_kwargs`/`_model_id` Cerebus-wiring
  pattern this plan's `main()` copies), `:91-94` (`--system-prompt`, the override-flag pattern
  AR-1.3's `--extraction-prompt` copies), tail (`except (ValueError, FileNotFoundError,
  RuntimeError)` — the error/exit contract AR-1.7 copies verbatim); `classify.py`/`experiment.py`
  (repo root) — the sys.path-injection wrapper shape this plan's new `extract_categories.py`
  copies verbatim.

## Interfaces

Pinned here because every later task builds against them.

```python
# schema.py -- additive
def build_category_extraction_model() -> type[BaseModel]:
    """Fixed (no arguments -- unlike build_induction_model's n_labels), create_model-built:
    category_description: str (required)
    labels: list[_StrictLabel] (required, no min_length/max_length)
    model_config = ConfigDict(extra="forbid")

    _StrictLabel = create_model("LabelStrict", __base__=categories.Label,
    __config__=ConfigDict(extra="forbid")) -- a STRICT LOCAL COPY of Label's shape,
    exactly mirroring build_batch_model's own `f"{row_model.__name__}Strict"` pattern
    (schema.py:174-178). This is necessary, not cosmetic: plain `categories.Label` has
    no `extra="forbid"` of its own (categories.py:30-34), so an outer `extra="forbid"`
    alone does NOT reject an extra/unexpected field nested inside one label -- verified
    live (plan critique round): a raw JSON response with a bogus key inside one label
    object round-trips through model_validate_json silently, discarding the extra key,
    when `labels: list[Label]` is used directly. Using `list[_StrictLabel]` instead
    closes that gap while leaving the shared `categories.Label` class completely
    unmodified (never touches categories.py). `_StrictLabel` never leaks past
    `Classifier.classify()`'s own return value: `classify()` returns
    `result.model_dump(mode="json")` (classifier.py's existing, unchanged contract) -- a
    plain dict, not a pydantic instance -- so Task 3's pure function reads
    `result["labels"]` as a plain list of `{"value": ..., "description": ...}` dicts and
    builds ordinary `categories.Label` instances from them directly. No conversion step
    is needed; `_StrictLabel` is purely a `schema.py`-internal validation detail.

    Verified live: this shape (list[_StrictLabel], no min/max length) serializes via
    litellm.utils.type_to_response_format_param with "strict": true and no
    minItems/maxItems on `labels` -- an unbounded list is usable directly under strict
    mode, unlike a length-bounded one (Category.labels' own min_length=1, or
    build_induction_model's fixed-arity positional-field workaround)."""

# resources.py -- additive
DEFAULT_CATEGORY_EXTRACTION_PROMPT_FILE = RESOURCES_DIR / "prompts" / "category_extraction_prompt.txt"

# prompts.py -- additive
def build_category_extraction_prompt(
    system_prompt_file: str | Path = DEFAULT_CATEGORY_EXTRACTION_PROMPT_FILE,
) -> str:
    """Zero-placeholder, read-the-file-verbatim -- identical shape to
    build_induction_prompt. No category/label data is templated in; the
    instruction text is the USER message, assembled by category_extraction.py,
    not this function."""

# category_extraction.py -- new module. May import EXACTLY these six:
# categories.py, classifier.py, schema.py, prompts.py, resources.py, cost.py -- NOT a
# leaf module, but ALSO not literally "the same breadth as cli.py/experiment.py" (those
# two have their own, mutually-different import lists, both broader than these six --
# e.g. both import pipeline.py/batching.py/debate.py, which this module must NOT).
# Must NOT import: pipeline.py, batching.py, debate.py, multi_model.py, cli.py,
# experiment.py.
_RESERVED_SENTINEL = "none"                # duplicated from experiment.py, not imported --
_RESERVED_SENTINEL_PREFIX = "none - "      # category_extraction.py must not import experiment.py
                                            # (AR-1.5); see Spec Deviations row 1.

def _check_reserved_sentinel(label_values: list[str]) -> None:
    """Byte-identical duplicate of experiment.py._check_reserved_sentinel (FR-1.4) --
    duplicated, not imported, since AR-1.5 forbids this module from importing
    experiment.py. A test asserts the two stay identical (mirrors spec 5's own
    AR-2.3-style duplicate-equality test for batching.py/induction.py's
    _DATA_START/_DATA_END)."""

def extract_category_from_prompt(
    instruction_text: str, category_name: str, classifier: Classifier
) -> Category:
    """Pure function -- no filesystem access, mirroring induction.py's own contract.
    1. result = classifier.classify(instruction_text)  # unchanged Classifier call
    2. labels_raw = result["labels"]; if not labels_raw: raise ValueError("at least
       one label is required, got zero") -- FR-1.3.
    3. values = [lbl["value"] for lbl in labels_raw]; reject case-sensitive-exact
       duplicates by name -- FR-1.4 (first half).
    4. _check_reserved_sentinel(values) -- FR-1.4 (second half).
    5. Reject blank/whitespace-only result["category_description"], or any label's
       value/description, after .strip() -- FR-1.5.
    6. Return Category(name=category_name, description=result["category_description"],
       labels=[Label(value=v, description=d) for v, d in ...]) -- FR-1.1.
    category_name is NOT validated here -- FR-1.2's validation is CLI-level (argparse
    time), before this function is ever called; this function trusts its caller."""

def _validate_category_name(name: str) -> None:
    """FR-1.2: raises ValueError (not called from argparse type= -- so the message can
    name which specific rule failed) unless name.isidentifier() and name != "__root__"
    and not name.startswith("model_"). Called from main() immediately after parsing,
    before any Classifier is constructed."""

def build_parser() -> argparse.ArgumentParser: ...  # AR-1.4's flag table
def main() -> None: ...
    # 1. Parse args; call _validate_category_name(args.category_name) -- FR-1.2.
    # 2. Resolve --output (default "categories.json"); if it exists and not
    #    --overwrite, or is a symlink/directory, or its parent dir is missing --
    #    raise before any LLM call -- FR-1.7.
    # 3. resources.check_default_resources_available() if args.extraction_prompt is
    #    None (bundled default in use) -- AR-1.3.
    # 4. Read --prompt-file via Path(...).read_text(encoding="utf-8"); if blank/
    #    whitespace-only after .strip(), raise before any LLM call -- FR-1.8.
    # 5. Build gateway_kwargs exactly like cli.py's own block (cli.py:614-629) --
    #    AR-1.4's --cerebus/--api-base support.
    # 6. classifier = Classifier(model_id=_model_id(args.model), system_prompt=
    #    build_category_extraction_prompt(args.extraction_prompt),
    #    classification_model=build_category_extraction_model(),
    #    max_retries=args.retries, **gateway_kwargs)
    # 7. category = extract_category_from_prompt(instruction_text, args.category_name,
    #    classifier) -- FR-1.1/1.3/1.4/1.5.
    # 8. cost.write_json_atomic(output_path, {"categories": [category.model_dump(
    #    mode="json")]}); load_categories(output_path) round-trip check -- FR-1.6.
    # 9. Print confirmation, exit 0. except (ValueError, FileNotFoundError,
    #    RuntimeError) as e: print(f"Error: {e}"); sys.exit(1) -- AR-1.7.
```

`--model`'s default is copied as a plain string literal from `experiment.py._DEFAULT_MODEL`
(`"azure/gpt-5-chat"`) -- not imported (this module must not import `experiment.py`, AR-1.5) and
not re-derived through `cli.py`'s `nargs="+"`/`_resolve_classifier_models` machinery, which
exists specifically for `--models`/`--n-classifiers` support this feature doesn't have.

## Project Constraints

- **INV-1** (one-directional acyclic imports): `category_extraction.py` is a new node with the
  same import breadth as `cli.py`/`experiment.py` (categories/classifier/schema/prompts/
  resources/cost) -- a **third full-access orchestrator**, not a restricted leaf (AR-1.1). It
  has zero outbound imports of `cli.py`, `experiment.py`, `pipeline.py`, `batching.py`,
  `debate.py`, or `multi_model.py` -- so the graph stays acyclic. ADR amends INV-1's rule text
  and the Module Boundary Map (adds a `category_extraction.py` row, alongside `cli.py`/
  `experiment.py`'s existing entries -- it does not edit either of their rows, since neither
  imports the new module).
- **INV-4**: no new broad `except Exception` is introduced by this plan's own new code --
  `extract_category_from_prompt` lets `classifier.classify()`'s own exceptions propagate
  unchanged (AR-1.7's contract is `main()`'s outer `except (ValueError, FileNotFoundError,
  RuntimeError)`, which is not a broad `except Exception` and needs no `# noqa: BLE001`).
- **INV-7**: `category_extraction.py`'s functions, `schema.build_category_extraction_model`, and
  `prompts.build_category_extraction_prompt` are not part of the public API (not re-exported by
  `__init__.py`) and `tests/test_category_extraction.py` needs to import them directly -- the
  same shape of exception every prior spec's own new internals received. ADR requests this
  (AR-1.6). See Spec Deviations.
- **Testing policy** (`AGENTS.md`): no live network/provider calls; the sole LLM call is faked.
  Tests ship in the same task as the code they cover -- no deferred test task. All new tests
  land in one new file, `tests/test_category_extraction.py` (FR-2.1's own explicit choice).
- **PEP 604 unions + `from __future__ import annotations`** in `category_extraction.py` (the
  only new module under `src/query_classification/`; `extract_categories.py`, the repo-root
  script, does not use type hints, matching `classify.py`/`experiment.py`'s own root wrappers).
- **Verification command**: `.venv/bin/python -m pytest`. No lint/typecheck/CI gate exists.

## Implementation Strategy

- **Size:** medium (5 tasks).
- **Execution mode:** solo, sequential.
- **Parallelizable work:** none. Every task adds tests to the same brand-new
  `tests/test_category_extraction.py` file -- a real, shared-file dependency across all five
  tasks, mirroring spec 7's own Task 3/4 reasoning (there: an *existing* shared file; here: a
  *new* one every task must extend rather than each independently create).
- **Sequential blockers:** Task 3 (the pure function) needs Task 1's schema shape only for its
  own tests (constructing a `Classifier` to fake `classify()` against) -- not a hard code
  dependency, but tested against the real schema for realism. Task 4 (`main()`) needs Tasks 1
  (schema), 2 (prompt), and 3 (pure function) to exist. Task 5 (root script + docs/ADR) needs
  the real, shipped shape from Tasks 1-4.
- **No expected-red checkpoints.** Every new file is purely additive; no existing test's
  behavior is touched by any task until Task 5's `spec/ARCHITECTURE.md` edit (docs-only).

## Implementation Tasks

### Task 1: `schema.py` -- `build_category_extraction_model()`
**Goal:** the fixed, unbounded-list, `extra="forbid"` response schema every later task's
`Classifier` is built against.
**Files:** `src/query_classification/schema.py`, `tests/test_category_extraction.py` (new)
**Dependencies:** None
**Do:**
- `schema.py` currently imports only `Category` from `categories.py` (`schema.py:14`) -- add
  `Label` to that same import line; it does not already import it.
- Build a strict local copy of `Label`'s shape first: `_LabelStrict = create_model(
  "LabelStrict", __base__=Label, __config__=ConfigDict(extra="forbid"))` -- exactly mirroring
  `build_batch_model`'s own `f"{row_model.__name__}Strict"` pattern (`schema.py:174-178`). Plain
  `categories.Label` has no `extra="forbid"` of its own (`categories.py:30-34`); without this
  local strict copy, an outer `extra="forbid"` alone does **not** reject an extra/unexpected
  field nested inside one label object (verified live during plan critique: a raw JSON response
  with a bogus key inside one label round-trips through `model_validate_json`, silently
  discarding the extra key, when `labels: list[Label]` is used directly). This never modifies
  the shared `categories.Label` class itself.
- Add `build_category_extraction_model() -> type[BaseModel]`: `create_model(
  "CategoryExtractionResult", __config__=ConfigDict(extra="forbid"),
  category_description=(str, ...), labels=(list[_LabelStrict], ...))`.
- No `n_labels`/other parameters -- unlike `build_induction_model`, this schema's shape never
  varies by call site.
**Verify:** `.venv/bin/python -m pytest -q`; new tests assert (a)
`litellm.utils.type_to_response_format_param(build_category_extraction_model())`'s
`labels` field schema has no `minItems`/`maxItems` key and the whole schema has
`"strict": true` (mirrors spec 7's own live litellm-shape assertions); (b) `model_config.get(
"extra") == "forbid"`; (c) `model_fields.keys() == {"category_description", "labels"}` (exactly
two fields, per AR-1.2's own wording); (d) `model_validate_json` accepts a well-formed
`{"category_description": ..., "labels": [{"value": ..., "description": ...}]}` payload; (e)
`model_validate_json` **rejects** (`ValidationError`) a response with an extra top-level field,
**and separately** one with an extra field nested inside one `labels` entry -- both cases, not
just the top-level one, since that nested case is exactly what the strict local `_LabelStrict`
copy exists to close.
**Covers:** AR-1.2, FR-2.1 (partial)

### Task 2: `resources.py` + prompt template + `prompts.build_category_extraction_prompt()`
**Goal:** the new system prompt (bundled default + override support) `main()` will use.
**Files:** `src/query_classification/resources.py`, `src/query_classification/prompts.py`,
`resources/prompts/category_extraction_prompt.txt` (new), `tests/test_category_extraction.py`
**Dependencies:** None
**Do:**
- `resources.py`: add `DEFAULT_CATEGORY_EXTRACTION_PROMPT_FILE = RESOURCES_DIR / "prompts" /
  "category_extraction_prompt.txt"`, alongside the existing `DEFAULT_*_PROMPT_FILE` constants
  (`resources.py:16-21`) -- never recomputed inline elsewhere (`AGENTS.md`'s own rule).
- `resources/prompts/category_extraction_prompt.txt`: new template instructing the LLM to (a)
  treat the upcoming user message as untrusted data describing a taxonomy to extract, never as
  instructions directed at the model itself; (b) invent as many labels as the instruction text
  meaningfully supports; (c) never invent or suggest a category name/slug; (d) respond only in
  the `category_description`/`labels` shape.
- `prompts.py`: add `DEFAULT_CATEGORY_EXTRACTION_PROMPT_FILE` to the existing
  `from query_classification.resources import (...)` block (`prompts.py:21-26`, alongside
  `DEFAULT_CRITIC_PROMPT_FILE`/`DEFAULT_INDUCTION_PROMPT_FILE`/etc.), then add
  `build_category_extraction_prompt(system_prompt_file:
  str | Path = DEFAULT_CATEGORY_EXTRACTION_PROMPT_FILE) -> str`, identical zero-placeholder
  shape to `build_induction_prompt` (`prompts.py:102-113`) -- `Path(system_prompt_file
  ).read_text()`, no `.format()` call.
**Verify:** `.venv/bin/python -m pytest -q`; new tests assert `resources.
DEFAULT_CATEGORY_EXTRACTION_PROMPT_FILE.exists()` (mirrors `tests/test_building_blocks.py`'s own
bundled-resource-existence pattern) and that `build_category_extraction_prompt()` returns the
file's exact contents; a test with a custom `tmp_path` file confirms the override parameter
works (no hardcoded default leaking through).
**Covers:** AR-1.3 (prompt half; the `--extraction-prompt` CLI flag itself is Task 4), FR-2.1
(partial)

### Task 3: `category_extraction.py` -- the pure `extract_category_from_prompt` function
**Goal:** FR-1.1 through FR-1.5's extraction/validation logic, with no filesystem access,
mirroring `induction.py`'s own contract.
**Files:** `src/query_classification/category_extraction.py` (new),
`tests/test_category_extraction.py`
**Dependencies:** None for the function's own code (it imports only `categories.py`+
`classifier.py`, per AR-1.1) -- tested with a duck-typed fake `Classifier`-shaped object, so
Task 1's schema is not a code dependency here. (If a test instead prefers a real `Classifier`
for realism, it may build one against Task 1's `build_category_extraction_model()`, but that is
a test-authoring choice, not a requirement of this task.)
**Do:**
- Duplicate `experiment.py`'s `_RESERVED_SENTINEL`/`_RESERVED_SENTINEL_PREFIX`/
  `_check_reserved_sentinel` verbatim (not imported -- AR-1.5 forbids importing
  `experiment.py`). A test asserts the duplicate stays byte-identical to
  `experiment.py._check_reserved_sentinel`'s behavior (mirrors spec 5's own
  `_DATA_START`/`_DATA_END` duplicate-equality precedent).
- `extract_category_from_prompt(instruction_text, category_name, classifier) -> Category`:
  call `classifier.classify(instruction_text)` (unchanged `classify()` signature/contract);
  reject zero labels (FR-1.3); reject duplicate label values by exact string match (FR-1.4,
  first half); reject reserved-sentinel label values via the duplicated check (FR-1.4, second
  half); reject a blank/whitespace-only `category_description` or any label's `value`/
  `description` after `.strip()` (FR-1.5); construct and return the `Category`.
- Deliberately do **not** validate `category_name` here (FR-1.2's validation is CLI-level,
  Task 4) -- this function trusts its caller, exactly as `induction.py`'s own pure function
  trusts `experiment.py` to have validated `--category-name`'s non-emptiness upstream.
**Verify:** `.venv/bin/python -m pytest -q`; new tests cover: 3 labels round-trip verbatim
(FR-1.1); zero labels raises `ValueError` mentioning "at least one label" (FR-1.3); two labels
both valued `"urgent"` raises `ValueError` naming `"urgent"` (FR-1.4); a label valued `"none"`/
`"NONE"`/`"none - invented"` raises `ValueError` naming the reserved value (FR-1.4); a blank
`category_description`, an empty label `value`, and a whitespace-only label `description` each
raise `ValueError` (FR-1.5); the duplicated `_check_reserved_sentinel`, called directly (not
through `extract_category_from_prompt`), behaves identically to `experiment.py`'s own function
for the same parametrized set of inputs (accept/reject cases side by side) -- a real parity
check, not merely "looks similar," per the plan critique's concern that behavioral-parity claims
need an actual comparison, not just independent assertions.
**Covers:** FR-1.1, FR-1.3, FR-1.4, FR-1.5, FR-2.1 (partial)

### Task 4: `category_extraction.py` -- `build_parser()`/`main()`
**Goal:** the CLI surface, `--category-name` validation, output-path handling, and the full
read -> classify -> validate -> write -> confirm flow.
**Files:** `src/query_classification/category_extraction.py`,
`tests/test_category_extraction.py`
**Dependencies:** Tasks 1, 2, 3
**Do:**
- `build_parser()`: AR-1.4's flag table exactly -- `--prompt-file`/`--category-name` (both
  required), `--output` (default `"categories.json"`), `--overwrite` (flag),
  `--extraction-prompt` (default `None`, resolved to
  `resources.DEFAULT_CATEGORY_EXTRACTION_PROMPT_FILE` only when omitted -- mirrors `cli.py`'s
  own `--system-prompt` convention exactly), `--model` (plain `type=str`, default
  `"azure/gpt-5-chat"` -- copied as a literal, not imported from `experiment.py`), `--retries`
  (default 3), `--cerebus` (flag), `--api-base`.
- `main()` calls `load_dotenv(override=True)` and the same quiet-logging setup as its first
  action, before argument parsing -- matching both `cli.py:473-479` and
  `experiment.py:1147-1153` exactly. Omitting this (an early draft of this task did) would mean
  `DEFAULT_LLM_PROVIDER=cerebus`/`.env`-based credentials behave differently in this entry point
  than in the other two, contradicting AR-1.4's own "matching `cli.py`/`experiment.py`" framing.
- `_validate_category_name(name)`: `str.isidentifier()`, `name != "__root__"`, not
  `name.startswith("model_")` -- raise `ValueError` naming which rule failed. Called
  immediately after parsing, before anything else (FR-1.2).
- Output-path preflight, before any LLM call (FR-1.7). **Ordering matters:** check
  `Path(args.output).is_symlink()` on the *lexical*, as-given path **first** -- `Path.resolve()`
  follows the final symlink, so resolving before checking would inspect the symlink's *target*,
  not the caller-supplied link itself, and could silently miss the refusal FR-1.7 requires. Only
  after that check passes, resolve the path to test existence/parent-directory/directory-ness:
  if it exists and `--overwrite` is not set, or it is an existing directory, or its parent
  directory does not exist -- raise a clear `ValueError`/`FileNotFoundError`.
- Read `--prompt-file` via `Path(...).read_text(encoding="utf-8")`; if blank/whitespace-only
  after `.strip()`, raise before any LLM call (FR-1.8).
- Build `gateway_kwargs` exactly like `cli.py:614-629`'s own block (`--cerebus`/`--api-base`
  handling, `_model_id` prefixing via `cerebus_model_id`) -- copy the pattern, not the code
  (`cli.py` is not imported, per AR-1.5).
- `args.cerebus = args.cerebus or cerebus_enabled_via_env()`, matching `cli.py`/`experiment.py`'s
  own `DEFAULT_LLM_PROVIDER=cerebus` opt-in convention.
- `--extraction-prompt` defaults to `None` in `build_parser()` (matching `cli.py`'s own
  `--system-prompt` convention, `cli.py:91-94`), resolved to
  `resources.DEFAULT_CATEGORY_EXTRACTION_PROMPT_FILE` only when omitted. When
  `args.extraction_prompt is None` (mirroring `cli.py`'s own `uses_bundled_defaults` check,
  `cli.py:482-488`), call `resources.check_default_resources_available()` before reading any
  prompt file, but catch the `FileNotFoundError` it raises and re-raise with this entry point's
  own remediation text (its message names `--categories`/`--system-prompt`/
  `--task-description`, `resources.py:34-40` -- none of which are flags on this tool; re-raising
  with `--extraction-prompt` named instead keeps the guard's mechanical behavior while fixing
  its advice for this specific caller, without editing the shared helper or its message for the
  two existing entry points) (AR-1.3).
- Construct the `Classifier` (Task 1's schema, Task 2's prompt), call
  `extract_category_from_prompt` (Task 3), write via `cost.write_json_atomic`, round-trip via
  `load_categories(output_path)` (FR-1.6), print a one-line confirmation naming the resolved
  output path, exit 0.
- Wrap the whole body in `except (ValueError, FileNotFoundError, RuntimeError) as e:
  print(f"Error: {e}"); sys.exit(1)` (AR-1.7) -- identical to `cli.py`/`experiment.py`'s own
  `main()` tail; no new exception sanitization.
**Verify:** `.venv/bin/python -m pytest -q`. Every distinct clause gets its own assertion (not
folded into one broad "it works" test):
- `build_parser().parse_args([])` raises `SystemExit` (missing required flags).
- `--category-name __root__`/`model_config`/`"not an identifier"` each fail with a clear error
  before any LLM call (call-counting fake shows **zero** calls); a valid name's resulting
  `Category.name` equals the exact string passed.
- Zero labels: a fake response with `"labels": []` fails, and the `--output` path does **not**
  exist afterward (the CLI-level counterpart of Task 3's own pure-function check).
- Running twice with the same `--output` and no `--overwrite` fails the second time (zero
  further LLM calls); `--overwrite` succeeds and the file's new content differs from the first
  run's (not just "the call succeeded").
- A symlinked `--output` (pointing at an existing file) fails before any LLM call, with or
  without `--overwrite`; separately, an `--output` pointing at an existing directory, and one
  whose parent directory doesn't exist, each fail before any LLM call.
- An empty (0-byte) and a whitespace-only `--prompt-file` each fail before any LLM call.
- A successful run's `load_categories(output_path)` returns one `Category` matching the run, and
  exactly one classifier call was made (call-counting fake).
- At least one test constructs a real `Classifier` via `main()`'s own path (monkeypatching
  `litellm.completion`, never `Classifier.classify`, so the fake sits below `classify()`'s own
  logic) and asserts the resulting `Classifier`'s `model_id`/system prompt content/
  `max_retries`/gateway kwargs are wired correctly; a `--cerebus` variant with a non-HTTPS
  `--api-base` fails via the existing `reject_insecure_cerebus_endpoint` guard, before any LLM
  call.
- On a handled failure, stdout contains the `Error: ...` line, no traceback is printed, and the
  process exits with code 1; on success, exit code is 0 and stdout contains the confirmation
  line naming the output path.
- A missing bundled `--extraction-prompt` default (simulate via monkeypatching
  `resources.RESOURCES_DIR` to a nonexistent path) fails with a message naming
  `--extraction-prompt`, not `--categories`/`--system-prompt`/`--task-description`.
**Covers:** FR-1.2, FR-1.6, FR-1.7, FR-1.8, AR-1.3 (flag half), AR-1.4, AR-1.7, FR-2.1
(remaining)

### Task 5: Root script + docs/ARCHITECTURE/ADR
**Goal:** the standalone entry point is runnable exactly like `classify.py`/`experiment.py`, and
the architecture docs/ADR reflect the new module.
**Files:** `extract_categories.py` (new, repo root), `spec/ARCHITECTURE.md`,
`spec/8-categories-from-prompt/ADR.md` (new), `README.md`,
`src/query_classification/cost.py` (docstring only, one line -- "the two entry points" becomes
"three entry points" or names all three)
**Dependencies:** Task 4
**Do:**
- `extract_categories.py`: byte-for-byte the same shape as `classify.py`/`experiment.py`'s own
  root wrappers -- `sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))`, `from
  query_classification.category_extraction import main`, `if __name__ == "__main__": main()`.
- `spec/ARCHITECTURE.md`: add `category_extraction.py` to the mermaid graph and the Module
  Boundary Map as a **third orchestrator row** -- "May import from" the exact six modules
  `categories.py`/`classifier.py`/`schema.py`/`prompts.py`/`resources.py`/`cost.py` (**not**
  "same breadth as `cli.py`/`experiment.py`" -- those two rows are broader, and mutually
  different from each other and from this one; e.g. both import `pipeline.py`/`batching.py`/
  `debate.py`, which this new module must not); "Must NOT import from" =
  `pipeline.py`/`batching.py`/`debate.py`/`multi_model.py`/`cli.py`/`experiment.py`. Amend INV-1's
  rule/evidence text with this exact list (not a vague "same as" reference), and amend `cost.py`
  row / `cost.py`'s own module docstring (`cost.py:1-8`, which currently says it serves only "the
  two entry points (`experiment.py`/`cli.py`)") to note a third importer now exists. Amend INV-7's
  `tests/` exception list + Module Boundary Map row to add `category_extraction` (all functions),
  `schema.build_category_extraction_model`, `prompts.build_category_extraction_prompt`.
- `spec/8-categories-from-prompt/ADR.md`: one record covering both INV-1 (new third-orchestrator
  module) and INV-7 (new test-import names), citing spec 5's/7's own ADR-001s as precedent for
  both, per AR-1.6.
- `README.md`: a short new-entry-point section (mirrors how the Experiment Runner section
  documents `experiment.py`) -- usage example, flag summary, and an explicit one-line note that
  this is a standalone tool today, with `experiment.py` integration deliberately deferred (per
  the spec's own "future, not yet written" Related Specs row).
**Verify:** `.venv/bin/python extract_categories.py --help` succeeds with no editable install;
`grep category_extraction.py spec/ARCHITECTURE.md` matches in the boundary map, the graph, and
INV-1; `grep` for the new names matches in the `tests/` boundary-map row and INV-7's evidence,
alongside (not replacing) every prior spec's own provenance note; `README.md` mentions
`extract_categories.py`.
**Covers:** AR-1.1, AR-1.5, AR-1.6

## Final Verification

- `.venv/bin/python -m pytest -q` -- the failing-test *names* (not just the count) are identical
  to this plan's own Readiness baseline (`grep FAILED` diffed against the pre-implementation run,
  matching how spec 7's own implementation verified this exact situation); this plan's new tests
  add to the passing count, and no existing test is removed or weakened.
- `.venv/bin/python extract_categories.py --help` -- run as an actual subprocess; this is the
  **only** check that literally invokes the root script as a separate process (no LLM call is
  reachable via `--help`, so no fake is needed for it).
- A faked-LLM end-to-end run, 3 labels, valid category name, fresh `--output` path, producing a
  `categories.json` `load_categories()` accepts with exactly those 3 labels in order -- this
  runs **in-process**, calling `category_extraction.main()` directly with `sys.argv` set and
  `litellm.completion` monkeypatched (a normal subprocess cannot see a pytest monkeypatch, so
  this must not be attempted against the root script as a subprocess -- Task 4's own tests
  already establish this in-process pattern; this is the same test, restated as a whole-flow
  sanity check, not a new mechanism).
- `grep -rn "import experiment" src/query_classification/category_extraction.py` -- no matches
  (confirms AR-1.5's "does not import `experiment.py`" holds; the reserved-sentinel check is a
  verified duplicate, not an import).
- `grep -n "def classify(" src/query_classification/classifier.py` plus
  `grep -n "\.classify(" src/query_classification/category_extraction.py` -- the new module's
  own call site passes only `instruction_text`, matching `classify()`'s existing, completely
  unchanged signature (AR-1.1 of spec 7, still holding; this plan touches `classifier.py` not at
  all).
- `grep -rn "litellm.completion(" tests/test_category_extraction.py` (excluding
  monkeypatch/fixture setup lines) -- no matches, confirming `AGENTS.md`'s "no live
  network/provider calls in tests" rule holds for the new file.

## Documentation

- Living docs: none exist under `spec/docs/` (never generated for this repo, same as specs 5/7).
- `spec/ARCHITECTURE.md` -- Module Boundary Map, mermaid graph, INV-1, INV-7 (Task 5).
- `README.md` -- new entry-point section (Task 5).
- `spec/8-categories-from-prompt/ADR.md` -- new, one record (Task 5).

## Spec Deviations

| # | Spec says | Plan does instead | Reason | Action required |
|---|-----------|-------------------|--------|-----------------|
| 1 | FR-1.4 says label values are "passed through `experiment.py`'s existing `_check_reserved_sentinel`" | Plan duplicates `_check_reserved_sentinel`'s exact logic (constants + function body) inside `category_extraction.py`, rather than importing it from `experiment.py` | AR-1.5 and INV-1, both also spec text, forbid `category_extraction.py` from importing `experiment.py` at all — so FR-1.4's literal phrasing and AR-1.5's own import restriction are in tension with each other, and one implementer reading FR-1.4 alone could reasonably try an import that AR-1.5 then blocks. This plan's critique round re-examined this specific tension (raised independently by the external Codex critique, which argued it should instead be sent back via `/spec-update`) and judged it resolvable at the plan level rather than the spec level: read together, the only non-contradictory interpretation of "passed through the existing `_check_reserved_sentinel`" + "must not import `experiment.py`" is "behaves identically to that function, without literally importing it" — exactly what duplication plus Task 3's own parametrized parity test (not just a vibes-based "looks similar" claim) delivers. This is the same shape of deviation spec 5 already made routine for `_DATA_START`/`_DATA_END` between `batching.py`/`induction.py`, which was likewise never escalated to `/spec-update` | None (see reasoning; a defensible, actively-reconsidered reading of two spec passages that would otherwise contradict each other, not a new behavior) |
| 2 | Spec doesn't explicitly require a README update | Task 5 adds a short new-entry-point section to `README.md` | Every prior spec that added a new entry point (`experiment.py`'s own Experiment Runner section) documented it in the README; the spec's own Rollout & migration checklist item implies this is expected, and the spec's own new "future, not yet written" Related Specs row is naturally documented here too | None |

## Risks

- **Risk: `_check_reserved_sentinel`'s duplicated logic silently drifts from
  `experiment.py`'s original if one is edited without the other.** Mitigated by Task 3's own
  duplicate-equality test, mirroring the exact mitigation spec 5 already uses for
  `batching.py`'s/`induction.py`'s own duplicated `_DATA_START`/`_DATA_END` constants.
- **Risk: `--category-name` validation (`isidentifier()` + two reserved-name rules) is
  narrower or wider than what actually crashes `build_classification_model`.** Low: Task 1's/
  Task 4's own tests reproduce the exact three live-verified crash cases from the spec's
  critique round; a future pydantic version changing its protected-namespace rules would need
  to be caught by a new test at that time, not by this plan.
- **Risk: the new module's "third full-access orchestrator" shape could be mistaken for a
  green light to casually import `cli.py`/`experiment.py`'s own internals.** Mitigated by
  AR-1.5's explicit "does not import `cli.py`, `experiment.py`, `pipeline.py`, `batching.py`,
  `debate.py`, `multi_model.py`" list, restated in this plan's own Task 5 ARCHITECTURE.md edit
  and checked by the Final Verification's `grep`.
- **Risk: this repo's established pattern of sequencing the ADR/architecture-doc update as the
  final task (mirrored here as Task 5) means Tasks 1-4's own commits temporarily add
  code/tests that `spec/ARCHITECTURE.md` doesn't yet describe.** This is not unique to this
  plan -- specs 2, 4, 5, and 7 all used the identical "docs/ADR last" task ordering and were
  each merged successfully with it. Accepted as consistent, working precedent, not fixed here;
  `spec/ARCHITECTURE.md` is a description of the *shipped* state after a spec closes, not a
  gate each intermediate commit must individually satisfy.
- **Rollback/checkpoint guidance:** each task is a commit on the spec worktree branch. Tasks 1-3
  are purely additive and independently revertible (new functions/resource file, touching no
  existing behavior). Task 4 depends on Tasks 1-3's actual symbols (imports
  `build_category_extraction_model`/`build_category_extraction_prompt`), so reverting any of
  Tasks 1-3 alone while keeping Task 4 would break those imports -- if a rollback is needed
  before Task 5 lands, revert Task 4 first, then whichever earlier task caused the issue. Task
  5's `spec/ARCHITECTURE.md`/`ADR.md`/`README.md`/`cost.py`-docstring edits are docs-only and
  trivially revertible on their own. A `categories.json` file a user already generated with this
  tool is an external artifact, not touched by any code rollback.
