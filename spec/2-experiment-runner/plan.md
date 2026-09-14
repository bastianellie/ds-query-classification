# Implementation Plan: 2-experiment-runner

**Status:** Ready
**Date:** 2026-09-14
**Spec:** spec/2-experiment-runner/spec.md
**Plan Critique:** spec/2-experiment-runner/plan-critique-consolidated-v-2.md

## Overview

Implements the 2026-09-14 post-close induction redesign only — the rest of spec 2 is
already shipped and must not move. Two coupled changes: the induction response contract
becomes **positional** (`description_1…description_N`, no label names, arity pinned by
required fields), and example sampling gains a **total-budget** mode
(`--induction-examples`) alongside the existing per-label cap. The interim
`--induction-retries` flag is removed, since the failure mode it was added for stops
existing.

Tasks 1–3 are **one atomic change** — the contract, the algorithm and the wiring cannot be
landed independently without leaving the suite red (see Implementation Strategy). Task 4 is
separable.

## Readiness

- Checklist: all 10 items closed `[x]` in the spec — no open gaps.
- Open questions: none. The four design decisions were resolved before planning (flag
  coexistence: added alongside, mutually exclusive; rounding: largest-remainder, exact
  total; shortfall: redistribute to labels with spare rows; response shape: N required
  `description_i` fields, not a length-bounded list). The one spec ambiguity the v-2
  critique surfaced — what `run_config.json` records for a `classify`-only run, which has
  no induction flags at all — was **fixed directly in the spec** during that round
  (Data Requirements + FR-2.1 now state both keys are null there). No `/spec-update` round
  is outstanding.
- External API/library verification: no new dependencies. Installed: pydantic 2.13.5,
  litellm 1.100.0. Verified live, twice independently (this plan and the Codex critique):
  `create_model` with dynamic `description_i` fields marks every field `required` with
  `additionalProperties: false`; `litellm.utils.type_to_response_format_param` emits
  `"strict": true`; a `min_length`/`max_length` list emits `minItems`/`maxItems`, which
  Azure's structured-output mode lists as unsupported array keywords — which is why AR-2.1
  uses required fields.
- Repository state: **dirty**, deliberately. Uncommitted work this plan builds on or must
  not clobber: the `--induction-retries` implementation it removes (`induction.py`,
  `experiment.py`, `tests/test_experiment.py`, `README.md`), the spec update it implements
  (`spec/2-experiment-runner/spec.md`), the two new experiment suites
  (`experiments/trec/`, `experiments/pubmed-rct/`, untracked), and pre-existing user edits
  to `experiments/ag-news/analyze_run.ipynb` plus the user-authored, already-executed
  `experiments/trec/analyze_run.ipynb` — **neither notebook is in scope and neither may be
  regenerated or reformatted.**

## Code Impact

- Modules/services: `schema.py` (builder signature + shape), `induction.py` (sampling
  algorithm, payload shape, Category construction, retry/reconciliation removal, module +
  function docstrings), `experiment.py` (two flags, one sizing resolver, FR-2.7 preflight
  relocation, `n_labels` threading, run_config, `_SCHEMA_VERSION`).
- Database/schema: `run_config.json` — `induction_examples` added, `examples_per_label`
  nullable, `induction_retries` removed; `_SCHEMA_VERSION` 3 → **4**.
- API/interfaces: `build_induction_model()` → `build_induction_model(n_labels)`;
  `induce()` loses `max_reconciliation_retries`, swaps `examples_per_label` for the two
  mutually exclusive sizing params, and gains a dependency on its classifier exposing
  `classification_model`; `_construct_classifiers()` gains `n_labels`. All internal
  (INV-7 test exceptions); none are in `__init__.py`'s public surface.
- Test infrastructure: `FakeInductionClassifier` (`tests/test_experiment.py:331`) must grow
  a `classification_model` attribute — see Task 1.
- UI/config/background jobs: `resources/prompts/induction_prompt.txt`, `README.md`, the
  three `experiments/*/run_experiment.sh` scripts and their READMEs.
- Key files: `src/query_classification/schema.py:118`,
  `src/query_classification/induction.py:1-2,54-207`,
  `src/query_classification/experiment.py:86,168-200,437-443,599-650,845-849,920-947`,
  `resources/prompts/induction_prompt.txt`, `tests/test_experiment.py:331-612,1197`,
  `README.md:243-245`, `spec/ARCHITECTURE.md:78`,
  `spec/2-experiment-runner/implementation-summary.md:36`.
- **Must NOT be touched:** `cli.py`, `pipeline.py`, `debate.py`, `multi_model.py`,
  `dataset_io.py`, `classifier.py` (AR-3.1/AR-3.2 keep `classify.py` frozen). Confirm with a
  zero-diff check at the end.

## Project Constraints

- **INV-1 is the binding one.** `induction.py` may import only `categories.py` +
  `classifier.py`, and is explicitly forbidden `schema.py` (`spec/ARCHITECTURE.md:78`), so
  the induction model **cannot** be built inside `induction.py`. It stays in
  `experiment.py`, which means `n_labels` must reach `_construct_classifiers`. Verified
  feasible without reordering: `project_and_filter` already runs at `experiment.py:920`,
  before `_construct_classifiers` at `:935`. **No ADR is required** — no import edge,
  module, or invariant changes.
- INV-11: the schema enforces shape and cardinality only. Required-field arity is
  cardinality, so AR-2.1 stays inside INV-11 — do not add label-value validation there.
- INV-7: tests may import `induction`/`experiment`/`schema.build_induction_model` directly
  (existing exceptions). Task 1's fake change stays inside them.
- INV-4: any new broad `except Exception` needs `# noqa: BLE001` + a reason.
  `induction.py:165`'s existing one stays.
- CLAUDE.md: `from __future__ import annotations` + PEP 604 unions in every
  `src/query_classification/` module; **no live network/provider calls in tests**;
  `pytest` must pass before a change is done; no lint/typecheck/CI gate exists.

## Implementation Strategy

- Size: medium — 4 tasks, but only 2 independently landable units.
- Execution mode: solo. **Tasks 1–3 are one atomic commit.**
- **Honest checkpoint story (corrected after the v-2 critique):** Tasks 1 and 2 each break
  callers that Task 3 fixes — Task 1 changes `build_induction_model`'s arity while
  `experiment.py:645` still calls it with no argument; Task 2 changes `induce()`'s signature
  while `experiment.py:938-948` still passes the old arguments. **The full suite is expected
  RED between Task 1 and the end of Task 3.** Each task's `Verify:` line below is therefore
  scoped to the subset that can legitimately pass at that point; only Task 3 gates on the
  whole suite. Do not "fix" an intermediate red by adding compatibility shims.
- Parallelizable work: Task 4 (scripts, README, ARCHITECTURE row, implementation-summary
  note) touches no source file Tasks 1–3 touch and can proceed as soon as the two flag
  names and the persisted key names are frozen. Tasks 1–3 share
  `tests/test_experiment.py` and a hard interface chain, so they do not parallelize — but
  note that is an *interface* dependency, not merely a shared-file one.
- Sequential blockers: Task 1 defines the `description_i` names and the fake's new
  attribute; Task 2 consumes both and defines `induce()`'s signature; Task 3 calls it and
  restores a green suite.

## Implementation Tasks

### Task 1: Response contract — `schema.py`, induction prompt, test fake
**Goal:** the induction output model is `category_description` plus N **required**
`description_i` fields; the prompt states that contract; the test fake can satisfy Task 2's
arity guard.
**Files:** `src/query_classification/schema.py`,
`resources/prompts/induction_prompt.txt`, `tests/test_experiment.py`
**Dependencies:** None
**Do:**
- `build_induction_model()` (`schema.py:118`) → `build_induction_model(n_labels: int)`,
  returning `create_model("InductionResult", category_description=(str, ...),
  **{f"description_{i}": (str, ...) for i in range(1, n_labels + 1)})`. Raise `ValueError`
  for `n_labels < 1`. Delete the `InductionLabel`/`labels` list model. Reuse `create_model`
  exactly as `build_classification_model` (`schema.py:63`) does.
- Rewrite only the closing response-format paragraph of `induction_prompt.txt` — it
  currently asks for `a "labels" array with one entry per label, each entry containing
  that label's exact name`. Replace with the positional contract (the payload lists each
  label with its 1-based position; the description for position `i` goes in
  `description_i`). **Leave the untrusted-data paragraph verbatim** (AR-2.2; it is the
  injection defense). This paragraph is load-bearing: if strict structured output is
  unavailable, `Classifier._attempt_completion` falls back to `json_object` mode where the
  prompt is the *only* statement of the field names.
- **Give `FakeInductionClassifier` (`tests/test_experiment.py:331`) a
  `classification_model` attribute**, defaulting to a real `build_induction_model(n)` for
  the label count each test uses. Verified necessary: the fake currently exposes only
  `.classify()`/`response`/`error`/`responses`/`calls`, so Task 2's arity guard would raise
  `AttributeError` in every direct `induce()` test. This widens the induction fake's
  contract beyond FR-3.7's `.classify(text) -> dict` minimum — an intentional, recorded
  consequence (Spec Deviations #2), not an accident.
- Replace `test_build_induction_model_list_shape_not_dynamic_field` (`:598`) — name and
  body both assert the deleted list shape. New tests: exact field set for `n_labels=1` and
  `n_labels=3`; a response missing `description_2` is rejected; `n_labels=0` raises.
- Add a **serialization-shape regression test** (not a provider test — it cannot prove
  provider rejection, only that the local schema stays strict-compatible): assert
  `litellm.utils.type_to_response_format_param(build_induction_model(3))` yields
  `strict is True`, every `description_i` in `required`, and **no** `minItems`/`maxItems`
  anywhere. Name/comment it so the limit of what it proves is explicit.
- Add a **template-contract test**: the rendered induction prompt mentions `description_`
  and no longer contains `exact name`. Without it a schema-changed/prompt-unchanged state
  passes the entire offline suite (the fakes never read the template) and fails only against
  a live provider.
- Leave `build_induction_prompt` (`prompts.py:102`) unchanged — placeholder-free template,
  no label parameters, nothing response-shape-dependent.
  `test_build_induction_prompt_non_empty_no_placeholders` (`:608`) must keep passing, which
  also guards against a stray `{`/`}` in the rewrite.
**Verify:** `.venv/bin/python -m pytest tests/test_experiment.py -q -k "build_induction or
induction_prompt"` passes. **The full suite is expected red until Task 3** — do not gate
here on it.
**Covers:** AR-2.1, AR-2.2 (system-prompt half)

### Task 2: `induction.py` — two sizing modes, positional reconstruction, retry removal
**Goal:** `induce()` samples by either cap or total budget, sends an explicitly positional
payload, rebuilds the `Category` from its own label list, and contains no retry or
label-set reconciliation code or prose.
**Files:** `src/query_classification/induction.py`, `tests/test_experiment.py`
**Dependencies:** Task 1
**Do:**
- Signature: drop `max_reconciliation_retries` (`:65`); replace `examples_per_label: int`
  with `examples_per_label: int | None` **and** `induction_examples: int | None`, exactly
  one non-`None` (raise naming both otherwise — `induce()` is called directly by tests as
  well as through the CLI). Task 3 owns resolving the default so the CLI never passes
  `(None, None)`.
- Quota computation, replacing the single comparison at `:123-127`:
  - cap mode — unchanged: `min(len(stable_positions), examples_per_label)`.
  - budget mode — every label gets `N // n_labels`, then the `N % n_labels` leftovers go one
    each to **the first `N % n_labels` labels in sorted order**. That is exactly equivalent
    to the spec's "largest fractional part, ties by sorted label value": because the budget
    is split *equally* rather than proportionally to row counts, every fractional part is
    identical (`rem/n_labels`), so sorted order is the entire rule. Do not compute
    fractional parts; do not infer a proportional weighting.
  - shortfall: clamp each label to its row count, then allocate **the current remaining
    shortfall as increments** among the labels that still have spare capacity, by the same
    sorted-order rule, clamp again, and repeat until the budget is met or no spare capacity
    remains. Phrase it as incremental allocation — "re-divided by the same rule" was
    ambiguous enough to be read as recomputing whole quotas over the remaining labels.
    Quotas must sum to `min(N, total usable rows)`. This was prototyped against 7 edge
    cases (including a cascading shortfall needing multiple passes) before finalizing;
    reproduce them as tests rather than re-deriving.
  - A label with **zero** usable rows is impossible by construction, not an edge case:
    `distinct_labels` comes from the column's own unique values.
  - Raise if `induction_examples < len(distinct_labels)` (FR-2.1's floor).
- Keep `rng = random.Random(seed)` as the single generator consumed in sorted-label order
  (`:118`), and preserve the at-or-below-quota "consumes no RNG state" branch — in cap mode
  that is FR-2.1's cross-label-independence guarantee, asserted by
  `test_induce_at_cap_label_consumes_no_rng_state` (`:404`); in budget mode it is
  explicitly given up, so only seed-determinism is asserted there.
- **Arity guard (no new import).** `induction.py` cannot import `schema.py` (INV-1), so read
  the model it was handed: compare the field names of
  `induction_classifier.classification_model.model_fields` against the **exact expected
  set** `{"category_description", "description_1", …, "description_N"}` and raise a clear
  internal-consistency error on mismatch. Compare the set, not a count: a count accepts
  malformed names (`description_0`, `description_x`) and defers failure to an opaque
  `KeyError`. Note a substring test (`"description" in name`) counts **4** for a 3-label
  model because `category_description` matches — verified live — so use exact names. Test
  both directions: the guard passes for a correctly-sized model *and* raises for a
  wrong-sized one; only the passing case catches that off-by-one.
- Payload (`:140`): replace `{"category_name", "labels": {label: [texts]}}` with an
  **ordered array** whose entries each carry their own 1-based `position` alongside the
  label value and examples (AR-2.2) — an object keyed by label value would rest the
  positional contract on JSON key ordering. Keep the `<<<DATA>>>` delimiters and
  `json.dumps(..., sort_keys=False)`.
- Preflight (`:155-161`): unchanged logic, but the error must name the sizing flag actually
  in effect rather than always `--examples-per-label` (FR-2.3's updated Verify). The two
  numeric defaults the old spec misstated are already correct in code
  (`--max-example-chars` 500, `--max-prompt-chars` 100000) — **no code change**, the spec
  text was wrong.
- Delete `:163-205`'s retry loop, the `expected`/`returned`/`missing`/`unexpected`
  reconciliation and `reconciliation_error` plumbing. Replace with one call, then build
  `Category` from `distinct_labels` zipped against `result[f"description_{i}"]` for
  `i` in `1..N`. Every `Label.value` comes from `distinct_labels`, never from `result`.
- **Rewrite the stale prose**: `induction.py`'s module docstring (`:1-2`) and `induce()`'s
  docstring (`:67-89`) still describe label-set reconciliation and multiple attempts. Both
  were written for the removed design and must describe the positional contract instead.
- Keep unchanged: the empty/zero-label guards (`:74-80` — but see Task 3, which must
  preflight them *earlier*), the FR-2.5 reserved-sentinel loop (`:82-88`), the
  `RuntimeError` sanitization of a failed call (FR-2.6), and `InductionOutcome`'s three
  fields (`sampled_examples` stays keyed by label value — the Data Requirements manifest
  shape is not part of this change).
- **Tests — delete** (subject no longer exists):
  `test_induce_rejects_missing_or_invented_labels` (`:443`),
  `test_induce_rejects_duplicate_label` (`:516`), and the four same-day retry tests at
  `:453`, `:477`, `:489`, `:501`.
- **Tests — update:** `_induction_response(labels)` (`:351`) and
  `FakeInductionClassifier`'s `responses` support (`:331`) produce the positional shape;
  every surviving `induce(...)` call site passes the new sizing kwargs
  (`:362,404,420,432,527,538,553,570,580`).
- **Tests — add a structural payload assertion** (this is FR-2.2's and AR-2.2's direct
  Verify condition and nothing else covers it):
  `test_induce_single_call_sees_all_labels_grouped` (`:580`) currently only asserts
  `f'"{label}"' in sent`, which still passes under the new payload while proving nothing.
  Parse the delimited JSON and assert the labels container is an **array** (not an object),
  entries are in sorted-label order, positions are exactly `1..N`, and each position is
  paired with its intended label and examples. A dict that merely preserves insertion order
  must fail this test.
- **Tests — add** beyond that: positional mapping (distinguishable `"d1"/"d2"/"d3"`
  descriptions land on the correctly sorted labels — catches an off-by-one or re-sort);
  quota `100/3 → 34/33/33` summing to `N`; cascading scarcity (capacities `(1,2,100)`,
  `N=20 → (1,2,17)`); all labels scarce / `N` above the whole split (`(2,3,4)`, `N=100 →
  (2,3,4)`); `N == n_labels` giving every label exactly 1; deterministic sorted-order
  tie-breaking; the `N < n_labels` floor raising with zero classifier calls; both-params and
  neither-param each raising; budget-mode seed-determinism **and** its explicit lack of
  cross-label independence (assert determinism, not isolation).
**Verify:** `.venv/bin/python -m pytest tests/test_experiment.py -q -k "induce and not
subcommand and not alone and not help"` — the direct-`induce()` unit tests pass. **Full
suite still expected red until Task 3.**
**Covers:** FR-2.1, FR-2.2, FR-2.3, FR-2.4, FR-2.5, FR-2.6 (preserved, not re-delivered),
FR-2.7 (in-`induce()` half), AR-2.2 (payload half), AR-2.3

### Task 3: `experiment.py` — flags, one sizing resolver, preflight order, run_config
**Goal:** both sizing flags exist and are mutually exclusive, the default cap is resolved in
exactly one place, FR-2.7's errors still fire before any model construction,
`--induction-retries` is gone, and `run_config.json` records the effective mode at
`schema_version: 4`. The full suite is green again at the end of this task.
**Files:** `src/query_classification/experiment.py`, `tests/test_experiment.py`
**Dependencies:** Tasks 1, 2
**Do:**
- Parser (`:168-200`, the `induction` group): add `--induction-examples` (`type=int`,
  `default=None`); change `--examples-per-label`'s `default=20` to `default=None`. **The
  default change is mandatory** — a flag that always arrives as `20` would collide with
  every `--induction-examples` invocation, the same mechanic spec 4's AR-1.8 needed for
  `--model`. Delete `--induction-retries` entirely.
- `_validate_args`: add the mutual-exclusion error naming both flags; add
  `--induction-examples` to the `>= 1` numeric loop (`:437-443`); drop
  `--induction-retries` from it; keep `--examples-per-label >= 1` applying only when given.
- **One authoritative sizing resolver.** Add a single helper that turns the parsed args into
  the effective pair, and use its result for *both* the `run_induction` call and the
  run_config record — never recompute it in two places:
  - neither flag given → `(examples_per_label=20, induction_examples=None)`. This step is
    what stops the ordinary default invocation from reaching Task 2's exactly-one check as
    `(None, None)`; without it the documented default `induce`/`run` invocation breaks.
  - `--examples-per-label N` → `(N, None)`; `--induction-examples N` → `(None, N)`.
  - **`classify`-only runs have neither attribute** (the induction group is not a parent of
    that subparser — verified: `hasattr(ns, "examples_per_label")` is `False`). The resolver
    must return `(None, None)` there and the run_config must record both as null, per the
    spec's Data Requirements. Do not record the literal `20` on a `classify` run; it would
    falsely claim a sizing mode ran.
- **Preflight ordering — preserve FR-2.7.** After `project_and_filter` (`:920`) and
  **before** `_construct_classifiers` (`:935`), compute
  `distinct_labels = sorted(train_df[label_column].unique().tolist())` and raise FR-2.7's
  existing errors (empty filtered train split; zero distinct usable labels) **there, with
  FR-2.7's wording**, then enforce FR-2.1's `--induction-examples >= len(distinct_labels)`
  floor. Without this, an all-rows-filtered split yields `n_labels == 0` and
  `build_induction_model(0)`'s generic `n_labels < 1` error fires first, replacing FR-2.7's
  actionable message — a regression Task 2 alone cannot prevent. Then pass
  `n_labels=len(distinct_labels)` into `_construct_classifiers`.
- `_construct_classifiers` (`:599`): add `n_labels: int | None = None`; call
  `build_induction_model(n_labels)` at `:645`; raise if `will_induce` and `n_labels is
  None`. The `will_classify` path is untouched.
- `run_induction(...)` (`:938`): pass the resolver's `examples_per_label=` and
  `induction_examples=`; drop `max_reconciliation_retries=`.
- `_SCHEMA_VERSION` (`:86`) 3 → **4**. In the config dict (`:845-849`): add
  `induction_examples`, remove `induction_retries`, record the resolver's effective
  `examples_per_label`.
- **Tests:** remove the CLI-level `--induction-retries` coverage; **update the existing
  `assert config["schema_version"] == 3` at `tests/test_experiment.py:1197` to `4`** (the
  only such assertion in the repo — verified — and the suite fails without it); add a
  mutual-exclusion CLI test (exit 1, `fake_classify["calls"] == 0`); a budget-floor CLI
  test; an **all-rows-filtered CLI test** asserting FR-2.7's message and zero LLM calls
  (the regression guard for the preflight ordering above); a **default-invocation test**
  proving neither-flag-given still works and records `examples_per_label == 20`; a
  `classify`-only run_config test asserting **both** sizing keys are null; run_config
  assertions for both inducing modes; `schema_version == 4`; and an end-to-end `induce` in
  budget mode writing a valid `categories.json`.
**Verify:** `.venv/bin/python -m pytest -q` — **full suite green**;
`.venv/bin/python experiment.py induce --help` and `run --help` each show
`--examples-per-label` and `--induction-examples` and **no** `--induction-retries`, while
`classify --help` shows none of the three.
**Covers:** FR-2.1, FR-2.4, FR-2.6, FR-2.7 (preflight half), FR-3.4, AR-2.1, AR-2.3, AR-3.4

### Task 4: Docs and experiment tooling
**Goal:** the new flag is documented and actually reachable through the shipped scripts;
stale prose about the removed flag and the removed reconciliation step is gone or explicitly
marked historical.
**Files:** `README.md`, `spec/ARCHITECTURE.md`,
`spec/2-experiment-runner/implementation-summary.md`,
`experiments/{ag-news,trec,pubmed-rct}/run_experiment.sh` and their `README.md`s
**Dependencies:** Tasks 1–3 for the final wording; the script work can start once the two
flag names are frozen.
**Do:**
- `README.md:243-245`: replace the `--induction-retries` sentence with
  `--induction-examples`, stating the mutual exclusion, the `>= n_labels` floor, and that a
  total budget makes prompt size independent of the label count.
- `spec/ARCHITECTURE.md:78`: the `induction.py` row describes its responsibility as
  `label-set reconciliation`, which no longer exists — reword to the positional description
  mapping. Row content only; INV-1's import-edge rule is unchanged, so this is a doc fix
  needing no ADR.
- `spec/2-experiment-runner/implementation-summary.md:36` also still says "label-set
  reconciliation". It is a close-time snapshot of the original implementation, so **mark it
  as historical** (a dated note that the induction contract was redesigned post-close, with
  a pointer to the spec's Change Log) rather than rewriting history — the same treatment
  spec 4's superseded summary got.
- **Scripts.** All three pass `--examples-per-label "$EXAMPLES_PER_LABEL"`
  **unconditionally** (`ag-news:62`, `trec:66`, `pubmed-rct:67`), and each advertises
  arbitrary trailing flags via `"$@"`. Two verified consequences to fix together:
  - `run_experiment.sh --induction-examples 500` would send **both** flags and fail the
    mutual-exclusion check. Detect `--induction-examples` among `"$@"` and suppress the
    default cap when present (in addition to adding an `INDUCTION_EXAMPLES` env var guarded
    like the existing `TEST_LIMIT` pattern).
  - `EXAMPLES_PER_LABEL= INDUCTION_EXAMPLES=60 run_experiment.sh` cannot work while the
    scripts use `${EXAMPLES_PER_LABEL:-20}`: verified that the colon form collapses an
    explicitly empty value back to `20`. Use `${EXAMPLES_PER_LABEL-20}` (no colon) so unset
    and empty are distinguishable.
  - Document both forms in each script header and experiment README. Keep the current
    effective default so existing invocations behave identically.
- Do **not** touch `experiments/ag-news/analyze_run.ipynb` or
  `experiments/trec/analyze_run.ipynb` — both carry user edits and executed outputs.
**Verify:** `bash -n` each modified script. For argv behavior, add a `DRY_RUN=1` mode that
prints the assembled argv instead of invoking Python (the plan previously said "inspect via
a dry-run echo", which was not executable — no such mode existed), then check three cases
offline with no provider call: default (cap 20 only), `INDUCTION_EXAMPLES=60` (budget only),
and a trailing `--induction-examples 500` (budget only, default cap suppressed).
**Covers:** FR-2.1 (downstream tooling consequence — see Spec Deviations #1)

## Requirements Unaffected By This Update

Spec 2 has 23 FRs and 10 ARs; the `Covers:` lines above name only the requirements this
redesign changes or must actively preserve. Everything below is **already shipped and
deliberately untouched** — listed so "intentionally unaffected" is distinguishable from
"forgotten", and so the `Covers:` audit is checkable:

- **All of Feature 1** (FR-1.1–FR-1.8, AR-1.1–AR-1.3) — dataset loading, split resolution,
  label normalization, row filtering, column projection. No task touches `dataset_io.py`.
- **All of Feature 3** except **FR-3.4** and **AR-3.4** (both cited by Task 3).
  **FR-3.7** (ship unit tests) is satisfied structurally: per CLAUDE.md each of Tasks 1–3
  carries its own tests, and Final Verification asserts the whole suite plus the specific
  pre-existing tests guarding unchanged behavior. Note Task 1 widens the induction fake's
  interface beyond FR-3.7's `.classify(text) -> dict` minimum (Spec Deviations #2).
- **FR-2.6** is listed in Task 2's `Covers:` as *preserved, not re-delivered* — it is the
  one unaffected requirement with real risk, because Task 2 deletes code immediately
  adjacent to its sanitized failure path.

## Final Verification

- `.venv/bin/python -m pytest -q` — full suite green. Baseline before this plan: **214
  passed** (independently confirmed during the v-2 critique round). Expected final total is
  `214 − 6 deleted + N new`; report that arithmetic explicitly in the implementation summary
  so a silent deletion is visible. If `N` is small, coverage has net dropped and Task 2's
  additions were insufficient.
- Help surface: `induce --help` and `run --help` each show `--examples-per-label` and
  `--induction-examples` and no `--induction-retries`; `classify --help` shows none of the
  three (the induction group is not a parent of that subparser).
- `git diff --stat` shows **zero** changes to `cli.py`, `pipeline.py`, `debate.py`,
  `multi_model.py`, `dataset_io.py`, `classifier.py` (AR-3.1/AR-3.2 hold).
- `grep -rn "induction.retries\|induction_retries\|max_reconciliation_retries"` over `src/`,
  `tests/`, `README.md`, `experiments/` returns nothing; the only surviving matches are the
  deliberate historical record in `spec/2-experiment-runner/spec.md` (AR-2.3 + Change Log).
- Inspect generated `run_config.json` for three shapes: cap mode
  (`examples_per_label == 20`, `induction_examples` null), budget mode (the inverse), and a
  `classify`-only run (**both** null). `schema_version == 4` in all three; no
  `induction_retries` key anywhere.
- Re-run the pre-existing tests guarding behavior this plan does not intend to change:
  `test_induce_at_cap_label_consumes_no_rng_state`, `test_induce_rng_exact_sequence`,
  `test_induce_truncation_marker`, `test_induce_prompt_size_preflight`,
  `test_induce_rejects_reserved_sentinel`, `test_induce_sanitizes_failure`,
  `test_induce_rejects_degenerate_input`, `test_induce_category_round_trips`.
- **Not verifiable in-session:** the real PubMed RCT induction that motivated this work
  needs live provider credentials. The acceptance test is
  `experiments/pubmed-rct/run_experiment.sh` completing induction where it previously died
  on `missing: ['conclusions']` — hand that to the user rather than claiming it passed.

## Documentation

- Living docs: none — `spec/docs/` does not exist in this repo (confirmed). Run
  `/spec-docs --full` first if living docs are wanted going forward.
- In-repo docs updated by Task 4: `README.md`, `spec/ARCHITECTURE.md:78`,
  `spec/2-experiment-runner/implementation-summary.md` (historical marking), the three
  experiment script headers and READMEs.

## Spec Deviations

| # | Spec says | Plan does instead | Reason | Action required |
|---|-----------|-------------------|--------|-----------------|
| 1 | Nothing — no FR/AR governs `experiments/` tooling | Task 4 suppresses the scripts' default cap when a budget flag is present (via `"$@"` detection or `INDUCTION_EXAMPLES`), and switches to `${VAR-20}` so an empty value is distinguishable from unset | Verified: the scripts pass `--examples-per-label` unconditionally and advertise trailing flags, so after FR-2.1's mutual-exclusion rule the new flag would be unreachable through the documented entry point. Restores access; changes no product behavior | None |
| 2 | FR-3.7 describes the induction test fake as `.classify(text) -> dict`; FR-2.4 specifies no guard against a model/label-count mismatch | Task 2 adds an internal arity guard reading `classification_model.model_fields` off the handed classifier, and Task 1 widens `FakeInductionClassifier` to carry that attribute | INV-1 forbids `induction.py` importing `schema.py`, so `induce()` cannot rebuild the model to check it; reading the object it already holds closes the only silent-misalignment path without a new import edge. Treats FR-3.7's fake shape as a **minimum**, not an exhaustive interface — if it were exhaustive this would need a `/spec-update`, so the reading is recorded here explicitly | None |

## Risks

- **The prompt rewrite is load-bearing.** If a provider rejects strict structured output,
  `Classifier._attempt_completion` falls back to `json_object` mode where
  `induction_prompt.txt` is the only statement of the `description_i` names. A half-done
  Task 1 (schema changed, prompt not) passes the whole offline suite and fails only against
  a real provider — hence Task 1's template-contract test.
- **Strict-mode property limit at high label counts.** The positional schema has `N + 1`
  properties, and Azure's structured-output mode caps an object at 100 properties. At ~99+
  labels the strict path will be rejected and silently fall back to `json_object` mode:
  correctness survives (Pydantic still enforces the required fields) but the run pays an
  extra failed round-trip and loses provider-side strictness. Accepted, not fixed — it is
  far outside this repo's current datasets (PubMed RCT 5, TREC coarse 6, TREC fine 50).
- **No observability when structured output falls back.** `classifier.py:312-316` swallows
  the `BadRequestError` and retries in JSON mode without logging which path ran. Pre-existing
  behavior and `classifier.py` is out of scope here, but the property limit above makes it
  newly relevant: do not claim all positional induction calls stay on the strict path.
- **`_SCHEMA_VERSION` 4 is a real persisted-shape change**, not additive — a key is removed
  and another becomes nullable. Verified no in-repo reader branches on it (the analyze
  notebooks read `classifier_config.critics` only), so there is no migration task here, but
  external readers pinned to 3 need updating.
- **Six deleted tests reduce the assertion count.** They go because their subject is gone,
  not because they failed. Task 2's additions must cover the replacement ground (positional
  mapping, payload structure, arity guard, quota math) or net coverage drops.
- **Budget mode weakens a reproducibility property on purpose** (cross-label independence).
  Specified in FR-2.1, but it means "adding train rows for one label changed another label's
  examples" is expected in budget mode and a real bug in cap mode. The tests must encode
  that distinction so the next implementer doesn't "fix" it.
- Rollback/checkpoint guidance: Tasks 1–3 land as **one commit** (the contract change is not
  independently shippable — their intermediate states are knowingly red), Task 4 separately.
  To undo, use **`git revert`** of those commits — never `git checkout`/`git reset --hard`,
  since the working tree holds unrelated uncommitted user work (both analyze notebooks, the
  two new experiment suites).
