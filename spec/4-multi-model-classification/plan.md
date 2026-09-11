# Implementation Plan: 4-multi-model-classification (redesign)

**Status:** Ready
**Date:** 2026-09-10
**Spec:** spec/4-multi-model-classification/spec.md
**Plan Critique:** spec/4-multi-model-classification/plan-critique-consolidated-v-2.md

## Overview

Implements the `--model`/`--models`/`--n-classifiers` redesign (spec's Change Log) on top
of the already-shipped, already-merged multi-model-voting feature. `multi_model.py` and
`pipeline.py`'s own contract need no changes (AR-1.1/AR-1.2) — but a **v1 draft of this
plan significantly understated how much of `experiment.py` actually branches on raw
`args.models` truthiness**, which a plan critique caught and this version fixes
throughout: three downstream sites (source-column collision, post-run completeness,
failure-count aggregation) plus several `args.model`-as-plain-string call sites in both
files all need updating, not just the validation block and `_construct_classifiers`.

## Readiness

- Checklist: all 10 items closed (`[x]`/`[N/A]`) in `spec.md`; no open items.
- Open questions: none. Three spec-level defects were found and fixed directly in
  `spec.md` while re-verifying this plan against a critique round (not left as plan-level
  workarounds): (1) FR-1.9/Data Requirements had the `model` field's null-condition
  **backwards** (said null at resolved-length-1, should be null at length-2+); (2) AR-1.8
  didn't enumerate the "`--models` given 2+ values, `--model` not given" case for
  induction-fallback resolution — added explicitly; (3) AR-1.4/AR-1.5 incorrectly claimed
  `experiment.py`'s post-run completeness and source-column collision checks were
  "unchanged" — they aren't, since both currently gate on raw `args.models` truthiness,
  which no longer means "multi-classifier mode active." AR-1.3 also gained an explicit
  fail-fast note for a pathological edge case (a supplied model id that already looks like
  `"<other-id>#N"`).
- External API/library verification: none needed.
- Repository state: **dirty** — `git status --short` shows `spec.md` modified (this
  session's fixes) **and `plan.md` modified** (this rewrite); one unrelated untracked file
  (`data/example_queries_classified.csv`) predates this work.
- **Codebase exploration:** `cli.py`, `experiment.py`, `pipeline.py`, `multi_model.py`,
  `tests/test_debate.py`, and `tests/test_experiment.py` were each read in full (multiple
  passes, including a full `grep -n "args\.model"` sweep of `experiment.py` to find every
  call site) during this session, immediately before and during this plan's writing —
  current on disk, not stale.

## Shared Resolution Truth Table

Both Task 1 (`cli.py`) and Task 2 (`experiment.py`) implement the **identical** algorithm
below, independently (Project Constraints — no shared code between the two entry points).
This table is the single source of truth both implementations must match; use it directly
for test cases rather than re-deriving cases independently, to prevent the two streams
from silently diverging on an edge case.

| Input | Resolved list | Outcome |
|---|---|---|
| (nothing given) | `[DEFAULT]` (len 1) | OK — today's existing plain/critics behavior |
| `--model x` | `[x]` (len 1) | OK |
| `--model x y` | — | **Error** — "`--model` accepts only one value; use `--models`" |
| `--n-classifiers 5` | `[DEFAULT]*5` (len 5) | OK — multi-classifier, all default model |
| `--model x --n-classifiers 5` | `[x]*5` (len 5) | OK — multi-classifier, `x` replicated |
| `--n-classifiers 0` | — | **Error** — "`--n-classifiers` must be >= 1" (always checked, regardless of `--models`) |
| `--models x` | `[x]` (len 1) | OK — equivalent to `--model x` (FR-1.1 point 3) |
| `--model x --models x` | `[x]` (len 1) | OK — no error even though both given (single-value `--models` override) |
| `--model x --models y` | `[y]` (len 1) | OK — `--models`' value wins silently, `x` is simply unused |
| `--model x --models y z` | — | **Error** — "`--model`/`--models` both given" (`--models` has 2+ values, so no override applies) |
| `--models x y z` | `[x,y,z]` (len 3) | OK — **the regression-guard case**: no `--n-classifiers` needed |
| `--models x y z --n-classifiers 5` | `[x,y,z]` (len 3) | OK — `--n-classifiers` is inapplicable here, **not** an error |
| `--models x y z --n-classifiers 0` | — | **Error** — the universal `>= 1` floor still applies even though the length-match check doesn't |
| `--models x x y` | `[x,x,y]` (len 3) | OK — duplicates allowed; classifier keys become `x#1`, `x#2`, `y` |
| `--models "" x` | — | **Error** — empty/whitespace value |
| `--models x y --critics` | — | **Error** — mutually exclusive, unconditionally (2+ `--models` values) |
| `--n-classifiers 3 --critics` (no `--models`, or `--models` w/ 1 value) | — | **Error** — mutually exclusive (only when `--n-classifiers` is what's driving the list) |
| `--model x --critics` (n-classifiers left at 1) | `[x]` (len 1) | OK — ordinary, **unaffected** `--critics` usage |
| `--models x y --n-classifiers 2` + `CEREBUS_MODE=azure` | `[x,y]` | **Error** — 2+ distinct models + Azure |
| `--model x --n-classifiers 3` + `CEREBUS_MODE=azure` | `[x,x,x]` | OK — all same model, Azure is fine |
| `--models x x "x#1"` | — | **Error** (AR-1.3's collision guard) — `x#1` (literal) would collide with `x`'s 1st occurrence-suffixed key |

## Code Impact

- **Modules:** modified `src/query_classification/cli.py`, `src/query_classification/
  experiment.py`. **Not modified:** `src/query_classification/pipeline.py`,
  `src/query_classification/multi_model.py` (AR-1.1, AR-1.2 — their own *contracts* need no
  change; verified by reading both files end-to-end).
- **Database/schema:** `run_config.json`'s `models.model_failure_counts` field's *key
  format* changes (bare id vs. occurrence-suffixed); `_SCHEMA_VERSION` bumps `2` → `3`.
- **API/interfaces:** `--model` changes from a plain single-value flag to `nargs="+"`,
  `default=None` on both entry points. New `--n-classifiers` flag on both. `--models` no
  longer requires unique values.
- **Key files (every one of these needs a code change — this list grew substantially from
  this plan's first draft after a critique found the gaps below):**
  - `src/query_classification/cli.py:58-62` (`--model` `add_argument`), `:123-136`
    (`--models` `add_argument`), `:232-264` (validation block + `allow_new_labels` line —
    **the `allow_new_labels` fix must keep `args.critics` in the condition**), `:310-338`
    (classifier construction — **the `else` single-classifier branch's `critic_model`/
    `reconciler_model` fallbacks at what are today's `:345,359` also read raw
    `args.model` and must switch to the resolved single-model value**)
  - `src/query_classification/experiment.py:139-143` (`--model`), `:221-229` (`--models`),
    `:325-340` (`_validate_args`'s `--models` block — point 1 unconditional, points 2-6
    gated behind `hasattr(args, "models")`), `:85` (`_SCHEMA_VERSION`), `:473-583`
    (`_construct_classifiers` — **`:492`'s `models["model"] = args.model` init and `:508`'s
    induction-fallback `... or args.model` both read raw `args.model` and must switch to
    the resolved single-model value; `:531`'s `if getattr(args, "models", None):` branch
    condition must switch to "resolved list length > 1", not raw truthiness; `:560-561`'s
    critic/reconciler fallbacks have the same raw-`args.model` problem as `cli.py`'s**),
    `:832` (the no-induction `models_final` fallback literal — same raw-`args.model`
    problem), `:889` (source-column collision check — `elif args.models:` must become
    "resolved list length > 1"), `:947` (post-run completeness check — same), `:961-966`
    (failure-count aggregation — `if args.models:` must become "resolved list length > 1"
    **and** the `{m: 0 for m in args.models}` pre-seed must use the actual constructed
    classifier keys, not raw `args.models`, or occurrence-suffixed failures are silently
    never counted)
  - `tests/test_debate.py:817,843` (existing reversal tests — update), and see Task 1 for
    3 more sites needing attention
  - `tests/test_experiment.py:627,1039,1051,1106,1134` (existing tests needing updates —
    5 sites, not the 2 a first draft of this plan named)
  - `README.md` (Options table, "Multi-model mode" **and** "Experiment runner" sections)

## Project Constraints

- **INV-1**: unaffected — no new module, no new import edge, no ADR needed (confirmed —
  `spec.md`'s own checklist states this).
- **cli.py/experiment.py don't share private code** (established convention) — the
  resolution algorithm and occurrence-suffix logic are implemented once in each file.
  **This plan's Shared Resolution Truth Table exists specifically to keep the two
  independent implementations behaviorally identical** despite not sharing code.
- **CLAUDE.md testing policy**: no live network/provider calls — mock `Classifier.classify`
  via each file's established pattern.
- **No lint/typecheck/CI** — `pytest` passing is the only gate. Use `.venv/bin/python`.

## Implementation Strategy

- Size: medium (2 files touched, ~9 FR/AR requirements, no new module) — larger in actual
  edit count than a first pass suggested, once every `args.models`/`args.model` call site
  was found.
- Execution mode: 2 parallel streams (Task 1, Task 2 — zero file overlap), then solo
  Task 3 (README).
- Sequential blockers: Task 3 depends on Tasks 1+2.
- **Both tasks must resolve the classifier list *once* per invocation** (a single
  `resolved_models: list[str]` value) **and thread that same value through every site**
  that currently reads `args.models`/`args.model` directly — validation, classifier
  construction, `models_final`/`run_config.json` bookkeeping, the source-column collision
  check, the post-run completeness check, and failure-count aggregation. Do not treat
  "replace the validation block" and "replace `_construct_classifiers`" as the whole job —
  the Key Files list above enumerates every site found; treat it as the checklist.

## Implementation Tasks

### Task 1: `cli.py` — `--model`/`--models`/`--n-classifiers` resolution and construction
**Goal:** `classify.py` implements the Shared Resolution Truth Table and AR-1.3's
occurrence-suffixed classifier construction, with `--critics` mode fully unaffected.
**Files:** `src/query_classification/cli.py`, `tests/test_debate.py`
**Dependencies:** None
**Do:**
- Change `--model`'s `add_argument` (`:58-62`) to `nargs="+", default=None`; update help
  text. Add `--n-classifiers` (`type=int, default=1`) near `--models` (`:123-136`).
- Replace the existing `--models`-only validation block (`:232-264`) with a helper (e.g.
  `_resolve_classifier_models(args) -> list[str]`) implementing the Shared Resolution
  Truth Table exactly — called once, early, in `main()`, before `load_categories`.
  **Explicitly**: the `--n-classifiers >= 1` check always runs, in every branch — only
  the *length-match-against-`--models`* check (which doesn't exist in this design at all)
  is what's "not applicable" when `--models` has 2+ values. Include the AR-1.3 collision
  guard: after building the occurrence-suffixed key dict, if its length is less than
  `len(resolved_models)`, raise a clear error (two different positions produced the same
  key — only reachable via a pathological input like `--models x x "x#1"`).
- Fix `allow_new_labels` (`:264`): the new condition is `args.allow_new_labels if
  (args.critics or len(resolved_models) > 1) else True` — **keep `args.critics` in the
  condition**; replacing it with only `len(resolved_models) > 1` (as a first draft of this
  plan said) would force `allow_new_labels=True` for ordinary `--critics` usage
  (a resolved single classifier), silently breaking existing `--critics` behavior.
- Replace the classifier-construction block (`:310-338`): if `len(resolved_models) == 1`,
  build `classifier` exactly as today, **including** its existing `temperature=
  args.sampling_temperature if args.critics else None` line (unchanged) — but its
  `model_id` must come from `resolved_models[0]`, not raw `args.model` (which is now a
  list or `None`). The `critic_classifiers`/`reconciler_classifiers` dict comprehensions
  just below (today's `:345,359`, `args.critic_model or args.model`) have the identical
  problem — `args.model` is no longer a plain string — switch both to `args.critic_model
  or resolved_models[0]` / `args.reconciler_model or resolved_models[0]`. When
  `len(resolved_models) > 1`, build `models_classifiers: dict[str, Classifier]` keyed by
  occurrence-suffixed id (bare when not repeated), each identically configured
  (`temperature=None` always — this branch is never reachable with `--critics=True`, per
  the truth table).
- **Update these existing tests** (assert now-reversed behavior):
  - `test_cli_models_fewer_than_two_exits_before_any_llm_call` (`:817`) — `--models a` (1
    value) must now succeed.
  - `test_cli_models_duplicate_value_exits_before_any_llm_call` (`:843`) — `--models a a
    b` must now succeed, producing occurrence-suffixed audit keys.
  - Confirm (don't just assume) `test_cli_models_mode_end_to_end_writes_audit_columns`
    and `test_cli_models_mode_honors_allow_new_labels_carve_out` still pass with the
    fixed `allow_new_labels` condition above.
- **New tests**, each proving actual behavior, not just "no error":
  - Every row of the Shared Resolution Truth Table above that doesn't already have
    coverage, especially: `--models a b c` alone (the regression guard — must succeed
    with no `--n-classifiers`); `--models a b c --n-classifiers 99` (must still succeed,
    resolving 3, not error); `--models a b c --n-classifiers 0` (must error — the
    universal floor, distinguishing it from the "not applicable" length-match); `--model
    a --n-classifiers 3` + `CEREBUS_MODE=azure` (must **not** error, same model repeated);
    `--models a b` + `CEREBUS_MODE=azure` (must error, 2 distinct models); the AR-1.3
    collision-guard case (`--models a a "a#1"`).
  - An **exact-call-count** test: mock `Classifier.classify` with a counter, confirm
    `--model x --n-classifiers 5` makes exactly 5 calls (not fewer, e.g. from an
    accidental dict-collapse bug) — this is what actually proves FR-1.2 for repeated
    instances, not just "constructs N classifiers."
  - An **out-of-order-completion tie-break** test for repeated instances: two classifier
    instances sharing a model id, mocked so the later-positioned one's call resolves
    first, asserting the tie is still broken toward the earlier position (proves FR-1.3
    for the specific case this redesign adds — same-model ties, not just distinct-model
    ties, which the original implementation already tested).
  - A **repeated-instance partial-failure** test: one specific occurrence of a repeated
    model fails (not all occurrences), asserting the correct occurrence-suffixed key
    appears in `_model_errors` and is absent from `_by_model`, while the other occurrence
    of the same model succeeds normally.
**Verify:** `.venv/bin/python classify.py --help` shows `--n-classifiers` and `--model`'s
updated help text; `.venv/bin/python -m pytest tests/test_debate.py -v` passes, including
updated/new tests, with no regressions among the pre-existing tests.
**Covers:** FR-1.1, FR-1.2 (`cli.py` surface), FR-1.3 (`cli.py` surface), FR-1.4/FR-1.7
(`cli.py` surface), FR-1.8 (`cli.py` surface), FR-1.10 (`cli.py` share), AR-1.3, AR-1.7
(`cli.py` surface), AR-1.8

### Task 2: `experiment.py` — same resolution, plus every downstream `args.models` site
**Goal:** `experiment.py classify`/`run` implement the same Shared Resolution Truth
Table, with **every** site that currently branches on `args.models`/reads `args.model`
as a plain string updated — not only `_validate_args`/`_construct_classifiers`.
**Files:** `src/query_classification/experiment.py`, `tests/test_experiment.py`
**Dependencies:** None (parallel with Task 1)
**Do:**
- Change `--model`'s `add_argument` (`:139-143`, in `common`) to `nargs="+", default=
  None`. Add `--n-classifiers` to the `classification` group (`:221-229`, alongside
  `--models` — not `common`/`induction`).
- In `_validate_args`, add the same resolution/validation algorithm as Task 1 (duplicated,
  not shared), with `experiment.py`'s specific wrinkle: **`--models`/`--n-classifiers`
  don't exist as `args` attributes for `induce`** (only `--model` lives in `common`). So:
  the "`--model` given >1 value" check runs unconditionally, for every subcommand; every
  other check (the `--models`/`--n-classifiers`/`--critics` interaction, points 2-6 of the
  algorithm) is gated behind `hasattr(args, "models")` (true only for `classify`/`run`).
  Store the resolved list somewhere `_construct_classifiers` and `main()` can reuse it
  (e.g. as a new `args.resolved_classifier_models` attribute set at the end of
  `_validate_args`, or recomputed identically in both places — either is fine as long as
  it's the *same* value both times, since `_validate_args` and `_construct_classifiers`
  are called at different points in `main()`).
- Add `_resolve_single_default_model(args) -> str`: returns `args.model[0]` if `--model`
  was given, else the literal default `"azure/gpt-5-chat"` — used for
  `--induction-model`'s fallback (`:508`, replacing `... or args.model`) **and** as the
  literal default in the "single resolved model" resolution branch. Per the spec's now-
  explicit 4th case (AR-1.8), this is the value used even when `--models` was given with
  2+ values for the classification role in the same invocation — never one of `--models`'
  entries.
- In `_construct_classifiers` (`:473-583`):
  - `:492`'s `"model": args.model` in the base dict init must become the resolved single-
    model value (via `_resolve_single_default_model`, or `resolved_models[0]` when
    `len(resolved_models) == 1`) — not raw `args.model`.
  - `:508`'s induction-fallback: use `_resolve_single_default_model(args)`.
  - `:531`'s `if getattr(args, "models", None):` branch condition must become "the
    resolved classifier list has length > 1" — not raw `args.models` truthiness. This is
    what makes `--model x --n-classifiers 5` (where `args.models is None`) correctly enter
    the multi-classifier construction path, and what makes `--models x` (single value,
    `args.models` truthy) correctly **not** enter it.
  - `:560-561`'s `args.critic_model or args.model` / `args.reconciler_model or args.model`
    (inside the `if args.critics:` block, only reachable when `len(resolved_models) ==
    1`) — switch both to use the resolved single-model value, same as Task 1's `cli.py`
    fix.
  - Widen the occurrence-suffix key-building logic identically to Task 1 (including the
    AR-1.3 collision guard).
- Fix the no-induction `models_final` fallback literal (`:832`, `"model": args.model`) to
  use the resolved single-model value.
- Fix all **three** downstream sites that currently gate on raw `args.models` truthiness
  (a first draft of this plan missed all three — found via a full `grep -n "args\.models"`
  sweep of `experiment.py`):
  - `:889` (pre-projection source-column collision check, `elif args.models:`) → gate on
    resolved-list-length > 1.
  - `:947` (post-run completeness check, `elif args.models:`) → gate on resolved-list-
    length > 1.
  - `:961-966` (failure-count aggregation) → gate the whole block on resolved-list-length
    > 1 (not `if args.models:`), **and** fix the pre-seed step: `model_failure_counts =
    {m: 0 for m in args.models}` collapses duplicates (for `--models a a b`, this produces
    only 2 keys, `"a"` and `"b"`, not the 3 occurrence-suffixed keys the audit trail
    actually uses) and is never entered at all for a `--model`+`--n-classifiers`
    replication (`args.models is None`). Pre-seed from the **actual constructed classifier
    keys** (the same occurrence-suffixed dict built in `_construct_classifiers`, threaded
    through the same way `classifiers.get("multi_model")` already is) instead of raw
    `args.models`.
- Bump `_SCHEMA_VERSION` (`:85`) from `2` to `3`.
- **Update these existing tests** (verified against actual test bodies, not just names):
  - `test_induce_alone_resolves_induction_model_fallback` (`:627`) — asserts `ns.model ==
    "base-model"`; under `nargs="+"`, parsing produces `ns.model == ["base-model"]`.
    Update the assertion.
  - `test_models_requires_at_least_two_values` (`:1039`) — `--models a` must now succeed.
  - `test_models_rejects_duplicate_value` (`:1051`) — `--models a a b` must now succeed.
  - `test_models_end_to_end_run_config` (`:1106`) — asserts `config["schema_version"] ==
    2`; update to `3`.
  - `test_models_run_with_induction_merges_all_fields` (`:1134`) — its argv combines
    `--model induction-model-id` with `--models a b` (2 values), which the new mutual-
    exclusion rule now rejects outright (a first draft of this plan incorrectly said this
    test "still passes, just extend it"). Rewrite it to use `--induction-model
    induction-model-id` instead of `--model induction-model-id`, so induction still gets a
    distinct, assertable value without triggering the now-illegal `--model`+`--models`(2+)
    combination; keep its existing assertions (`model is None`, `classification_models ==
    ["a","b"]`).
- **New tests**, mirroring Task 1's list adapted to `_run_main`/`fake_classify`, plus:
  - A test proving the three downstream-site fixes: `--model x --n-classifiers 3` (no
    `--models`) produces the multi-classifier audit columns, passes the post-run
    completeness check correctly, and gets a non-null `model_failure_counts` with 3
    (occurrence-suffixed) keys — this directly targets the exact regression Task 2 v1
    would have shipped.
  - A repeated-instance failure-count test: one specific occurrence of a repeated model
    fails; assert the correct suffixed key shows a nonzero count in `model_failure_counts`
    **and** the other occurrence of the same model shows an explicit `0` — this is what
    actually proves the pre-seed fix, not just that failure counting works at all for
    distinct models (already covered by the existing `test_models_failure_counts_per_model`).
**Verify:** `.venv/bin/python experiment.py classify --help` / `run --help` show
`--n-classifiers`; `induce --help` omits it but still shows `--model`;
`.venv/bin/python -m pytest tests/test_experiment.py -v` passes, including updated/new
tests, with no regressions among the pre-existing tests.
**Covers:** FR-1.1, FR-1.2/FR-1.3/FR-1.4/FR-1.7/FR-1.8 (`experiment.py` surface), FR-1.9,
FR-1.10 (`experiment.py` share), AR-1.3, AR-1.4 (`experiment.py` portion), AR-1.5, AR-1.7
(`experiment.py` surface), AR-1.8

### Task 3: `README.md` — document the redesigned flags
**Goal:** README reflects `--n-classifiers`, the revised `--model`/`--models` rules, and
the occurrence-suffix audit-key format, across **all three** sections AR-1.9 names.
**Files:** `README.md`
**Dependencies:** Task 1, Task 2
**Do:**
- Update the "Multi-model mode" section: `--n-classifiers`; confirm `--models a b c`
  alone is unaffected; a worked example for `--model gpt-4o-mini --n-classifiers 5`; the
  occurrence-suffix audit-key format, contrasted with the unsuffixed all-distinct case.
  Change existing "number of models" call-volume/concurrency wording to "number of
  classifier instances" throughout this section, so same-model replication isn't
  mischaracterized as only applying to distinct models.
- Update the **"Experiment runner" section** (a first draft of this plan named only the
  Options table and Multi-model mode section — AR-1.9 explicitly also requires this one):
  state that `--n-classifiers` (like `--models`) is available on `classify`/`run` only,
  not `induce`.
- Update the Options table: `--model`'s row notes it accepts only one value; add
  `--n-classifiers`'s row.
**Verify:** manual read-through confirms AR-1.9's three sections are all updated; no
command to run.
**Covers:** AR-1.9

## Requirements Unaffected By This Update

FR-1.5 (`--restore` audit-column integration), FR-1.6 (partial/total classifier-failure
handling in `multi_model.py`), and AR-1.6 (concurrency/index-ordering/never-raising) need
no code changes — they operate on whatever already-unique `dict[str, Classifier]` keys
they're handed, agnostic to content, and live entirely in `pipeline.py`/`multi_model.py`,
which this update doesn't touch. (AR-1.4 and AR-1.5 do **not** belong in this list — see
Task 2; a first draft of this plan incorrectly included them here.) Final Verification's
full-suite run confirms these still hold; no dedicated new test is needed for
requirements whose implementation is genuinely unchanged.

## Final Verification

- `.venv/bin/python -m pytest` — full suite passes (all existing + updated + new tests).
- `.venv/bin/python classify.py --help`, `.venv/bin/python experiment.py classify --help`,
  `.venv/bin/python experiment.py run --help` all show `--n-classifiers`;
  `.venv/bin/python experiment.py induce --help` does not, but still shows `--model`.
- Targeted re-run of the specific tests mapped to FR-1.5/FR-1.6/AR-1.6 in the existing
  suite (the restore tests, partial/total-failure tests, and concurrency/tie-break tests
  already added by the original implementation) — confirms "no changes needed" is
  actually still true post-edit, not just asserted.
- A `run_config.json` inspection across all three resolved shapes in one pass: a
  length-1 run (`model` holds a string, both new fields `null`), a length-3-distinct run
  (`model_failure_counts` bare keys), and a length-3-repeated run (`model_failure_counts`
  occurrence-suffixed keys, all summing correctly against `_model_errors`) — this is what
  actually catches a stale `models_final` value, not a single spot-check.
- Confirm `src/query_classification/pipeline.py` and `src/query_classification/
  multi_model.py` have **no diff** against their state before this plan's tasks.

## Documentation

- Living docs: none — `spec/docs/` still doesn't exist in this repo.

## Spec Deviations

None identified. The one candidate deviation from this plan's first draft (induction-
fallback resolution for the "`--models` given 2+ values" case) is no longer a deviation —
it's now explicit spec text (AR-1.8's added fourth case, fixed directly in `spec.md`
during this session rather than left as an inferred plan-level choice).

## Risks

- **Risk:** the `experiment.py` downstream-site gaps this version of the plan fixes
  (source-column collision, post-run completeness, failure-count aggregation, several
  raw-`args.model` reads) were **missed entirely by this plan's first draft** and caught
  only by an explicit plan-critique pass that re-derived every `args.models`/`args.model`
  call site from a fresh `grep`, rather than trusting the first draft's own file/line
  citations. **Mitigation:** the Key Files list and Task 2's Do bullets now enumerate
  every site found; treat that enumeration as the actual scope, not a starting point.
- **Risk:** Tasks 1 and 2 implement the identical resolution algorithm independently
  (by design — see Project Constraints) and could still drift on an edge case despite the
  Shared Resolution Truth Table. **Mitigation:** both tasks' test lists explicitly
  reference the same table; a final cross-check (run the same input through both
  `classify.py` and `experiment.py classify` and diff the resulting resolved-list
  behavior) is cheap and worth doing once both land, even though it's not a formal task.
- **Risk:** `_SCHEMA_VERSION`'s bump to `3` is not merely additive/informational the way
  the plan's first draft characterized it — occurrence-suffixed `model_failure_counts`
  keys are an externally persisted shape change. Reverting code does not retroactively fix
  already-produced `schema_version: 3` artifacts with suffixed keys; any external consumer
  parsing this field needs to handle both key shapes going forward. **Mitigation:** none
  needed for this repo (no in-tree consumer), but noted here rather than glossed over.
- **Rollback/checkpoint guidance:** commit each task's work as its own commit; use
  `git revert` for rollback, **not** `git checkout -- <file>` (AGENTS.md/session rules:
  don't discard changes via `checkout`/`reset`/`clean` without explicit confirmation, since
  the working tree may contain other, unrelated user edits — this plan's first draft's
  rollback wording was too casual about this). No data migration occurs at any point.
