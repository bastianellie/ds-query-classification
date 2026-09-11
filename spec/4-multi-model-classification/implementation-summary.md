# Implementation Summary: 4-multi-model-classification

**Status:** Completed
**Date:** 2026-09-11
**Worktree:** `/Users/bastianellie/work/elsevier/projects/general/codebase/ds-query-classification-worktrees/4-multi-model-classification` on branch `spec/4-multi-model-classification`

> **Supersedes** the original 2026-09-10 implementation summary below this note. That
> summary documented the first shipped design (`--models` only, 2+ distinct models
> required, mutually exclusive with `--critics`). This entry documents the **redesign**
> requested immediately afterward: `--model`/`--models`/`--n-classifiers` as three
> independent flags that resolve to one ordered classifier list, per
> `spec/4-multi-model-classification/spec.md`'s "Reopened 2026-09-10" section and
> `plan.md`. The original summary is left in place below for history, since most of its
> "Files Created"/architectural content (`multi_model.py`, `pipeline.py` wiring) is still
> accurate and unchanged by this redesign.

## Overview

Redesigned the CLI surface for multi-classifier voting mode. `--model` now takes exactly
one value (`nargs="+"` so a second value is a clear, named error rather than a silent
drop); `--models` takes 2+ values (duplicates allowed) or exactly 1 (silently overrides
`--model`); a new `--n-classifiers` (default 1) replicates a single resolved model into N
independent classifier instances without requiring `--critics`. All three resolve, once,
to an ordered `list[str]` — length 1 is today's existing plain/`--critics` behavior
unchanged, length 2+ engages multi-classifier voting. Repeated model ids get
occurrence-suffixed audit keys (`"<id>#1"`, `"<id>#2"`) so duplicates stay individually
addressable. `pipeline.py`/`multi_model.py` needed **zero changes** — confirmed via `git
diff --stat` showing no diff on either file — since both already operate on opaque
`dict[str, Classifier]` keys.

## Team Execution

Solo, sequential (no subagents, no parallel streams) — the resolution algorithm's
demonstrated bug density across two prior critique rounds (spec-level and plan-level, see
`plan-critique-consolidated-v-2.md`) made it safer for one person to implement both
`cli.py` and `experiment.py`'s independently-duplicated copies with direct, literal
behavioral parity via `plan.md`'s Shared Resolution Truth Table, rather than risking two
subagents drifting apart on an edge case.

1. Task 1 — `cli.py` + `tests/test_debate.py`
2. Task 2 — `experiment.py` + `tests/test_experiment.py`
3. Task 3 — `README.md`

## Files Modified

- `src/query_classification/cli.py` — `--model` → `nargs="+", default=None`; new
  `--n-classifiers`; `_resolve_classifier_models(args)` (validation + resolution per the
  Shared Resolution Truth Table) and `_build_classifier_dict(resolved_models, build_one)`
  (occurrence-suffixed key construction with a collision guard) replace the old
  `--models`-only validation block; `allow_new_labels` condition fixed to
  `args.critics or len(resolved_models) > 1` (keeping `args.critics` — dropping it would
  have forced `allow_new_labels=True` for ordinary `--critics` usage); classifier
  construction and `critic_model`/`reconciler_model` fallbacks now read from
  `resolved_models` instead of raw `args.model` (no longer a plain string).
- `src/query_classification/experiment.py` — identical redesign, duplicated per this
  project's established small-duplication convention (module docstring): `--model` →
  `nargs="+", default=None` in the `common` group; `--n-classifiers` added to the
  `classification` group (not `common`/`induction`, since `induce` has no classification
  role); `_resolve_classifier_models`/`_build_classifier_dict`/
  `_resolve_single_default_model` added; `_validate_args` stores the resolved list as
  `args.resolved_classifier_models` (computed once, reused by `_construct_classifiers` and
  `main()`); **every** site that branched on raw `args.models` truthiness now gates on
  `len(resolved_models) > 1` instead — `_construct_classifiers`'s multi-classifier branch,
  the no-induction `models_final` fallback, the pre-projection source-column collision
  check, the post-run completeness check, and the failure-count aggregation's pre-seed
  step (now seeded from the actual constructed classifier keys, not raw `--models`, so
  occurrence-suffixed duplicates are no longer silently collapsed/dropped). `induction`
  model fallback now uses `_resolve_single_default_model(args)` (never one of `--models`'
  2+ entries, per spec's AR-1.8 4th case). `_SCHEMA_VERSION` bumped `2` → `3`
  (`model_failure_counts`' keys may now be occurrence-suffixed).
- `README.md` — "Multi-model mode" retitled "Multi-classifier mode", rewritten for both
  entry paths (`--models` with 2+ values, or `--model`+`--n-classifiers`); Options table
  updated (`--model`'s one-value constraint, `--models` duplicates-allowed, new
  `--n-classifiers` row); Experiment runner section's flag list, data-egress note, and
  analysis-notebook-compatibility note updated; project-layout entry for `multi_model.py`.
- `tests/test_debate.py` / `tests/test_experiment.py` — see Test Results below.

## Test Results

`.venv/bin/python -m pytest -q` (full suite, in the worktree): **210 passed**, 0 failures
(baseline before this redesign: 191). `pipeline.py`/`multi_model.py` have zero diff.

**Existing tests updated** (behavior reversed by the redesign, verified against actual
bodies, not just names):
- `test_cli_models_fewer_than_two_exits_before_any_llm_call` /
  `test_cli_models_duplicate_value_exits_before_any_llm_call` (`test_debate.py`) → renamed
  and rewritten as `test_cli_models_single_value_overrides_model_silently` /
  `test_cli_models_duplicate_value_uses_occurrence_suffixed_keys` (both now assert success).
- `test_models_requires_at_least_two_values` / `test_models_rejects_duplicate_value`
  (`test_experiment.py`) → same rename/rewrite pattern.
- `test_induce_alone_resolves_induction_model_fallback` — `ns.model` assertion updated for
  `nargs="+"` (`"base-model"` → `["base-model"]`).
- `test_models_end_to_end_run_config` — `schema_version` assertion `2` → `3`.
- `test_models_run_with_induction_merges_all_fields` — argv rewritten from `--model
  induction-model-id --models a b` (now illegal — `--model` explicit + `--models` 2+
  values) to `--induction-model induction-model-id --models a b`.
- `test_induce_help_does_not_list_models` — loosened from a raw `"--models" not in`
  substring check (which broke on `--model`'s own help text legitimately cross-referencing
  `--models` in prose) to checking for the flag's own definition line (`"--models MODEL"`).
- `test_plain_mode_closed_vocabulary` — its hand-built `argparse.Namespace` updated
  (`model="m"` → `model=["m"]`) to match the new `nargs="+"` shape; caught independently
  during implementation (not flagged by either critique round) by reasoning through what
  `_construct_classifiers` would do with a bare one-character string under the new list-
  based resolution (silently correct only by coincidence for a 1-character model id).

**New tests** (Shared Resolution Truth Table coverage, repeated-instance behavior, and the
three previously-missed `experiment.py` downstream sites), in both test files:
`_resolve_classifier_models`/`_build_classifier_dict` direct unit tests (regression guard
for `--models a b c` alone, `--n-classifiers` irrelevance once `--models` has 2+ values,
the universal `--n-classifiers >= 1` floor, `--model`+`--n-classifiers` under
`CEREBUS_MODE=azure` succeeding vs. `--models` with 2+ distinct models under the same
erroring, the pathological-model-id collision guard); an exact-call-count test for
`--model x --n-classifiers 5`; an out-of-order-completion tie-break test and a
partial-failure test for two instances sharing one model id; and, in `test_experiment.py`
specifically, a `--model x --n-classifiers 3` end-to-end test asserting the multi-
classifier audit columns/completeness check/`model_failure_counts` all engage correctly
with `args.models is None`, plus a dedicated source-column-collision regression test for
that same `args.models is None` case (the exact scenario the three previously-missed
downstream sites would have silently mishandled).

CLI surface verified directly: `classify.py --help`, `python -m query_classification
--help`, `experiment.py classify --help`, `experiment.py run --help` all show
`--n-classifiers` and `--model`'s updated one-value help text; `experiment.py induce
--help` shows `--model` but not `--n-classifiers`'s own flag definition.

## Spec Adherence

| Requirement | Status | Implementation | Test |
|---|---|---|---|
| FR-1.1 | Done | `cli.py::_resolve_classifier_models`, `experiment.py::_resolve_classifier_models` | Shared Resolution Truth Table tests in both `test_debate.py` and `test_experiment.py` (`_resolve`/`test_resolve_*`, `test_experiment_resolve_*`) |
| FR-1.2 | Done | shared `temperature=None`/prompt/schema per instance (`_build_classifier_dict`) | `test_cli_n_classifiers_replication_calls_every_instance_exactly_once` |
| FR-1.3 | Done | unaffected in `multi_model.py`; new same-model tie-break case | `test_cli_n_classifiers_repeated_instance_tie_break_by_construction_order` |
| FR-1.4 / FR-1.7 | Done | `_build_classifier_dict`'s occurrence-suffixed keys, both files | `test_cli_models_duplicate_value_uses_occurrence_suffixed_keys`, `test_models_duplicate_value_uses_occurrence_suffixed_keys`, `test_cli_n_classifiers_repeated_instance_partial_failure_keeps_suffixed_keys_distinct`, `test_n_classifiers_repeated_instance_failure_count_keeps_keys_distinct` |
| FR-1.5 / FR-1.6 | Unaffected (confirmed) | `pipeline.py`/`multi_model.py` zero diff | pre-existing `test_multi_model.py` suite, still passing |
| FR-1.8 | Done | both entry points redesigned identically | full suite + `--help` checks above |
| FR-1.9 | Done | `experiment.py::_construct_classifiers`/`main`, `_SCHEMA_VERSION = 3` | `test_models_end_to_end_run_config`, `test_models_duplicate_value_uses_occurrence_suffixed_keys` (asserts `model is None`), `test_n_classifiers_replication_produces_multi_classifier_run_config`, `test_n_classifiers_repeated_instance_failure_count_keeps_keys_distinct`, `test_models_non_models_run_has_null_keys` (unaffected case) |
| FR-1.10 | Done | — | this test suite |
| AR-1.1 / AR-1.2 | Confirmed unaffected | — | `git diff --stat` shows zero diff on `pipeline.py`/`multi_model.py` |
| AR-1.3 | Done | `_build_classifier_dict` (both files), collision guard | `test_build_classifier_dict_collision_guard_rejects_pathological_model_id`, `test_experiment_build_classifier_dict_collision_guard_rejects_pathological_model_id` |
| AR-1.4 / AR-1.5 | Done | `experiment.py`'s completeness check and source-column collision check, both switched to `len(resolved_models) > 1` | `test_n_classifiers_replication_produces_multi_classifier_run_config`, `test_n_classifiers_replication_source_column_collision` |
| AR-1.6 | Unaffected (confirmed) | `multi_model.py` zero diff | pre-existing concurrency/tally tests, still passing |
| AR-1.7 | Done | identical `gateway_kwargs`/prompt/schema shared across every constructed instance | pre-existing Cerebus-routing and `allow_new_labels` carve-out tests, still passing |
| AR-1.8 | Done | `--model` `nargs="+"`, `default=None`, in both files; 4-case induction-fallback resolution | `test_experiment_resolve_induce_ignores_models_attribute_entirely`, `test_induce_alone_resolves_induction_model_fallback`, `test_resolve_model_replication_with_cerebus_azure_mode_succeeds_for_same_model` |
| AR-1.9 | Done | `README.md` | manual read-through per plan Task 3 |

All 10 FRs and 9 ARs implemented/confirmed-unaffected and verified. No skipped or
partially-implemented requirements.

## Deviations from Spec

None. `plan.md`'s Spec Deviations table was empty going into implementation (both genuine
spec-level defects the plan critique found — the backwards `model`-null condition, the
missing 4th induction-fallback case — were fixed directly in `spec.md` before this
implementation began, not worked around here).

---

# Implementation Summary: 4-multi-model-classification (original, 2026-09-10, superseded above)

**Status:** Completed
**Date:** 2026-09-10
**Worktree:** `/Users/bastianellie/work/elsevier/projects/general/codebase/ds-query-classification-worktrees/4-multi-model-classification` on branch `spec/4-multi-model-classification`

## Overview

Implemented `--models`: a standalone classification mode (mutually exclusive with
`--critics`) where 2+ distinct LiteLLM models classify each row once, independently, and
their answers are merged per category by plurality vote on each model's top label —
ties broken by `--models` list order. Available on both `classify.py` and `experiment.py
classify`/`run` (not `induce`). Adds a per-model/vote-tally/failure audit trail to the
output CSV and to `experiment.py`'s `run_config.json`.

## Team Execution

| Stream | Scope | Completed by |
|---|---|---|
| Task 1 | `multi_model.py` module + ADR + `ARCHITECTURE.md` amendment + orchestration tests | Solo (main session) |
| Task 2 | `pipeline.py` wiring + end-to-end/restore/public-API tests | Solo (main session) |
| Task 3 | `cli.py` wiring + tests | Parallel subagent |
| Task 4 | `experiment.py` wiring + tests | Parallel subagent |
| Task 5 | `README.md` documentation | Solo (main session) |

**Sequential phases:** Task 1 → Task 2 (Task 2 imports/dispatches to Task 1's module) →
{Task 3, Task 4} → Task 5 (documents the finished flag behavior).
**Parallel phase:** Tasks 3 and 4 ran concurrently as two subagents — confirmed
zero file overlap (`cli.py`+`tests/test_debate.py` vs. `experiment.py`+
`tests/test_experiment.py`) before dispatching. Both subagents' diffs were independently
re-reviewed against the plan and the actual current file content before committing (not
just trusted from their self-reports).

## Files Created
- `src/query_classification/multi_model.py` — vote-tally/tie-break/audit-column
  orchestration for `--models` mode; mirrors `debate.py`'s role, reuses
  `debate._vote_bucket`.
- `tests/test_multi_model.py` — orchestration-level tests (direct `run_multi_model`
  calls) plus end-to-end/restore tests (via `classify_csv`).
- `spec/4-multi-model-classification/ADR.md` — INV-1/INV-7 amendment record.

## Files Modified
- `src/query_classification/pipeline.py` — `classify_csv`'s `classifier` param is now
  `Classifier | None` (same position, still required-by-value); new `models:
  dict[str, Classifier] | None` param; validates the mode-exclusivity/count/non-empty
  invariants itself; every `critics`-gated branch (column creation, restore-seed,
  restore-completeness, non-restore reset, the inter-category/text-column collision
  check) now also branches for `models`, via one shared `audit_suffixes` tuple.
- `src/query_classification/cli.py` — `--models` flag, 5-case validation, per-model
  `Classifier` construction (temperature unset, identical prompt/schema/gateway config),
  `--allow-new-labels` carve-out.
- `src/query_classification/experiment.py` — `--models` on `classify`/`run`; same 5
  validations; `_construct_classifiers`'s `models` dict now unconditionally carries
  `classification_models`/`model_failure_counts` (`null` unless `--models` is used);
  fixed the `run`-with-induction merge to also carry `model` across (previously only
  `critic_model`/`reconciler_model` were copied); `model_failure_counts` computed into
  `models_final` directly (not `config["models"]`, which doesn't exist as a dict until
  later) and pre-seeded to 0 per model; post-run completeness check and source-column
  collision check extended for all 3 multi-model audit suffixes; `_SCHEMA_VERSION` bumped
  `1` → `2`.
- `spec/ARCHITECTURE.md` — INV-1 (new `multi_model.py` Module Boundary Map row, Mermaid
  edges, `pipeline.py`/`experiment.py` "May import from" additions) and INV-7 (new
  `tests/` direct-import exception for `multi_model.py`), per `ADR.md`.
- `README.md` — new "Multi-model mode" section, `--models` Options-table row, Experiment
  runner section updates (availability, `--restore` scope, data-egress/CSV-injection/
  `analyze_run.ipynb` notes), project-layout entry.

## Test Results

`.venv/bin/python -m pytest -q` (full suite): **191 passed**, 0 failures (baseline before
this spec: 148).

Per-scope runs during implementation, all green:
- `tests/test_multi_model.py` — orchestration + end-to-end/restore tests.
- `tests/test_debate.py` — `classify_csv` public-API validation + `cli.py`-level tests.
- `tests/test_experiment.py` — `experiment.py`-level validation/end-to-end/`run_config.json`
  tests.
- `tests/test_building_blocks.py` — unaffected, still passing (regression check).

CLI surface verified directly: `classify.py --help`, `experiment.py classify --help`,
`experiment.py run --help` all list `--models`; `experiment.py induce --help` does not.

## Spec Adherence

| Requirement | Status | Implementation | Test |
|---|---|---|---|
| FR-1.1 | Done | `cli.py::main` (5 validations), `experiment.py::_validate_args` | `test_classify_csv_rejects_*`, `test_models_and_critics_mutually_exclusive` + 4 siblings in both test files |
| FR-1.2 | Done | `cli.py`/`experiment.py` classifier construction (`temperature` unset, `allow_new_labels` carve-out) | `test_calls_every_model_exactly_once`, `--allow-new-labels` carve-out tests in both CLI test files |
| FR-1.3 | Done | `multi_model.run_multi_model` (index-ordered tallying, `Counter.most_common()`) | `test_tie_break_by_list_order_regardless_of_completion_order` |
| FR-1.4 | Done | `multi_model.run_multi_model` (`_votes`/`_by_model` columns) | `test_by_model_audit_column_has_every_successful_model`, `test_audit_columns_end_to_end_via_classify_csv` |
| FR-1.5 | Done (`classify.py`/library only, confirmed N/A on `experiment.py`) | `pipeline.py`'s restore branches | `test_restore_requires_all_multi_model_audit_columns_not_just_category_value`, `test_restore_from_separate_completed_output_skips_done_rows` |
| FR-1.6 | Done | `multi_model.run_multi_model` (never raises, total-failure shape) | `test_total_failure_never_raises_and_records_every_model`, `test_total_model_failure_increments_classified_not_failed_counter` |
| FR-1.7 | Done | `multi_model._sanitize_error` (`"Classification"` stage) | `test_partial_failure_excludes_failed_model_and_sanitizes_error` |
| FR-1.8 | Done | `cli.py`/`experiment.py` `build_parser` | `--help` checks above; `test_induce_help_does_not_list_models` |
| FR-1.9 | Done | `experiment.py::_construct_classifiers`/`main` | `test_models_end_to_end_run_config`, `test_models_non_models_run_has_null_keys`, `test_models_run_with_induction_merges_all_fields`, `test_models_failure_counts_per_model` |
| FR-1.10 | Done | — | `tests/test_multi_model.py` (new), additions to `tests/test_debate.py`/`tests/test_experiment.py` |
| AR-1.1 | Done | `src/query_classification/multi_model.py`, `spec/ARCHITECTURE.md`, `ADR.md` | import graph re-verified manually (no import-boundary linter in this repo) |
| AR-1.2 | Done | `pipeline.py::classify_csv` signature + validation | `test_classify_csv_rejects_*` (6 cases) |
| AR-1.3 | Done | `pipeline.py` (5 branches), `experiment.py` post-run completeness check | full `test_multi_model.py`/`test_experiment.py` suites |
| AR-1.4 | Done | `experiment.py` source-column collision check | `test_models_source_column_collision_all_suffixes` (parametrized, 3 suffixes) |
| AR-1.5 | Done | `multi_model.run_multi_model` (nested executor, index preallocation, never-raise) | `test_bounded_concurrency_all_models_run_at_once`, `test_tie_break_by_list_order_regardless_of_completion_order` |
| AR-1.6 | Done | shared `temperature=None`/prompt/schema/gateway config in both entry points | Cerebus routing test in `test_debate.py`; shared-config assertions in `test_multi_model.py` |
| AR-1.7 | Done | `--model`'s existing code paths untouched | full suite regression (no pre-existing test broke) |
| AR-1.8 | Done | `README.md` | manual read-through per plan Task 5 |

All 10 FRs and 8 ARs implemented and verified. No skipped or partially-implemented
requirements.

## Deviations from Spec

None. The plan's own Spec Deviations table (`spec/4-multi-model-classification/plan.md`)
listed none, and implementation didn't surface any new ones — every subtlety encountered
(the `classify_csv` signature constraint, the `run_config.json` merge/KeyError bugs, the
duplicate-detection list-vs-dict distinction) was already anticipated and resolved during
the two-round spec critique and one-round plan critique before implementation began.

## Notes for follow-up (not blocking, not part of this spec)

- `spec/docs/` doesn't exist in this repo yet (confirmed) — living-docs update (Step 9)
  was skipped per `/spec-implement`'s own instructions for that case. Run `/spec-docs
  --full` first if living docs are wanted going forward.
- The repository has pre-existing untracked files unrelated to this spec
  (`data/example_queries_classified.csv`, `experiments/`, `spec/initial-classification-
  with-critics-spec.md`) — left untouched in both the main checkout and this worktree, as
  they were out of scope for this implementation.
