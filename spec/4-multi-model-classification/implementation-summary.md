# Implementation Summary: 4-multi-model-classification

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
