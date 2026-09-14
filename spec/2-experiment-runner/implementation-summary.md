# Implementation Summary: 2-experiment-runner

**Status:** Completed
**Date:** 2026-09-08
**Worktree:** `/Users/bastianellie/work/elsevier/projects/general/codebase/ds-query-classification-worktrees/2-experiment-runner` on branch `spec/2-experiment-runner`

> **Historical snapshot, superseded in part on 2026-09-14.** This summary describes the
> induction pipeline as it was *originally* implemented and closed: an open-ended
> `labels: [{label, description}]` response, reconciled against the dataset's label set
> after the call (see e.g. "label-set reconciliation" below). That design was replaced by
> a **positional** response contract (`description_1..description_N`, no label names,
> arity pinned by required fields) — see `spec/2-experiment-runner/spec.md`'s Change Log
> entry "Update: induction re-engineering — positional descriptions + total-budget
> sampling (2026-09-14, post-close)" for the motivation and design, and this spec's own
> updated FR-2.1/FR-2.4/AR-2.1/AR-2.3 for current behavior. Left as written below rather
> than rewritten, per this project's convention for superseded summaries (see spec 4's).

## Overview

Implemented the experiment runner: a separate `experiment.py` entry point with
`induce`/`classify`/`run` subcommands that loads a dataset (local CSV pair or
HuggingFace Hub dataset), induces a `categories.json` from a labeled train
split via one LLM call, classifies a test split with any classifier mode the
repo offers (plain or `--critics`), and writes a self-contained, auditable run
directory. No existing module's behavior changed (`pipeline.py`/`cli.py`/
`classifier.py`/`categories.py`/`debate.py`/`classify.py` are byte-identical
to `main`, confirmed via `git diff --exit-code`).

## Team Execution

| Agent | Role | Task(s) completed |
|---|---|---|
| Claude (in-session) | Orchestrator | Task 5 (ADR.md), Task 4 (experiment.py + root entry point + README), Task 6 (test suite), integration debugging |
| Subagent A | dataset_io.py | Task 1 |
| Subagent B | schema/prompts/resources | Task 2 |
| Subagent C | induction.py | Task 3 |

**Parallel phase:** Tasks 1/2/3 (three subagents) ran concurrently — the
plan's file-conflict analysis confirmed zero file overlap and zero functional
dependency between them; Task 5 (ADR.md) was written solo in the same window.
**Sequential phase:** Task 4 (depends on all of 1/2/3) was written solo, then
Task 6 (depends on Task 4) was written solo.

## Files Created

- `src/query_classification/dataset_io.py` — local/HF dataset loading, HF-id validation, label normalization (`ClassLabel`/withheld handling), column projection metadata, row filtering
- `src/query_classification/induction.py` — seeded sampling, prompt-size preflight, one induction call, label-set reconciliation (pure function, no filesystem access)
- `src/query_classification/experiment.py` — subcommand parser, classifier construction for every role, run-directory orchestration, collision/completeness checks
- `experiment.py` (repo root) — thin entry point mirroring `classify.py`
- `resources/prompts/induction_prompt.txt` — bundled induction system prompt
- `tests/test_experiment.py` — 68 tests
- `spec/2-experiment-runner/ADR.md` — INV-1/INV-7 amendments

## Files Modified

- `src/query_classification/schema.py` — added `build_induction_model`
- `src/query_classification/prompts.py` — added `build_induction_prompt`
- `src/query_classification/resources.py` — added `DEFAULT_INDUCTION_PROMPT_FILE`
- `pyproject.toml` — added `[project.optional-dependencies] hf = ["datasets>=4"]`
- `README.md` — experiment runner usage, run-directory layout, plain-mode divergence, data-egress note, updated project layout

## Test Results

`.venv/bin/python -m pytest -q` → **104 passed** (36 pre-existing + 68 new).
`git diff --exit-code -- src/query_classification/{pipeline,cli,classifier,categories,debate}.py classify.py src/query_classification/{__init__,__main__}.py` → exit 0 (byte-unchanged). Both `python experiment.py --help` and `python -m query_classification.experiment --help` work. A mutation-testing spot check (reintroducing the original per-label-RNG design) confirmed `test_induce_rng_exact_sequence` genuinely fails on that regression, not just passing trivially.

## Spec Adherence

| Requirement | Status | Implementation | Test |
|---|---|---|---|
| FR-1.1 | Done | `experiment.py::_resolve_mode`, `dataset_io.load_local_split` | `test_run_end_to_end`, `test_split_availability_error_cases` |
| FR-1.2 | Done | `dataset_io.validate_hf_id`, `load_hf_splits` | `test_validate_hf_id_*`, `test_load_hf_splits_missing_split_lists_available`, `test_load_hf_splits_resolved_sha_best_effort` |
| FR-1.3 | Done | `dataset_io.resolve_hf_labels`, `load_local_split` | `test_load_local_split_verbatim_numeric_labels`, `test_resolve_hf_labels_unsupported_feature_type` |
| FR-1.4 | Done | `experiment.py` missing-column check, `_validate_args` | `test_missing_column_errors_list_available`, `test_text_label_column_must_differ` |
| FR-1.5 | Done | `dataset_io.load_hf_splits` (lazy import), `pyproject.toml` | `test_load_hf_splits_missing_extra_hint` |
| FR-1.6 | Done | `experiment.py::_resolve_mode` | `test_split_availability_error_cases` (3 cases) + `test_run_end_to_end`/classify tests (other 2) |
| FR-1.7 | Done | `dataset_io.project_and_filter` | `test_project_and_filter_drops_empty_rows_once`, `test_project_and_filter_label_only_dropped_for_train_role` |
| FR-1.8 | Done | `dataset_io.resolve_hf_labels`, `experiment.py` test_labels derivation | `test_resolve_hf_labels_mixed_withheld`, `test_run_end_to_end_withheld_test_labels`, `test_unseen_test_labels_recorded` |
| FR-2.1 | Done | `induction.induce` (single `random.Random(seed)`) | `test_induce_rng_exact_sequence`, `test_induce_at_cap_label_consumes_no_rng_state` |
| FR-2.2 | Done | `induction.induce` (one call, all labels grouped) | `test_induce_single_call_sees_all_labels_grouped` |
| FR-2.3 | Done | `induction.induce` (truncation + serialized-payload preflight) | `test_induce_truncation_marker`, `test_induce_prompt_size_preflight` |
| FR-2.4 | Done | `induction.induce` (set + length reconciliation) | `test_induce_rejects_missing_or_invented_labels`, `test_induce_rejects_duplicate_label` |
| FR-2.5 | Done | `induction.induce`, `experiment.py::_check_reserved_sentinel` | `test_induce_rejects_reserved_sentinel`, `test_classify_rejects_supplied_sentinel` |
| FR-2.6 | Done | `induction.induce` (sanitized `RuntimeError`) | `test_induce_sanitizes_failure` |
| FR-2.7 | Done | `induction.induce` (degenerate-input checks) | `test_induce_rejects_degenerate_input` |
| FR-3.1 | Done | `experiment.py::main` (`induce` dispatch) | `test_induce_subcommand_writes_categories_only` |
| FR-3.2 | Done | `experiment.py::main` (single-category check) | `test_classify_rejects_multi_category_file` |
| FR-3.3 | Done | `experiment.py::main` (`run` dispatch) | `test_run_end_to_end` |
| FR-3.4 | Done | `experiment.py::main` (run-dir writing, atomic JSON) | `test_run_dir_overwrite_preserves_unrelated_files`, `test_overwrite_same_path_categories_noop`, `test_run_dir_refuses_nonempty_without_overwrite`, `test_supplied_categories_byte_verbatim_copy` |
| FR-3.5 | Done | `experiment.py::_construct_classifiers` | `test_plain_mode_closed_vocabulary`, `test_limit_truncates_before_classifying`, `test_run_subcommand_has_induction_flags` |
| FR-3.6 | Done | `experiment.py::main` (collision checks against `source_column_names`) | `test_gold_column_rename_and_collision`, `test_collision_against_projected_away_column`, `test_critics_audit_column_collision` |
| FR-3.7 | Done | `tests/test_experiment.py` | this file (104 tests total) |
| FR-3.8 | Done | `experiment.py::main` (completeness check) | `test_completeness_check_partial_failure`, `test_completeness_check_allow_partial` |
| AR-1.1 | Done | module named `dataset_io.py` | N/A (naming, verified by import working) |
| AR-1.2 | Done | `spec/2-experiment-runner/ADR.md` | N/A (doc) |
| AR-1.3 | Done | `experiment.py` column projection (both HF and local paths) | `test_load_hf_splits_overwrites_label_column_with_resolved_values`, `test_collision_against_projected_away_column` |
| AR-2.1 | Done | `schema.build_induction_model` (fixed list, not dynamic field) | `test_build_induction_model_list_shape_not_dynamic_field` |
| AR-2.2 | Done | `prompts.build_induction_prompt`, `resources/prompts/induction_prompt.txt` | `test_build_induction_prompt_non_empty_no_placeholders` |
| AR-2.3 | Done | `induction.py` imports `Classifier` unmodified | verified by `git diff --exit-code` on `classifier.py` |
| AR-3.1 | Done | `experiment.py` calls `classify_csv` unmodified | `git diff --exit-code` on `pipeline.py` |
| AR-3.2 | Done | root `experiment.py`, `python -m query_classification.experiment` | `test_module_invocation_help` |
| AR-3.3 | Done | no `console_scripts`/entry-points added | manual check of `pyproject.toml` |
| AR-3.4 | Done | `experiment.py::_construct_classifiers`, `_validate_args` | manual + `test_plain_mode_closed_vocabulary` |

## Deviations from Spec

None. One deviation from the **plan** (not the spec) was found and fixed during Task 4 integration: the plan never explicitly assigned column-projection responsibility (AR-1.3) for the **local-CSV** path to any function — `dataset_io.load_local_split` reads the whole file as-is, and `project_and_filter` only filters rows, never columns. This was caught via end-to-end smoke testing (a pre-existing extra column leaked into `test.csv`), and fixed by adding an explicit projection step in `experiment.py::main` for both sources uniformly, right after the missing-column check and before `project_and_filter`. No spec or plan text needed to change — this is an internal data-flow detail, and both AR-1.3's and FR-3.6's acceptance criteria hold with this fix in place (see `test_collision_against_projected_away_column`).

---

# Implementation Summary: 2-experiment-runner — induction redesign (2026-09-14)

**Status:** Completed
**Date:** 2026-09-14
**Worktree:** `/Users/bastianellie/work/elsevier/projects/general/codebase/ds-query-classification-worktrees/2-experiment-runner` on branch `spec/2-experiment-runner`
**Plan:** `spec/2-experiment-runner/plan.md` (Ready, critique: `plan-critique-consolidated-v-2.md`)

## Overview

Implements the induction redesign appended to this spec's Change Log on 2026-09-14: the
induction LLM response becomes **positional** (`description_1..description_N`, no label
names — a missing/invented label is now structurally impossible instead of detected by a
post-hoc set comparison), a new `--induction-examples` total-budget sampling mode is added
alongside the existing `--examples-per-label` per-label cap, and the interim
`--induction-retries` flag (added the same day, before this redesign was chosen) is
removed.

Motivated by a real production failure: `experiments/pubmed-rct` runs with 100
examples/label repeatedly failed with `induction response label set does not match the
dataset's labels (missing: ['conclusions'], ...)` — the old schema let the model
silently drop a label from its echoed-back response, an open-ended list with no arity
bound.

## Team Execution

Solo, sequential, exactly as planned — Tasks 1–3 landed as one continuous unit (the
contract, algorithm, and wiring have a hard interface chain; the plan explicitly declared
the full suite **expected red** between Task 1 and the end of Task 3, which held: 220
passed / 4 failed after Tasks 1–2, all four failures tracing to the same root cause
(`build_induction_model()` still being called with no argument) and resolving together
once Task 3 landed). Task 4 (docs/scripts) followed once the two flag names were frozen.

## Files Modified

- `src/query_classification/schema.py` — `build_induction_model()` → `build_induction_model(n_labels)`:
  `category_description` plus N **required** `description_i` fields via `create_model`,
  replacing the open-ended `labels: list[{label, description}]` model. Verified live
  against the installed stack (litellm 1.100.1, pydantic 2.13.5) that a length-bounded list
  is not an option: `litellm.utils.type_to_response_format_param` serializes
  `"strict": true`, and strict structured output rejects `minItems`/`maxItems` — required
  fields is the only mechanism that pins arity without losing strict-mode compatibility.
- `resources/prompts/induction_prompt.txt` — closing response-format paragraph rewritten
  to state the positional contract (`description_i` for the label at position `i`);
  untrusted-data rules paragraph preserved verbatim (this is the injection defense, AR-2.2).
- `src/query_classification/induction.py` — `_resolve_quotas()` (new): per-label cap or
  largest-remainder total-budget quotas with shortfall redistribution to labels with spare
  rows, prototyped and verified against 7 edge cases (including cascading shortfall)
  independently by both plan-critique passes before implementation. `induce()`: signature
  takes `examples_per_label`/`induction_examples` (exactly one non-`None`) instead of
  `max_reconciliation_retries`; payload is now an ordered array of
  `{position, label, examples}` entries (not a dict keyed by label); an internal arity
  guard compares the handed classifier's exact field set against
  `{category_description, description_1..N}` (not a count — a substring/count check would
  also match `category_description` itself, verified live); the retry loop and label-set
  reconciliation are deleted entirely; `Category` is built directly from the runner's own
  sorted label list zipped against the response's `description_i` values. Module and
  function docstrings rewritten to describe the positional contract, not reconciliation.
- `src/query_classification/experiment.py` — `--induction-examples` flag added;
  `--examples-per-label`'s argparse default changed `20` → `None` (mandatory, not
  cosmetic: otherwise it collides with every `--induction-examples` invocation);
  `--induction-retries` removed. `_validate_args` gained the mutual-exclusion check and a
  new `_resolve_induction_sizing(args)` helper, called once and stored as
  `args.resolved_induction_sizing` — the single authoritative source for both
  `run_induction()`'s arguments and the run_config record (a `classify`-only run has
  neither attribute at all and resolves to `(None, None)`). `main()` gained an FR-2.7
  preflight (empty train split / zero distinct labels / `--induction-examples <
  n_labels` floor) relocated *ahead of* `_construct_classifiers`, specifically so an
  all-filtered train split still gets FR-2.7's actionable message instead of
  `build_induction_model(0)`'s generic arity error. `_construct_classifiers` gained an
  `n_labels` parameter threaded through to `build_induction_model`. `_SCHEMA_VERSION`
  bumped `3` → `4` (a real shape change: `induction_examples` added,
  `examples_per_label` nullable, `induction_retries` removed).
- `tests/test_experiment.py` — `FakeInductionClassifier` gained a `classification_model`
  attribute (inferred from the response's own `description_i` keys, or explicit
  `n_labels=`) so it satisfies the new arity guard; `_induction_response()` produces the
  positional shape; `fake_classify`'s induction branch generalized to any
  `description_i`-shaped field set (was a fixed `{category_description, labels}` check).
  6 tests deleted (their subject — the label-echo reconciliation and its retry — no longer
  exists); 22 added (schema arity/strict-compatibility, prompt-contract regression,
  positional-payload structure, positional mapping under out-of-sorted-order source rows,
  arity-guard both directions, 6 quota/budget-mode edge cases mirroring the ones verified
  by hand before implementation, mutual-exclusion/budget-floor/all-rows-filtered CLI
  regressions, a below-cap-label coverage gap found during spec-adherence review, and
  `schema_version`/default-invocation/classify-only-nulls run_config assertions).
- `tests/test_cerebus.py` — `recorded_classifier_inits`'s fake classify (a second,
  independent copy of the same induction-role detection) updated identically; this file
  wasn't in the plan's file list but broke for the same root cause and was fixed as an
  obviously-correct, same-shape mechanical fix.
- `README.md` — Options prose for `induce`/`run`'s induction flags rewritten.
- `spec/ARCHITECTURE.md:78` — `induction.py`'s Module Boundary Map responsibility
  description reworded (row content only; INV-1's import-edge rule unchanged, no ADR).
- `spec/2-experiment-runner/implementation-summary.md` — this file: the original
  2026-09-08 summary marked historical rather than rewritten (matches spec 4's precedent
  for a superseded summary), this section appended.
- `experiments/{ag-news,trec,pubmed-rct}/run_experiment.sh` — added `INDUCTION_EXAMPLES`
  (guarded like the existing `TEST_LIMIT` pattern) and a `DRY_RUN=1` argv-printing mode;
  switched `EXAMPLES_PER_LABEL`'s default expansion from `${VAR:-20}` to `${VAR-20}` (the
  colon form silently collapses an explicitly-empty override back to `20`, verified live);
  detect a trailing `--induction-examples` flag in `"$@"` and suppress the default
  `--examples-per-label` in that case too. **Found and fixed during this task's own
  verification** (not anticipated by the plan): the scripts' `ARGS=(...)` array still had
  its own unconditional `--examples-per-label "$EXAMPLES_PER_LABEL"` line, which the new
  conditional-append logic didn't remove — every dry-run case sent a duplicate
  `--examples-per-label` (or one alongside `--induction-examples`) until removed.
- `experiments/{ag-news,trec,pubmed-rct}/README.md` — documented `INDUCTION_EXAMPLES` and
  `DRY_RUN`.

## Test Results

`.venv/bin/python -m pytest -q`: **229 passed** (baseline before this task: 214; 6 tests
deleted because their subject no longer exists, 21 added — net +15).

Specific pre-existing tests re-run to confirm unaffected behavior survived:
`test_induce_at_cap_label_consumes_no_rng_state`, `test_induce_rng_exact_sequence`,
`test_induce_truncation_marker`, `test_induce_prompt_size_preflight`,
`test_induce_rejects_reserved_sentinel`, `test_induce_sanitizes_failure`,
`test_induce_rejects_degenerate_input`, `test_induce_category_round_trips` — all pass.

CLI surface verified directly: `induce --help`/`run --help` show `--examples-per-label`
and `--induction-examples`, no `--induction-retries`; `classify --help` shows none of the
three (the induction argument group is not a parent of that subparser).

`git diff --stat` confirms **zero** changes to `cli.py`, `pipeline.py`, `debate.py`,
`multi_model.py`, `dataset_io.py`, `classifier.py` (AR-3.1/AR-3.2 hold).

Orphan sweep (`grep -rn "induction.retries\|induction_retries\|max_reconciliation_retries"`
over `src/`, `tests/`, `README.md`, `experiments/`) returns nothing except the generated
`egg-info/PKG-INFO` build artifact (regenerates automatically; not hand-edited, per
AGENTS.md).

**Not verifiable in this session:** the real PubMed RCT induction that motivated this work
needs live provider credentials. The acceptance test is `experiments/pubmed-rct/
run_experiment.sh` (optionally with `INDUCTION_EXAMPLES=` to reproduce the original 100
examples/label scenario in budget form) completing induction where it previously died on
`missing: ['conclusions']`.

## Spec Adherence

| Requirement | Status | Implementation | Test |
|---|---|---|---|
| FR-2.1 | Done | `induction.py::_resolve_quotas`/`induce`, `experiment.py::_resolve_induction_sizing` | `test_induce_budget_quotas_sum_to_exactly_n_largest_remainder`, `test_induce_budget_n_equals_n_labels_gives_every_label_one`, `test_induce_budget_redistributes_shortfall_to_labels_with_spare_rows`, `test_induce_budget_cascading_shortfall_across_multiple_scarce_labels`, `test_induce_budget_all_labels_scarce_sums_to_available_rows`, `test_induce_budget_below_label_count_rejected_with_zero_calls`, `test_induce_budget_mode_is_seed_deterministic_but_not_cross_label_independent`, `test_induce_requires_exactly_one_sizing_mode`, `test_induce_examples_flags_mutually_exclusive`, `test_induce_below_cap_label_contributes_all_its_examples`, `test_induce_at_cap_label_consumes_no_rng_state` (unaffected, re-verified) |
| FR-2.2 | Done | `induction.py::induce` (ordered payload construction) | `test_induce_payload_is_an_ordered_positional_array_not_an_object`, `test_induce_single_call_sees_all_labels_grouped` |
| FR-2.3 | Done (unaffected logic, sizing-flag-aware error message) | `induction.py::induce` preflight | `test_induce_prompt_size_preflight` (re-verified) |
| FR-2.4 | Done | `induction.py::induce` (positional `Category` construction), `schema.py::build_induction_model` | `test_induce_positional_mapping_survives_out_of_order_completion_risk`, `test_build_induction_model_positional_required_fields`, `test_build_induction_model_rejects_short_response`, `test_induce_category_round_trips` (re-verified) |
| FR-2.5 | Unaffected (confirmed) | `induction.py::induce` reserved-sentinel loop | `test_induce_rejects_reserved_sentinel` (re-verified) |
| FR-2.6 | Unaffected (confirmed) | `induction.py::induce` `RuntimeError` sanitization | `test_induce_sanitizes_failure` (re-verified) |
| FR-2.7 | Done | `experiment.py::main` preflight (relocated ahead of classifier construction) + `induction.py::induce`'s own guards | `test_induce_all_rows_filtered_reports_fr_2_7_error_not_arity_error`, `test_induce_budget_below_label_count_rejected`, `test_induce_rejects_degenerate_input` (re-verified) |
| AR-2.1 | Done | `schema.py::build_induction_model` | `test_build_induction_model_positional_required_fields`, `test_build_induction_model_rejects_short_response`, `test_build_induction_model_rejects_zero_labels`, `test_build_induction_model_serialization_stays_strict_compatible` |
| AR-2.2 | Done | `resources/prompts/induction_prompt.txt`, `induction.py::induce` payload | `test_build_induction_prompt_states_positional_contract_not_label_echo`, `test_induce_payload_is_an_ordered_positional_array_not_an_object` |
| AR-2.3 | Done (retry removed, `Classifier` reuse unchanged) | `induction.py::induce`, `experiment.py` (flag/validation removal) | orphan sweep in Final Verification; full suite green with no `induction_retries` references |
| FR-3.4 (`experiment.py` portion) | Done | `experiment.py::main` config dict, `_SCHEMA_VERSION = 4` | `test_models_end_to_end_run_config` (schema_version), `test_run_end_to_end`/`test_models_non_models_run_has_null_keys`/`test_induce_budget_mode_end_to_end_writes_valid_categories` (three run_config sizing shapes) |
| AR-3.4 (`experiment.py` portion) | Done | `experiment.py::_validate_args`, `_resolve_induction_sizing`, `_construct_classifiers` | same as FR-3.4 plus `test_induce_examples_flags_mutually_exclusive`, `test_induce_budget_below_label_count_rejected` |

**Requirements unaffected by this update** (per `plan.md`'s own accounting, re-confirmed
here): all of Feature 1 (FR-1.1–FR-1.8, AR-1.1–AR-1.3) — `dataset_io.py` untouched; all of
Feature 3 except FR-3.4/AR-3.4 above; FR-3.7 (ship unit tests) satisfied structurally, per
CLAUDE.md, by folding tests into Tasks 1–3 rather than a dedicated task.

## Deviations from Spec

Both rows from `plan.md`'s Spec Deviations table, both classified **None** (no
`/spec-update` needed), carried through as implemented:

1. **Experiment scripts.** No FR/AR governs `experiments/` tooling. The scripts needed
   fixing to make `--induction-examples` actually reachable (they passed
   `--examples-per-label` unconditionally) — implemented as planned, plus one bug found
   only by running the offline dry-run checks live (see Files Modified above).
2. **Arity guard.** `induction.py`'s internal arity guard (comparing the handed
   classifier's field set against the expected `description_i` set) is defensive code with
   no CLI-reachable path, required by INV-1 (which forbids `induction.py` importing
   `schema.py`, so it cannot rebuild the model to check it) — not a violation of FR-2.4/AR-2.1.
   Its test-double consequence (widening `FakeInductionClassifier` beyond FR-3.7's
   `.classify(text) -> dict` minimum) is implemented and documented in the fake's own
   docstring.

No other deviations. The plan's "expected red between Task 1 and Task 3" checkpoint story
held exactly as predicted; no compatibility shims were needed or written.
