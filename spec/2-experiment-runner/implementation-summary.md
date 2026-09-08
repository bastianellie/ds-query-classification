# Implementation Summary: 2-experiment-runner

**Status:** Completed
**Date:** 2026-09-08
**Worktree:** `/Users/bastianellie/work/elsevier/projects/general/codebase/ds-query-classification-worktrees/2-experiment-runner` on branch `spec/2-experiment-runner`

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
