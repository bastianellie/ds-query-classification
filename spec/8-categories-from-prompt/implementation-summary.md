# Implementation Summary: 8-categories-from-prompt

**Status:** Completed
**Date:** 2026-09-17
**Worktree:** `/Users/bastianellie/work/elsevier/projects/general/codebase/ds-query-classification-worktrees/8-categories-from-prompt` on branch `spec/8-categories-from-prompt`

## Overview

Implements a new, standalone entry point (`extract_categories.py` / `category_extraction.py`)
that extracts a single `Category` taxonomy — LLM-authored label values and descriptions — from a
free-text instruction file, via one `Classifier.classify()` invocation. No labeled training data
is involved, unlike `experiment.py`'s existing `induce` subcommand. Neither existing entry point
(`cli.py`, `experiment.py`) is modified or imports the new module.

## Team Execution

Solo, sequential — matching the plan's Implementation Strategy. All five tasks add tests to the
same brand-new `tests/test_category_extraction.py` file, so sequential execution avoided any
file-conflict risk within that shared file.

**Sequential phases:** Task 1 (`schema.py`'s new schema builder) → Task 2 (`resources.py`/
`prompts.py`'s new prompt) → Task 3 (`category_extraction.py`'s pure extraction function) → Task
4 (`category_extraction.py`'s CLI) → Task 5 (root script + docs/ADR).

## Files Created

- `src/query_classification/category_extraction.py` — the new module: `extract_category_from_prompt`
  (pure function, FR-1.1/1.3/1.4/1.5), `_validate_category_name` (FR-1.2), `build_parser()`/
  `main()` (the CLI, FR-1.6/1.7/1.8/AR-1.3/1.4/1.7).
- `resources/prompts/category_extraction_prompt.txt` — the new system prompt template.
- `extract_categories.py` (repo root) — sys.path-injection wrapper, mirroring `classify.py`/
  `experiment.py`.
- `tests/test_category_extraction.py` — 42 tests covering every FR/AR Verify clause.
- `spec/8-categories-from-prompt/ADR.md` — INV-1/INV-7 amendments for the new module.

## Files Modified

- `src/query_classification/schema.py` — `build_category_extraction_model()`: a
  `create_model`-built schema with `category_description: str` + `labels: list[_LabelStrict]`
  (a strict local copy of `Label`'s shape, mirroring `build_batch_model`'s own precedent — needed
  because plain `categories.Label` has no `extra="forbid"` of its own, so an outer
  `extra="forbid"` alone doesn't reject an extra field nested inside one label). Added `Label` to
  the existing `categories` import (previously imported only `Category`).
- `src/query_classification/prompts.py` — `build_category_extraction_prompt()`, added
  `DEFAULT_CATEGORY_EXTRACTION_PROMPT_FILE` to the existing `resources` import block.
- `src/query_classification/resources.py` — new `DEFAULT_CATEGORY_EXTRACTION_PROMPT_FILE`
  constant.
- `src/query_classification/cost.py` — one-line docstring update noting a third module
  (`category_extraction.py`) now reuses `write_json_atomic`.
- `spec/ARCHITECTURE.md` — `category_extraction.py` added to the mermaid graph, Module Boundary
  Map (as a third, narrower orchestrator — exactly six allowed imports, not "the same breadth"
  as `cli.py`/`experiment.py`), INV-1 (new module + import list), INV-7 (three new test-import
  exceptions), `cost.py`'s own row/evidence note.
- `README.md` — new "Extracting categories from a prompt" section, explicitly noting this is
  standalone today with `experiment.py` integration deliberately deferred.

## Test Results

`.venv/bin/python -m pytest -q`: **34 failed, 438 passed** (up from the pre-implementation
baseline of 34 failed, 396 passed). The failing-test *names* are byte-identical to the baseline
(diffed and confirmed) — all 34 are the pre-existing, environment-only
`botocore.exceptions.TokenRetrievalError`/`InvalidGrantException` issue from an expired local AWS
SSO session, documented identically in specs 5/6/7's own summaries. 42 new tests added in
`tests/test_category_extraction.py`. No existing test was removed or weakened.

## Spec Adherence

| Requirement | Status | Implementation | Test |
|---|---|---|---|
| FR-1.1 | Done | `category_extraction.py::extract_category_from_prompt` | `test_extract_category_round_trips_three_labels_verbatim` |
| FR-1.2 | Done | `category_extraction.py::_validate_category_name` | `test_cli_rejects_invalid_category_name_before_any_llm_call` |
| FR-1.3 | Done | `extract_category_from_prompt`'s zero-labels check | `test_extract_category_zero_labels_raises`, `test_cli_zero_labels_fails_and_writes_no_output` |
| FR-1.4 | Done | duplicate-value check + duplicated `_check_reserved_sentinel` | `test_extract_category_duplicate_label_values_raise`, `test_extract_category_reserved_sentinel_values_raise`, `test_check_reserved_sentinel_matches_experiment_module` |
| FR-1.5 | Done | blank-string checks in `extract_category_from_prompt` | `test_extract_category_blank_category_description_raises`, `_blank_label_value_raises`, `_blank_label_description_raises` |
| FR-1.6 | Done | `main()`'s `cost.write_json_atomic` + `load_categories` round-trip | `test_cli_successful_run_produces_valid_category_name` |
| FR-1.7 | Done | `main()`'s output-path preflight (symlink-lexical-first ordering) | `test_cli_output_already_exists_without_overwrite_fails_second_run`, `test_cli_symlinked_output_fails_before_any_llm_call`, `test_cli_output_is_existing_directory_fails_before_any_llm_call`, `test_cli_output_missing_parent_directory_fails_before_any_llm_call` |
| FR-1.8 | Done | `main()`'s `Path(...).read_text(encoding="utf-8")` + empty-check | `test_cli_empty_or_whitespace_prompt_file_fails_before_any_llm_call`, `test_extract_category_round_trips_three_labels_verbatim` (verifies exact text passed to `classify()`) |
| AR-1.1 | Done | `category_extraction.py` module shape | N/A (structural; verified via `grep` in Final Verification) |
| AR-1.2 | Done | `schema.build_category_extraction_model` | `test_build_category_extraction_model_*` (6 tests) |
| AR-1.3 | Done | `prompts.build_category_extraction_prompt` + `--extraction-prompt` + resource guard | `test_default_category_extraction_prompt_file_exists`, `test_build_category_extraction_prompt_*`, `test_cli_missing_bundled_resource_names_extraction_prompt_flag` |
| AR-1.4 | Done | `build_parser()`'s flag table | `test_build_parser_requires_prompt_file_and_category_name`, `test_cli_main_constructs_classifier_correctly` |
| AR-1.5 | Done | import list | verified via `grep -rn "import experiment"` (no matches) |
| AR-1.6 | Done | `spec/8-categories-from-prompt/ADR.md` | N/A (docs) |
| AR-1.7 | Done | `main()`'s `except (ValueError, FileNotFoundError, RuntimeError)` | `test_cli_handled_failure_prints_error_no_traceback_exit_1`, `test_cli_success_prints_confirmation_with_output_path_and_exits_zero` |
| FR-2.1 | Done | `tests/test_category_extraction.py` (42 tests) | full suite: `.venv/bin/python -m pytest -q` |

## Deviations from Spec

Carried forward from the plan's Spec Deviations table (both "Action required: None"):

1. FR-1.4 says label values are "passed through `experiment.py`'s existing
   `_check_reserved_sentinel`" — implemented as a duplicate of that function's exact logic
   inside `category_extraction.py`, not an import, since AR-1.5/INV-1 forbid importing
   `experiment.py`. A parity test (`test_check_reserved_sentinel_matches_experiment_module`)
   asserts the duplicate behaves identically to `experiment.py`'s own function across a
   parametrized set of inputs.
2. No FR/AR requires a README change — added a short "Extracting categories from a prompt"
   section anyway, matching every prior spec's own entry-point documentation precedent.

No other deviations.
