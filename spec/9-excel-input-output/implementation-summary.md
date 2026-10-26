# Implementation Summary: 9-excel-input-output

**Status:** Completed
**Date:** 2026-10-26
**Worktree:** `/Users/bastianellie/work/elsevier/projects/general/codebase/ds-query-classification-worktrees/9-excel-input-output` on branch `spec/9-excel-input-output`

## Overview
Added Excel (`.xlsx`/`.xlsm`) input, `--sheet`, `--keep-columns`, blank-text row skipping, and an optional end-of-run `.xlsx` export (`--output-format xlsx`) to the classifier. CSV stays the default output and the resume format. All table I/O lives in a new leaf module, `table_io.py`.

## Team Execution
Solo, sequential (Tasks 1 -> 2 -> 3 -> 4), as `plan.md` prescribed: each task depended on the previous one's API.

## Files Created
- `src/query_classification/table_io.py` — read (CSV/Excel, sheet resolution, column selection), option validation, blank-text mask, row-limit check, sanitized atomic `.xlsx` export, `ExcelExportError`.
- `tests/test_excel_io.py` — 52 tests (Tasks 1-3).
- `spec/9-excel-input-output/ADR.md` — ADR-001 (`table_io` leaf, INV-1), ADR-002 (test imports, INV-7).

## Files Modified
- `src/query_classification/pipeline.py` — `classify_csv` gains `sheet`, `keep_columns`, `output_format`; reads via `table_io`; blank-row skipping before `limit`; `.xlsx` export on both return paths (normal and `--batch`).
- `src/query_classification/cli.py` — `--sheet`, `--keep-columns`, `--output-format`; data-free checks before any classifier is built; export failure is held until the cost report is written, then re-raised (exit 1).
- `requirements.txt`, `pyproject.toml` — `openpyxl`.
- `README.md`, `spec/ARCHITECTURE.md` — flags, Excel section, `table_io` in the graph/boundary map/INV-1/INV-7/CLI contract.

## Test Results
Baseline on clean `HEAD`: 34 failed, 438 passed. All 34 are in `tests/test_experiment.py` (32) and `tests/test_cerebus.py` (2) and fail for lack of a Cerebus API key (expired AWS SSO token), unrelated to this work.
After implementation: **34 failed, 490 passed**, and the failing set is identical to the baseline (compared by test id). `tests/test_excel_io.py`: 52 passed. One `PytestUnraisableExceptionWarning` comes from `test_cli_export_failure_...`, which deliberately breaks `openpyxl.Workbook.save`.

## Spec Adherence
| Requirement | Status | Implementation | Test |
|-------------|--------|---------------|------|
| FR-1.1 | Done | `table_io.is_excel_path`, `read_table` | `test_csv_and_xlsx_read_to_equal_tables`, `test_xls_is_rejected_with_supported_formats`, `test_csv_and_xlsx_input_classify_identically` |
| FR-1.2 | Done | `read_table` (`dtype=object`) | `test_excel_string_cells_are_not_reinterpreted` |
| FR-1.3 | Done | `table_io._resolve_sheet` | `test_sheet_selection_by_name_index_or_default`, `test_unknown_sheet_lists_available_names`, `test_sheet_name_wins_over_index` |
| FR-1.4 | Done | `table_io.validate_options`; called in `cli.main` and `classify_csv` | `test_sheet_with_csv_input_raises_before_any_classification`, `test_cli_data_free_errors_exit_1_before_any_classifier_is_built` |
| FR-1.5 | Done | same | `test_excel_input_without_output_never_overwrites_the_workbook`, CLI test above |
| FR-1.6 | Done | `pipeline.classify_csv` (blank mask before `limit`) | `test_blank_text_rows_are_skipped_kept_in_order_and_not_retried`, `test_restore_limit_counts_only_remaining_classifiable_rows` |
| FR-1.7 | Done | existing missing-column error; empty `work_idx` | `test_xlsx_missing_text_column_and_header_only_sheet` |
| FR-1.8 | Done | unchanged restore block | `test_xlsx_with_interior_and_trailing_empty_rows_restores_by_position`, `test_restore_with_wrong_length_prior_output_still_raises` |
| FR-1.9 | Done | `_require_openpyxl`; both dependency files | `test_csv_read_never_needs_openpyxl_and_excel_read_reports_it`, `test_importing_table_io_does_not_import_openpyxl`, `test_openpyxl_declared_in_both_dependency_files` |
| FR-1.10 | Done | pandas row domain; `read_table` | `test_empty_rows_and_duplicate_headers`, restore test above |
| FR-2.1 | Done | `parse_keep_columns`, `read_table` | `test_parse_keep_columns_strips_dedupes_and_drops_empties`, `test_keep_columns_input_order_and_text_always_kept` |
| FR-2.2 | Done | `read_table` (incl. header read on the error path for Excel) | `test_unknown_keep_column_lists_unknown_and_available`, `test_unknown_keep_column_makes_zero_calls` |
| FR-2.3 | Done | `classify_csv` collision check | `test_keep_columns_limits_output_and_collision_is_rejected` |
| FR-2.4 | Done | columns dropped at read time | `test_dropped_columns_never_reach_any_written_file` |
| FR-3.1 | Done | `validate_options` | `test_output_path_with_excel_suffix_is_rejected`, `test_validate_options_data_free_checks`, `test_invalid_output_format_is_rejected` |
| FR-3.2 | Done | `pipeline._export_xlsx`, `table_io.write_xlsx` | `test_xlsx_export_happens_once_after_the_final_csv`, `test_xlsx_export_also_happens_on_the_batch_path`, `test_write_xlsx_is_atomic_and_leaves_no_temp_file`, `test_write_xlsx_failure_raises_cleans_temp_and_keeps_old_file`, `test_cli_export_failure_keeps_csv_writes_cost_report_exits_1` |
| FR-3.3 | Done | `table_io.check_row_limit` | `test_row_limit_fails_fast_with_zero_calls`, `test_check_row_limit_boundary` |
| FR-3.4 | Done | `write_xlsx` (`data_type = "s"`) | `test_write_xlsx_roundtrip_literal_text_and_lists`, `test_xlsx_export_happens_once_after_the_final_csv` |
| FR-3.5 | Done | cost-report path untouched | `test_cli_excel_in_csv_and_xlsx_out_with_keep_columns` |
| FR-3.6 | Done | `_clean_cell` | `test_write_xlsx_sanitizes_illegal_chars_and_truncates` |
| FR-4.1 | Done | `tests/test_excel_io.py` | suite passes |
| FR-4.2 | Done | README, `spec/ARCHITECTURE.md`, `ADR.md` | verified by inspection |
| AR-1.1 | Done | `table_io.py` | `test_table_io_is_a_leaf_module` |
| AR-1.2 | Done | `read_table` (callable `usecols`, one `ExcelFile`) | `test_excel_keep_columns_passes_callable_usecols` |
| AR-1.3 | Done | keyword params with defaults | `test_old_style_call_is_unchanged`, full existing suite |

## Deviations from Spec
None. (Plan-level note, not a deviation: the export failure is signalled with `ExcelExportError`, caught in `cli.main` so the cost report is written first.)

## Real-workbook check
Read-only run of `table_io.read_table` on the 145 MB, 629,058-row August log with `--keep-columns CONVERSATION_ID,TIMESTAMP` and text column `GENAI_INPUT`: 63 s, 629,058 rows, columns `TIMESTAMP, CONVERSATION_ID, GENAI_INPUT` (input order), 60 blank query rows. No LLM calls; the workbook was not modified or copied into the repo.

## Notes
- Beyond the spec's wording, `write_xlsx` renders list-valued cells (the category columns, which are Python lists in memory) with `str(...)` so the `.xlsx` shows the same text as the CSV; openpyxl cannot store lists.
- On the error path only, a missing `--keep-columns` name in an Excel file triggers a lightweight header read so the message can list the sheet's real columns.
