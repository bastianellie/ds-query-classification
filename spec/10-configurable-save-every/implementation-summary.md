# Implementation Summary: 10-configurable-save-every

**Status:** Completed
**Date:** 2026-10-26
**Worktree:** `/Users/bastianellie/work/elsevier/projects/general/codebase/ds-query-classification-worktrees/10-configurable-save-every` on branch `spec/10-configurable-save-every`

## Overview
Added `--save-every N` to the CLI (default 20, must be >= 1) and validation of `save_every` in `classify_csv`. In `--batch` mode the interim save now happens at most once per consumed batch, when the completed count crossed a multiple of `save_every` during that batch. Plain, `--critics`, and `--models` cadence and every final write are unchanged.

## Team Execution
Solo, sequential (Tasks 1 -> 2 -> 3), as `plan.md` prescribed.

## Files Created
- `tests/test_save_every.py` — 24 tests (library validation, plain and batch cadence, CLI flag).
- `spec/10-configurable-save-every/ADR.md` — ADR-001 (INV-9 amendment), ADR-002 (spec 5 FR-3.1 flush-count equivalence).

## Files Modified
- `src/query_classification/pipeline.py` — `validate_save_every` helper, called first in `classify_csv`; the batch branch captures `completed` before each consumed future and saves once if `completed // save_every` increased.
- `src/query_classification/cli.py` — `--save-every` flag, `validate_save_every` call right after the other data-free checks, pass-through to `classify_csv`.
- `README.md` — Options row and a "Saving progress on large inputs" section.
- `spec/ARCHITECTURE.md` — INV-9 (title, bullet, amendment note), Stable Contracts CLI surface, Known Gaps bullet.

## Test Results
Baseline on clean `HEAD` in this worktree: 524 passed, 0 failed. After: **548 passed, 0 failed** (+24 new). Verified that the 8 new tests that depend on the new behavior (library validation x5, batch once-per-batch, multi-multiple crossing, batch-boundary saves) fail against the original `pipeline.py` and pass with the change; the rest pin behavior that must not change and pass on both. `classify.py --help` lists `--save-every`.

## Spec Adherence
| Requirement | Status | Implementation | Test |
|-------------|--------|---------------|------|
| FR-1.1 | Done | `cli.build_parser` | `test_parser_default_is_20_and_value_is_parsed`, `test_non_integer_save_every_is_an_argparse_error` |
| FR-1.2 | Done | `cli.main` (`validate_save_every` after `validate_options`) | `test_cli_rejects_save_every_below_one_before_any_classifier_is_built` |
| FR-1.3 | Done | `cli.main` passes `save_every=args.save_every` | `test_cli_passes_save_every_through_to_classify_csv` |
| FR-1.4 | Done | `pipeline.validate_save_every`, first statement of `classify_csv` | `test_classify_csv_rejects_invalid_save_every_before_any_work` |
| FR-2.1 | Done | `pipeline.classify_csv` batch branch | `test_batch_writes_once_per_batch_not_once_per_row`, `test_batch_with_large_save_every_writes_only_the_final_file`, `test_batch_partial_last_batch_only_saves_if_it_crosses_a_multiple`, `test_batch_crossing_several_multiples_writes_once`, `test_batch_size_not_dividing_save_every_saves_at_batch_boundaries`, `test_batch_future_level_failure_still_counts_rows_and_can_save` |
| FR-2.2 | Done | non-batch loop untouched | `test_plain_run_saves_every_n_rows_then_final` |
| FR-2.3 | Done | final writes and `_export_xlsx` untouched | `test_save_every_larger_than_row_count_writes_only_once`, `test_cli_xlsx_export_happens_once_after_the_final_csv_write` |
| FR-2.4 | Done | work-set order from spec 9 unchanged | `test_limit_smaller_than_save_every_writes_only_once`, `test_restore_run_restarts_the_cadence_over_remaining_rows`, `test_blank_and_restored_rows_do_not_advance_the_cadence` |
| FR-3.1 | Done | README | verified by inspection |
| FR-3.2 | Done | `ADR.md`, `spec/ARCHITECTURE.md` | verified by inspection (no remaining "no CLI flag exposes it" / `ZeroDivisionError` claim) |
| FR-3.3 | Done | `tests/test_save_every.py` | suite passes; existing flush-count test passes unmodified |
| AR-1.1 | Done | only `cli.py`/`pipeline.py` edited; no new imports beyond `validate_save_every` from `pipeline` | `git diff` |
| AR-1.2 | Done | no change to the executor structure or the `df.to_csv` mechanism | existing INV-6 behavior, full suite |

## Deviations from Spec
None. (Plan note: the "blank and already-restored rows" edge case is tested with blank rows; already-restored rows are covered by the restore test, since the cadence restarts over remaining rows only.)
