# Implementation Plan: 10-configurable-save-every

**Status:** Ready
**Date:** 2026-10-26
**Spec:** spec/10-configurable-save-every/spec.md
**Plan Critique:** spec/10-configurable-save-every/plan-critique-consolidated-v-1.md

## Overview
Add a validated `save_every` check and the "save at most once per consumed batch future" rule to `pipeline.classify_csv`, expose `--save-every` in `cli.py` (flag, early validation, pass-through), and update README, `spec/ARCHITECTURE.md`, and an ADR. Two small source edits plus tests and docs.

## Readiness
- Checklist: all ten items closed in the spec; one critique round (Claude, Codex) consolidated with no blocking items.
- Open questions: none.
- External API/library verification: no new libraries. Local facts confirmed in the code: `classify_csv` already has `save_every: int = 20` (`pipeline.py`), the non-batch loop saves with `if completed % save_every == 0` after each row, and the batch branch does the same check per row inside the per-outcome loop before its final `df.to_csv`; `cli.main` calls `table_io.validate_options(...)` and `table_io.parse_keep_columns(...)` right after the bundled-resources check, before any classifier is built; the existing `test_classify_csv_batched_flush_positions_match_unbatched_run` asserts only equal write counts (200 rows, fixed batch 7).
- Repository state: `main` at `a7dd1d7`, with the user's uncommitted edits to `classifier.py`, `tests/test_batching.py`, two notebooks and untracked data files. The worktree will be cut from `HEAD` and will not contain them; this plan touches none of those files.

## Code Impact
- Modules/services: `src/query_classification/pipeline.py` (`classify_csv`, one new module-level helper), `src/query_classification/cli.py` (`build_parser`, `main`).
- Database/schema: none.
- API/interfaces: new CLI flag `--save-every`; `classify_csv`'s existing `save_every` parameter now validated (`ValueError` for `< 1` or non-int/bool).
- UI/config/background jobs: none.
- Key files: `pipeline.py`, `cli.py`, `tests/test_save_every.py` (new), `README.md`, `spec/ARCHITECTURE.md`, `spec/10-configurable-save-every/ADR.md` (new).
- Must NOT touch: `experiment.py`, `table_io.py`, `batching.py`, `classifier.py` (which has unrelated uncommitted user edits in the main checkout), existing tests.

## Project Constraints
- INV-1: `cli.py` already imports `pipeline.py`; a new helper in `pipeline.py` that `cli.py` calls adds no import edge.
- INV-6: DataFrame mutation stays on the main thread; do not alter the executor structure.
- INV-7: tests import the public API (`classify_csv`, `Category`, `Label`) plus `cli.build_parser`/`cli.main` (already allowed). Do not import `pipeline` directly in tests; `_make_batch_runner` from `tests/test_batching.py` is reused as `tests/test_excel_io.py` does.
- INV-9: amended by the ADR in Task 3 (title and the `save_every` bullets).
- AGENTS.md: no live LLM calls in tests; `pytest` must pass; new module code not needed (`from __future__ import annotations` already present in edited modules); broad `except Exception` needs `# noqa: BLE001` + comment (none planned).
- Verification commands: `.venv/bin/python -m pytest`; `.venv/bin/python classify.py --help`.

## Implementation Strategy
- Size: small (3 tasks).
- Execution mode: solo, sequential. Task 2 calls Task 1's helper; Task 3 documents the final behavior.
- Parallelizable work: none worth the coordination cost.
- Sequential blockers: Task 1 -> Task 2 -> Task 3.

## Implementation Tasks

### Task 1: `pipeline.py` validation and batch save rule
**Goal:** `classify_csv` validates `save_every` and the batch branch saves at most once per consumed future.
**Files:** `src/query_classification/pipeline.py`, `tests/test_save_every.py` (new, part 1).
**Dependencies:** None
**Do:**
- Add a module-level `validate_save_every(value)` raising `ValueError(f"--save-every must be >= 1, got {value}")` for anything that is not an `int` (explicitly excluding `bool`) or is `< 1`. Call it as the first statement of `classify_csv`, before `Path` handling, any read, or any write.
- In the `--batch` branch, capture `completed_before = completed` when a future is consumed (before the per-outcome loop), drop the per-row `if completed % save_every == 0: df.to_csv(...)` from that loop, and after the loop write once if `completed // save_every > completed_before // save_every`. Leave the plain/`--critics`/`--models` loop's per-row check and every final write untouched.
- Tests: FR-1.4 (`0`, `-3`, `2.5`, `True` raise `ValueError`, zero classifier calls, no output file); FR-2.1 (`--batch` fixed 10, `save_every=5`, 40 rows: 4 interim writes to the output path; `save_every=100`: zero interim; 45 rows with a partial last batch; a patched `BatchRunner.run` that raises still counts rows and can trigger the save; a batch crossing several multiples writes once; actual completed-row count recorded at each save for a batch size that does not divide `save_every`); FR-2.2 (plain 10 rows, `save_every=3`: writes at 3, 6, 9 + final); FR-2.3 (5 rows, `save_every=1000`: exactly one write to the output path); FR-2.4 (`limit=3`/`limit=0`: one write; restore run with 4 remaining rows and `save_every=2`: two interim writes; blank and already-restored rows ahead of the work rows do not advance the cadence; `--batch 25` with `save_every=10` over 50 rows: two interim writes). The new test file is created in this task and extended in Task 2. Spies wrap `pandas.DataFrame.to_csv`, normalize the first argument (positional `path_or_buf` or keyword) to a `Path` before comparing with the output path, always delegate to the captured original, ignore fixture writes, and record `self["c"].notna().sum()` (in-memory classified count, as the existing test does) at each output-path write. The "actual completed-row count" assertion uses that recorded count for interim writes only (the final write is excluded by taking all but the last). For the future-level-exception case, patch the runner instance's `run` to raise for one selected batch (not an LLM failure, which `BatchRunner.run` turns into per-row failures) and assert the failure accounting and that the save still fires.
**Verify:** `.venv/bin/python -m pytest tests/test_save_every.py tests/test_batching.py tests/test_excel_io.py`, then the full suite compared against the baseline failure list.
**Covers:** FR-1.4, FR-2.1, FR-2.2, FR-2.3, FR-2.4, FR-3.3 (tests for these), AR-1.1, AR-1.2
**Notes:** Batch futures are consumed in completion order (`as_completed`); the crossing is evaluated against the count of outcomes applied, not an input prefix. The existing test `test_classify_csv_batched_flush_positions_match_unbatched_run` must pass unmodified.

### Task 2: CLI flag
**Goal:** `--save-every` works end to end through `cli.main`.
**Files:** `src/query_classification/cli.py`, `tests/test_save_every.py` (part 2).
**Dependencies:** Task 1
**Do:**
- Add `--save-every` (`type=int`, `default=20`, `metavar="N"`, help text pointing at large inputs) to `build_parser`.
- In `main`, immediately after `table_io.validate_options(...)` / `parse_keep_columns`, call `validate_save_every(args.save_every)` (imported from `pipeline`); pass `save_every=args.save_every` to `classify_csv`.
- Tests via `cli.main` with a faked `Classifier.classify`, following `tests/test_excel_io.py`'s `cli_env` pattern: parser default 20 and value 5000 (FR-1.1), `--save-every abc` exits via argparse (code 2), `--save-every 0` exits 1 with the message, no `OVERWRITTEN` warning and no `Classifier.__init__` call (FR-1.2), `--save-every 3` over 7 rows gives interim writes at 3 and 6 plus the final write, and a recorder wrapped around `cli.classify_csv` asserts it received `save_every=3` directly (FR-1.3), and `--output-format xlsx --save-every 2` over 6 rows writes the `.xlsx` once, after the final CSV write (FR-2.3).
**Verify:** `.venv/bin/python -m pytest tests/test_save_every.py` and `.venv/bin/python classify.py --help | grep -A3 save-every`.
**Covers:** FR-1.1, FR-1.2, FR-1.3, FR-2.3 (CLI/xlsx case), FR-3.3 (tests for these), AR-1.1
**Notes:** `experiment.py` must keep its own default and is not edited. `cli.main` already runs the bundled-resources availability guard (`check_default_resources_available`, a file-exists check) before `table_io.validate_options`; the save-every check stays immediately after `validate_options` as FR-1.2 specifies, so the guard precedes it on a broken install by design (it is not categories/prompt loading). Test with the bundled resources present.

### Task 3: Documentation, ADR, architecture
**Goal:** README, `spec/ARCHITECTURE.md`, and the ADR match the shipped behavior.
**Files:** `README.md`, `spec/ARCHITECTURE.md`, `spec/10-configurable-save-every/ADR.md` (new).
**Dependencies:** Task 2
**Do:**
- README: add the `--save-every` row to the Options table and a short note (default 20 for small inputs; about 5000 for hundreds of thousands of rows; crash loses the rows completed since the last save, up to N or N plus one batch in `--batch` mode; `--restore` resumes from the last saved file unless that write itself was truncated, since writes are not atomic; cloud-synced output folders make each save slower, so write locally and copy afterwards).
- `spec/ARCHITECTURE.md`: amend INV-9 (title and the `save_every` bullet: CLI flag exists, `< 1` raises `ValueError`), the Stable Contracts CLI surface, and the Known Gaps "No atomicity" bullet only where it assumes an unconfigurable `save_every`.
- ADR-001: INV-9 amendment; ADR-002: spec 5 FR-3.1's flush-count equivalence now holds only for batch sizes below `save_every`.
**Verify:** `grep -n "save-every" README.md`; read the INV-9 section of `spec/ARCHITECTURE.md` and confirm the title and bullets no longer say that `save_every` has no CLI flag or raises `ZeroDivisionError`; read the ADR and confirm ADR-001 states the INV-9 amendment and ADR-002 states spec 5 FR-3.1's equivalence holds only for batch sizes below `save_every`; run `grep -n "^from query_classification\|^import query_classification" tests/test_save_every.py` and confirm no `pipeline` import (INV-7); full `pytest` still passes.
**Covers:** FR-3.1, FR-3.2 (and a final FR-3.3 completeness check that `tests/test_save_every.py` has a test for every Verify clause in FR-1.1 through FR-2.4)
**Notes:** Review the ARCHITECTURE edits against the code after Task 2.

## Final Verification
- Before Task 1, in the worktree, record the baseline: `.venv/bin/python -m pytest -q 2>&1 | grep ^FAILED | sed 's/ - .*//' | sort > <scratchpad>/baseline_s10.txt`. After each task, repeat into another file and `diff` against it; the failing set must be identical (the known baseline is 34 failures from the missing Cerebus API key) and all new tests must pass.
- `.venv/bin/python classify.py --help` lists `--save-every`.
- Coverage check: every FR/AR appears in at least one task's `Covers:` line.

## Documentation
- README.md, spec/ARCHITECTURE.md, spec/10-configurable-save-every/ADR.md (Task 3). No `spec/docs/` exists.

## Spec Deviations
None identified.

## Risks
- Risks: batch saves land at batch boundaries, so intermediate files hold slightly more rows than the multiple of `save_every` (documented); the pre-existing batched-vs-unbatched write-count test must stay green; a large `save_every` raises the crash-loss bound.
- Rollback/checkpoint guidance: commit per task with a green suite. The default of 20 reproduces current behavior, so reverting Task 1's batch-branch edit alone restores the old per-row batch cadence.
