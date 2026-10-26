# Implementation Plan: 9-excel-input-output

**Status:** Ready
**Date:** 2026-10-26
**Spec:** spec/9-excel-input-output/spec.md
**Plan Critique:** spec/9-excel-input-output/plan-critique-consolidated-v-1.md

## Overview
Add a new leaf module `table_io.py` that owns reading CSV/Excel tables (sheet resolution, column selection) and the sanitized atomic `.xlsx` export. Wire it into `pipeline.classify_csv` (new keyword params, blank-row skipping, end-of-run export) and `cli.py` (three flags plus early validations), then update README, `spec/ARCHITECTURE.md`, and write the ADR.

## Readiness
- Checklist: all ten items closed in the spec (`[x]` with evidence); two critique rounds (Claude, Codex) consolidated with no blocking items.
- Open questions: none.
- External API/library verification: verified live in this venv's stack (pandas 3.0.5, openpyxl 3.1.5): `pd.ExcelFile(path, engine="openpyxl")` context manager with `.sheet_names` and `.parse(sheet, dtype=object, usecols=callable)`; `dtype=object` keeps `"007"`; interior empty rows kept, trailing dropped; duplicate headers become `a`, `a.1`; `to_excel` stores `=1+1` as a formula cell and raises `IllegalCharacterError` on `\x0b`; cells over 32,767 chars are truncated with a warning. `openpyxl` is **not yet installed in the repo's `.venv`** (only in a scratch dir), so Task 1 installs it.
- Repository state: spec folder only has spec/critique files; no source changes yet.

## Code Impact
- Modules/services: new `src/query_classification/table_io.py`; edits to `pipeline.py` (`classify_csv`) and `cli.py` (`build_parser`, `main`).
- Database/schema: none.
- API/interfaces: `classify_csv` gains keyword params `sheet`, `keep_columns`, `output_format` (defaults reproduce current behavior); CLI gains `--sheet`, `--keep-columns`, `--output-format`.
- UI/config/background jobs: `requirements.txt` + `pyproject.toml` (`openpyxl`).
- Key files: `src/query_classification/{table_io,pipeline,cli}.py`, `tests/test_excel_io.py` (new), `README.md`, `spec/ARCHITECTURE.md`, `spec/9-excel-input-output/ADR.md` (new).
- Must NOT touch: `experiment.py`, `dataset_io.py`, `classifier.py`, `batching.py`, `debate.py`, `multi_model.py`, `cost.py`.

## Project Constraints
- INV-1: `table_io.py` must be a leaf (stdlib + pandas + lazy openpyxl, no `query_classification` imports); `pipeline.py` and `cli.py` may import it; this edge requires the ADR in Task 4. `pipeline.py` must not import `openpyxl` directly.
- INV-4: any broad `except Exception` carries `# noqa: BLE001` plus a one-line reason (the export wrapper in `table_io` likely needs one).
- INV-6: DataFrame mutation stays on the main thread; do not touch the worker-thread structure.
- INV-7: tests import the public API by default; `table_io` (all functions) and `cli.main`/`cli.build_parser` need the ADR's INV-7 exception (cli ones already granted by spec 3).
- INV-8/INV-10: restore stays positional with the row-count check; category-vs-`--column` collision check stays first.
- AGENTS.md: `from __future__ import annotations` and PEP 604 unions in new modules; no live LLM calls in tests (fake `litellm.completion` as `tests/test_cost.py`/`tests/test_cerebus.py` do, reusing `_classifier`/`_Fake*` from `tests/test_cerebus.py`); `pytest` must pass before done.
- Dependency lists are duplicated by hand in `requirements.txt` and `pyproject.toml`; change both.
- Verification commands: `.venv/bin/python -m pytest`, `.venv/bin/python classify.py --help`.

## Implementation Strategy
- Size: medium (4 tasks).
- Execution mode: solo, sequential. Tasks 2 and 3 both depend on Task 1's API, and Task 3 depends on Task 2's `classify_csv` signature and exception type, so there is no safe parallel stream.
- Parallelizable work: Task 4 (docs/ADR) touches only docs and could overlap Task 3, but the final flag text depends on Task 3's help strings, so it runs last.
- Sequential blockers: Task 1 -> Task 2 -> Task 3 -> Task 4.

## Implementation Tasks

### Task 1: `table_io.py` leaf module and dependency
**Goal:** All table reading and `.xlsx` writing logic exists, tested in isolation.
**Files:** `src/query_classification/table_io.py` (new), `requirements.txt`, `pyproject.toml`, `tests/test_excel_io.py` (new, part 1).
**Dependencies:** None
**Do:**
- Add `openpyxl` to `requirements.txt` and `pyproject.toml` and install it into `.venv` (`.venv/bin/python -m pip install openpyxl`).
- Implement: `is_excel_path(path)` (`.xlsx`/`.xlsm` case-insensitive; `.xls` raises `ValueError`); `read_table(path, sheet, keep_columns, text_column)` returning a DataFrame (CSV: `pd.read_csv` as today, then column selection; Excel: one `pd.ExcelFile` context, name-then-digits sheet resolution with a `ValueError` listing sheet names, `parse(dtype=object, usecols=callable)` when keep_columns given, unknown-name `ValueError` listing unknown and available columns, comma-split/strip/dedupe parsing, text column always kept, input column order preserved); a lazy `openpyxl` import helper raising `RuntimeError("... pip install openpyxl")`; `blank_text_mask(series)` (null/empty/whitespace after normal read); `EXCEL_MAX_ROWS = 1_048_576` and the over-limit check helper.
- Implement `write_xlsx(df, path)`: copy values, strip control characters other than tab/newline/CR, truncate cells to 32,767 chars, count affected cells and print `Excel export: sanitized N cells` when N > 0, write every string as a literal string cell (post-write `data_type = 's'` or equivalent) so `=1+1` never becomes a formula, sheet `Classified`, no index, temp file in the same directory then `os.replace`, remove the temp on failure, raise `ExcelExportError(RuntimeError)` wrapping any failure (message names the CSV-kept path via a parameter).
- Wrap workbook open/parse failures (corrupt, password-protected) in `ValueError` naming the file path (spec Constraints), so `cli.main`'s existing `ValueError` handler prints it.
- Tests (same task): FR-1.1, 1.2 (`"007"`), 1.3 (name/index/missing/digit-named sheet), unreadable-workbook `ValueError` with path, 1.9 (CSV read never imports `openpyxl`, checked via `sys.modules`; Excel read with the import patched to fail raises the prescribed `RuntimeError`; dependency files list `openpyxl`), 1.10 (interior/trailing empty rows, duplicate header), 2.1, 2.2, AR-1.1 (no internal imports), AR-1.2 (callable `usecols` captured), FR-3.2 (atomic, no stray temp), 3.4, 3.6.
**Verify:** `.venv/bin/python -m pytest tests/test_excel_io.py` passes; `.venv/bin/python -m pytest` still passes.
**Covers:** FR-1.1, FR-1.2, FR-1.3, FR-1.9, FR-1.10, FR-2.1, FR-2.2, FR-3.4, FR-3.6, AR-1.1, AR-1.2 (and the mechanics of FR-3.2, FR-3.3)
**Notes (literal-string cells):** apply the literal-string guarantee to every string column in the exported frame, including category and audit columns, not only the query column. Spike the mechanism on a few hundred thousand rows before settling on it (a write-only openpyxl workbook may be needed for memory); the tests pin only the observable result.
**Notes:** Spec has no mechanism for literal-string cells; the tests pin the observable behavior (`data_type 's'` after reload) so any mechanism that passes is acceptable.

### Task 2: `classify_csv` integration
**Goal:** `classify_csv` accepts the new parameters and applies read, keep-columns, blank-row skipping, and the optional export.
**Files:** `src/query_classification/pipeline.py`, `tests/test_excel_io.py` (part 2).
**Dependencies:** Task 1
**Do:**
- Append keyword params `sheet: str | None = None`, `keep_columns: list[str] | None = None`, `output_format: str = "csv"` to `classify_csv`. Ordering: the existing category-name-vs-`--column` collision guard (INV-8) stays the first validation; immediately after it, and before any table I/O, add the new data-free checks. `sheet` with a non-Excel input raises `ValueError` (FR-1.4); Excel input with `output_path` omitted raises `ValueError` (FR-1.5, before `output_path` defaults to `input_path`); `output_path` ending `.xlsx`/`.xlsm`/`.xls` raises `ValueError` (FR-3.1); `output_format` outside `{csv, xlsx}` raises `ValueError`.
- Replace the input `pd.read_csv(input_path)` with `table_io.read_table(...)`; keep the existing missing-column error; add the FR-2.3 collision check (kept column equals a category or audit column, only when `keep_columns` is given); add the FR-3.3 row-limit check when `output_format == "xlsx"`; leave the restore-block `pd.read_csv(output_path)` as is.
- Compute the blank-text mask once after reading; remove blank rows from `work_idx` in both the restore and non-restore branches **before** `work_idx[:limit]`; print `Skipped N rows with empty text` when N > 0; header-only input writes the header-only CSV and makes zero calls (FR-1.7).
- Export on **both** exits of `classify_csv`: the `--batch` branch writes its CSV and returns early (`return output_path` inside `if have_batch:`), and the common path returns at the end. Factor a small local helper (or a single shared tail) so that after each final `df.to_csv(output_path, index=False)`, when `output_format == "xlsx"`, `table_io.write_xlsx(df, output_path.with_suffix(".xlsx"))` runs. The incremental `df.to_csv` calls stay untouched.
- Tests: FR-1.4, 1.5, 1.6 (both tests), 1.7, 1.8 (`--limit` then `--restore`, wrong-length prior), 2.3, 2.4 (`save_every=1`, sentinel in no produced file), 3.1, 3.2 (no `.xlsx` until return; **also a `batch_runner` run asserting the `.xlsx` exists**), 3.3 (patched limit constant), AR-1.3 (old-arguments call unchanged), CSV/Excel equivalence.
- Immediately after the blank-row change, run the full suite first and treat any failure as FR-1.6 fallout to be understood, not masked: `tests/test_batching.py` and `tests/test_multi_model.py` call `classify_csv` widely and were not inspected for NaN-text fixtures.
**Verify:** `.venv/bin/python -m pytest` (full suite, including `tests/test_experiment.py`, which exercises `experiment.py`'s `classify_csv` call), then `.venv/bin/python -m pytest tests/test_excel_io.py`.
**Covers:** FR-1.4, FR-1.5, FR-1.6, FR-1.7, FR-1.8, FR-2.3, FR-2.4, FR-3.1, FR-3.2, FR-3.3, AR-1.3
**Notes:** Blank-row exclusion changes existing CSV behavior by design (spec FR-1.6); confirm no existing test feeds NaN text rows before relying on the suite as the regression check (grep tests for empty-text fixtures).

### Task 3: CLI flags and early validation
**Goal:** `--sheet`, `--keep-columns`, `--output-format` work end to end through `cli.main`.
**Files:** `src/query_classification/cli.py`, `tests/test_excel_io.py` (part 3).
**Dependencies:** Task 2
**Do:**
- Add the three flags to `build_parser` (update the `--input`/`--output` help text to mention `.xlsx` and that `--output` is required for Excel input and names a CSV); split `--keep-columns` on commas in `main` and pass `sheet`, `keep_columns`, `output_format` to `classify_csv`.
- In `main`'s `try`, immediately after the bundled-resources check and before any classifier construction or the overwrite warning, run the data-free checks (FR-1.4, FR-1.5, FR-3.1) by calling shared `table_io` helpers, so failures exit 1 with `Error: ...` before any client is built.
- Export failure path (FR-3.2), a sequencing note rather than a spec deviation: `classify_csv` owns the export while `cli.main` writes the cost report afterwards, so wrap the `classify_csv` call to catch `ExcelExportError`, still assemble and write the cost report, then re-raise so the existing `except (ValueError, FileNotFoundError, RuntimeError)` prints `Error: ...` and exits 1. Cost report path stays `csv_path.with_suffix(".cost_report.json")` from `--output` (FR-3.5). The test asserts the report is written exactly once and the printed error is the export message.
- Tests via `cli.main` with `monkeypatch.setattr(sys, "argv", ...)` and a fake `litellm.completion`, following `tests/test_cerebus.py`'s `cli_main` tests: FR-1.4 and FR-1.5 (classifier constructor patched to fail if called; workbook byte-identical), FR-3.1, FR-3.5, forced export failure (CSV kept, temp removed, exit 1, cost report written).
**Verify:** `.venv/bin/python -m pytest` (all) and `.venv/bin/python classify.py --help` lists the three flags.
**Covers:** FR-1.4, FR-1.5, FR-3.1, FR-3.5, FR-3.2 (failure path), AR-1.3
**Notes:** The spec says data-dependent checks run before any provider request but after client construction; do not move table reading ahead of client construction.

### Task 4: Documentation, architecture, ADR
**Goal:** Docs, `spec/ARCHITECTURE.md`, and the ADR match the shipped behavior.
**Files:** `README.md`, `spec/ARCHITECTURE.md`, `spec/9-excel-input-output/ADR.md` (new).
**Dependencies:** Task 3
**Do:**
- README: Options table rows for `--sheet`, `--keep-columns`, `--output-format`; a short Excel section covering the `--output`-required rule, blank-row skip, formula-safe `.xlsx` (CSV verbatim), `--restore` re-parsing cost and the positional-restore caveat for changed sheet/keep list, `.xlsm` discards macros, and the resource expectation for large workbooks.
- `spec/ARCHITECTURE.md`: add `table_io.py` to the mermaid graph and Module Boundary Map, INV-1 and INV-7 amendment notes, the Stable Contracts CLI surface, and the shared dependency files note (`openpyxl`).
- ADR with two decisions: the `table_io.py` leaf and its `pipeline.py`/`cli.py` edges (INV-1), and the INV-7 test-import exception for `table_io`.
**Verify:** README Options table lists the three flags; `grep -n table_io spec/ARCHITECTURE.md` hits the boundary map; ADR exists and names both decisions; full `pytest` still passes.
**Covers:** FR-4.1 (suite complete), FR-4.2
**Notes:** Do not edit `.egg-info/`, `build/`, `output/`.

## Final Verification
- `.venv/bin/python -m pytest` passes, including all of `tests/test_excel_io.py`.
- `.venv/bin/python classify.py --help` shows `--sheet`, `--keep-columns`, `--output-format`.
- Manual sanity (no LLM): run `table_io.read_table` with `--keep-columns` on the real `01_row_classified.xlsx` log (read-only; do not copy it into the repo) and confirm dropped columns are absent and the row count is 629,058; timing the full read is informational only, not an acceptance criterion.
- Review the `spec/ARCHITECTURE.md` edits against the actual imports and the ADR so the graph, boundary map, INV-1/INV-7 notes, and CLI contract agree.
- Coverage check: every FR/AR in the spec appears in at least one task's `Covers:`.

## Documentation
- README.md, spec/ARCHITECTURE.md, spec/9-excel-input-output/ADR.md (Task 4). No `spec/docs/` exists yet.

## Spec Deviations
None identified. (The `ExcelExportError` catch-write-rethrow in Task 3 is an implementation mechanism for the spec's FR-3.2 observable behavior, recorded as a sequencing note, not a deviation.)

## Risks
- Risks: blank-row skipping changes CSV behavior and could alter an existing test's expectations; the literal-string cell mechanism is left to the implementer (tests pin behavior); `.xlsx` write time/memory for 629k rows is unmeasured.
- Rollback/checkpoint guidance: each task ends with a green full `pytest`; commit per task so a bad task can be reverted without losing earlier ones. All new parameters default to current behavior, so reverting Tasks 2-3 restores the old CLI.
