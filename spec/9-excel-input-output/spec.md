# Spec 9: Excel Input, Column Selection, and Optional Excel Output

## Overview

`classify.py` reads only CSV files (`pd.read_csv(input_path)`, `pipeline.py:179`), but the logs
this tool is used on arrive as `.xlsx` workbooks (e.g. a 145 MB, 629k-row, 44-column log with
account names, cookie IDs, and free-text feedback beside the query column). This spec lets the
CLI read an `.xlsx` input directly, choose its sheet, carry only chosen input columns into the
output, and optionally export the final result as `.xlsx`. The output stays CSV by default and
the CSV remains the incremental/resume format.

## Goals

- Read `.xlsx`/`.xlsm` input with the same CLI and the same row semantics as CSV input.
- Let users keep only the columns they need in the output (privacy, size), for any input type.
- Optionally export the final result as `.xlsx`, written once at the end, never incrementally.
- Leave every existing CSV-in/CSV-out invocation behaving exactly as today.

---

## Feature 1: Excel input and sheet selection

**Who & why:** Analysts receive monthly query logs as Excel workbooks. Today they must hand-convert
to CSV first, which is slow for a 145 MB workbook, easy to get wrong (IDs such as `007` silently
become `7`), and leaves a second unmanaged copy of customer data on disk. They need
`--input log.xlsx` to just work, picking the right sheet when a workbook has several.

### Functional Requirements

#### FR-1.1: Input format is chosen by file extension
`classify_csv` loads the input via a single new helper (see AR-1.1). A path whose suffix,
case-insensitively, is `.xlsx` or `.xlsm` is read as an Excel workbook; `.xls` raises `ValueError`
naming the supported formats; every other suffix is read as CSV exactly as today
(`pd.read_csv(input_path)`), so extensionless and unusual CSV names keep working.
**Verify:** A test writes the same table to `t.csv` and `t.xlsx` and `classify_csv` (with a fake
classifier) yields equal category columns for both; a `.xls` path raises `ValueError`.

#### FR-1.2: Excel cells are read without type coercion
Excel input is read with `dtype=object` so that cell values keep their native Python type and
string cells are never reinterpreted (verified: with default inference, a string cell `"007"`
comes back as float `7.0`; with `dtype=object` it comes back as `"007"`). The text column is
still converted with `str(...)` at classification time, as for CSV.
**Verify:** A test round-trips a workbook containing the string `"007"` in a kept column and asserts
the output CSV contains `007`, not `7` or `7.0`.

#### FR-1.3: `--sheet` selects the worksheet
New flag `--sheet NAME_OR_INDEX` (library parameter `sheet: str | None = None`). Resolution order:
(1) exact match against the workbook's sheet names; (2) otherwise, if the value is all digits, a
zero-based sheet index; (3) otherwise `ValueError` listing the available sheet names. When omitted,
the first sheet is used. The workbook is opened once, and sheet names are read from that same open
handle (no second parse). A sheet named `"2024"` therefore wins over index `2024`.
**Verify:** Tests on a 3-sheet workbook: `--sheet Second` and `--sheet 1` select the second sheet;
omitted selects the first; `--sheet Missing` and `--sheet 7` raise `ValueError` that lists the real
sheet names; a workbook with a sheet literally named `"1"` resolves `--sheet 1` to that sheet.

#### FR-1.4: `--sheet` with CSV input is an error
Passing `--sheet` when the input is not an Excel file raises `ValueError` ("--sheet only applies to
Excel input") before any LLM client is constructed.
**Verify:** `cli.main` with `--input x.csv --sheet A` exits 1 with that message and constructs no
classifier (test monkeypatches the classifier constructor to fail if called).

#### FR-1.5: Excel input requires an explicit `--output`
For Excel input, an omitted `--output` raises `ValueError` (the CLI otherwise overwrites the input
in place, which would replace a workbook with CSV bytes). The check runs in `cli.py` before the
overwrite warning and before any client construction, and in `classify_csv` for library callers.
CSV-input behavior with omitted `--output` is unchanged.
**Verify:** `cli.main` with `--input x.xlsx` and no `--output` exits 1 and `x.xlsx` is byte-identical
afterwards; CSV input with no `--output` still prints the existing overwrite warning.

#### FR-1.6: Blank query cells are skipped, not classified
For both CSV and Excel input, a row whose text-column value is null, empty, or whitespace-only is
excluded from the rows sent to the LLM (today it would be sent as the string `"nan"`). "Null" means
null after the normal read: pandas' default NA parsing applies to both formats, so NA-like cell
strings that pandas already turns into NaN (for example `NA`) are blank, which keeps CSV and Excel
consistent. Such rows
stay in the output in their original position with null category columns (this also affects
`experiment.py`'s classify step, which calls `classify_csv`; its generated test CSV is already
filtered by `dataset_io.project_and_filter`, so no behavior change is expected there), are excluded from
`--restore`'s remaining-work set (so they are not retried on every resume), and the run prints
`Skipped N rows with empty text`. `--limit` counts only classifiable rows. Under `--restore`, `--limit N` keeps today's meaning: at
most N remaining (not yet classified, non-blank) rows are attempted in this run.
**Verify:** A test with 5 rows (2 blank, 1 whitespace-only) asserts the fake classifier is called
exactly twice, output row count is 5 in original order, blank rows have null labels, and a second run
with `--restore` makes zero classifier calls. A second test with blank rows before completed and
unfinished rows asserts `--restore --limit 1` classifies exactly one remaining row.

#### FR-1.10: The row domain of an Excel sheet is what `pd.read_excel` returns
Row positions (the contract behind `--restore`, INV-10) are the positions in the DataFrame returned
for the selected sheet. Verified: pandas keeps fully empty interior rows as all-null rows and drops
trailing fully empty rows. The spec adopts that as authoritative; the same input read twice must give
the same row domain. Duplicate header cells are disambiguated by pandas (`a`, `a.1`); `--keep-columns`
matches those post-parse names, so `--keep-columns a` selects only the first `a` (documented).
**Verify:** A test with a workbook containing an interior empty row, two trailing empty rows, and a
duplicated header asserts: the interior empty row is present as a skipped blank row (FR-1.6), the
trailing rows are absent, `--restore` resumes correctly on it, and `--keep-columns a` yields one `a`
column.

#### FR-1.7: Missing text column and empty workbooks fail clearly
A missing `--column` raises the existing "Column ... not found. Available columns: [...]"
`ValueError` for Excel input as it does for CSV. A sheet with no data rows (header only) produces an
output CSV with the header and zero data rows, no classifier calls, and exit code 0.
**Verify:** Tests for both cases against an `.xlsx` fixture.

#### FR-1.8: `--restore` works with Excel input
`--restore` re-reads the input workbook on each run and seeds category columns from the prior output
CSV by row position, exactly as INV-10 specifies for CSV, including the row-count mismatch
`ValueError`. The row count compared is the count of the data rows read from the selected sheet.
The positional contract is unchanged and unverified beyond row count: resuming with a different
`--sheet`, a different `--keep-columns` list, or edited input of the same length silently pairs old
labels with the current rows, and carried columns always come from the current input. This is
documented in the README as a user responsibility; no restore metadata is added.
**Verify:** A test runs half a workbook with `--limit`, resumes with `--restore`, and asserts only
the remaining rows are classified; a test with a prior output of the wrong length raises the INV-10
`ValueError`.

#### FR-1.9: Excel dependency is lazy and declared
`openpyxl` is added to `requirements.txt` and `pyproject.toml`. It is imported only on the Excel code
paths (reading and writing `.xlsx`), so CSV-only runs never import it. If it is missing when needed,
a `RuntimeError` says to run `pip install openpyxl` (the CLI already prints
`RuntimeError`s as `Error: ...` and exits 1).
**Verify:** A test monkeypatches the import to fail and asserts the `RuntimeError` message for Excel
input, while CSV input still succeeds in the same test; another test reads `requirements.txt` and
`pyproject.toml` and asserts both list `openpyxl`.

---

## Feature 2: Selecting columns to carry into the output

**Who & why:** The classifier copies every input column into the output. The real logs contain
account names, cookie IDs, countries, and free-text feedback, none of which the classification
needs, and 44 columns inflate a 600k-row output. Analysts need to keep just the identifiers they
join on, without a separate preprocessing step, and need the dropped columns to never be written to
the output file.

### Functional Requirements

#### FR-2.1: `--keep-columns` limits the input columns carried to the output
New flag `--keep-columns COL1,COL2,...` (library parameter `keep_columns: list[str] | None = None`).
Names are split on commas and stripped of surrounding whitespace; empty tokens and duplicates are
ignored. The `--column` text column is always kept even if not listed. Output column order follows
the input's column order, then the generated category/audit columns. When the flag is omitted all
input columns are carried over, exactly as today. The flag applies to CSV and Excel input alike.
A column name containing a comma cannot be selected via this flag; this is a documented limitation.
**Verify:** A test with columns `a,b,c,text` and `--keep-columns c,a` asserts output columns are
`a,c,text` followed by the category columns; omitting the flag yields all of them.

#### FR-2.2: Unknown names in `--keep-columns` fail before any LLM call
If any requested name is not an input column (exact, case-sensitive match), `ValueError` lists the
unknown names and the available columns. The check runs right after the table is read and before
any category column is added or any classifier call is made. "Before any LLM call" means no provider request; in `cli.py`, classifier
objects are constructed before `classify_csv` reads the table, so data-dependent checks (unknown
sheet, unknown keep names, missing text column, empty workbook, row limit) run after client
construction but before any request. Only the data-free checks in FR-1.4, FR-1.5, and FR-3.1 run
before client construction.
**Verify:** `--keep-columns a,zzz` raises `ValueError` mentioning `zzz` and listing available columns,
and the fake classifier records zero calls.

#### FR-2.3: A kept column must not collide with a generated column
When `--keep-columns` is given and a kept column equals a category name or an audit column this run
will generate, `ValueError` names the collision (a non-`--restore` run would otherwise null that
column). When the flag is omitted, behavior is unchanged from today.
**Verify:** A test with input column `writing_intent`, category `writing_intent`, and
`--keep-columns writing_intent,text` raises `ValueError`; the same input without the flag behaves as
before (existing tests unchanged).

#### FR-2.4: Dropped columns never reach the output
Columns not selected are dropped from the DataFrame immediately after reading (and, for Excel, are
not materialized at all, see AR-1.2), so they cannot appear in any incremental write, the final CSV,
or the optional `.xlsx`.
**Verify:** A test with a sentinel value in a dropped column asserts the sentinel string appears in
none of the files the run produced, including mid-run incremental writes (test uses `save_every=1`).

---

## Feature 3: Optional Excel output

**Who & why:** Analysts who live in Excel want to open the classified result there directly. The
pipeline rewrites its output file every `save_every` rows for crash recovery and resumes from it,
which is safe for CSV and slow and corruption-prone for a large `.xlsx`. The CSV must therefore stay
the working format, with `.xlsx` as an end-of-run export.

### Functional Requirements

#### FR-3.1: `--output-format {csv,xlsx}` with CSV as the default
New flag `--output-format` (library parameter `output_format: str = "csv"`). `--output` always names
the CSV file, in both formats; an `--output` whose suffix is `.xlsx`, `.xlsm`, or `.xls` raises
`ValueError` (use `--output-format xlsx` instead). The default produces exactly today's outputs.
**Verify:** `--output out.xlsx` raises `ValueError` before any LLM call; default format produces only
the CSV and the existing `.cost_report.json`.

#### FR-3.2: `xlsx` format exports once, after the CSV is final
With `--output-format xlsx`, after the final CSV write the same DataFrame is also written to
`<output stem>.xlsx` in the same directory (`out.csv` -> `out.xlsx`) as a single sheet named
`Classified` with no index column. The `.xlsx` is never written during incremental saves. The CSV is
always produced and remains the file `--restore` reads. The write is atomic (temp file in the same
directory, then `os.replace`), so a crash never leaves a truncated `.xlsx`. An existing `.xlsx` at
that path is replaced. If the export itself fails (disk full, permission, replace error), the temp
file is removed, the CSV is left intact, the error is printed as `Error: Excel export failed: ...
(the classified CSV was kept at <path>)`, and the CLI exits 1 (the cost report is still written
first). This is the one case in which a written CSV does not mean exit 0.
**Verify:** A test with `save_every=1` and a fake classifier that records the directory listing on
each call asserts no `.xlsx` exists until the run returns; after the run, the workbook's rows equal
the CSV's rows.

#### FR-3.3: Excel row-limit failure is fast
Excel worksheets hold at most 1,048,576 rows including the header. When `xlsx` output is requested
and the row count read from the input exceeds 1,048,575, `ValueError` is raised right after reading,
before any LLM call.
**Verify:** A test patches the limit constant to a small value, passes a table over it, and asserts
`ValueError` with zero classifier calls.

#### FR-3.4: Cell values are written as literal text, never as formulas
Query text from real logs can begin with `=`, `+`, `-`, or `@`. Verified: pandas `to_excel` via
openpyxl stores a string beginning with `=` as a formula cell (`data_type 'f'`), which then reads
back empty and could execute in Excel. The exporter must write every string cell as a literal
string, so a value such as `=1+1` is stored and displayed as that text.
This protection covers the `.xlsx` artifact only. The CSV is written verbatim (it is the
classification record and the resume format), so a CSV opened in a spreadsheet application can still
interpret values beginning with `=`, `+`, `-`, or `@`; making the CSV formula-safe is out of scope and
the README says so.
**Verify:** A test exports a row whose text is `=1+1`, reloads the workbook with openpyxl, and asserts
the cell's `data_type` is `'s'` and its value is `=1+1`; the CSV still contains `=1+1` unchanged.

#### FR-3.5: The cost report path is unchanged
`cli.py`'s cost report is still written next to the CSV (`<output stem>.cost_report.json`), derived
from the `--output` CSV path, never from the `.xlsx`.
**Verify:** A `cli.main` test with `--output-format xlsx` asserts `out.cost_report.json` exists and
`out.xlsx.cost_report.json` does not.

#### FR-3.6: Unwritable characters and oversized cells do not crash the export
Verified: openpyxl raises `IllegalCharacterError` for control characters such as `\x0b` in a string
cell, and silently truncates cells over 32,767 characters (with a `UserWarning`). Query logs can
contain both. The exporter removes characters Excel cannot store (control characters other than tab,
newline, and carriage return) from string cells in the `.xlsx` only (the CSV is never altered), and
truncates cells to 32,767 characters explicitly. If either happened, the run prints one summary line
with the number of affected cells (`Excel export: sanitized N cells`). Sanitizing and truncation
never fail the export; any other export failure follows FR-3.2.
**Verify:** A test exports a row containing `a\x0bb` and a 40,000-character string; the export
succeeds, the workbook cells are `ab` and 32,767 characters, the CSV still holds the originals, and the
summary line reports 2 affected cells.

---

## Feature 4: Tests, documentation, and contracts

**Who & why:** The implementer sees only this spec. Without explicit requirements, tests and
doc/contract updates get skipped, and the repo's rule is that README flags and ARCHITECTURE.md stay in sync with `cli.py`.

### Functional Requirements

#### FR-4.1: Ship unit tests with the code
The implementation ships a new `tests/test_excel_io.py` run by `pytest`, with at least one test for
every **Verify:** condition in this spec and for the edge cases above (multi-sheet, missing sheet,
unknown `--keep-columns`, missing text column, `--restore` with Excel input, blank text cells,
output-extension mismatch, over-row-limit, unwritable characters/oversized cells, export failure,
interior/trailing empty rows, duplicate headers). Tests build workbooks with `pandas`/`openpyxl` in
`tmp_path`, fake the classifier (never call `litellm`), and follow the patterns of
`tests/test_cost.py`/`tests/test_batching.py`. The full existing suite must still pass unmodified.
**Verify:** `.venv/bin/python -m pytest` passes, and `tests/test_excel_io.py` exists with tests
mapped to FR-1.1 to FR-3.6, including a test where the `.xlsx` export is forced to fail (CSV kept,
temp file removed, exit code 1).

#### FR-4.2: README and architecture documents are updated
`README.md` documents `--sheet`, `--keep-columns`, `--output-format`, the Excel-input
`--output`-required rule, the blank-row skip, the formula-safe export, and the known cost that
`--restore` re-parses the workbook. `spec/ARCHITECTURE.md` is updated per its own header: the Module
Boundary Map and INV-1 import graph (new leaf module, AR-1.1), the INV-7 test-import exception list,
the Stable Contracts CLI flag surface, and a note on the `pyproject.toml`/`requirements.txt` shared
file. Changes to INV-1/INV-7 are made through an ADR as that file requires (`spec/9-excel-input-output/ADR.md`).
The ADR records two decisions: the new `table_io.py` leaf and its `pipeline.py`/`cli.py` import edges
(INV-1), and the INV-7 test-import exception for `table_io`.
**Verify:** README's Options table lists all three flags; `spec/ARCHITECTURE.md` mentions the new
module in the boundary map; `spec/9-excel-input-output/ADR.md` exists and names both decisions.

---

### Architectural Requirements

#### AR-1.1: A new leaf module owns all table I/O
New `src/query_classification/table_io.py` (stdlib + pandas + lazily `openpyxl` only; no internal
imports), mirroring `dataset_io.py`'s leaf-module precedent, provides: reading a table from CSV or
Excel with sheet resolution and optional column selection; and the atomic `.xlsx` export. It is
imported by `pipeline.py` and `cli.py` (for the early fail-fast checks). `experiment.py`/`dataset_io.py`
are not modified. Every module starts with `from __future__ import annotations` and uses PEP 604
unions.
**Verify:** An import-graph test (or grep in the test) asserts `table_io.py` has no
`query_classification` imports and `pipeline.py` does not import `openpyxl` directly.

#### AR-1.2: Column selection happens at read time for Excel
For Excel input, selected columns are passed to `pd.read_excel` via a callable `usecols`, so dropped
columns never become DataFrame cells (memory and privacy). Unknown requested names are detected by
comparing the requested set to the columns actually returned. The workbook is opened once with
`pd.ExcelFile(path, engine="openpyxl")` as a context manager and closed after parsing.
**Verify:** A test patches `pd.read_excel`/`ExcelFile.parse` to capture `usecols` and asserts it is
supplied when `--keep-columns` is given.

#### AR-1.3: `classify_csv` signature stays backward compatible
`classify_csv` (re-exported public API, `__init__.py`) gains the keyword parameters `sheet`,
`keep_columns`, and `output_format` appended with defaults that reproduce current behavior; no
existing parameter changes position or meaning. `cli.py` passes them through. Validations that need no
data (FR-1.4, FR-1.5, FR-3.1) run in `cli.py` before client construction, and the same checks run in
`classify_csv` for library callers. The incremental `df.to_csv(...)` calls and the INV-6 main-thread
mutation rule are untouched.
**Verify:** Existing tests pass unmodified; calling `classify_csv` with only the old arguments
yields identical output to before.

---

## Data Requirements

- Input: `.csv`, `.xlsx`, `.xlsm`. Output: a CSV (always) plus an optional `.xlsx` sibling.
- Excel dates/numbers in carried columns are written to CSV via pandas' default serialization
  (e.g. `2026-08-31 00:00:00`). No locale handling is added.

## Integration Points

- In `pipeline.classify_csv`, the `pd.read_csv(input_path)` input read is replaced by the `table_io`
  read call; the `pd.read_csv(output_path)` read of the prior output in the restore block stays CSV.
- `cli.build_parser`'s flag surface and `cli.main`'s cost-report path derivation
  (`csv_path.with_suffix(".cost_report.json")`).
- `experiment.py`'s `classify_csv` call is symbol-level unaffected; it passes none of the new
  parameters and its test CSVs are pre-filtered by `dataset_io` (`project_and_filter` drops rows with
  missing text).

## Related Specs

| Spec | Relationship | Affected Requirements |
|------|-------------|---------------------|
| Spec 1: Initial classification with critics | **Modifies** — `classify_csv` input handling and the restore semantics it established (INV-10) | FR-1.6, FR-1.8 |
| Spec 7: Cost and timing logging | **References** — cost report path rule | FR-3.5 |
| Spec 2: Experiment runner | **References** — `dataset_io.py` leaf-module precedent; `experiment.py` unchanged | AR-1.1 |

## Constraints

- No new network or LLM behavior; no live provider calls in tests (AGENTS.md).
- `openpyxl` is the only new dependency.
- Unreadable or password-protected workbooks surface pandas'/openpyxl's own error, wrapped as `ValueError` with the file path.
- Only parsed cell values of the selected sheet are retained. Macros (`.xlsm`), formulas (cached values are read), formatting, hidden sheets, and workbook metadata are discarded, and output is always a new `.xlsx`.
- Resource expectation for the representative 629k-row, 145 MB workbook: the read takes minutes (measured) and the full table lives in memory during the run; the optional export needs a temporary file of workbook size on the output volume. The README documents this; no hard cap beyond FR-3.3 is imposed.

## Out of Scope

- Reading `.xls`, `.ods`, or other spreadsheet formats; multiple sheets in one run.
- Writing `.xlsx` incrementally, or an export-only mode that converts an existing CSV without re-reading the input.
- Caching a parsed workbook between runs (`--restore` re-parses it each run; known cost).
- Changing `experiment.py` or `dataset_io.py`; renaming `classify_csv`.
- Column renaming, reordering by flag, or regex/glob selection in `--keep-columns`.

## Spec Completeness Checklist

- [x] **Scope & acceptance criteria** — scope and non-scope above; every FR has a **Verify:** line.
- [x] **Testing strategy** — FR-4.1 requires `tests/test_excel_io.py` mapped to each Verify condition, with faked classifier.
- [x] **Existing patterns** — AR-1.1 follows `dataset_io.py`'s leaf-module pattern; FR-1.8 reuses INV-10; AR-1.3 follows the keyword-defaults style of earlier specs.
- [x] **Dependencies** — FR-1.9: only `openpyxl`, lazily imported, declared in both dependency files.
- [x] **Architecture & interfaces** — AR-1.1 to AR-1.3 and FR-4.2 (ADR for INV-1/INV-7).
- [x] **Error handling & failure modes** — FR-1.3, 1.4, 1.5, 1.7, 2.2, 2.3, 3.1, 3.3; atomic write in FR-3.2.
- [x] **Security review** — FR-3.4 (formula injection), FR-2.4 (dropped columns never written), FR-1.5 (no in-place overwrite of a workbook); no auth/network surface added.
- [x] **Performance impact** — AR-1.2 (read-time column selection, single open); FR-3.2/FR-3.3; known re-parse cost on `--restore` is documented in FR-4.2 and Out of Scope. Observed: openpyxl reads the 629k-row workbook in minutes.
- [x] **Rollout & migration** — all new parameters default to current behavior; the one deliberate behavior change for existing users is FR-1.6 (blank text rows no longer sent as `"nan"`), called out in README.
- [x] **Assumptions & risks** — Assumes users accept `--output` must name a CSV; `.xlsx` row cap handled by FR-3.3; `--keep-columns` cannot select names containing commas; risk that FR-1.6 changes row counts sent to the LLM for existing CSV users (strictly fewer wasted calls).
