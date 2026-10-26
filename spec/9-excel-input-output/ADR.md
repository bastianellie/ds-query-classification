# Architectural Decision Records: 9-excel-input-output

## ADR-001: `table_io.py` as a new leaf module imported by `pipeline.py` and `cli.py`
- **Date:** 2026-10-26
- **Status:** Accepted
- **Context:** Spec 9 (AR-1.1) needs CSV/Excel reading, column selection, option validation, and a sanitized atomic `.xlsx` export. `pipeline.py` only reads CSV today (`pd.read_csv`), and `cli.py` must run the data-free option checks before building any classifier. INV-1 lists the exact allowed imports of `pipeline.py` and `cli.py`, so a new module they both import is an amendment to that rule.
- **Decision:** Add `src/query_classification/table_io.py` as a **leaf** module: stdlib + pandas, with `openpyxl` imported lazily on Excel paths only, and no imports from the rest of the package (enforced by `tests/test_excel_io.py::test_table_io_is_a_leaf_module`). `pipeline.py` and `cli.py` may import it; `pipeline.py` never imports `openpyxl` directly. `experiment.py` and `dataset_io.py` are untouched.
- **Rationale:** Mirrors `dataset_io.py`'s leaf-module precedent, adds nodes without creating a cycle, keeps `pipeline.py` free of Excel-library details, and lets CSV-only runs work without `openpyxl` installed.
- **Consequences:** Two new inbound edges (`table_io -> pipeline`, `table_io -> cli`). `ARCHITECTURE.md`'s graph, boundary map, and INV-1 text are updated accordingly.
- **Invariants affected:** INV-1

## ADR-002: Direct test imports of `table_io`
- **Date:** 2026-10-26
- **Status:** Accepted
- **Context:** INV-7 limits tests to the public API plus an enumerated list of internal names. `table_io` primitives (sheet resolution, `read_table`, `write_xlsx`, `ExcelExportError`, `EXCEL_MAX_ROWS`) are internal and not re-exported, but the spec requires a test per Verify condition for them, including one that monkeypatches `table_io.EXCEL_MAX_ROWS`.
- **Decision:** Tests may import `query_classification.table_io` directly (all functions/classes/constants), mirroring the existing `dataset_io`/`batching`/`cost` exceptions. `classify_csv` and `Category`/`Label` are still imported through the public API, and `cli.main`/`cli.build_parser` fall under the exception already granted by spec 3.
- **Rationale:** The Excel behaviors are only reachable below `classify_csv` (e.g. the row-limit constant, the atomic writer), and precise failure localization matters for I/O code.
- **Consequences:** INV-7's exception list grows by one module.
- **Invariants affected:** INV-7
