"""Table input/output: CSV-or-Excel reading, column selection, option checks, and
the sanitized atomic ``.xlsx`` export (spec 9).

A leaf module: stdlib + pandas only, with ``openpyxl`` imported lazily on the
Excel code paths so CSV-only runs never need it. It deliberately imports nothing
from the rest of the package (see ``spec/ARCHITECTURE.md`` INV-1), mirroring
``dataset_io.py``'s precedent.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

EXCEL_SUFFIXES = (".xlsx", ".xlsm")
OUTPUT_FORMATS = ("csv", "xlsx")
# Excel worksheets hold at most this many rows, header included.
EXCEL_MAX_ROWS = 1_048_576
XLSX_SHEET_NAME = "Classified"
_MAX_CELL_CHARS = 32_767
# Control characters Excel cannot store (tab, newline and carriage return are fine).
_ILLEGAL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class ExcelExportError(RuntimeError):
    """The final ``.xlsx`` export failed; the classified CSV was already written."""


def _require_openpyxl() -> None:
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Excel input/output needs the 'openpyxl' package. Run: pip install openpyxl"
        ) from exc


def is_excel_path(path: str | Path) -> bool:
    """True for ``.xlsx``/``.xlsm``; ``.xls`` is rejected; anything else is CSV."""
    suffix = Path(path).suffix.lower()
    if suffix == ".xls":
        raise ValueError(
            f"Legacy .xls files are not supported ('{path}'). Supported formats: "
            f"CSV, {', '.join(EXCEL_SUFFIXES)}. Re-save the workbook as .xlsx."
        )
    return suffix in EXCEL_SUFFIXES


def parse_keep_columns(value: str | None) -> list[str] | None:
    """Split a ``--keep-columns`` value on commas; strip, drop empties/duplicates."""
    if value is None:
        return None
    names: list[str] = []
    for token in value.split(","):
        name = token.strip()
        if name and name not in names:
            names.append(name)
    return names


def validate_options(
    input_path: str | Path,
    output_path: str | Path | None,
    sheet: str | None,
    output_format: str,
) -> bool:
    """Data-free checks run before any client is built (FR-1.4, 1.5, 3.1).

    Returns whether the input is an Excel workbook.
    """
    excel_input = is_excel_path(input_path)
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(
            f"--output-format must be one of {', '.join(OUTPUT_FORMATS)}, got '{output_format}'"
        )
    if sheet is not None and not excel_input:
        raise ValueError("--sheet only applies to Excel input (.xlsx/.xlsm)")
    if excel_input and not output_path:
        raise ValueError(
            "Excel input requires an explicit --output (a CSV path); otherwise the "
            "workbook would be overwritten in place."
        )
    if output_path and Path(output_path).suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        raise ValueError(
            f"--output must name a CSV file, got '{output_path}'. Use "
            f"--output-format xlsx to also get an .xlsx export next to the CSV."
        )
    if output_path:
        # Fail before any LLM call: otherwise the first incremental save raises an
        # uncaught OSError after the whole run has already been paid for. The
        # directory is never auto-created (same stance as extract_categories.py).
        out = Path(output_path)
        if out.is_dir():
            raise ValueError(f"--output '{out}' is a directory, not a file")
        if not out.parent.is_dir():
            raise FileNotFoundError(f"--output parent directory does not exist: {out.parent}")
    return excel_input


def blank_text_mask(series: pd.Series) -> pd.Series:
    """True where the value is null, empty, or whitespace-only (FR-1.6)."""
    return series.isna() | series.astype(str).str.strip().eq("")


def check_row_limit(n_rows: int) -> None:
    """Fail fast when a table cannot fit in one Excel sheet (FR-3.3)."""
    if n_rows > EXCEL_MAX_ROWS - 1:
        raise ValueError(
            f"Input has {n_rows} data rows, more than an Excel sheet can hold "
            f"({EXCEL_MAX_ROWS - 1}). Use the default CSV output instead."
        )


def _resolve_sheet(sheet: str | None, sheet_names: list[str]) -> str:
    if sheet is None:
        return sheet_names[0]
    if sheet in sheet_names:
        return sheet
    if sheet.isdigit() and int(sheet) < len(sheet_names):
        return sheet_names[int(sheet)]
    raise ValueError(f"Sheet '{sheet}' not found. Available sheets: {sheet_names}")


def _excel_header(path: Path, sheet_name: str) -> list[str]:
    """Best-effort header read (error messages only; the happy path never calls it)."""
    try:
        import openpyxl

        wb = openpyxl.load_workbook(path, read_only=True)
        try:
            first = next(wb[sheet_name].iter_rows(max_row=1, values_only=True), ())
        finally:
            wb.close()
        return [str(v) for v in first if v is not None]
    except Exception:  # noqa: BLE001 - diagnostics must never mask the real error
        return []


def _select_columns(df: pd.DataFrame, wanted: set[str]) -> pd.DataFrame:
    return df[[c for c in df.columns if c in wanted]]


def read_table(
    path: str | Path,
    *,
    sheet: str | None = None,
    keep_columns: list[str] | None = None,
    text_column: str | None = None,
) -> pd.DataFrame:
    """Read a CSV or Excel table, optionally keeping only some columns.

    CSV is read exactly as before (``pd.read_csv``). Excel is opened once, the
    sheet is resolved by name then zero-based index, and cells are read with
    ``dtype=object`` so string cells such as ``"007"`` are never reinterpreted.
    When ``keep_columns`` is given, only those columns (plus ``text_column``)
    are kept, in input order; for Excel the others are never materialized.
    """
    path = Path(path)
    wanted: set[str] | None = None
    if keep_columns is not None:
        wanted = set(keep_columns) | ({text_column} if text_column else set())

    if not is_excel_path(path):
        df = pd.read_csv(path)
        available = list(df.columns)
    else:
        _require_openpyxl()
        try:
            with pd.ExcelFile(path, engine="openpyxl") as workbook:
                sheet_name = _resolve_sheet(sheet, list(workbook.sheet_names))
                usecols = (lambda c: c in wanted) if wanted is not None else None
                df = workbook.parse(sheet_name, dtype=object, usecols=usecols)
        except (ValueError, RuntimeError):
            raise
        except Exception as exc:  # noqa: BLE001 - re-raised as a path-bearing ValueError
            raise ValueError(f"Cannot read workbook '{path}': {exc}") from exc
        available = list(df.columns)
        if wanted is not None and not wanted.issubset(available):
            available = _excel_header(path, sheet_name) or available

    if wanted is not None:
        missing = [c for c in [*(keep_columns or []), *([text_column] if text_column else [])]
                   if c not in df.columns]
        if missing:
            unknown = list(dict.fromkeys(missing))
            raise ValueError(
                f"Column(s) {unknown} not found. Available columns: {available}"
            )
        df = _select_columns(df, wanted)
    return df


def _clean_cell(value: Any, counter: list[int]) -> Any:
    """Make one value safe for an Excel cell (lists rendered as in the CSV)."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, dict, set)):
        value = str(value)
    elif isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    else:
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if hasattr(value, "item") and not isinstance(value, (str, bytes, dt.datetime)):
            value = value.item()
    if isinstance(value, str):
        cleaned = _ILLEGAL_CHARS.sub("", value)
        if len(cleaned) > _MAX_CELL_CHARS:
            cleaned = cleaned[:_MAX_CELL_CHARS]
        if cleaned != value:
            counter[0] += 1
        return cleaned
    return value


def write_xlsx(df: pd.DataFrame, path: str | Path, csv_path: str | Path) -> Path:
    """Write ``df`` as a single-sheet ``.xlsx`` atomically (FR-3.2, 3.4, 3.6).

    Strings are stored as literal text (never formulas), characters Excel cannot
    store are dropped and oversized cells truncated in the workbook only. Any
    failure raises ``ExcelExportError`` and leaves no partial file behind.
    """
    path = Path(path)
    tmp_name: str | None = None
    workbook = None
    try:
        _require_openpyxl()
        from openpyxl import Workbook
        from openpyxl.cell import WriteOnlyCell

        counter = [0]
        workbook = Workbook(write_only=True)
        sheet = workbook.create_sheet(XLSX_SHEET_NAME)
        sheet.append([str(c) for c in df.columns])
        for row in df.itertuples(index=False, name=None):
            cells = []
            for value in row:
                value = _clean_cell(value, counter)
                cell = WriteOnlyCell(sheet, value=value)
                if isinstance(value, str):
                    cell.data_type = "s"  # literal text: "=1+1" must not become a formula
                cells.append(cell)
            sheet.append(cells)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        os.close(fd)
        workbook.save(tmp_name)
        os.replace(tmp_name, path)
        tmp_name = None
        if counter[0]:
            print(f"Excel export: sanitized {counter[0]} cells")
        return path
    except Exception as exc:  # noqa: BLE001 - any export failure keeps the CSV and exits 1
        if workbook is not None:
            try:
                workbook.close()  # releases the write-only workbook's scratch files
            except Exception:  # noqa: BLE001 - best-effort cleanup must not mask the real error
                pass
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
        raise ExcelExportError(
            f"Excel export failed: {exc} (the classified CSV was kept at {csv_path})"
        ) from exc
