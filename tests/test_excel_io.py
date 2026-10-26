"""Tests for spec 9 (excel-input-output): `table_io` primitives (Task 1),
`classify_csv` integration (Task 2) and the CLI flags (Task 3). No live
network/provider calls -- every LLM call is faked.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import openpyxl
import pandas as pd
import pytest

from query_classification import table_io
from query_classification.table_io import (
    EXCEL_MAX_ROWS,
    ExcelExportError,
    blank_text_mask,
    check_row_limit,
    is_excel_path,
    parse_keep_columns,
    read_table,
    validate_options,
    write_xlsx,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_workbook(path, sheets):
    """Write {sheet name: [header, *rows]} to an .xlsx via openpyxl."""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for row in rows:
            ws.append(row)
    wb.save(path)
    return path


# --- Task 1: reading --------------------------------------------------------


def test_is_excel_path_by_extension_case_insensitive(tmp_path):
    assert is_excel_path("a.xlsx") and is_excel_path("A.XLSM")
    assert not is_excel_path("a.csv") and not is_excel_path("noextension")


def test_xls_is_rejected_with_supported_formats():
    with pytest.raises(ValueError, match=r"\.xlsx"):
        is_excel_path("legacy.xls")


def test_csv_and_xlsx_read_to_equal_tables(tmp_path):
    rows = [["text", "n"], ["hello", 1], ["world", 2]]
    pd.DataFrame(rows[1:], columns=rows[0]).to_csv(tmp_path / "t.csv", index=False)
    make_workbook(tmp_path / "t.xlsx", {"S": rows})
    csv_df = read_table(tmp_path / "t.csv")
    xlsx_df = read_table(tmp_path / "t.xlsx")
    assert list(csv_df.columns) == list(xlsx_df.columns)
    assert csv_df["text"].tolist() == xlsx_df["text"].tolist()
    assert csv_df["n"].tolist() == xlsx_df["n"].astype(int).tolist()


def test_excel_string_cells_are_not_reinterpreted(tmp_path):
    make_workbook(tmp_path / "t.xlsx", {"S": [["id", "text"], ["007", "a"], ["8", "b"]]})
    df = read_table(tmp_path / "t.xlsx")
    assert df["id"].tolist() == ["007", "8"]


@pytest.fixture
def three_sheets(tmp_path):
    return make_workbook(
        tmp_path / "multi.xlsx",
        {
            "First": [["text"], ["one"]],
            "Second": [["text"], ["two"]],
            "Third": [["text"], ["three"]],
        },
    )


@pytest.mark.parametrize(
    "sheet,expected",
    [(None, "one"), ("Second", "two"), ("1", "two"), ("2", "three")],
)
def test_sheet_selection_by_name_index_or_default(three_sheets, sheet, expected):
    assert read_table(three_sheets, sheet=sheet)["text"].tolist() == [expected]


@pytest.mark.parametrize("sheet", ["Missing", "7"])
def test_unknown_sheet_lists_available_names(three_sheets, sheet):
    with pytest.raises(ValueError, match="First.*Second.*Third"):
        read_table(three_sheets, sheet=sheet)


def test_sheet_name_wins_over_index(tmp_path):
    path = make_workbook(
        tmp_path / "t.xlsx",
        {"A": [["text"], ["a"]], "1": [["text"], ["named-one"]], "Z": [["text"], ["z"]]},
    )
    # "1" is both a sheet name (position 1) and an index; add a sheet so index 1
    # would otherwise be "1" too -- reorder so the name and index disagree.
    wb = openpyxl.load_workbook(path)
    wb.move_sheet("1", offset=1)  # order is now A, Z, 1
    wb.save(path)
    assert read_table(path, sheet="1")["text"].tolist() == ["named-one"]


def test_unreadable_workbook_raises_valueerror_with_path(tmp_path):
    bad = tmp_path / "broken.xlsx"
    bad.write_bytes(b"this is not a zip file")
    with pytest.raises(ValueError, match="broken.xlsx"):
        read_table(bad)


def test_csv_read_never_needs_openpyxl_and_excel_read_reports_it(tmp_path, monkeypatch):
    pd.DataFrame({"text": ["a"]}).to_csv(tmp_path / "t.csv", index=False)
    make_workbook(tmp_path / "t.xlsx", {"S": [["text"], ["a"]]})
    monkeypatch.setitem(sys.modules, "openpyxl", None)  # makes `import openpyxl` fail
    assert read_table(tmp_path / "t.csv")["text"].tolist() == ["a"]
    with pytest.raises(RuntimeError, match="pip install openpyxl"):
        read_table(tmp_path / "t.xlsx")


def test_importing_table_io_does_not_import_openpyxl():
    code = (
        "import sys; sys.path.insert(0, 'src'); "
        "import query_classification.table_io; "
        "assert 'openpyxl' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_openpyxl_declared_in_both_dependency_files():
    requirements = (REPO_ROOT / "requirements.txt").read_text().splitlines()
    assert "openpyxl" in [line.strip() for line in requirements]
    assert '"openpyxl"' in (REPO_ROOT / "pyproject.toml").read_text()


def test_empty_rows_and_duplicate_headers(tmp_path):
    path = make_workbook(
        tmp_path / "t.xlsx",
        {
            "S": [
                ["q", "a", "a"],
                ["r1", 1, 2],
                [None, None, None],
                ["r3", 3, 4],
                [None, None, None],
                [None, None, None],
            ]
        },
    )
    df = read_table(path)
    assert len(df) == 3  # interior empty row kept, trailing empty rows dropped
    assert blank_text_mask(df["q"]).tolist() == [False, True, False]
    assert list(df.columns) == ["q", "a", "a.1"]
    kept = read_table(path, keep_columns=["a"], text_column="q")
    assert list(kept.columns) == ["q", "a"]


# --- Task 1: column selection ----------------------------------------------


def test_parse_keep_columns_strips_dedupes_and_drops_empties():
    assert parse_keep_columns(" c, a ,,c ") == ["c", "a"]
    assert parse_keep_columns(None) is None


@pytest.mark.parametrize("fmt", ["csv", "xlsx"])
def test_keep_columns_input_order_and_text_always_kept(tmp_path, fmt):
    rows = [["a", "b", "c", "text"], [1, 2, 3, "x"], [4, 5, 6, "y"]]
    if fmt == "csv":
        pd.DataFrame(rows[1:], columns=rows[0]).to_csv(tmp_path / "t.csv", index=False)
    else:
        make_workbook(tmp_path / "t.xlsx", {"S": rows})
    path = tmp_path / f"t.{fmt}"
    df = read_table(path, keep_columns=["c", "a"], text_column="text")
    assert list(df.columns) == ["a", "c", "text"]
    assert list(read_table(path).columns) == ["a", "b", "c", "text"]


@pytest.mark.parametrize("fmt", ["csv", "xlsx"])
def test_unknown_keep_column_lists_unknown_and_available(tmp_path, fmt):
    rows = [["a", "text"], [1, "x"]]
    if fmt == "csv":
        pd.DataFrame(rows[1:], columns=rows[0]).to_csv(tmp_path / "t.csv", index=False)
    else:
        make_workbook(tmp_path / "t.xlsx", {"S": rows})
    with pytest.raises(ValueError) as exc:
        read_table(tmp_path / f"t.{fmt}", keep_columns=["a", "zzz"], text_column="text")
    assert "zzz" in str(exc.value) and "'text'" in str(exc.value)


def test_excel_keep_columns_passes_callable_usecols(tmp_path, monkeypatch):
    path = make_workbook(tmp_path / "t.xlsx", {"S": [["a", "b", "text"], [1, 2, "x"]]})
    seen = {}
    original = pd.ExcelFile.parse

    def spy(self, *args, **kwargs):
        seen.update(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pd.ExcelFile, "parse", spy)
    df = read_table(path, keep_columns=["a"], text_column="text")
    assert callable(seen["usecols"]) and seen["dtype"] is object
    assert list(df.columns) == ["a", "text"]


def test_table_io_is_a_leaf_module():
    tree = ast.parse((REPO_ROOT / "src/query_classification/table_io.py").read_text())
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        assert not any(n.startswith("query_classification") for n in names)


def test_validate_options_data_free_checks():
    assert validate_options("a.csv", "out.csv", None, "csv") is False
    assert validate_options("a.xlsx", "out.csv", "S", "xlsx") is True
    assert validate_options("a.csv", None, None, "csv") is False  # CSV may overwrite
    with pytest.raises(ValueError, match="--sheet only applies"):
        validate_options("a.csv", "out.csv", "S", "csv")
    with pytest.raises(ValueError, match="explicit --output"):
        validate_options("a.xlsx", None, None, "csv")
    with pytest.raises(ValueError, match="must name a CSV"):
        validate_options("a.csv", "out.xlsx", None, "csv")
    with pytest.raises(ValueError, match="--output-format"):
        validate_options("a.csv", "out.csv", None, "parquet")


def test_blank_text_mask_covers_null_empty_and_whitespace():
    series = pd.Series(["ok", None, "", "   ", float("nan")], dtype=object)
    assert blank_text_mask(series).tolist() == [False, True, True, True, True]


def test_check_row_limit_boundary():
    check_row_limit(EXCEL_MAX_ROWS - 1)
    with pytest.raises(ValueError, match="Excel sheet"):
        check_row_limit(EXCEL_MAX_ROWS)


# --- Task 1: export --------------------------------------------------------


def _reload(path):
    ws = openpyxl.load_workbook(path).active
    return ws, [[c.value for c in row] for row in ws.iter_rows()]


def test_write_xlsx_roundtrip_literal_text_and_lists(tmp_path):
    df = pd.DataFrame(
        {
            "text": ["=1+1", "+x", "plain", None],
            "label": [["a"], ["b", "c"], None, ["d"]],
            "n": [1, 2, 3, 4],
        }
    )
    out = write_xlsx(df, tmp_path / "o.xlsx", tmp_path / "o.csv")
    ws, rows = _reload(out)
    assert ws.title == "Classified"
    assert rows[0] == ["text", "label", "n"]
    assert [r[0] for r in rows[1:]] == ["=1+1", "+x", "plain", None]
    assert rows[1][1] == "['a']" and rows[2][1] == "['b', 'c']"
    assert ws["A2"].data_type == "s"  # a literal, not a formula
    assert [r[2] for r in rows[1:]] == [1, 2, 3, 4]


def test_write_xlsx_sanitizes_illegal_chars_and_truncates(tmp_path, capsys):
    df = pd.DataFrame({"text": ["a\x0bb", "x" * 40_000, "tab\tok"]})
    out = write_xlsx(df, tmp_path / "o.xlsx", tmp_path / "o.csv")
    _, rows = _reload(out)
    assert rows[1][0] == "ab"
    assert len(rows[2][0]) == 32_767
    assert rows[3][0] == "tab\tok"
    assert "sanitized 2 cells" in capsys.readouterr().out
    assert df["text"][0] == "a\x0bb"  # the source frame (and so the CSV) is untouched


def test_write_xlsx_is_atomic_and_leaves_no_temp_file(tmp_path):
    out = tmp_path / "o.xlsx"
    out.write_bytes(b"old")
    write_xlsx(pd.DataFrame({"text": ["a"]}), out, tmp_path / "o.csv")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["o.xlsx"]
    assert _reload(out)[1][1] == ["a"]


def test_write_xlsx_failure_raises_cleans_temp_and_keeps_old_file(tmp_path, monkeypatch):
    out = tmp_path / "o.xlsx"
    out.write_bytes(b"old")

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(table_io.os, "replace", boom)
    with pytest.raises(ExcelExportError, match=r"disk full.*o\.csv"):
        write_xlsx(pd.DataFrame({"text": ["a"]}), out, tmp_path / "o.csv")
    assert out.read_bytes() == b"old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["o.xlsx"]
