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


# --- Task 2: classify_csv integration ---------------------------------------

from query_classification import Category, Label, classify_csv  # noqa: E402

CATS = [Category(name="label", description="d", labels=[Label(value="x", description="d")])]


class _FakeClassifier:
    """Stands in for `Classifier`: records the texts it is asked to classify."""

    def __init__(self, on_call=None):
        self.calls = []
        self.on_call = on_call

    def classify(self, text):
        self.calls.append(text)
        if self.on_call:
            self.on_call(text)
        return {"label": ["x"]}


def _csv(path, **cols):
    pd.DataFrame(cols).to_csv(path, index=False)
    return path


def test_csv_and_xlsx_input_classify_identically(tmp_path):
    rows = [["text", "id"], ["hello", "007"], ["world", "8"]]
    pd.DataFrame(rows[1:], columns=rows[0]).to_csv(tmp_path / "t.csv", index=False)
    make_workbook(tmp_path / "t.xlsx", {"S": rows})
    out_csv = classify_csv(tmp_path / "t.csv", "text", _FakeClassifier(), CATS, output_path=tmp_path / "a.csv")
    out_xlsx = classify_csv(tmp_path / "t.xlsx", "text", _FakeClassifier(), CATS, output_path=tmp_path / "b.csv")
    a, b = pd.read_csv(out_csv, dtype=str), pd.read_csv(out_xlsx, dtype=str)
    assert a["label"].tolist() == b["label"].tolist() == ["['x']", "['x']"]
    assert b["id"].tolist() == ["007", "8"]  # not reinterpreted as 7 / 8


def test_old_style_call_is_unchanged(tmp_path):
    src = _csv(tmp_path / "in.csv", text=["a", "b"], other=[1, 2])
    out = classify_csv(src, "text", _FakeClassifier(), CATS, output_path=tmp_path / "out.csv")
    df = pd.read_csv(out)
    assert list(df.columns) == ["text", "other", "label"]
    assert not list(tmp_path.glob("*.xlsx"))


def test_sheet_with_csv_input_raises_before_any_classification(tmp_path):
    src = _csv(tmp_path / "in.csv", text=["a"])
    fake = _FakeClassifier()
    with pytest.raises(ValueError, match="--sheet only applies"):
        classify_csv(src, "text", fake, CATS, output_path=tmp_path / "o.csv", sheet="S")
    assert fake.calls == []


def test_excel_input_without_output_never_overwrites_the_workbook(tmp_path):
    wb = make_workbook(tmp_path / "t.xlsx", {"S": [["text"], ["a"]]})
    before = wb.read_bytes()
    fake = _FakeClassifier()
    with pytest.raises(ValueError, match="explicit --output"):
        classify_csv(wb, "text", fake, CATS)
    assert wb.read_bytes() == before and fake.calls == []


def test_output_path_with_excel_suffix_is_rejected(tmp_path):
    src = _csv(tmp_path / "in.csv", text=["a"])
    fake = _FakeClassifier()
    with pytest.raises(ValueError, match="must name a CSV"):
        classify_csv(src, "text", fake, CATS, output_path=tmp_path / "o.xlsx")
    assert fake.calls == []


def test_blank_text_rows_are_skipped_kept_in_order_and_not_retried(tmp_path, capsys):
    src = tmp_path / "in.csv"
    pd.DataFrame({"text": ["a", None, "b", "", "   "], "id": [1, 2, 3, 4, 5]}).to_csv(src, index=False)
    fake = _FakeClassifier()
    out = classify_csv(src, "text", fake, CATS, output_path=tmp_path / "out.csv")
    assert sorted(fake.calls) == ["a", "b"]
    assert "Skipped 3 rows with empty text" in capsys.readouterr().out
    df = pd.read_csv(out)
    assert df["id"].tolist() == [1, 2, 3, 4, 5]
    assert df["label"].isna().tolist() == [False, True, False, True, True]
    again = _FakeClassifier()
    classify_csv(src, "text", again, CATS, output_path=out, restore=True)
    assert again.calls == []


def test_restore_limit_counts_only_remaining_classifiable_rows(tmp_path):
    src = _csv(tmp_path / "in.csv", text=[None, "a", "b", "c"])
    out = tmp_path / "out.csv"
    first = _FakeClassifier()
    classify_csv(src, "text", first, CATS, output_path=out, limit=1)
    assert first.calls == ["a"]
    second = _FakeClassifier()
    classify_csv(src, "text", second, CATS, output_path=out, restore=True, limit=1)
    assert second.calls == ["b"]


def test_xlsx_with_interior_and_trailing_empty_rows_restores_by_position(tmp_path):
    wb = make_workbook(
        tmp_path / "t.xlsx",
        {"S": [["text"], ["a"], [None], ["b"], ["c"], [None], [None]]},
    )
    out = tmp_path / "out.csv"
    classify_csv(wb, "text", _FakeClassifier(), CATS, output_path=out, limit=1)
    assert len(pd.read_csv(out)) == 4  # trailing empty rows are not part of the sheet
    rest = _FakeClassifier()
    classify_csv(wb, "text", rest, CATS, output_path=out, restore=True)
    assert sorted(rest.calls) == ["b", "c"]
    assert pd.read_csv(out)["label"].isna().tolist() == [False, True, False, False]


def test_xlsx_missing_text_column_and_header_only_sheet(tmp_path):
    wb = make_workbook(tmp_path / "t.xlsx", {"S": [["other"], ["a"]]})
    with pytest.raises(ValueError, match="Column 'text' not found"):
        classify_csv(wb, "text", _FakeClassifier(), CATS, output_path=tmp_path / "o.csv")
    empty = make_workbook(tmp_path / "e.xlsx", {"S": [["text", "id"]]})
    fake = _FakeClassifier()
    out = classify_csv(empty, "text", fake, CATS, output_path=tmp_path / "e.csv")
    assert fake.calls == [] and list(pd.read_csv(out).columns) == ["text", "id", "label"]
    assert len(pd.read_csv(out)) == 0


def test_restore_with_wrong_length_prior_output_still_raises(tmp_path):
    wb = make_workbook(tmp_path / "t.xlsx", {"S": [["text"], ["a"], ["b"]]})
    prior = _csv(tmp_path / "out.csv", text=["a"], label=["['x']"])
    with pytest.raises(ValueError, match="Cannot restore"):
        classify_csv(wb, "text", _FakeClassifier(), CATS, output_path=prior, restore=True)


def test_keep_columns_limits_output_and_collision_is_rejected(tmp_path):
    src = _csv(tmp_path / "in.csv", a=[1], b=[2], text=["x"], label=["old"])
    out = classify_csv(
        src, "text", _FakeClassifier(), CATS, output_path=tmp_path / "o.csv", keep_columns=["a"]
    )
    assert list(pd.read_csv(out).columns) == ["a", "text", "label"]
    fake = _FakeClassifier()
    with pytest.raises(ValueError, match="generates"):
        classify_csv(
            src, "text", fake, CATS, output_path=tmp_path / "o2.csv", keep_columns=["label"]
        )
    assert fake.calls == []
    # without the flag the pre-existing behavior is unchanged: the column is reused
    classify_csv(src, "text", _FakeClassifier(), CATS, output_path=tmp_path / "o3.csv")


def test_unknown_keep_column_makes_zero_calls(tmp_path):
    src = _csv(tmp_path / "in.csv", a=[1], text=["x"])
    fake = _FakeClassifier()
    with pytest.raises(ValueError, match="zzz"):
        classify_csv(src, "text", fake, CATS, output_path=tmp_path / "o.csv", keep_columns=["a", "zzz"])
    assert fake.calls == []


def test_dropped_columns_never_reach_any_written_file(tmp_path):
    src = tmp_path / "in.csv"
    pd.DataFrame({"text": ["a", "b", "c"], "secret": ["SENTINEL-1", "SENTINEL-2", "SENTINEL-3"]}).to_csv(
        src, index=False
    )
    out = tmp_path / "out.csv"
    snapshots = []

    def snapshot(_text):
        if out.exists():
            snapshots.append(out.read_text())

    classify_csv(
        src, "text", _FakeClassifier(on_call=snapshot), CATS, output_path=out,
        save_every=1, workers=1, keep_columns=["text"], output_format="xlsx",
    )
    texts = [out.read_text(), *snapshots]
    assert not any("SENTINEL" in t for t in texts)
    _, rows = _reload(tmp_path / "out.xlsx")
    assert "secret" not in rows[0]


def test_xlsx_export_happens_once_after_the_final_csv(tmp_path):
    src = _csv(tmp_path / "in.csv", text=["=1+1", "b", "c"])
    out = tmp_path / "out.csv"
    seen = []

    def watch(_text):
        seen.append(sorted(p.name for p in tmp_path.iterdir()))

    classify_csv(
        src, "text", _FakeClassifier(on_call=watch), CATS, output_path=out,
        save_every=1, workers=1, output_format="xlsx",
    )
    assert all("out.xlsx" not in names for names in seen)
    ws, rows = _reload(tmp_path / "out.xlsx")
    csv_df = pd.read_csv(out, dtype=str)
    assert [r[0] for r in rows[1:]] == csv_df["text"].tolist()  # incl. the literal "=1+1"
    assert ws["A2"].data_type == "s"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["in.csv", "out.csv", "out.xlsx"]


def test_xlsx_export_also_happens_on_the_batch_path(monkeypatch, tmp_path):
    from tests.test_batching import _make_batch_runner

    runner, _stats, categories = _make_batch_runner(monkeypatch, mode="fixed", fixed_size=2)
    src = _csv(tmp_path / "in.csv", text=["q1", "q2", "q3"])
    out = classify_csv(
        src, "text", None, categories, output_path=tmp_path / "out.csv",
        batch_runner=runner, output_format="xlsx",
    )
    _, rows = _reload(out.with_suffix(".xlsx"))
    assert len(rows) == 4 and rows[0] == ["text", "c"]


def test_row_limit_fails_fast_with_zero_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(table_io, "EXCEL_MAX_ROWS", 3)
    src = _csv(tmp_path / "in.csv", text=["a", "b", "c", "d"])
    fake = _FakeClassifier()
    with pytest.raises(ValueError, match="Excel sheet"):
        classify_csv(src, "text", fake, CATS, output_path=tmp_path / "o.csv", output_format="xlsx")
    assert fake.calls == []
    classify_csv(src, "text", fake, CATS, output_path=tmp_path / "o.csv")  # CSV is unaffected
    assert len(fake.calls) == 4


def test_invalid_output_format_is_rejected(tmp_path):
    src = _csv(tmp_path / "in.csv", text=["a"])
    with pytest.raises(ValueError, match="--output-format"):
        classify_csv(src, "text", _FakeClassifier(), CATS, output_path=tmp_path / "o.csv", output_format="parquet")
