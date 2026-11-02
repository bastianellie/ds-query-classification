"""classify.py fails fast on an unusable --output, before any LLM call.

Regression: a missing --output parent directory used to surface as an uncaught
OSError from the first incremental save, after every LLM call had been paid for.
All tests here fake litellm.completion -- no live provider calls.
"""

from __future__ import annotations

import json
import sys

import pytest

from query_classification import classifier as classifier_module
from query_classification.cli import main as cli_main
from query_classification.table_io import validate_options


@pytest.fixture
def llm_calls(monkeypatch):
    """Fake litellm.completion that records every call it receives."""
    import litellm

    calls: list[dict] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        fields = list(kwargs["response_format"].model_fields)
        payload = {fields[0]: ["x"]}

        class _Msg:
            content = json.dumps(payload)

        class _Choice:
            message = _Msg()
            finish_reason = "stop"

        class _Usage:
            prompt_tokens = 10
            completion_tokens = 5

        class _Resp:
            choices = [_Choice()]
            usage = _Usage()

        return _Resp()

    # See cli_fake_completion_with_usage in test_cerebus.py: litellm's import-time
    # load_dotenv can leak the real .env, so neutralize it and the Cerebus resolver.
    monkeypatch.setattr(classifier_module, "_resolve_cerebus_api_key", lambda: "test-key")
    monkeypatch.setattr("query_classification.cli.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(litellm, "completion", fake_completion)
    return calls


@pytest.fixture
def inputs(tmp_path):
    (tmp_path / "in.csv").write_text("text\na\nb\nc\n")
    cats = tmp_path / "cats.json"
    cats.write_text(
        '{"categories": [{"name": "c", "description": "d", '
        '"labels": [{"value": "x", "description": "d"}]}]}'
    )
    return tmp_path / "in.csv", cats


def _run(monkeypatch, input_path, cats, *extra):
    monkeypatch.setattr(
        sys, "argv",
        ["classify.py", "--input", str(input_path), "--column", "text",
         "--categories", str(cats), "--model", "gpt-4o-mini", *extra],
    )
    cli_main()


def _assert_clean_failure(llm_calls, capsys, expected):
    out = capsys.readouterr()
    assert llm_calls == []
    assert out.out.startswith("Error:") or "\nError:" in out.out
    assert expected in out.out
    assert "Traceback" not in out.out + out.err


def test_missing_output_parent_fails_before_any_llm_call(
    tmp_path, inputs, llm_calls, capsys, monkeypatch
):
    in_csv, cats = inputs
    missing = tmp_path / "does_not_exist" / "out.csv"
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, in_csv, cats, "--output", str(missing), "--save-every", "1")
    assert exc.value.code == 1
    _assert_clean_failure(llm_calls, capsys, "--output parent directory does not exist")
    assert not missing.parent.exists()  # never auto-created


def test_output_that_is_a_directory_fails_before_any_llm_call(
    tmp_path, inputs, llm_calls, capsys, monkeypatch
):
    in_csv, cats = inputs
    out_dir = tmp_path / "outdir"
    out_dir.mkdir()
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, in_csv, cats, "--output", str(out_dir))
    assert exc.value.code == 1
    _assert_clean_failure(llm_calls, capsys, "is a directory, not a file")


def test_excel_input_with_missing_output_parent_fails_before_any_llm_call(
    tmp_path, inputs, llm_calls, capsys, monkeypatch
):
    _, cats = inputs
    xlsx = tmp_path / "in.xlsx"  # never opened: the check precedes any read
    xlsx.write_bytes(b"")
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, xlsx, cats, "--output", str(tmp_path / "nope" / "out.csv"))
    assert exc.value.code == 1
    _assert_clean_failure(llm_calls, capsys, "--output parent directory does not exist")


def test_existing_output_parent_still_works(tmp_path, inputs, llm_calls, monkeypatch):
    in_csv, cats = inputs
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _run(monkeypatch, in_csv, cats, "--output", str(out_dir / "out.csv"), "--save-every", "1")
    assert len(llm_calls) == 3
    assert (out_dir / "out.csv").exists()


def test_omitted_output_overwrites_input_in_place(tmp_path, inputs, llm_calls, monkeypatch):
    in_csv, cats = inputs
    _run(monkeypatch, in_csv, cats)
    assert len(llm_calls) == 3
    assert "c" in in_csv.read_text().splitlines()[0].split(",")


def test_bare_relative_output_filename_is_accepted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert validate_options("in.csv", "out.csv", None, "csv") is False


def test_pipeline_classify_csv_also_rejects_missing_parent(tmp_path, inputs):
    """Library callers get the same fail-fast, before the classifier is touched."""
    from query_classification import classify_csv
    from query_classification.categories import load_categories

    in_csv, cats = inputs

    class _Boom:
        def classify(self, *a, **k):
            raise AssertionError("classifier must not be called")

    with pytest.raises(FileNotFoundError, match="parent directory does not exist"):
        classify_csv(
            input_path=in_csv, column="text", classifier=_Boom(),
            categories=load_categories(cats),
            output_path=tmp_path / "missing" / "out.csv",
        )
