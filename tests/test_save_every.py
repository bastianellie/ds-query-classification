"""Tests for spec 10 (configurable-save-every): `classify_csv`'s `save_every`
validation and batch save rule (Task 1), and the `--save-every` CLI flag
(Task 2). No live network/provider calls -- every LLM call is faked.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from query_classification import Category, Label, classify_csv

CATS = [Category(name="c", description="d", labels=[Label(value="x", description="d")])]


class _FakeClassifier:
    """Stands in for `Classifier`: records the texts it is asked to classify."""

    def __init__(self):
        self.calls = []

    def classify(self, text):
        self.calls.append(text)
        return {"c": ["x"]}


@pytest.fixture
def writes(monkeypatch):
    """Spy on `DataFrame.to_csv`: logs (target path, classified rows) per call.

    Always delegates to the real method; the target is normalised to a `Path`
    from either the positional or keyword argument so fixture writes can be
    filtered out by path.
    """
    log = []
    real = pd.DataFrame.to_csv

    def spy(self, path_or_buf=None, *args, **kwargs):
        target = path_or_buf if path_or_buf is not None else kwargs.get("path_or_buf")
        classified = int(self["c"].notna().sum()) if "c" in self.columns else None
        log.append((Path(target) if isinstance(target, (str, Path)) else None, classified))
        return real(self, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", spy)
    return log


def _output_writes(log, out):
    """Classified-row counts of every write to `out`, in order (last = final write)."""
    return [classified for path, classified in log if path == Path(out)]


def _csv(path, **cols):
    pd.DataFrame(cols).to_csv(path, index=False)
    return path


def _plain_run(tmp_path, texts, writes, **kwargs):
    src = _csv(tmp_path / "in.csv", text=texts)
    out = tmp_path / "out.csv"
    writes.clear()
    kwargs.setdefault("workers", 1)
    classify_csv(src, "text", _FakeClassifier(), CATS, output_path=out, **kwargs)
    return out


# --- FR-1.4: library validation ----------------------------------------------


@pytest.mark.parametrize("bad", [0, -3, 2.5, True, "5"])
def test_classify_csv_rejects_invalid_save_every_before_any_work(tmp_path, bad):
    src = _csv(tmp_path / "in.csv", text=["a", "b"])
    fake = _FakeClassifier()
    out = tmp_path / "out.csv"
    with pytest.raises(ValueError, match=r"--save-every must be >= 1, got"):
        classify_csv(src, "text", fake, CATS, output_path=out, save_every=bad)
    assert fake.calls == [] and not out.exists()


# --- FR-2.2 / FR-2.3 / FR-2.4: non-batch cadence ------------------------------


def test_plain_run_saves_every_n_rows_then_final(tmp_path, writes):
    out = _plain_run(tmp_path, [f"q{i}" for i in range(10)], writes, save_every=3)
    assert _output_writes(writes, out) == [3, 6, 9, 10]


def test_save_every_larger_than_row_count_writes_only_once(tmp_path, writes):
    out = _plain_run(tmp_path, [f"q{i}" for i in range(5)], writes, save_every=1000)
    assert _output_writes(writes, out) == [5]
    assert len(pd.read_csv(out)) == 5


@pytest.mark.parametrize("limit,expected", [(3, [3]), (0, [0])])
def test_limit_smaller_than_save_every_writes_only_once(tmp_path, writes, limit, expected):
    out = _plain_run(tmp_path, [f"q{i}" for i in range(8)], writes, save_every=10, limit=limit)
    assert _output_writes(writes, out) == expected


def test_restore_run_restarts_the_cadence_over_remaining_rows(tmp_path, writes):
    texts = [f"q{i}" for i in range(6)]
    src = _csv(tmp_path / "in.csv", text=texts)
    out = tmp_path / "out.csv"
    classify_csv(src, "text", _FakeClassifier(), CATS, output_path=out, limit=2, workers=1)
    writes.clear()
    second = _FakeClassifier()
    classify_csv(src, "text", second, CATS, output_path=out, restore=True, save_every=2, workers=1)
    assert len(second.calls) == 4
    assert _output_writes(writes, out) == [4, 6, 6]  # interim after 2 and 4 remaining rows, then final


def test_blank_and_restored_rows_do_not_advance_the_cadence(tmp_path, writes):
    out = _plain_run(
        tmp_path, [None, "a", None, "b", "c", "d"], writes, save_every=2
    )
    # four work rows -> interim writes after the 2nd and 4th, plus the final write
    assert _output_writes(writes, out) == [2, 4, 4]


# --- FR-2.1 / FR-2.4: batch saves at most once per consumed batch --------------


def _batch_run(monkeypatch, tmp_path, writes, rows, fixed_size, save_every, runner_patch=None):
    from tests.test_batching import _make_batch_runner

    runner, _stats, categories = _make_batch_runner(monkeypatch, mode="fixed", fixed_size=fixed_size)
    if runner_patch is not None:
        runner_patch(runner)
    src = _csv(tmp_path / "in.csv", text=[f"q{i}" for i in range(rows)])
    out = tmp_path / "out.csv"
    writes.clear()
    classify_csv(
        src, "text", None, categories, output_path=out,
        batch_runner=runner, save_every=save_every, workers=1,
    )
    return out


def test_batch_writes_once_per_batch_not_once_per_row(monkeypatch, tmp_path, writes):
    out = _batch_run(monkeypatch, tmp_path, writes, rows=40, fixed_size=10, save_every=5)
    # four batches, each crossing at least one multiple of 5: one interim write each
    # (the per-row rule would have written eight times), then the final write
    assert _output_writes(writes, out) == [10, 20, 30, 40, 40]


def test_batch_with_large_save_every_writes_only_the_final_file(monkeypatch, tmp_path, writes):
    out = _batch_run(monkeypatch, tmp_path, writes, rows=40, fixed_size=10, save_every=100)
    assert _output_writes(writes, out) == [40]


def test_batch_partial_last_batch_only_saves_if_it_crosses_a_multiple(monkeypatch, tmp_path, writes):
    out = _batch_run(monkeypatch, tmp_path, writes, rows=45, fixed_size=10, save_every=10)
    # batches of 10,10,10,10,5: the last one (40 -> 45) crosses no new multiple of 10
    assert _output_writes(writes, out) == [10, 20, 30, 40, 45]


def test_batch_crossing_several_multiples_writes_once(monkeypatch, tmp_path, writes):
    out = _batch_run(monkeypatch, tmp_path, writes, rows=50, fixed_size=25, save_every=10)
    assert _output_writes(writes, out) == [25, 50, 50]


def test_batch_size_not_dividing_save_every_saves_at_batch_boundaries(monkeypatch, tmp_path, writes):
    out = _batch_run(monkeypatch, tmp_path, writes, rows=60, fixed_size=7, save_every=20)
    # saves land at batch ends (21, 42, 60 completed rows), not at exact multiples
    assert _output_writes(writes, out) == [21, 42, 60, 60]


def test_batch_future_level_failure_still_counts_rows_and_can_save(monkeypatch, tmp_path, writes):
    def make_run_raise(runner):
        def boom(texts):
            raise RuntimeError("the batching layer itself blew up")

        monkeypatch.setattr(runner, "run", boom)

    out = _batch_run(
        monkeypatch, tmp_path, writes, rows=20, fixed_size=10, save_every=10,
        runner_patch=make_run_raise,
    )
    assert _output_writes(writes, out) == [0, 0, 0]  # two batches cross a multiple, plus final
    assert pd.read_csv(out)["c"].isna().all()


def test_batch_and_plain_runs_write_the_same_number_of_times_for_small_batches(
    monkeypatch, tmp_path, writes
):
    plain_out = _plain_run(tmp_path, [f"q{i}" for i in range(60)], writes, save_every=20)
    plain_count = len(_output_writes(writes, plain_out))
    batch_out = _batch_run(monkeypatch, tmp_path, writes, rows=60, fixed_size=7, save_every=20)
    assert len(_output_writes(writes, batch_out)) == plain_count
