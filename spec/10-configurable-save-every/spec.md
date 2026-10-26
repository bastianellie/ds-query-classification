# Spec 10: Configurable Save Frequency (`--save-every`)

> **Status: CLOSED** — Implemented and verified on 2026-10-26.
> Implementation summary: `spec/10-configurable-save-every/implementation-summary.md`
> Merged from worktree branch `spec/10-configurable-save-every` on 2026-10-26.

## Overview

`classify_csv` rewrites the whole output CSV every `save_every` completed rows (default 20), but
the CLI exposes no way to change it. On a 629,058-row input one full rewrite took about 2 s in a
local measurement (210 MB, 9 columns, a warm repeat on one laptop; more with 13 columns or a
cloud-synced folder), so a full run needs about 31,000 rewrites. That is illustrative, not a
benchmark, but it explains why the progress bar appears stuck. In `--batch`
mode the check also runs once per *row* of each batch, so one batch can trigger several full
rewrites back to back. This spec adds a `--save-every N` flag and makes batch mode save at most
once per completed batch.

## Goals

- Let users of very large inputs choose how often the output CSV is rewritten.
- Keep every existing invocation's behavior unchanged (default 20, non-batch cadence untouched).
- Stop `--batch` from rewriting the file several times for one batch.
- Replace the library's `ZeroDivisionError` on `save_every=0` with a clear `ValueError`.

---

## Feature 1: The `--save-every` flag

**Who & why:** Analysts classifying hundreds of thousands of rows are blocked by the default of
20: the run spends most of its time rewriting a 200+ MB file, and a run in a synced folder is
slower still. They need to trade crash-recovery granularity for throughput (for example save
every 5,000 rows) without writing a wrapper script.

### Functional Requirements

#### FR-1.1: `--save-every N` with a default of 20
New flag `--save-every N` (`type=int`, default `20`, `metavar N`) on the `classify.py`/
`python -m query_classification` CLI. Omitting it yields exactly today's cadence. A non-integer
value is rejected by argparse's own `type=int` error.
**Verify:** `build_parser().parse_args([...])` gives `save_every == 20` without the flag and `5000`
with `--save-every 5000`; `--save-every abc` exits with argparse's usage error.

#### FR-1.2: Values below 1 are rejected before any client is built
`--save-every 0` or a negative value raises `ValueError` ("--save-every must be >= 1, got N"),
printed as `Error: ...` with exit code 1. The check runs in `cli.main` immediately after the
`table_io.validate_options` call (before model resolution, resource loading, and any
`Classifier` construction, and before the in-place-overwrite warning). It raises inside
`main`'s existing `try`, so the existing handler prints it and exits 1; an argparse error (FR-1.1)
exits 2.
**Verify:** `cli.main` with `--save-every 0` exits 1 with that message, the output does not
contain the `OVERWRITTEN` warning, and a patched `Classifier.__init__` that records calls is never
invoked.

#### FR-1.3: The flag is passed through to `classify_csv`
`cli.main` passes `save_every=args.save_every` to `classify_csv`. `experiment.py` is unchanged and
keeps its current behavior.
**Verify:** a `cli.main` test with `--save-every 3` and a 7-row input counts the interim and final
CSV writes and sees the cadence of 3 (writes at 3 and 6 completed rows, plus the final write).

#### FR-1.4: The library rejects `save_every < 1`
`classify_csv(..., save_every=0)` (or negative, or any non-integer including `bool`) raises
`ValueError` with the same message verbatim instead of `ZeroDivisionError`. The check is the first
statement of `classify_csv`, before the input is read and before any output write or classifier
call. This supersedes the "still raises `ZeroDivisionError`" note in INV-9.
**Verify:** `classify_csv(..., save_every=0)`, `save_every=2.5`, and `save_every=True` raise
`ValueError`, a fake classifier records zero calls, and no output file is created.

---

## Feature 2: Batch mode saves at most once per batch

**Who & why:** With `--batch dynamic` a batch holds up to 50 rows. The batch branch evaluates the
save condition after every row of every batch, so with the default of 20 one batch triggers two to
three full-file rewrites back to back, which multiplies the cost the user is trying to avoid. They
need at most one rewrite per batch regardless of its size.

### Functional Requirements

#### FR-2.1: Crossing semantics in the batch branch
Batch futures are consumed in completion order (`as_completed`), not input order, so `completed` is
a count of outcomes applied so far, not a contiguous input prefix. After all rows of one
consumed batch future have been applied (row-at-a-time accounting is unchanged: `completed` still
advances by one per row, for successes and for failures, including a future-level exception that is
converted to one failure per covered row), the batch branch writes the CSV once if
`completed // save_every` is greater than its value captured before applying that future's rows.
It never writes more than once per future. Only rows in the selected work set count (see FR-2.4).
**Verify:** with `--batch 10` (fixed), `save_every=5`, and 40 rows, the number of interim CSV writes
is 4 (one per batch), not 8; with `save_every=100` and 40 rows there are zero interim writes; with
45 rows (a partial last batch of 5) and `save_every=10` there is one interim write per batch that
crossed a multiple; a batch whose future raises still counts its rows and can trigger the save.

#### FR-2.2: Non-batch cadence is untouched
The non-batch loops (plain, `--critics`, `--models`) keep the exact `completed % save_every == 0`
check after every row.
**Verify:** a plain run with 10 rows and `save_every=3` has interim writes at 3, 6, and 9 completed
rows, plus the final write.

#### FR-2.3: The final write always happens
Every run, batch or not, ends with the final CSV write (`DataFrame.to_csv` to the output path),
followed by the optional `.xlsx` export of spec 9, regardless of `save_every`. The `.xlsx` is
exported once after the final CSV write and never on interim saves, whatever `save_every` is. When
`save_every` is larger than the number of classifiable rows, the only CSV write is the final one.
**Verify:** a run with 5 rows and `save_every=1000` produces exactly one CSV write to the output
path (writes of fixture files are excluded) and the complete output file; a `cli.main` run with
`--output-format xlsx --save-every 2` over 6 rows writes the `.xlsx` once, after the final CSV
write.

#### FR-2.4: Edge cases keep their current meaning
The save cadence counts only the selected work rows. The work set is built in this order: rows not
already completed (`--restore`), then rows with non-blank text (spec 9 FR-1.6), then the first
`--limit` of those. Blank and already-restored rows never advance `completed`; failed outcomes do.
`--limit` smaller than `save_every` (including `--limit 0`): only the final write. A `--restore`
run restarts the cadence from zero. A batch larger than `save_every` writes once per batch
(FR-2.1).
**Verify:** tests for `limit=3, save_every=10` (one write) and `limit=0` (one write); for a restore
run that classifies 4 remaining rows with `save_every=2` (two interim writes in plain mode); for
an input with blank and already-restored rows ahead of the work rows (only work rows count); and
for `--batch 25` with `save_every=10` over 50 rows (two interim writes).

---

## Feature 3: Documentation, contracts, and tests

**Who & why:** The implementer sees only this spec. README flags and `spec/ARCHITECTURE.md` must
stay in sync with `cli.py`, INV-9 currently documents the opposite of this change, and the safety
trade-off of a large value needs to be written down for users.

### Functional Requirements

#### FR-3.1: README documents the flag and the trade-off
`README.md` gains a `--save-every` row in the Options table and a short note: the default of 20
suits small inputs; for inputs of hundreds of thousands of rows use a larger value such as 5000;
a crash loses the rows completed since the last save (up to N, or up to N plus one batch's rows
in `--batch` mode, because the save happens at a batch boundary) and `--restore` resumes from the
last saved file, provided that file was not itself truncated by a crash mid-write (writes are not
atomic, an existing Known Gap); a
cloud-synced output folder makes each save slower, so writing locally and copying afterwards is
advisable.
**Verify:** the Options table lists `--save-every` and the note mentions the crash-loss bound,
`--restore` with its non-atomic caveat, and the synced-folder advice.

#### FR-3.2: Architecture documents are updated through an ADR
`spec/10-configurable-save-every/ADR.md` records the amendment of INV-9 (the `save_every`
parameter now has a CLI flag and `save_every < 1` raises `ValueError`; INV-9's title and its
"remaining three are unvalidated" wording are updated accordingly) and of spec 5's FR-3.1
flush-count equivalence, which now holds only for batch sizes below `save_every` (see Related
Specs). `spec/ARCHITECTURE.md` is updated to match: INV-9, the
Stable Contracts CLI surface, and the Known Gaps sentence that mentions `save_every`.
**Verify:** the ADR exists and names INV-9; `spec/ARCHITECTURE.md` no longer says that no CLI flag
exposes `save_every`.

#### FR-3.3: Ship unit tests with the code
The implementation ships `tests/test_save_every.py`, run by `pytest`, with at least one test per
**Verify:** condition in this spec and for the edge cases above. Tests fake the LLM (no live calls),
count writes by spying on `pandas.DataFrame.to_csv` (the pattern of
`test_classify_csv_batched_flush_positions_match_unbatched_run` in `tests/test_batching.py`), and
use `cli.main`/`cli.build_parser` as `tests/test_excel_io.py` does. The full existing suite must
still pass unmodified, including `test_classify_csv_batched_flush_positions_match_unbatched_run`
(which asserts only that the numbers of interim writes are equal for batch size 7 over 200 rows;
that holds under FR-2.1). The new tests also check the actual number of completed rows in the saved
file at each interim write for a batch size that does not divide `save_every` (saves now land at
batch boundaries, so the file may hold slightly more rows than the multiple) and for a batch that
crosses several multiples at once. Spies scope to writes whose target is the output path.
**Verify:** `.venv/bin/python -m pytest` passes, and `tests/test_save_every.py` maps to FR-1.1
through FR-2.4.

---

### Architectural Requirements

#### AR-1.1: No new module, no new import edges
The change touches `cli.py` (flag, validation, pass-through) and `pipeline.py` (`classify_csv`
validation and the batch-branch save rule) only. It adds no import, so INV-1's graph is
unchanged.
**Verify:** `git diff` for the implementation touches only `cli.py`, `pipeline.py`, tests, README,
and spec documents, and no new `import` lines appear in either module.

#### AR-1.2: INV-6 and the save mechanism are preserved
All DataFrame mutation stays on the main thread, and the save is still a plain
`df.to_csv(output_path, index=False)` (atomic writes are out of scope and remain an existing Known
Gap). The check for FR-1.2 follows the data-free-check placement used by spec 9 in `cli.main`.
**Verify:** no code outside the main-thread `as_completed` loops writes to `df`; the batch and
non-batch final writes are the same call as before.

---

## Data Requirements

None. No output schema change; the CSV content is identical for any `save_every`.

## Integration Points

- `cli.build_parser` (new flag) and `cli.main` (validation before client construction, pass-through
  to `classify_csv`).
- `pipeline.classify_csv`: the `save_every` parameter (validation) and the `--batch` branch's save
  check inside the per-row outcome loop.

## Related Specs

| Spec | Relationship | Affected Requirements |
|------|-------------|---------------------|
| Spec 5: Batched classification | **Modifies** — FR-3.1 required batched runs to flush as often as unbatched ones; with batch size >= `save_every` a batch now writes at most once | FR-2.1, FR-2.4 |
| Spec 9: Excel input/output | **References** — data-free validation placement in `cli.main`; `.xlsx` export still happens once after the final write | FR-1.2, FR-2.3, AR-1.2 |
| Spec 1: Initial classification with critics | **References** — INV-9 and `--restore` semantics | FR-1.4, FR-2.4 |

## Constraints

- Output data, classifier calls, and non-batch write cadence are unchanged for every existing
  invocation. In `--batch` mode the number of interim writes is unchanged for batch sizes smaller
  than `save_every` (each multiple is crossed in a different batch), but the exact moment of each
  write moves to the end of its batch, so intermediate file contents may differ slightly. For batch
  sizes of at least `save_every`, fewer writes happen by design.
- No live LLM calls in tests (AGENTS.md).

## Out of Scope

- Atomic temp-file-then-rename CSV writes (existing Known Gap).
- Incremental or append-only writes, or any change that avoids rewriting the whole file.
- Time-based save intervals.
- A `--save-every` flag for `experiment.py`.

## Spec Completeness Checklist

- [x] **Scope & acceptance criteria** — scope and non-scope above; every FR has a **Verify:** line.
- [x] **Testing strategy** — FR-3.3 requires `tests/test_save_every.py` mapped to each Verify, with `to_csv` spying and a faked LLM.
- [x] **Existing patterns** — FR-1.2 follows spec 9's data-free check placement; FR-3.3 follows `tests/test_batching.py`'s `to_csv` spy; FR-1.4 mirrors INV-5's constructor-guard style.
- [x] **Dependencies** — none added.
- [x] **Architecture & interfaces** — AR-1.1/AR-1.2; FR-3.2 amends INV-9 through an ADR and notes the spec 5 FR-3.1 change.
- [x] **Error handling & failure modes** — FR-1.1, FR-1.2, FR-1.4 (argparse error, `ValueError`, exit 1); FR-2.3/FR-2.4 edge cases.
- [x] **Security review** — N/A for auth/injection; no new data exposure, the CSV content is unchanged (explicit in Data Requirements).
- [x] **Performance impact** — Overview and FR-2.1: measured ~2 s per 210 MB rewrite; a larger N cuts rewrites from ~31,000 to ~126 at N=5000.
- [x] **Rollout & migration** — default 20 keeps current behavior; the only behavior change is fewer writes for batch size >= `save_every` (Constraints, ADR).
- [x] **Assumptions & risks** — a larger N means a crash loses up to N rows of progress (FR-3.1 documents it); the non-atomic write Known Gap remains; the ~2 s figure is measured on 9 columns.
