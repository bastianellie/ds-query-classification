# Architectural Decision Records: 10-configurable-save-every

## ADR-001: `save_every` gets a CLI flag and is validated (amends INV-9)
- **Date:** 2026-10-26
- **Status:** Accepted
- **Context:** INV-9 recorded that the library-only `save_every` parameter had no CLI flag and raised `ZeroDivisionError` for `0`. On very large inputs the default of 20 means a full-file rewrite every 20 rows, which dominates run time, so users need to set it.
- **Decision:** Expose `--save-every N` (default 20). Validate with `pipeline.validate_save_every`: anything other than a positive `int` (`bool` excluded) raises `ValueError("--save-every must be >= 1, got ...")`, called in `cli.main` right after the other data-free checks and as the first statement of `classify_csv`.
- **Rationale:** Keeps today's behavior by default, fails fast and clearly instead of a crash deep in the loop, and reuses the data-free-check placement introduced by spec 9.
- **Consequences:** INV-9's title and `save_every` bullet are updated; `experiment.py` is unchanged and keeps the library default.
- **Invariants affected:** INV-9

## ADR-002: Batch mode saves at most once per consumed batch (amends spec 5 FR-3.1)
- **Date:** 2026-10-26
- **Status:** Accepted
- **Context:** Spec 5 FR-3.1 required row-at-a-time accounting so that the `completed % save_every == 0` flush check fired exactly as often as in an unbatched run. In batch mode that check ran after every row, so one batch of up to 50 rows could trigger several back-to-back full-file rewrites.
- **Decision:** Keep row-at-a-time accounting (`completed`, progress, failure counts unchanged) but evaluate the save once per consumed batch future: write if `completed // save_every` increased while applying that batch's rows. The final write is unchanged.
- **Rationale:** The number of interim writes is identical to the unbatched run whenever the batch size is smaller than `save_every` (each multiple is crossed in a different batch), so FR-3.1's equivalence still holds there; for larger batches it intentionally drops redundant rewrites.
- **Consequences:** Interim writes land at batch boundaries, so an intermediate file can hold slightly more rows than the multiple of `save_every`; the crash-loss bound in batch mode is up to `save_every` plus one batch's rows. The existing count-only test `test_classify_csv_batched_flush_positions_match_unbatched_run` is unchanged and passes.
- **Invariants affected:** None (INV-6 is preserved: all DataFrame mutation stays on the main thread)
