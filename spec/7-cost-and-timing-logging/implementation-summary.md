# Implementation Summary: 7-cost-and-timing-logging

**Status:** Completed
**Date:** 2026-09-16
**Worktree:** `/Users/bastianellie/work/elsevier/projects/general/codebase/ds-query-classification-worktrees/7-cost-and-timing-logging` on branch `spec/7-cost-and-timing-logging`

## Overview

Implements per-run cost and timing logging for both entry points (`experiment.py`,
`classify.py`/`cli.py`): total experiment cost, category-induction cost, per-query/per-batch
cost statistics (mean/stddev, exact where possible, estimated under `--critics`/`--models`),
and total wall-clock time, written to a new `cost_report.json` artifact.

## Team Execution

Solo, sequential — matching the plan's Implementation Strategy. No parallel streams were used;
Task 3 (`pipeline.py`) and Task 4 (`batching.py`) share `tests/test_batching.py`, so running
them sequentially (as planned) avoided any file-conflict risk.

**Sequential phases:** Task 1 (`classifier.py` usage capture) → Task 2 (`cost.py` primitives)
→ Tasks 3/4 (`pipeline.py`/`batching.py` wiring) → Tasks 5/6 (`experiment.py`/`cli.py` report
assembly) → Task 7 (docs/ADR).

## Files Created

- `src/query_classification/cost.py` — pricing resolution (`PricingCache`, duplicating
  `batching._candidate_ids`'s walk per FR-2.3), `compute_cost`, `stats_from_samples`,
  `uniform_estimate`, `expand_batch_costs_to_query_samples`, `Timer`, `write_json_atomic`,
  `QueryCostCollector`.
- `tests/test_cost.py` — Task 1 (usage capture/`track_usage()`) and Task 2 (`cost.py`
  primitives) tests.
- `spec/7-cost-and-timing-logging/ADR.md` — INV-1/INV-7 amendments for the new module.

## Files Modified

- `src/query_classification/classifier.py` — `_UsageSink`, module-level thread-local,
  `usage_totals()`, `track_usage()` context manager, `_capture_usage_from_response` hook
  called once in `_attempt_completion` before `_extract_content`. `classify()`'s own call
  signature is untouched everywhere.
- `src/query_classification/batching.py` — `BatchStats.record_cost`/`snapshot()["batch_costs"]`,
  `BatchRunner.__init__`'s `cost_fn` callable, `BatchRunner.run`'s `_top_level` parameter
  (only the top-level first attempt's cost is recorded), `BatchRunner.total_usage()`.
- `src/query_classification/pipeline.py` — `classify_csv`'s `cost_collector`/`cost_per_token`
  parameters; `_classify_tracked` wrapper for plain-mode exact per-row cost;
  `cost_collector.set_rows_attempted()` called unconditionally in every mode.
- `src/query_classification/experiment.py` — pre-try-block initialization of
  `pricing_cache`/`timer`/`induction_classifier`/`classifiers`/`query_cost_collector`;
  induction-classifier capture before reassignment; `_assemble_cost_report()`; report written
  on both success and failure paths; `_write_json_atomic` replaced by `cost.write_json_atomic`
  (used for `run_config.json`/`categories.json`/`sampled_examples.json` too).
- `src/query_classification/cli.py` — mirrors `experiment.py`'s wiring; `_assemble_cost_report()`
  (cli.py's own variant, `induction_cost_usd` always `null`); report written on success only,
  to `{stem}.cost_report.json` via `Path.with_suffix`.
- `tests/test_batching.py`, `tests/test_experiment.py`, `tests/test_cerebus.py` — new tests
  for each task (see Test Results below).
- `spec/ARCHITECTURE.md` — `cost.py` added to the mermaid graph, Module Boundary Map (new row;
  `pipeline.py`/`cli.py`/`experiment.py` rows' import lists edited; `batching.py`'s row
  deliberately unchanged), INV-1 (new module + three inbound edges), INV-7 (`cost` + new
  `classifier._capture_usage_from_response` exception), Stable Contracts (`cost_report.json`
  shape note).
- `README.md` — "Cost and timing" section.

## Test Results

`.venv/bin/python -m pytest -q`: **34 failed, 396 passed** (up from the pre-implementation
baseline of 34 failed, 342 passed — all 34 failures are the identical, pre-existing
`botocore.exceptions.TokenRetrievalError`/`InvalidGrantException` environmental issue from an
expired local AWS SSO session, verified via `diff` against the same file's failure set on a
clean `main` checkout: byte-identical). 54 new tests added across `tests/test_cost.py` (30),
`tests/test_batching.py` (+10), `tests/test_experiment.py` (+8), `tests/test_cerebus.py` (+6).
No existing test was removed or weakened.

## Spec Adherence

| Requirement | Status | Implementation | Test |
|---|---|---|---|
| FR-1.1 | Done | `classifier.py` `_capture_usage_from_response` | `test_capture_usage_from_real_response_updates_instance_total` |
| FR-1.2 | Done | same hook, called before every `_extract_content` return | `test_unsupported_params_retry_reports_only_the_successful_attempts_usage`, `test_retries_across_validation_failures_sum_every_response_producing_attempt` |
| FR-1.3 | Done | `Classifier.track_usage()` | `test_track_usage_sink_reports_only_calls_made_inside_the_with_block`, `test_two_threads_each_see_only_their_own_calls_usage` |
| FR-1.4 | Done | `Classifier.usage_totals()` | `test_capture_usage_from_real_response_updates_instance_total` |
| AR-1.1 | Done | `classify()` signature unchanged; verified via `grep` and `test_wholesale_mocked_classify_wrapped_in_track_usage_observes_none` | same |
| AR-1.2 | Done | `classifier.py` imports unchanged (stdlib/litellm/pydantic only) | N/A (structural) |
| FR-2.1 | Done | `PricingCache.resolve()` | `test_pricing_cache_resolves_same_model_id_only_once` |
| FR-2.2 | Done | `cost.compute_cost` | `test_compute_cost_arithmetic`, `test_compute_cost_none_when_pricing_unresolved` |
| FR-2.3 | Done | `cost._candidate_ids` duplicated from `batching._candidate_ids` | `test_pricing_cache_candidate_walk_matches_batching_module` |
| AR-2.1 | Done | new `cost.py` module | `test_cost.py` (whole file) |
| FR-3.1 | Done | `pipeline._classify_tracked` + `QueryCostCollector` | `test_classify_csv_plain_mode_records_exact_per_row_costs` |
| FR-3.2 | Done | `BatchRunner.run`'s `_top_level` gating + `BatchStats.record_cost` | `test_run_records_cost_of_only_the_first_top_level_attempt`, `test_bisected_batch_only_first_attempts_cost_appears_in_batch_costs` |
| FR-3.3 | Done | `cost.uniform_estimate` + `QueryCostCollector.rows_attempted()` | `test_critics_cost_report_uniform_estimate` |
| FR-3.4 | Done | `_assemble_cost_report`'s per-classifier-then-sum aggregation; induction captured before reassignment | `test_run_end_to_end_cost_report_nonzero_induction_distinct_from_classification` |
| AR-3.1 | Done | `BatchStats`/`BatchRunner` additive changes | `test_total_usage_sums_usage_across_every_cached_arity_classifier`, `test_run_default_top_level_argument_behaves_exactly_as_before` |
| FR-4.1 | Done | `cost.Timer`, wired at `experiment.py`'s `stage = "starting"` and `cli.py`'s post-`load_categories` point | `test_timer_reports_at_least_injected_sleep_duration`, `test_run_end_to_end_cost_report_nonzero_induction_distinct_from_classification` |
| FR-5.1 | Done | `experiment.py`'s `_assemble_cost_report` + `cost.write_json_atomic` | `test_induce_only_cost_report_query_cost_zero_batch_null`, `test_unresolvable_model_cost_report_monetary_fields_null` |
| FR-5.2 | Done | `cli.py` writes `{stem}.cost_report.json` on success only | `test_cli_produces_cost_report_json_sibling_to_output`, `test_cli_failure_before_classify_csv_writes_no_cost_report` |
| FR-5.3 | Done | one `print(f"Cost: ... | Time: ...")` line per successful run | `test_stdout_prints_exactly_one_cost_time_line` |
| AR-5.1 | Done | `pipeline.py`/`cli.py`/`experiment.py` import `cost.py`; `batching.py` does not | ADR-001; `grep` verification |
| AR-5.2 | Done | `cost.write_json_atomic` (temp file + `Path.replace`, `allow_nan=False`) | `test_write_json_atomic_leaves_no_partial_file_on_mid_write_failure`, `test_write_json_atomic_rejects_non_finite_values` |
| FR-6.1 | Done | tests ship alongside every task's code | 54 new tests across 4 files |

## Deviations from Spec

Carried forward from the plan's Spec Deviations table (all "Action required: None"):

1. Feature 6 has no dedicated AR for the INV-7 test-import boundary — this ADR amends INV-7
   anyway, since FR-6.1 is unsatisfiable without it.
2. No FR/AR requires a README change — added a short "Cost and timing" note regardless, since
   a permanent new artifact deserves at least a pointer.
3. A plain-mode row that fails with zero billable attempts is excluded from
   `QueryCostCollector` entirely (not recorded as `None`), to avoid poisoning the whole run's
   `mean_usd` over one row that never reached the provider.
4. `pricing` is reported `null` when a run's classifiers resolve to 2+ distinct model ids
   (`--critics`/`--models` with different role-specific model ids) — a single
   `resolved_input_model`/`resolved_output_model` pair cannot truthfully represent several
   actually-used models.

No other deviations.
