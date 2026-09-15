# Implementation Summary: 5-batched-classification

**Status:** Completed
**Date:** 2026-09-15
**Worktree:** `/Users/bastianellie/work/elsevier/projects/general/codebase/ds-query-classification-worktrees/5-batched-classification` on branch `spec/5-batched-classification`

## Overview

Implements `--batch`: batching multiple queries into one classification call
(dynamic, token-budget-sized, or fixed-size), cutting LLM call volume roughly
by the batch size. Built on spec 6's already-shipped failure-classification
primitives (`classify_failure`/`FailureKind`, `TruncatedResponseError`,
`_DROPPABLE_PARAMS`) rather than re-deriving litellm-hierarchy knowledge — a
new `batching.py` module owns token-budget resolution, batch sizing, payload
assembly, the arity-keyed classifier cache, and the bisection-on-failure
routine; `pipeline.py`, `cli.py`, and `experiment.py` each gain a thin
integration layer around it.

## Team Execution

Solo, sequential — no parallel work streams (the plan's own analysis found no
independent file-touch sets; every task built on the previous one's shipped
interface).

**Sequential phases:** Task 1 (schema builder + `Classifier.max_tokens`) →
Task 2 (`batching.py` budget resolution/sizing/payload) → Task 3
(`BatchRunner`/`BatchStats`/bisection) → Task 4 (`pipeline.classify_csv`
integration) → Task 5a (`experiment.py` CLI wiring) → Task 5b (`cli.py` CLI
wiring, mirroring 5a) → Task 6 (docs/scripts/ARCHITECTURE/ADR).

## Files Created

- `src/query_classification/batching.py` - Token-budget resolution
  (`resolve_token_budgets`, `_candidate_ids`), batch sizing (`plan_batch`,
  `BatchPlan`), payload assembly (`_render_payload`,
  `_measure_rendered_payload`, output-estimate helpers), the arity-keyed
  classifier cache and batched call with bisection (`BatchRunner`), and
  thread-safe outcome counters (`BatchStats`).
- `tests/test_batching.py` - Unit tests for all of the above plus
  `classify_csv`'s batch integration (50 tests).
- `spec/5-batched-classification/ADR.md` - INV-1 (new module + 3 inbound
  edges) and INV-7 (3 new test-import names) amendments.

## Files Modified

- `src/query_classification/schema.py` - `build_batch_model(row_model, n)`:
  strict positional batch response schema (FR-2.1/AR-2.1/AR-2.2).
- `src/query_classification/classifier.py` - `Classifier.__init__` gains
  `max_tokens: int | None = None`, emitted conditionally by
  `_completion_kwargs`; `_DROPPABLE_PARAMS` gains a prepended `"max_tokens"`
  entry (FR-2.5). No control-flow changes — spec 6's ladder/truncation
  detection already applies unconditionally.
- `src/query_classification/pipeline.py` - `classify_csv` gains an optional
  `batch_runner` parameter; when set, batches (from
  `BatchRunner.iter_batches`) are submitted to the existing
  `ThreadPoolExecutor` instead of rows, and each batch's per-row outcomes are
  applied one at a time on the main thread, preserving the existing
  `completed`/`save_every` flush cadence exactly. Mutually exclusive with
  `classifier`/`models`/`critics`.
- `src/query_classification/experiment.py` - Four new flags
  (`--batch`/`--batch-max-size`/`--batch-max-input-tokens`/
  `--batch-max-output-tokens`) on the `classification` parent group;
  `_resolve_batch_sizing`/`_resolve_batch_max_size` resolvers; validation
  matrix and token-budget resolution inside `_validate_args` (before any LLM
  call); `_construct_classifiers` builds a `BatchRunner`/`BatchStats`/
  classifier factory instead of a single `Classifier` when `--batch` is
  active; `_SCHEMA_VERSION` bumped 4→5; `run_config.json` gains a `batch`
  block (`_initial_batch_config`, overwritten from the stats snapshot after
  `classify_csv` returns); the comparability caveat prints once.
- `src/query_classification/cli.py` - Same four flags, resolvers, validation
  matrix, diagnostics, and factory construction as `experiment.py`
  (duplicated per INV-1 — the two entry points share no code).
- `tests/test_experiment.py` - 27 new tests for the CLI-level `--batch`
  surface (resolver shapes, validation matrix, FR-1.3's diagnostics, `run_config.json`
  provenance, the caveat, `--test-limit`+`--batch`); the pre-existing
  `schema_version == 4` assertion updated to `5`.
- `tests/test_cerebus.py` - 9 new tests for `cli.py`'s `--batch` surface,
  including the Cerebus gateway carry-through into the batch classifier
  factory.
- `README.md` - New "Batched classification" section, four new Options rows,
  `batching.py` in the project layout, `--batch` mentioned in the Experiment
  runner section.
- `experiments/{ag-news,trec,pubmed-rct}/run_experiment.sh` - `BATCH`/
  `CRITICS` env vars; the previously-hardcoded `--critics` trio is now
  conditional on `CRITICS` (default on); setting `BATCH` requires
  `CRITICS=0`, enforced both for the env var and for a literal `--critics`
  passed through in `"$@"`. Argv token order is unchanged when no variables
  are set (verified against a pre-edit baseline).
- `experiments/{ag-news,trec,pubmed-rct}/README.md` - `BATCH`/`CRITICS`
  documented.
- `spec/ARCHITECTURE.md` - `batching.py` added to the mermaid graph, Module
  Boundary Map, and INV-1; INV-7's `tests/` exceptions and evidence gain this
  spec's own new names, alongside (not duplicating) spec 6's.

## Test Results

`.venv/bin/python -m pytest -q`: **376 passed**, 0 failed (baseline before
this spec: 290 passed, 21 pre-existing environmental failures in
`tests/test_experiment.py` from an expired local AWS SSO token — resolved on
its own mid-session and confirmed unrelated to this spec's changes; the same
269+21 baseline was reproduced identically in this worktree before Task 1
began).

Manual verification beyond `pytest`:
- `experiment.py run --help` / `classify.py --help`: all four flags present;
  `experiment.py induce --help`: none present.
- `bash -n` passes on all three experiment scripts.
- No-vars-set `DRY_RUN` argv identical to the pre-Task-1 baseline (captured
  before any script edit), after normalizing `--run-dir`'s embedded
  timestamp, for all three scripts.
- `BATCH=4 CRITICS=0 DRY_RUN=1` → `--batch 4`, no `--critics`; `BATCH=4
  DRY_RUN=1` alone → exits 1 naming `BATCH`/`CRITICS`; `CRITICS=0 DRY_RUN=1`
  alone → neither `--critics` nor `--batch`; `CRITICS=0 BATCH=4 DRY_RUN=1
  ./run_experiment.sh --critics` → exits 1 via the `"$@"` scan.
- Live-verified against the installed litellm (1.101.0): the FR-1.2 chunk-down
  candidate walk, `AR-1.3`'s bare `Exception` on an unmapped model, and
  `AR-2.1`'s single-`minItems` claim.
- Manual no-batch smoke test: a faked-LLM 50-row run with `save_every=10`
  flushes at rows 10/20/30/40/50 (plus the final flush), matching the
  unbatched path's existing cadence exactly (the unbatched code path itself
  was never touched by this spec, and the full suite's 340+ unbatched-path
  tests stayed green through every task commit).

## Spec Adherence

| Requirement | Status | Implementation | Test |
|---|---|---|---|
| FR-1.1 | Done | `experiment.py::_resolve_batch_sizing`, `cli.py::_resolve_batch_sizing` | `test_batch_resolver_three_shapes`, `test_batch_resolver_rejects_invalid_values`, `test_cli_batch_flags_present_on_parser` |
| FR-1.2 | Done | `batching.py::resolve_token_budgets`/`_candidate_ids` | `test_resolve_token_budgets_chunk_down_openai_prefixed_workspace_slug`, `_workspace_slug_without_openai_prefix`, `_chunks_down_to_a_real_model`, `_total_failure_resolves_neither_side`, `_asymmetric_input_known_output_missing` |
| FR-1.3 | Done | `experiment.py::_validate_args`, `cli.py::main` (fatal message + diagnostics) | `test_batch_dynamic_unresolvable_input_budget_is_fatal_with_zero_llm_calls`, `_prints_verbatim_sentence`, `_unresolvable_output_budget_warns_and_proceeds`, `test_cli_batch_dynamic_unresolvable_input_is_fatal` |
| FR-1.4 | Done | `resolve_token_budgets`'s override params; `--batch-max-*` flags | `test_resolve_token_budgets_override_skips_walk_for_that_side_only`, `test_batch_max_input_tokens_override_sizes_batches`, `test_batch_sizing_flags_require_batch`, `test_cli_batch_max_size_without_batch_rejected` |
| FR-1.5 | Done | `plan_batch`'s `target_before_budget` cap; CLI validation | `test_plan_batch_dynamic_cap_enforced_and_never_measures_beyond_cap`, `test_plan_batch_max_size_10_caps_dynamic_arity`, `test_batch_int_exceeding_max_size_rejected`, `test_cli_batch_exceeding_max_size_rejected` |
| FR-1.6 | Done | `plan_batch` (dynamic growth + exact-measurement reduction), `_measure_rendered_payload` | `test_plan_batch_dynamic_reduces_when_additive_estimate_underestimates_real_payload`, `test_plan_batch_oversized_single_row_warns_and_sends_alone` |
| FR-1.7 | Done | `plan_batch` (fixed-mode trim/floor-of-1) | `test_plan_batch_fixed_mode_trims_below_int_when_budget_exceeded`, `_full_int_when_budget_not_exceeded`, `_short_final_iteration_is_not_a_trim` |
| AR-1.1 | Done | `src/query_classification/batching.py` (new module); `pipeline.py` imports it | `spec/5-batched-classification/ADR.md` ADR-001 (INV-1) |
| AR-1.2 | Done | `resolve_token_budgets` sets `litellm.suppress_debug_info = True` | verified live (module docstring/comment cites the behavior) |
| AR-1.3 | Done | `resolve_token_budgets`'s `except Exception` per candidate (`# noqa: BLE001`) | `test_resolve_token_budgets_total_failure_resolves_neither_side` |
| FR-2.1 | Done | `schema.py::build_batch_model` | `test_build_batch_model_accepts_exactly_n_results`, `_raises_missing_for_short_response`, `_raises_extra_forbidden_for_long_response`, `_raises_extra_forbidden_for_unexpected_nested_field`, `test_build_classification_model_output_still_ignores_unexpected_fields` |
| FR-2.2 | Done | `batching.py::_render_payload` | `test_render_payload_adversarial_text_survives_as_a_single_string_value` |
| FR-2.3 | Done | `experiment.py`/`cli.py` build the batch prompt from the single-row model + `_BATCH_PROMPT_SUFFIX` | `test_batch_prompt_contains_task_description_and_independence_instruction` |
| FR-2.4 | Done | `BatchRunner._get_classifier` (lock-protected cache), `classifier_factory` closures in `experiment.py`/`cli.py` | `test_concurrent_misses_build_exactly_one_classifier_per_arity`, `test_classifier_factory_cerebus_settings_survive_into_cached_classifier`, `test_cli_batch_classifier_carries_cerebus_gateway_headers_and_max_tokens` |
| FR-2.5 | Done | `Classifier.max_tokens`, `_DROPPABLE_PARAMS = ("max_tokens", "temperature")` | `test_classifier_completion_kwargs_omits_max_tokens_by_default`, `_includes_max_tokens_when_set`, `test_droppable_params_drops_max_tokens_before_temperature` |
| AR-2.1 | Done | `build_batch_model`'s positional (not `list`) fields | `test_build_batch_model_strict_serialization_has_one_min_items` |
| AR-2.2 | Done | `build_batch_model` in `schema.py`, alongside `build_classification_model`/`build_induction_model` | `test_build_batch_model_does_not_mutate_row_model`, `test_build_batch_model_zero_raises_value_error` |
| AR-2.3 | Done | `batching.py`'s own `_DATA_START`/`_DATA_END` copies | `test_delimiters_match_induction_py` |
| FR-3.1 | Done | `pipeline.classify_csv`'s `have_batch` branch (batches submitted to `ThreadPoolExecutor`, applied row-at-a-time) | `test_classify_csv_batched_produces_expected_batch_count_and_preserves_order`, `test_classify_csv_batched_flush_positions_match_unbatched_run`, `test_classify_csv_batch_future_raising_leaves_accounting_consistent` |
| FR-3.2 | Done | `BatchRunner.run` (isolable→bisect, retryable→bounded retry, neither→fail), via `classifier.classify_failure` | `test_run_isolates_exactly_one_bad_row_via_validation_error`, `_all_failing_batch_of_4_makes_exactly_7_calls_and_4_failures`, `_context_window_exceeded_bisects_with_no_fallback_attempt`, `_rate_limit_retries_bounded_no_split`, `_authentication_error_fails_immediately_no_split_no_retry`, `_single_row_failure_returns_exception_without_raising` |
| FR-3.3 | Done | `classify_csv`'s mutual-exclusion checks; `experiment.py`/`cli.py`'s validation matrix | `test_classify_csv_batch_runner_with_critics_raises_value_error`, `_with_models_raises_value_error`, `test_batch_with_critics_rejected`, `_with_resolved_multi_models_rejected`, `_with_n_classifiers_rejected`, `_with_single_value_models_is_accepted`, `test_cli_batch_and_critics_rejected`, `test_cli_batch_with_models_2plus_rejected`, `test_cli_batch_single_value_models_accepted` |
| FR-3.4 | Done | `pipeline.classify_csv` forms batches from post-`--restore`-filtered `work_idx` | `test_classify_csv_restore_classifies_only_remaining_rows` |
| FR-4.1 | Done | `experiment.py::_initial_batch_config`, `BatchStats.snapshot()`, post-`classify_csv` overwrite | `test_batch_run_config_schema_version_5_and_null_when_no_batch`, `_records_call_count_and_arity_range`, `test_batch_classify_less_run_has_batch_null`, `test_batch_failed_run_still_writes_batch_key_with_null_observed_fields` |
| FR-4.2 | Done | one `print()` in `experiment.py`/`cli.py`, guarded by `batch_runner is not None` | `test_batch_caveat_printed_exactly_once`, `_no_batch_run_prints_no_caveat`, `test_cli_batch_caveat_printed_once` |
| FR-4.3 | Done | `README.md`, three `run_experiment.sh` scripts, `spec/ARCHITECTURE.md` | manual: `bash -n`, `DRY_RUN` argv identity (see Test Results) |
| FR-5.1 | Done | `tests/test_batching.py` (50 tests), additions to `tests/test_experiment.py` (27) and `tests/test_cerebus.py` (9) | this table |
| AR-5.1 | Done | `spec/5-batched-classification/ADR.md` ADR-001 (INV-7) | direct imports in `tests/test_batching.py` of `batching`'s functions/`_DATA_START`/`_DATA_END`, `induction._DATA_START`/`_DATA_END`, `schema.build_batch_model` |

## Deviations from Spec

None. The Spec Deviations table in `plan.md` records "None identified," and
no deviation was introduced during implementation.
