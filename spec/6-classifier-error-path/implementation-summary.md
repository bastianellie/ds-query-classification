# Implementation Summary: 6-classifier-error-path

**Status:** Completed
**Date:** 2026-09-15
**Worktree:** `/Users/bastianellie/work/elsevier/projects/general/codebase/ds-query-classification-worktrees/6-classifier-error-path` on branch `spec/6-classifier-error-path`

## Overview

Replaced `Classifier`'s uniform-retry error path with an explicit two-axis failure
classification (`retryable`, `isolable`), a narrowed and guarded JSON-mode fallback reached
only via ladder exhaustion, ordered response checks that make a truncated/policy-refused/empty
response distinguishable from a malformed one, and a `classify()` retry loop that only retries
what can actually succeed on retry. All seven verified defects from the spec are closed; no
public API, CLI flag, or persisted artifact changed.

## Team Execution

Solo implementation, following `plan.md`'s five sequential tasks exactly (single-file rewrite,
no parallelizable streams).

| Phase | Work |
|---|---|
| Task 1 | `FailureKind`, `classify_failure`/`_classify_failure_verbose`, three response-failure exception types, `_DROPPABLE_PARAMS`, `_log_classified_failure` — all additive |
| Task 2 | `_extract_content`'s four ordered checks; `test_cerebus.py`'s `_FakeChoice`/`_FakeResponse` fixtures gained `finish_reason`/`choices` overrides |
| Task 3 | `_attempt_completion`'s except-chain rewrite (narrowed exclusions, ordered ladder, guarded+chained fallback); one `test_cerebus.py` test replaced by two per FR-2.3 |
| Task 4 | `classify()`'s retry loop now consults `retryable` only; `_log_classified_failure` is the sole logging call site |
| Task 5 | `spec/ARCHITECTURE.md` (INV-7 correction + amendment, `tests/` row, Known Gap removed); `ADR.md` |

**Sequential throughout** — each task's tests passed and the full suite stayed green before the
next began, per the plan's "no expected-red checkpoints" design.

## Files Created

- `tests/test_classifier_errors.py` — 60 tests covering FR-1.1, FR-1.2 (28 parametrized
  per-type cases + 4 hierarchy-trap tests), FR-2.1, FR-2.2, FR-2.4, FR-2.5, FR-3.1, FR-3.2.
- `spec/6-classifier-error-path/ADR.md` — one record (INV-7 amendment + correction).

## Files Modified

- `src/query_classification/classifier.py` — `FailureKind`, `classify_failure`,
  `_classify_failure_verbose`, `TruncatedResponseError`/`PolicyRefusalError`/
  `EmptyResponseError`, `_DROPPABLE_PARAMS`, `_log_classified_failure`, `_extract_content`,
  rewritten `_attempt_completion`/`_fallback_completion` (dropped the `allow_temperature_drop`
  parameter — verified nothing else in the tree referenced it), rewritten `classify()`.
- `tests/test_cerebus.py` — `_FakeChoice` gained `finish_reason`; `_FakeResponse` gained
  `finish_reason`/`choices` override parameters;
  `test_complete_reraises_unsupported_params_error_when_no_temperature_to_drop` replaced by
  `test_complete_reaches_fallback_when_ladder_exhausted_and_fallback_succeeds` and
  `test_complete_chains_fallback_failure_without_a_third_attempt` per FR-2.3. The other two
  existing `_complete` tests are unmodified and still pass — one no longer exercises anything
  new (succeeds before the one-entry ladder is ever exhausted), the other now makes 3 calls
  instead of 2 (ladder exhaustion routes to a fallback attempt that also fails) but was never
  asserting a call count, so it stays green unchanged, exactly as the plan predicted.
- `spec/ARCHITECTURE.md` — INV-7 (accepted-exception list extended, stale "tested only via the
  public API" sentence deleted), the `tests/` Module Boundary Map row, removed the
  "Fallback-trigger scope is broader than 'schema unsupported'" Known Gap (closed by FR-2.1).

## Test Results

- Baseline (before this spec): 229 passed.
- After implementation: **290 passed**, 0 failed, 0 skipped.
- No test invokes a real `litellm.completion`; every LLM call is faked via `monkeypatch`.

## Spec Adherence

| Requirement | Status | Implementation | Test |
|---|---|---|---|
| FR-1.1 | Done | `classifier.py::FailureKind`, `classify_failure` | `test_classify_failure_every_named_type[_mk_pydantic_validation-...]` (both axes), `test_classify_failure_default_for_unrecognized_type` |
| FR-1.2 | Done | `classifier.py::_classify_failure_verbose`'s ordered chain | `test_classify_failure_every_named_type` (28 parametrized cases) + 4 `test_hierarchy_trap_*` tests |
| AR-1.1 | Done | No new module; `classifier.py` imports unchanged (stdlib + litellm + pydantic) | verified via `git diff` — no new import added beyond `NamedTuple`/`ValidationError` |
| AR-1.2 | Done | `spec/6-classifier-error-path/ADR.md` ADR-001; `spec/ARCHITECTURE.md` INV-7 + `tests/` row | N/A (documentation) |
| FR-2.1 | Done | `classifier.py::_attempt_completion`'s explicit exclusion `except` clause | `test_excluded_types_never_reach_fallback[...]` (2 cases), `test_plain_bad_request_error_reaches_fallback` |
| FR-2.2 | Done | `classifier.py::_fallback_completion` (classify, chain, re-raise fallback's own exception) | `test_fallback_failure_is_chained_not_wrapped`, `test_classify_logs_exactly_once_for_a_chained_fallback_failure` |
| FR-2.3 | Done | `_attempt_completion`'s ladder-exhaustion fallthrough | `test_complete_reaches_fallback_when_ladder_exhausted_and_fallback_succeeds`, `test_complete_chains_fallback_failure_without_a_third_attempt` |
| FR-2.4 | Done | `classifier.py::_extract_content`'s 4 ordered checks | 7 tests incl. both overlap cases (`test_overlap_length_with_none_content_...`, `test_overlap_content_filter_with_none_content_...`) |
| FR-2.5 | Done | `_DROPPABLE_PARAMS` + the ladder loop in `_attempt_completion` | `test_ladder_drops_two_entries_in_order_across_three_calls` (explicit 3-call proof), `test_ladder_skips_entry_absent_from_kwargs_without_an_extra_call`, `test_fallback_kwargs_after_exhaustion_contain_no_ladder_params` |
| AR-2.1 | Done | Chain/except ordering places all 5 `BadRequestError` subclasses before the base | `test_hierarchy_trap_bad_request_subclasses_do_not_collapse` + FR-2.1's tests |
| AR-2.2 | Done | `_extract_content` raises instead of returning `None`/indexing an empty list | `test_empty_choices_raises_empty_response`, `test_content_none_with_stop_raises_empty_response` |
| FR-3.1 | Done | `classify()`'s rewritten except block, consulting `kind.retryable` only | `test_classify_retries_retryable_failure_up_to_max_retries`, `test_classify_retries_validation_error_up_to_max_retries` (regression guard), `test_classify_fails_fast_on_non_retryable_failure`, `test_classify_fails_fast_on_truncated_response` |
| FR-3.2 | Done | `_log_classified_failure`, sole call site in `classify()` | `test_log_classified_failure_*` (unit) + `test_classify_logs_authentication_error_...`, `test_classify_logs_unrecognized_type_as_default_branch`, `test_classify_never_logs_configured_secrets` (end-to-end) |
| AR-3.1 | Done | `# noqa: BLE001` + updated comments on `classify`'s and `_fallback_completion`'s broad catches | verified manually (code inspection) |
| FR-4.1 | Done | `tests/test_classifier_errors.py` (60 tests) + `tests/test_cerebus.py` edits, all faked, no live calls | `.venv/bin/python -m pytest -q` — 290 passed |

## Deviations from Spec

None. No Spec Deviations were recorded in `plan.md` requiring implementation-time action beyond
what the plan already specified (its own single "None identified" — the earlier line-number
citation row was removed during plan critique as unfounded).

One minor implementation detail not dictated by the spec, recorded here for completeness:
`_attempt_completion`'s tail was factored into a standalone `_extract_content(response)`
function rather than inlined, to keep FR-2.4's four-step check independently readable. This
function is not part of INV-7's accepted-import list (tests exercise it only indirectly, through
`_attempt_completion`, which the spec did authorize) — a deliberate choice to avoid growing the
test-import surface beyond what the spec's AR-1.2 named.
