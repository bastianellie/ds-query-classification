# Implementation Summary: 1-initial-classification-with-critics

**Status:** Completed
**Date:** 2026-09-08
**Worktree:** `/Users/bastianellie/work/elsevier/projects/general/codebase/ds-query-classification-worktrees/1-initial-classification-with-critics` on branch `spec/1-initial-classification-with-critics`

## Overview

Implements the opt-in `--critics` mode: each row is sampled multiple times against
the initial classifier, votes are tallied per category, and categories without
consensus are escalated to a Critic (devil's advocate) and, if it raises a real
challenge, a Reconciler — with a full audit trail written to new CSV columns.
Entirely opt-in; the existing single-classification path is unchanged when
`--critics` isn't passed.

## Team Execution

Solo implementation — no team/task-tracking tools (`TeamCreate`/`TaskCreate`) were
available in this environment, and the plan's design is tightly interlocking
(Task 3's prompt fix, Task 2's validator, Task 4's consumption of both), so
delegating pieces to subagents lacking that context would have cost more in
re-explaining than it saved in wall-clock time. Followed the plan's own task
order and parallelism annotations conceptually (Tasks 1/2/3/5 have no
inter-dependencies) while executing them sequentially myself.

**Sequential phases:** Task 1 (classifier.py) → Task 2 (schema.py) → Task 3
(prompts.py/resources.py/templates) → Task 5 (ADR) → Task 4 (debate.py) → Task 6
(pipeline.py) → Task 7 (cli.py/README) → Task 8 (tests/test_debate.py). Deviated
from the plan's stated Task 4-before-Task-5 ordering only in *when I personally
typed the ADR* (wrote it right after Tasks 1-3, before Task 4's code) — the ADR's
content doesn't depend on Task 4's code existing (only its design, already fixed
by the plan), so this matches the plan's own stated rationale for making Task 5
independent, not a deviation from it.

## Files Created

- `src/query_classification/debate.py` — sampling/voting/routing/Critic/Reconciler orchestration (AR-1.2)
- `resources/prompts/critic_prompt.txt`, `resources/prompts/reconciler_prompt.txt` — bundled role prompt templates (AR-2.2)
- `spec/1-initial-classification-with-critics/ADR.md` — architecture-graph and test-import-boundary decision record (AR-1.6, AR-1.7)
- `tests/test_debate.py` — 30 new tests (FR-2.9)

## Files Modified

- `src/query_classification/classifier.py` — optional `temperature` param (AR-1.1)
- `src/query_classification/schema.py` — `allow_new_labels` param on `build_classification_model`; new `build_critic_model`/`build_reconciler_model` with a challenges/proposed_label validator (FR-1.2, AR-2.1)
- `src/query_classification/prompts.py` — `allow_new_labels` param on `build_system_prompt`; new `build_critic_prompt`/`build_reconciler_prompt` (FR-1.2, AR-2.2)
- `resources/prompts/system_prompt.txt` — hardcoded invention instruction replaced with `{invention_policy}` placeholder (FR-1.2 — the critical fix; a schema-only change would not have worked)
- `src/query_classification/resources.py` — `DEFAULT_CRITIC_PROMPT_FILE`/`DEFAULT_RECONCILER_PROMPT_FILE` constants (INV-2)
- `src/query_classification/pipeline.py` — `critics`/`critic_classifiers`/`reconciler_classifiers`/`sampling_runs`/`consensus_threshold`/`allow_new_labels` params; collision/creation/reset/restore-copy/completeness logic extended to audit columns (AR-1.3, AR-1.5, FR-1.6, FR-1.7, FR-2.6)
- `src/query_classification/cli.py` — 7 new flags, sampling/critic/reconciler `Classifier` construction, resource-guard scope fix (FR-1.7, FR-2.4, FR-2.5)
- `README.md` — new flags documented (plus 2 pre-existing undocumented ones, `--api-base`/`--workers`), `--critics` usage example, `debate.py` added to Project layout
- `tests/test_building_blocks.py` — extended for the new bundled resources and the `allow_new_labels` prompt/schema toggle
- `.gitignore` — fixed a pattern bug from an earlier session (critique files without an `init-`/`plan-` prefix weren't actually being ignored)

## Test Results

`.venv/bin/python -m pytest -v`: **36 passed**, 0 failures (6 pre-existing + 30 new
in `tests/test_debate.py`). No live network calls anywhere in the suite.

Additionally verified manually (not via pytest, per the plan's ad-hoc-verification-
during-implementation approach) and caught two real bugs before they reached the
test suite:
1. A `pandas.errors.LossySetitemError` when resetting audit columns on a fresh
   non-restore run, if a prior run's CSV had already inferred a plain (non-
   nullable) `bool` dtype for `_challenged`/`_reconciled` — fixed by casting to
   `object` dtype before assigning `None` (pipeline.py).
2. My own first draft of the `allow_new_labels` toggle test only changed the
   system-prompt half of the constraint, not the schema half — a real
   integration gap between Task 2 and Task 3 that a less literal test would have
   missed.

## Spec Adherence

| Requirement | Status | Implementation | Test |
|---|---|---|---|
| FR-1.1 | Done | `debate.py::run_debate` (sampling loop); `pipeline.py` (conditional submission) | `test_sampling_runs_calls_classifier_n_times` |
| FR-1.2 | Done | `schema.py::build_classification_model(allow_new_labels=)`; `prompts.py::build_system_prompt(allow_new_labels=)`; `system_prompt.txt`'s `{invention_policy}` | `test_vote_tally_on_top_label`, `test_build_system_prompt_allow_new_labels_toggle` |
| FR-1.3 | Done | `debate.py::run_debate` (consensus bypass) | `test_consensus_bypass_skips_critic` |
| FR-1.4 | Done | `debate.py::run_debate` (leading/runner-up, no-runner-up fallback) | `test_escalation_includes_leading_and_runner_up_in_critic_message`, `test_no_runner_up_fallback_when_too_few_distinct_candidates` |
| FR-1.5 | Done | `debate.py::_vote_bucket` | `test_allow_new_labels_true_buckets_invented_suggestions_together`, `test_allow_new_labels_false_keeps_invented_suggestions_distinct` |
| FR-1.6 | Done | `debate.py::run_debate` (all-failed raise); `pipeline.py` (stale reset) | `test_all_samples_failed_raises`, `test_partial_sample_failures_still_produce_a_tally`, `test_stale_audit_values_reset_on_fresh_non_restore_run` |
| FR-1.7 | Done | `pipeline.py` (sampling_runs/consensus_threshold); `cli.py` (sampling_temperature) | `test_invalid_consensus_threshold_raises_before_any_call`, `test_cli_invalid_sampling_temperature_exits_before_any_llm_call` |
| FR-2.1 | Done | `debate.py::_debate_one_category` | `test_escalation_includes_leading_and_runner_up_in_critic_message`, `test_none_top_label_is_challengeable`, `test_critic_decline_preserves_leading_and_records_rationale` |
| FR-2.2 | Done | `debate.py::_debate_one_category` | `test_critic_challenge_triggers_reconciliation` |
| FR-2.3 | Done | `debate.py::_debate_one_category` | `test_critic_failure_degrades_to_leading_with_sanitized_error`, `test_reconciler_failure_degrades_to_leading_but_records_challenge` |
| FR-2.4 | Done | `cli.py::main` (critic/reconciler model routing) | `test_cli_default_and_overridden_model_routing` |
| FR-2.5 | Done | `pipeline.py` (column creation gated on `critics`) | `test_full_audit_trail_columns_present_only_with_critics` |
| FR-2.6 | Done | `pipeline.py` (restore-copy + completeness check extended) | `test_restore_requires_votes_column_not_just_category_value`, `test_restore_from_separate_completed_output_skips_done_rows` |
| FR-2.8 | Done | `debate.py::_sanitize_error` | `test_critic_failure_degrades_to_leading_with_sanitized_error`, `test_reconciler_failure_degrades_to_leading_but_records_challenge` |
| FR-2.9 | Done | `tests/test_debate.py` (this file, 30 tests) | itself |
| AR-1.1 | Done | `classifier.py::Classifier.__init__`/`_completion_kwargs` | `test_sampling_temperature_included_in_completion_kwargs` |
| AR-1.2 | Done | `debate.py::run_debate` (flat dict return) | all `debate.run_debate` tests |
| AR-1.3 | Done | `pipeline.py::classify_csv` signature + submission branch | `test_full_audit_trail_columns_present_only_with_critics` + 6 unmodified baseline tests |
| AR-1.4 | Done | `debate.py` (context-manager `ThreadPoolExecutor`, both sampling and per-category debate loops) | `test_result_correct_regardless_of_completion_order`, `test_multiple_escalated_categories_debate_concurrently` |
| AR-1.5 | Done | `pipeline.py` (collision check extended to audit columns) | `test_category_name_collides_with_generated_audit_column_raises` |
| AR-1.6 | Done | `spec/1-initial-classification-with-critics/ADR.md` ADR-001 | verified by inspection (before/after INV-1 wording, new Module Boundary Map row) |
| AR-1.7 | Done | `ADR.md` ADR-002 | exercised directly by every `test_debate.py` import of `debate`/`build_critic_model`/etc. |
| AR-2.1 | Done | `schema.py::build_critic_model`/`build_reconciler_model` | `test_build_critic_model_validator`, `test_build_reconciler_model_enforces_label_bounds` |
| AR-2.2 | Done | `prompts.py::build_critic_prompt`/`build_reconciler_prompt`; new template files | `test_build_critic_and_reconciler_prompts_are_nonempty`, `test_bundled_resources_exist` |
| AR-2.3 | Done | `cli.py` constructs plain `Classifier` instances for Critic/Reconciler roles; `debate.py` only calls `.classify()` | verified by code inspection (no new retry/fallback code in `debate.py`) + all Critic/Reconciler tests |
| AR-2.4 | Done | `debate.py::AUDIT_COLUMN_SUFFIXES`, consumed by `pipeline.py::_audit_columns` | `test_full_audit_trail_columns_present_only_with_critics`, `test_category_name_collides_with_generated_audit_column_raises` |
| AR-2.5 | Done | `debate.py` (`json.dumps` for `{cat}_votes`) | every test asserting via `json.loads(result["sentiment_votes"])` |

## Deviations from Spec

None from the spec itself. One deviation from the plan's exact assumption, not the
spec: `debate.py` imports only `categories.py` and `classifier.py`, not
`schema.py`/`prompts.py` as the plan's Task 4 assumed — the per-call Critic/
Reconciler user messages are assembled from plain `Category`/`Label` fields
directly, with no need to call into `schema.py`/`prompts.py` at runtime (those are
only used once, upfront, in `cli.py`, to construct each category's `Classifier`).
`ADR.md`'s authorized import set (`categories.py`+`classifier.py`+`schema.py`+
`prompts.py`) is therefore a strict superset of what's actually used — an
intentional, safe ceiling rather than an inaccuracy, since "may import from" is a
permission, not a requirement. No spec or architecture change needed.

Also fixed (not a deviation, but worth flagging): the `spec/1-initial-classification-
with-critics/` directory's `spec.md`/`plan.md`/critique files existed only as
*untracked* files in the main checkout before this implementation started — they
were never part of any git history. Copied `spec.md`/`plan.md` into this worktree
and committed them alongside the code (so the design record travels with the
implementation it produced); left the critique files out of version control,
matching the project's established preference for that file class, and fixed a
`.gitignore` pattern bug along the way (the same fix is still needed in the main
checkout — flagged in the final report, not applied there since that's outside
this worktree).
