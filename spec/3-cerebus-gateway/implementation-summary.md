# Implementation Summary: 3-cerebus-gateway

**Status:** Completed
**Date:** 2026-09-09
**Worktree:** N/A — implemented directly in the main checkout, then formalized via
`/spec-write` after the fact (this spec was written retroactively; there was no
`/spec-plan`/`/spec-implement` cycle in the usual order).

## Overview

Implemented an opt-in `--cerebus` flag on both `classify.py`/`cli.py` and
`experiment.py` that routes every LLM call through Cerebus (an internal Portkey-based
gateway) instead of a direct provider, with credential resolution from `CEREBUS_API_KEY`
falling back to AWS Secrets Manager. Written first as an ad hoc addition (flagged to the
user per `AGENTS.md`'s "do not implement unspecced features without flagging it" rule),
then formalized via `/spec-write` at the user's request. The spec-write critique pass
(Claude + Codex) found and fixed three genuine problems in the as-shipped code: an
unvalidated `--api-base` override could leak the gateway credential to an arbitrary
endpoint, an error path appended raw (unsanitized) exception text, and the test suite
violated `INV-7`'s test-import-boundary rule — all three fixed before this summary was
written, so "Completed" reflects the corrected implementation, not the original draft.

## Team Execution

Solo (no worktree, no subagents) — implemented directly, then critiqued and corrected in
the same session per `/spec-write`'s standard flow (Claude in-session critique + Codex in
background).

## Files Created

- `tests/test_cerebus.py` — 32 tests
- `spec/3-cerebus-gateway/spec.md`, `ADR.md`, `critique-v-1-claude.md`,
  `critique-v-1-codex.md`, `critique-consolidated-v-1.md`

## Files Modified

- `src/query_classification/classifier.py` — `_resolve_cerebus_api_key`,
  `build_cerebus_completion_kwargs`, `cerebus_model_id`,
  `reject_insecure_cerebus_endpoint`; `Classifier` gains `api_key`/`extra_headers`
- `src/query_classification/cli.py` — `--cerebus` flag; gateway kwargs resolved once in
  `main()`, applied to every `Classifier` construction
- `src/query_classification/experiment.py` — `--cerebus` flag on the shared `common`
  group; `_resolve_gateway_kwargs` resolves once in `main()`, threaded through both
  `_construct_classifiers` calls; `run_config.json` gains a `cerebus` field
- `pyproject.toml` — `[project.optional-dependencies] cerebus = ["boto3"]`
- `.env.example`, `.env` — new `CEREBUS_*` placeholder variables
- `README.md` — new "Cerebus / Portkey gateway" section, Options table row

## Test Results

`.venv/bin/python -m pytest -q` → **136 passed** (104 pre-existing + 32 new). A
mutation-testing spot check (breaking `--reconciler-model`'s `openai/` prefixing)
confirmed the new end-to-end wiring tests genuinely fail on that regression.

## Spec Adherence

| Requirement | Status | Implementation | Test |
|---|---|---|---|
| FR-1.1 | Done | `cli.py`/`experiment.py` `build_parser()` | `test_cli_parser_cerebus_defaults_false`, `test_experiment_parser_cerebus_defaults_false` |
| FR-1.2 | Done | `classifier.build_cerebus_completion_kwargs` | `test_build_kwargs_requires_valid_mode`, `test_build_kwargs_azure_mode_*`, `test_build_kwargs_direct_mode_*` |
| FR-1.3 | Done | `classifier.cerebus_model_id`, `_model_id()` wrappers | `test_cerebus_model_id`, `test_cli_critics_run_prefixes_and_shares_headers_across_every_role`, `test_experiment_run_prefixes_and_shares_headers_across_every_role` |
| FR-1.4 | Done | `classifier.reject_insecure_cerebus_endpoint`, both `main()`s | `test_reject_insecure_cerebus_endpoint`, `test_cli_main_rejects_insecure_api_base_override_under_cerebus`, `test_experiment_main_rejects_insecure_api_base_override_under_cerebus` |
| FR-1.5 | Done | `experiment._resolve_gateway_kwargs` (resolved once, threaded through) | `test_cli_critics_run_prefixes_and_shares_headers_across_every_role`, `test_experiment_run_prefixes_and_shares_headers_across_every_role` |
| FR-1.6 | Done | Both `main()`s' early validation phase | `test_cli_main_reports_clean_error_on_missing_cerebus_mode`, `test_experiment_main_reports_clean_error_on_missing_cerebus_mode` |
| AR-1.1 | Done | `Classifier.__init__`/`_completion_kwargs` | `test_classifier_completion_kwargs_omit_gateway_fields_by_default`, `test_classifier_completion_kwargs_include_gateway_fields_when_set` |
| AR-1.2 | Done | `classifier.build_cerebus_completion_kwargs` | covered by FR-1.2's tests |
| AR-1.3 | Done | `classifier.py` remains import-free of siblings | verified via code inspection (no ADR needed) |
| AR-1.4 | Done | `spec/3-cerebus-gateway/ADR.md` | N/A (doc) |
| FR-2.1 | Done | `classifier._resolve_cerebus_api_key` | `test_resolve_key_from_env_var_skips_aws` |
| FR-2.2 | Done | same | `test_resolve_key_falls_back_to_aws_secrets_manager`, `test_resolve_key_uses_documented_default_aws_coordinates`, `test_resolve_key_passes_through_overridden_aws_coordinates` |
| FR-2.3 | Done | same | `test_resolve_key_missing_boto3_raises_actionable_error`, `test_resolve_key_missing_transitive_dependency_is_not_misreported_as_boto3` |
| FR-2.4 | Done | `classifier._cerebus_key_help_message` | `test_resolve_key_aws_failure_raises_clear_error_with_remediation`, `test_resolve_key_missing_json_key_raises` |
| FR-2.5 | Done | `_cerebus_api_key_cache` | `test_resolve_key_caches_after_first_aws_fetch`, `test_env_file_never_modified_by_aws_fallback_resolution` |
| AR-2.1 | Done | `fake_boto3` fixture (sys.modules injection) | all AWS-path tests |
| AR-2.2 | Done | `tests/test_cerebus.py` | this file |

## Deviations from Spec

None — this spec was written to match the corrected implementation, so there is nothing
left unreconciled. See `critique-consolidated-v-1.md` for the three blocking issues found
and fixed during the spec-write critique pass itself.
