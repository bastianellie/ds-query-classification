# Architectural Decision Records: 3-cerebus-gateway

## ADR-001: INV-7 gains direct-import test exceptions for `classifier.py`'s Cerebus functions and `cli.py`

- **Date:** 2026-09-09
- **Status:** Accepted
- **Context:** `tests/test_cerebus.py` directly imports `query_classification.classifier`
  (for `_resolve_cerebus_api_key`, `build_cerebus_completion_kwargs`, `cerebus_model_id`,
  `reject_insecure_cerebus_endpoint`, and the module-level `_cerebus_api_key_cache` reset
  fixture) and `query_classification.cli` (`build_parser`, `main`), both of which INV-7's
  current wording explicitly forbids ("Do not import `categories.py`/`classifier.py`/
  `pipeline.py`/`cli.py` directly"). This was caught during this spec's own critique
  (Codex, `plan-critique`-equivalent pass) — a real conflict, not a hypothetical one, since
  the test file already exists and the full suite already passes with these imports in
  place.

  Testing the Cerebus functions through the public API alone isn't practical: none of
  `_resolve_cerebus_api_key`, `build_cerebus_completion_kwargs`, `cerebus_model_id`, or
  `reject_insecure_cerebus_endpoint` are part of `__init__.py`'s re-exported surface (nor
  should they be — they're internal to how `--cerebus` wires a `Classifier` together, not
  building blocks a library consumer would call directly), and `cli.py`'s `build_parser`/
  `main` are the only way to exercise the actual CLI-level wiring (flag defaults, the
  fail-fast validation order, per-role model-id prefixing across every constructed
  classifier) that FR-1.3/FR-1.5/FR-1.6 depend on.

- **Decision:** Extend INV-7's accepted direct-import exception list with:
  `query_classification.classifier`'s Cerebus-specific surface (`_resolve_cerebus_api_key`,
  `build_cerebus_completion_kwargs`, `cerebus_model_id`, `reject_insecure_cerebus_endpoint`,
  and the `_cerebus_api_key_cache` module attribute, for test-isolation resets only) and
  `query_classification.cli` (`build_parser`, `main`). This does **not** open up the rest
  of `classifier.py` (e.g. `Classifier` itself is already tested via the public API) or
  `cli.py` (no other function needs direct-import test access) — the exception is scoped
  to exactly what `tests/test_cerebus.py` uses, mirroring how Spec 1's exception for
  `schema.py`/`prompts.py` named specific functions rather than opening the whole module.
- **Rationale:** identical to Spec 1's ADR-002 and Spec 2's ADR-002 rationale — precise
  failure localization for functionality that isn't part of the public API surface, and in
  `cli.py`'s case, no practical way to exercise the CLI wiring itself (as opposed to the
  building blocks it calls) without importing it directly.
- **Consequences:** `tests/test_cerebus.py`'s existing imports are retroactively
  sanctioned by this ADR rather than needing to change. Any *new* test wanting to import
  something else from `classifier.py`/`cli.py` directly (rather than through the public
  API) needs its own justification and its own ADR/amendment — this exception is not a
  blanket opening of either module.
- **Invariants affected:** INV-7 (amended, not violated).

## References
- Spec: `spec/3-cerebus-gateway/spec.md` — AR-2.1 (test isolation via faked `boto3`),
  AR-1.4 (this ADR's own trigger).
- Precedent: `spec/1-initial-classification-with-critics/ADR.md` (ADR-002),
  `spec/2-experiment-runner/ADR.md` (ADR-002), both folded into `spec/ARCHITECTURE.md` by
  `/spec-close` for their respective specs.
