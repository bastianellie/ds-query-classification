# Architectural Decision Records: 6-classifier-error-path

## ADR-001: Extend INV-7's accepted direct-import list, and correct its stale claim

- **Date:** 2026-09-15
- **Status:** Accepted
- **Context:** This spec adds `classifier.py`-internal failure-classification primitives
  (`FailureKind`, `classify_failure`, `TruncatedResponseError`, `PolicyRefusalError`,
  `EmptyResponseError`, `_DROPPABLE_PARAMS`) and requires tests to exercise
  `Classifier._complete`/`_attempt_completion` directly, since FR-2.1–FR-2.5 describe behavior
  below `classify()` that isn't reachable through it without a live provider (the project's
  hard testing rule forbids live provider calls in tests). INV-7 restricts `tests/` to the
  public API plus a named list of accepted exceptions, and did not previously name any of
  these. Separately, INV-7's own text asserted "`Classifier` itself is still tested only via
  the public API" — verified false at the time this spec was written: `tests/test_cerebus.py`
  has called `clf._complete(...)` directly in three tests since spec 3, and that sentence was
  never updated to match.
- **Decision:** INV-7's accepted-exception list gains, from spec 6: `classifier.FailureKind`,
  `classify_failure`, `TruncatedResponseError`, `PolicyRefusalError`, `EmptyResponseError`,
  `_DROPPABLE_PARAMS`, and `Classifier._complete`/`_attempt_completion`. The contradicted
  sentence is deleted rather than merely appended around. The Module Boundary Map's `tests/`
  row is updated to match. `tests/test_classifier_errors.py` (new) is the primary consumer of
  this amendment.
- **Rationale:** Mirrors the precedent set by every prior spec that added internal test-import
  exceptions (spec 1's `debate.py`, spec 2's `dataset_io.py`/`induction.py`/`experiment.py`,
  spec 3's Cerebus functions, spec 4's `multi_model.py`) — each gives far more precise failure
  localization for internals with no public-API equivalent than testing only through
  `classify()` would. Correcting the stale sentence (rather than leaving it to accumulate a
  second contradiction) keeps INV-7 an accurate description of the actual test suite, which is
  the whole point of the invariant.
- **Consequences:** `Classifier`'s effectively-public surface for test purposes widens beyond
  `classify()`. This is an accepted, bounded widening — the same shape as every prior INV-7
  amendment — not a general license to import arbitrary `classifier.py` internals; anything not
  named in the amended list remains off-limits per INV-7's own rule.
- **Invariants affected:** INV-7 (amended, per above). INV-1 is unaffected — this spec adds no
  new module and no new import edge; `classifier.py` remains a leaf (stdlib + litellm +
  pydantic only).
