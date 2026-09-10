# Architectural Decision Records: 4-multi-model-classification

## ADR-001: INV-1 gains a new module, `multi_model.py`, sibling to `debate.py`
- **Date:** 2026-09-10
- **Status:** Accepted
- **Context:** Spec 4 (AR-1.1) adds a `--models` classification mode — N distinct models
  classify a row independently and are merged by plurality vote — as a standalone
  alternative to `--critics`. The vote-tally/tie-break/audit-column orchestration for this
  mode needs its own module (mirroring `debate.py`'s role for `--critics`), and reuses
  `debate.py`'s existing `_vote_bucket` none-label-merging convention rather than
  duplicating it, which requires `multi_model.py` to import `debate.py` — a new edge not
  previously authorized by INV-1's Module Boundary Map.
- **Decision:**
  - `multi_model.py` may import from: `categories.py`, `classifier.py`, `schema.py`,
    `prompts.py`, and `debate.py` (specifically its `_vote_bucket` function). It must not
    import `pipeline.py`, `cli.py`, or `experiment.py` — the same ceiling `debate.py`
    itself has.
  - `pipeline.py`'s and `experiment.py`'s "May import from" lists both gain
    `multi_model.py`, alongside their existing `debate.py` entry.
  - The new `debate.py → multi_model.py` edge (in this project's Mermaid-diagram
    convention, "A --> B" means "B imports A") is acyclic: `debate.py` itself has zero
    outbound imports of `pipeline.py`/`cli.py`/`experiment.py`/`multi_model.py` today, so
    a new *importer* of `debate.py` doesn't create a cycle — regardless of how many
    existing modules (`pipeline.py`, `experiment.py`) already import `debate.py`.
- **Rationale:** identical in shape to Spec 1's `debate.py` precedent — a new module
  composing existing building blocks (plus one sibling module, `debate.py`, for a single
  reused helper) without collapsing module boundaries or granting broad access. Reusing
  `_vote_bucket` instead of duplicating it keeps the none-label-invention convention
  defined in exactly one place.
- **Consequences:** `spec/ARCHITECTURE.md`'s INV-1 rule, Module Boundary Map, and Mermaid
  graph gain a row/edges for `multi_model.py`. Any future module wanting to import
  `debate.py`'s helpers needs its own such row, keeping the exception list explicit.
- **Invariants affected:** INV-1 (amended, not violated).

## ADR-002: INV-7 gains a direct-import test exception for `multi_model.py`
- **Date:** 2026-09-10
- **Status:** Accepted
- **Context:** `multi_model.py` is internal (not re-exported from `__init__.py`) but is
  unit-tested directly by `tests/test_multi_model.py` (FR-1.10), following the same
  rationale Spec 1 established for `debate.py`: direct imports give far more precise
  failure localization for this feature's orchestration logic than testing only through
  the public API (`classify_csv`).
- **Decision:** Extend INV-7's accepted direct-import exception list with
  `query_classification.multi_model` (all functions), alongside the existing `debate.py`/
  `dataset_io.py`/`induction.py`/`experiment.py`/Cerebus-function/`cli.py` exceptions —
  this extends the existing exception list rather than replacing it.
- **Rationale:** identical to Spec 1's ADR-002 rationale.
- **Consequences:** `tests/test_multi_model.py` may import `multi_model` directly. It must
  not import `categories.py`/`classifier.py`/`pipeline.py`/`cli.py` directly (unchanged
  from the existing rule).
- **Invariants affected:** INV-7 (amended, not violated).

## References
- Spec: `spec/4-multi-model-classification/spec.md` — AR-1.1 (requires this ADR).
- Plan: `spec/4-multi-model-classification/plan.md` — Task 1, Project Constraints (INV-1/
  INV-7 sections).
- Precedent: `spec/1-initial-classification-with-critics/ADR.md` (ADR-001/ADR-002) and
  `spec/2-experiment-runner/ADR.md` (ADR-001/ADR-002), both folded into
  `spec/ARCHITECTURE.md` by `/spec-close` for their respective specs.
