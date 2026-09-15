# Architectural Decision Records: 5-batched-classification

## ADR-001: New `batching.py` node (INV-1), plus this spec's own INV-7 test-import names

- **Date:** 2026-09-15
- **Status:** Accepted
- **Context:** This spec adds `--batch` support: a new module, `src/query_classification/batching.py`,
  owns token-budget resolution (FR-1.2), batch sizing (FR-1.5–FR-1.7), payload assembly
  (FR-2.2), the arity-keyed classifier factory/cache (FR-2.4), and the batched call with
  bisection on failure (FR-3.2, via spec 6's already-amended `classifier.classify_failure`).
  `pipeline.py` gains an import of it (to run batches); `cli.py` and `experiment.py` each gain
  an import of it (to construct a `BatchRunner`, `BatchStats`, and the classifier factory).
  INV-1's rule text and the Module Boundary Map are closed per-module lists, so all three
  inbound edges — not just one — must be recorded, or the invariant would contradict the
  shipped import graph. Separately, `tests/test_batching.py` needs direct imports of
  `batching` (all functions, plus its private `_DATA_START`/`_DATA_END` constants for AR-2.3's
  delimiter-equality test) and `induction._DATA_START`/`_DATA_END` (the other half of that
  test), and `schema.build_batch_model` (not re-exported) — none of which INV-7 previously
  named. Spec 6's own ADR-001 already amended INV-7 for the classifier-side failure-
  classification primitives this spec consumes (`FailureKind`/`classify_failure`/the three
  response-failure exception types/`_DROPPABLE_PARAMS`/`Classifier._complete`/
  `_attempt_completion`); this spec introduces no new names for those and does not re-request
  them.
- **Decision:** INV-1's rule text and the Module Boundary Map gain `batching.py` as a new node
  (importing only `categories.py`+`classifier.py` in practice, a subset of what AR-1.1
  authorizes — it never needs `schema.py`/`prompts.py` directly, since the batch model
  builder and system prompt are both supplied by the caller), plus the three inbound edges
  `batching.py --> pipeline.py`, `batching.py --> cli.py`, `batching.py --> experiment.py`.
  INV-7's accepted-exception list and the `tests/` Module Boundary Map row gain, from this
  spec: `batching` (all functions + the private `_DATA_START`/`_DATA_END` constants),
  `induction._DATA_START`/`_DATA_END`, and `schema.build_batch_model`. Spec 6's own INV-7
  provenance note and text are left exactly as-is — this amendment only appends this spec's
  own names alongside them, never duplicates or re-amends the classifier-side names.
- **Rationale:** Mirrors the precedent set by spec 4's `multi_model.py` (a new sibling module
  to `debate.py`, with the same "new node + inbound edges onto the two CLI entry points and
  the CSV loop" shape) for INV-1, and by every prior spec's own INV-7 amendment (spec 1's
  `debate.py`, spec 2's `dataset_io.py`/`induction.py`/`experiment.py`, spec 3's Cerebus
  functions, spec 4's `multi_model.py`, spec 6's failure-classification primitives) for INV-7 —
  each gives tests direct access to internals with no public-API equivalent, rather than
  forcing indirect exercise through `classify_csv`/a live provider. Reusing spec 6's
  `classify_failure` directly (instead of re-deriving litellm-hierarchy knowledge inside
  `batching.py`) is itself the reason this spec's own INV-7 ask is small: an earlier draft of
  this spec built its own three-way failure taxonomy from scratch and got the litellm
  hierarchy wrong twice, in exactly the ways spec 6 was independently critiqued into
  correctness for — reusing the already-verified classifier removes an entire class of defect
  this spec would otherwise be exposed to a third time.
- **Consequences:** `batching.py` becomes a new, independently-testable building block
  alongside `debate.py`/`multi_model.py` — a `--batch`-mode sibling to `--critics`/`--models`.
  Its effectively-public surface for test purposes widens INV-7's list by three names; nothing
  else in `classifier.py`/`categories.py`/`schema.py` becomes importable beyond what's named.
  The graph stays acyclic: `batching.py` has zero outbound imports of `pipeline.py`,
  `cli.py`, `experiment.py`, `debate.py`, or `multi_model.py`.
- **Invariants affected:** INV-1 (amended, new module + three inbound edges) and INV-7
  (amended, three new test-import names — distinct from, and not duplicating, spec 6's
  already-recorded classifier-side names).
