# Architectural Decision Records: 7-cost-and-timing-logging

## ADR-001: New `cost.py` node (INV-1), plus this spec's own INV-7 test-import names

- **Date:** 2026-09-16
- **Status:** Accepted
- **Context:** This spec adds cost/timing logging: a new module,
  `src/query_classification/cost.py`, owns per-model pricing resolution (`PricingCache`,
  FR-2.1/FR-2.3), cost arithmetic (`compute_cost`, FR-2.2), sample statistics
  (`stats_from_samples`/`uniform_estimate`/`expand_batch_costs_to_query_samples`, FR-3.x), a
  live-readable run timer (`Timer`, FR-4.1), the shared atomic JSON writer
  (`write_json_atomic`, AR-5.2), and the per-row cost collector (`QueryCostCollector`,
  FR-3.1/FR-3.3). `pipeline.py`, `cli.py`, and `experiment.py` each gain an import of it — the
  same three-inbound-edge shape spec 5's `batching.py` established. `batching.py` itself gains
  no new import: `BatchRunner` instead takes a plain injected `cost_fn: Callable[[int, int],
  float | None] | None` from its caller, so as not to add a fourth, undeclared
  `batching.py -> cost.py` edge that INV-1 does not authorize. INV-1's rule text and the
  Module Boundary Map are closed per-module lists, so all three inbound edges — not just one —
  must be recorded, or the invariant would contradict the shipped import graph.

  Separately, `tests/test_cost.py` needs a direct import of `cost` (all functions/classes,
  none of which are re-exported by `__init__.py`), and `Classifier` gains one new private
  method, `_capture_usage_from_response` (the single hook point `_attempt_completion` calls
  before returning, which reads `response.usage` and feeds both the instance's running total
  and any active `track_usage()` sink). Spec 6's own ADR-001 already amended INV-7 for
  `Classifier._complete`/`_attempt_completion`, but it named those specific existing methods —
  it is not a blanket grant for every future private `Classifier` method, and
  `_capture_usage_from_response` did not exist when that ADR was written. This spec's own
  tests exercise it indirectly (via `track_usage()`, itself public), so no test imports the
  name directly today, but the ADR still records it as an accepted INV-7 exception in case a
  future test needs to assert against it directly, mirroring how spec 6 named
  `_attempt_completion` even though most of its own tests exercise it via `_complete`.

- **Decision:** INV-1's rule text and the Module Boundary Map gain `cost.py` as a new node
  (importing only `categories.py`+`classifier.py`, mirroring `batching.py`'s own minimal
  import list), plus the three inbound edges `cost.py --> pipeline.py`, `cost.py --> cli.py`,
  `cost.py --> experiment.py`. `batching.py`'s own row is explicitly **not** edited to add
  `cost.py` as an import — it remains `categories.py`+`classifier.py`+`schema.py`+`prompts.py`
  only, unchanged from spec 5. INV-7's accepted-exception list and the `tests/` Module
  Boundary Map row gain, from this spec: `cost` (all functions/classes) and
  `classifier._capture_usage_from_response`.

- **Rationale:** Mirrors the precedent set by spec 5's `batching.py` (a new sibling module,
  with the same "new node + inbound edges onto the two CLI entry points and the CSV loop"
  shape) for INV-1, and by every prior spec's own INV-7 amendment for exercising internals
  with no public-API equivalent. The `cost_fn`-callable design (rather than a direct
  `batching.py -> cost.py` import) was chosen specifically to avoid widening `batching.py`'s
  declared import boundary a second time in as many specs — `BatchRunner` already accepts
  injected collaborators (`classifier_factory`, `stats`), so a plain cost-computing callable
  is a natural, minimal addition to that same pattern rather than a new dependency edge.

- **Consequences:** `cost.py` becomes a new, independently-testable building block alongside
  `batching.py`/`debate.py`/`multi_model.py`. Its effectively-public surface for test purposes
  widens INV-7's list by `cost` (in full) and one new `classifier.py` private method; nothing
  else in `classifier.py`/`categories.py`/`schema.py`/`batching.py` becomes importable beyond
  what's named. The graph stays acyclic: `cost.py` has zero outbound imports of `pipeline.py`,
  `cli.py`, `experiment.py`, or `batching.py`.

- **Invariants affected:** INV-1 (amended, new module + three inbound edges) and INV-7
  (amended, two new test-import names — distinct from, and not duplicating, any prior spec's
  own recorded names).
