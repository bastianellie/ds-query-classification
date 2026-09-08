# Architectural Decision Records: 2-experiment-runner

## ADR-001: INV-1 gains a second full-access orchestrator, `experiment.py`
- **Date:** 2026-09-08
- **Status:** Accepted
- **Context:** Spec 2 (AR-1.2, AR-3.1, AR-3.4) adds three new modules — `dataset_io.py`,
  `induction.py`, `experiment.py` — the last of which must construct every classifier role
  (plain, `--critics`, induction) and call `pipeline.classify_csv` directly, duplicating
  `cli.py`'s classifier-construction responsibility end-to-end for its own subcommands
  (AR-3.1 requires `pipeline.py`/`cli.py` to stay unmodified, so the experiment runner
  cannot reuse `cli.py`'s private construction code — it needs the same import breadth
  instead). Today's INV-1 states `cli.py` is the *only* module allowed to import all
  others; a second module needing that same breadth is a genuine invariant change, not an
  incidental import.
- **Decision:**
  - `dataset_io.py` may import from: stdlib, `pandas`, `huggingface_hub` (lazily, inside
    functions, per FR-1.5) — no internal `query_classification` imports. It is a leaf
    module, like `categories.py`/`classifier.py`.
  - `induction.py` may import from: `categories.py`, `classifier.py` only (for the
    `Category`/`Label` construction and the `Classifier` type hint respectively) — not
    `schema.py`/`prompts.py`, since it receives an already-constructed `Classifier` as a
    plain argument rather than building one itself (mirroring `debate.py`'s established
    precedent of importing a subset of what its ADR authorizes).
  - `experiment.py` may import from, enumerated explicitly: `dataset_io.py`,
    `induction.py`, `schema.py`, `prompts.py`, `resources.py`, `pipeline.py`, `debate.py`,
    `categories.py`, `classifier.py` — the same breadth as `cli.py`. It does **not** import
    `cli.py`, and `cli.py` does not import it.
  - The root `experiment.py` script (repo root) is a thin entry point mirroring
    `classify.py`'s existing `sys.path.insert` + `main()` pattern — not a package-internal
    import source, exactly as `classify.py` is already treated.
  - **Revised rule:** "`cli.py` and `experiment.py` are the two modules allowed to import
    all others; neither imports the other."
- **Rationale:** Spec 1's `debate.py` precedent already established that a new module can
  compose existing building blocks without collapsing module boundaries; `experiment.py`
  needs strictly broader access because it is a second orchestrator, not a composed
  building block — but the same principle (an ADR-authorized, explicitly enumerated import
  list rather than a blanket "may import everything") applies.
- **Consequences:** `spec/ARCHITECTURE.md`'s INV-1 rule and Module Boundary Map gain rows
  for the three new modules and the revised two-orchestrator statement. Any future module
  needing this same breadth requires its own ADR, keeping the exception list explicit
  rather than accumulating silently.
- **Invariants affected:** INV-1 (amended, not violated).

## ADR-002: INV-7 gains direct-import test exceptions for the three new modules
- **Date:** 2026-09-08
- **Status:** Accepted
- **Context:** `dataset_io.py`, `induction.py`, and `experiment.py` are internal
  (not re-exported from `__init__.py`) but are unit-tested directly by
  `tests/test_experiment.py` (FR-3.7), following the same rationale Spec 1 established for
  `debate.py`: direct imports give far more precise failure localization for a
  many-moving-parts feature than testing only through the public API. `schema.py` and
  `prompts.py` also gain one new function each (`build_induction_model`,
  `build_induction_prompt`) that need the same direct-import test access Spec 1 already
  granted their `build_critic_model`/`build_reconciler_model`/`build_critic_prompt`/
  `build_reconciler_prompt` siblings.
- **Decision:** Extend INV-7's accepted direct-import exception list with:
  `query_classification.dataset_io` (all functions), `query_classification.induction`
  (all functions), `query_classification.experiment` (all functions), and, specifically,
  `schema.build_induction_model` and `prompts.build_induction_prompt`. This extends Spec
  1's existing exception list (`debate.py` + the four critic/reconciler builder functions)
  rather than replacing it — both sets of exceptions remain in force.
- **Rationale:** identical to Spec 1's ADR-002 rationale — precise failure localization
  for internal orchestration modules that are not part of the public API surface.
- **Consequences:** `tests/test_experiment.py` may import `dataset_io`, `induction`,
  `experiment`, `schema.build_induction_model`, and `prompts.build_induction_prompt`
  directly. It must not import `categories.py`/`classifier.py`/`pipeline.py`/`cli.py`
  directly (unchanged from the existing rule) — `dataset_io.py`/`induction.py`/
  `experiment.py` are the sanctioned direct-import surface for this spec's own modules.
- **Invariants affected:** INV-7 (amended, not violated).

## References
- Spec: `spec/2-experiment-runner/spec.md` — AR-1.2 (requires this ADR), AR-2.1/AR-2.2
  (the two new schema/prompts functions), AR-3.1/AR-3.4 (the experiment runner's
  classifier-construction breadth).
- Plan: `spec/2-experiment-runner/plan.md` — Task 5, Project Constraints (INV-1/INV-7
  sections), and the Module Boundary Map entries fixed by Tasks 1-4's design.
- Precedent: `spec/1-initial-classification-with-critics/ADR.md` (ADR-001/ADR-002), folded
  into `spec/ARCHITECTURE.md` by `/spec-close` for that spec.
