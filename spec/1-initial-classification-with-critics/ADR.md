# Architectural Decision Records: 1-initial-classification-with-critics

## ADR-001: New `debate.py` module and its dependency edges

- **Date:** 2026-09-08
- **Status:** Accepted
- **Context:** Implementing the `--critics` self-consistency-sampling + Critic/Reconciler
  debate mechanism (spec `spec/1-initial-classification-with-critics/spec.md`, AR-1.2)
  requires a new orchestration module that composes `categories.py`, `classifier.py`,
  `schema.py`, and `prompts.py` for a single row, and that `pipeline.py` calls into it.
  `spec/ARCHITECTURE.md`'s current INV-1 fixes `pipeline.py`'s allowed imports to
  `categories.py` + `classifier.py` only, and the Module Boundary Map has no `debate.py`
  row — this ADR is the record required before that invariant changes, per
  `spec/ARCHITECTURE.md`'s own header ("Changes to invariants require an ADR").
- **Decision:** Add `src/query_classification/debate.py`, a new module with:
  - **May import from:** `categories.py`, `classifier.py`, `schema.py`, `prompts.py`.
  - **Must NOT import from:** `pipeline.py`, `cli.py`.
  `pipeline.py`'s allowed-imports set gains `debate.py` (in addition to its existing
  `categories.py`/`classifier.py`).

  **INV-1 before:**
  > `categories.py` and `classifier.py` never import any sibling module; `schema.py`
  > depends only on `categories.py`; `prompts.py` depends only on `schema.py`+
  > `resources.py`; `pipeline.py` depends only on `categories.py`+`classifier.py`;
  > `__init__.py` depends on exactly those 5 modules; `cli.py` is the only module allowed
  > to import all 6.

  **INV-1 after:**
  > `categories.py` and `classifier.py` never import any sibling module; `schema.py`
  > depends only on `categories.py`; `prompts.py` depends only on `schema.py`+
  > `resources.py`; `debate.py` depends only on `categories.py`+`classifier.py`+
  > `schema.py`+`prompts.py`; `pipeline.py` depends only on `categories.py`+
  > `classifier.py`+`debate.py`; `__init__.py` depends on exactly the original 5 modules
  > (unchanged — `debate.py` is not re-exported, see ADR-002); `cli.py` is the only module
  > allowed to import all 7.

  **New Module Boundary Map row:**

  | Module / path | Responsibility | May import from | Must NOT import from |
  |---|---|---|---|
  | `src/query_classification/debate.py` | Per-row self-consistency sampling, consensus voting, and Critic/Reconciler debate orchestration for the `--critics` mode | `categories.py`, `classifier.py`, `schema.py`, `prompts.py` | `pipeline.py`, `cli.py` |

  `pipeline.py`'s Module Boundary Map row's "May import from" cell gains `debate.py`
  alongside its existing `categories.py`, `classifier.py`.
- **Rationale:** `debate.py` needs schema/prompt builders (`build_critic_model`,
  `build_reconciler_model`, `build_critic_prompt`, `build_reconciler_prompt`) that
  `pipeline.py` itself must not depend on directly (INV-1's existing separation of
  concerns between orchestration and schema/prompt construction) — introducing one new
  module that owns this composition, rather than having `pipeline.py` import
  `schema.py`/`prompts.py` directly, keeps that separation intact.
- **Consequences:** `pipeline.py` is no longer limited to `categories.py`+`classifier.py`;
  this is a one-time, scoped widening for this exact new dependency, not a general
  loosening of the rule (the "must not import `schema.py`/`prompts.py`/`cli.py`" halves
  of `pipeline.py`'s and `schema.py`'s/`prompts.py`'s rows are otherwise unchanged).
- **Invariants affected:** INV-1 (see before/after above).

## ADR-002: `debate.py` and new schema/prompt builder functions as accepted test-import exceptions

- **Date:** 2026-09-08
- **Status:** Accepted
- **Context:** `spec/ARCHITECTURE.md` INV-7 restricts new tests to the public API
  (`query_classification`) plus `query_classification.resources` as the one accepted
  direct-import exception. This spec's plan (`spec/1-initial-classification-with-critics/
  plan.md`, AR-1.7) requires directly unit-testing `debate.py`'s functions and the new
  `schema.py`/`prompts.py` builder functions (`build_critic_model`,
  `build_reconciler_model`, `build_critic_prompt`, `build_reconciler_prompt`) — none of
  which are re-exported from `__init__.py` (see ADR-001; they remain internal
  implementation detail, not part of the supported public API).
- **Decision:** INV-7's accepted direct-import exception list grows from just
  `resources.py` to also include:
  - `debate.py` (all of its functions), and
  - the *specific new* functions `build_critic_model`/`build_reconciler_model` in
    `schema.py` and `build_critic_prompt`/`build_reconciler_prompt` in `prompts.py`.

  The pre-existing `build_classification_model`/`build_system_prompt`/`schema_description`
  remain public-API-only for tests, per today's INV-7 — this exception is scoped to the
  new functions this spec adds, not a blanket exception for all of `schema.py`/
  `prompts.py`.

  **INV-7 before:**
  > New tests should import from `query_classification` (the package `__init__.py`);
  > importing `query_classification.resources` directly is an accepted exception (it's
  > not re-exported by `__init__.py`, but the resource-path constants are needed for
  > path-existence assertions). Do not import `categories.py`/`schema.py`/`classifier.py`/
  > `pipeline.py`/`cli.py` directly.

  **INV-7 after:**
  > New tests should import from `query_classification` (the package `__init__.py`).
  > Accepted direct-import exceptions: `query_classification.resources` (resource-path
  > constants needed for path-existence assertions); `query_classification.debate` (all
  > functions — internal orchestration, not part of the public API); and, specifically,
  > `schema.build_critic_model`/`build_reconciler_model` and
  > `prompts.build_critic_prompt`/`build_reconciler_prompt` (new internal builder
  > functions introduced alongside `debate.py`, not re-exported). The pre-existing
  > `build_classification_model`/`build_system_prompt`/`schema_description` remain
  > public-API-only. Do not import `categories.py`/`classifier.py`/`pipeline.py`/`cli.py`
  > directly.
- **Rationale:** these functions are genuinely internal (composing them is `debate.py`'s
  job, not something a library caller does directly), but unit-testing them in isolation
  (rather than only indirectly through `debate.run_debate`) gives much more precise
  failure localization for a feature with this many interacting pieces.
- **Consequences:** slightly widens what tests may import directly; scoped narrowly to
  the specific new functions this spec adds, not a general relaxation of INV-7.
- **Invariants affected:** INV-7 (see before/after above).
