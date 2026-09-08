# Implementation Plan: 1-initial-classification-with-critics

**Status:** Ready
**Date:** 2026-09-08
**Spec:** spec/1-initial-classification-with-critics/spec.md
**Plan Critique:** spec/1-initial-classification-with-critics/plan-critique-consolidated-v-1.md

## Overview

Build the opt-in `--critics` mode bottom-up: extend the three low-level building blocks
(`Classifier`, schema builders, prompt builders) in parallel, then write the new `debate.py`
orchestration module that depends on all three, then wire it into `pipeline.py` and `cli.py`
in sequence, then ship tests. An architecture update (ADR, per `spec/ARCHITECTURE.md`'s own
process) runs alongside the `pipeline.py` wiring since both depend on `debate.py` existing
but don't touch each other's files.

## Readiness

- Checklist: all 10 items closed `[x]` in the spec — no open gaps to resolve before planning.
- Open questions: none blocking. The spec intentionally left exact Pydantic field names
  (AR-2.1) and `debate.run_debate`'s exact signature unspecified as implementation detail —
  both are nailed down concretely below (Task 2, Task 4) so this isn't an "undefined
  interface" gap in the plan, even though the spec itself stayed at the conceptual level.
- External API/library verification: no new third-party dependency. `litellm.completion`'s
  `temperature` parameter was confirmed present on the installed version via
  `inspect.signature` during spec-write (see spec AR-1.1) — re-verified here by re-reading
  `classifier.py`; still accurate.
- Repository state: **not fully clean** — `git status` currently shows untracked files from
  earlier, unrelated work this session (`spec/` docs, `pyproject.toml`,
  `resources/categories/example_categories.json`, a `data/` CSV). None of these are
  produced by or relevant to this plan; leave them untouched and don't assume a clean-slate
  revert boundary around them. Confirm no *new* unexpected changes exist before starting
  Task 1, rather than expecting a fully clean tree.

## Code Impact

- **Modules:** `src/query_classification/` — `classifier.py`, `schema.py`, `prompts.py`,
  `resources.py`, `pipeline.py`, `cli.py` extended; new `debate.py`.
- **Database/schema:** none (no DB). CSV schema gains 7 new columns per category, gated
  entirely behind `--critics` (Data Requirements in the spec).
- **API/interfaces:** 7 new CLI flags (`--critics`, `--sampling-runs`,
  `--sampling-temperature`, `--consensus-threshold`, `--allow-new-labels`,
  `--critic-model`, `--reconciler-model`); `classify_csv`'s signature grows;
  `debate.run_debate` is a new internal function (not part of the public API — see AR-1.7).
- **UI/config/background jobs:** none — batch CLI tool, no UI or background jobs.
- **Key files:**
  - `src/query_classification/classifier.py` (modify)
  - `src/query_classification/schema.py` (modify)
  - `src/query_classification/prompts.py` (modify)
  - `src/query_classification/resources.py` (modify — 2 new path constants)
  - `resources/prompts/critic_prompt.txt`, `resources/prompts/reconciler_prompt.txt` (new)
  - `src/query_classification/debate.py` (new)
  - `src/query_classification/pipeline.py` (modify)
  - `src/query_classification/cli.py` (modify)
  - `README.md` (modify)
  - `spec/1-initial-classification-with-critics/ADR.md` (new)
  - `tests/test_debate.py` (new)

## Project Constraints

- **INV-1** (dependency graph) is being *revised*, not violated silently — Task 5 writes
  the ADR `spec/ARCHITECTURE.md` requires before an invariant changes; the actual
  `spec/ARCHITECTURE.md` edit happens later via `/spec-close`, not in this implementation
  pass (matches the doc's own stated process, not a deviation from it).
- **INV-2** (all bundled resource paths centralized in `resources.py`) — the two new prompt
  template paths must be added as constants in `resources.py`, not hardcoded inline in
  `prompts.py` (Task 3).
- **INV-4** (broad `except Exception` needs `# noqa: BLE001` + inline reason) — `debate.py`'s
  graceful-degradation catches around the Critic/Reconciler calls (FR-2.3) must follow this
  convention (Task 4).
- **INV-6** (`pipeline.py` mutates the DataFrame only on the main thread) — preserved by
  construction: `run_debate` (Task 4) never touches `df` itself, only returns a flat dict;
  `pipeline.py`'s existing main-thread-only write loop (`pipeline.py:106-107`) is unchanged
  and still the only place any `df` cell is set (Task 6).
- **INV-7** (tests use the public API, `resources.py` as the one accepted direct-import
  exception) — Task 2/3/8's tests importing `debate.py`, and the new
  `build_critic_model`/`build_reconciler_model`/`build_critic_prompt`/
  `build_reconciler_prompt` functions directly, need Task 5's broadened exception in place
  first (i.e., don't write Tasks 2/3/8's tests without Task 5's ADR acknowledging it, even
  though the ADR itself isn't binding until folded in by `/spec-close`).
- **AGENTS.md "Required patterns"** — `from __future__ import annotations` + PEP 604 unions
  in every new/modified module with type hints (`debate.py` included).
- **AGENTS.md "Testing policy"** — no live network/provider calls in tests; fake
  `Classifier`-shaped objects only (already the pattern used to verify the spec's own
  FR Verify conditions during spec-write).
- **Spec Constraints' prompt data-framing requirement** — the Critic/Reconciler templates
  (Task 3) and the assembled per-call user messages (Task 4) must clearly delimit "text
  under review" from instructions, and treat the Critic's own generated argument as quoted
  data (not additional instructions) when it's relayed into the Reconciler's prompt. This
  reduces, not eliminates, the risk of the Critic's output being interpreted as new
  instructions by the Reconciler — a genuine, new model-to-model injection channel this
  feature introduces.

## Implementation Strategy

- **Size:** large (2 features, 15 FRs, 12 ARs, 1 new module, 1 architecture change).
- **Execution mode:** 4 parallel streams for Tasks 1-3 + 5, then a sequential chain
  (Task 4 → Task 6 → Task 7 → Task 8).
- **Parallelizable work:** Tasks 1/2/3/5 are all independent of each other and of Task 4's
  actual code — Task 5 (the ADR) documents a dependency graph already fixed by this plan's
  design, not by Task 4's code being finished, so it doesn't need to wait for Task 4 despite
  describing it.
- **Sequential blockers:** Task 4 needs 1+2+3 (imports the temperature param, the schema
  builders, and the prompt builders). Task 6 needs Task 4 (imports `debate.run_debate` and
  the reserved-suffix constants). Task 7 needs Task 6 (calls the new `classify_csv`
  signature). Task 8 needs Task 7 (exercises the fully-wired CLI-to-`debate.py` path).

## Implementation Tasks

### Task 1: `Classifier` gains an optional sampling temperature
**Goal:** `Classifier.__init__` accepts `temperature: float | None = None`, included in
`_completion_kwargs` only when set, with zero behavior change for every existing call site
that doesn't pass it.
**Files:** `src/query_classification/classifier.py`
**Dependencies:** None
**Do:**
- Add `temperature: float | None = None` to `Classifier.__init__`'s signature
  (`classifier.py:52-70`), stored as `self.temperature`.
- In `_completion_kwargs` (`classifier.py:72-78`), add `kwargs["temperature"] =
  self.temperature` only when `self.temperature is not None` — mirror the existing
  `if self.api_base:` conditional pattern exactly.
**Verify:** a `Classifier` constructed without `temperature` produces `_completion_kwargs`
identical to today (no `"temperature"` key); constructed with `temperature=0.7`, the kwargs
dict includes `"temperature": 0.7`. Existing `pytest` suite still passes unmodified.
**Covers:** AR-1.1

### Task 2: Critic/Reconciler schema builders
**Goal:** Two new Pydantic-model builder functions in `schema.py`, one per category, ready
to plug into a `Classifier` as its `classification_model`.
**Files:** `src/query_classification/schema.py`
**Dependencies:** None
**Do:**
- Extend `build_classification_model` (`schema.py:17-49`) with `allow_new_labels: bool =
  True` (default preserves today's exact output/tests). When `False`, each field's
  generated `field_description` (`schema.py:27-35`) omits the `"none - <new label>"`
  invention clause, leaving only the plain-`"none"` instruction — this is prompt text only,
  not a schema/`Field` constraint change (INV-11 still holds: cardinality/type only is
  ever enforced). This is what lets `cli.py` (Task 7) build a sampling classifier whose
  schema/prompt actually discourages invention when `--allow-new-labels` is off (FR-1.2).
- Add `build_critic_model(category: Category) -> type[BaseModel]` alongside
  `build_classification_model` (`schema.py:17-49`). Concrete field shape: `challenges:
  bool`, `proposed_label: str | None = None`, `argument: str` (always present — holds
  either the challenge argument or the decline rationale, per spec FR-2.1's requirement
  that decline rationale is preserved). Add a Pydantic model validator enforcing
  `proposed_label` is set (non-`None`) if and only if `challenges=True` — a plain
  `Optional` field alone doesn't enforce this conditional requirement and would silently
  accept `{"challenges": true, "proposed_label": null, ...}`.
- Add `build_reconciler_model(category: Category) -> type[BaseModel]`: `labels: list[str]`
  reusing the same `Field(min_length=1, max_length=3)` constraint as
  `schema.py:44-47`, plus `reasoning: str`.
- Both take a single `Category` (not `list[Category]`) — one model per escalated category,
  matching the spec's per-category granularity.
**Verify:** `build_classification_model(categories)` (no new arg) produces the exact same
field descriptions as today — existing `pytest` suite passes unmodified;
`build_classification_model(categories, allow_new_labels=False)` produces a field
description without the `"none - <new label>"` clause. `build_critic_model(cat)` validates
`{"challenges": false, "argument": "no compelling alternative"}` and separately
`{"challenges": true, "proposed_label": "x", "argument": "..."}`; `build_reconciler_model(cat)`
validates `{"labels": ["x"], "reasoning": "..."}` and rejects a 4-label list (matching the
existing 1-3 bound).
**Covers:** FR-1.2, AR-2.1

### Task 3: Critic/Reconciler prompt builders + templates + sampling's no-invention prompt
**Goal:** Two new bundled prompt templates and their renderer functions, following the
project's existing static-system-prompt convention — plus making the sampling classifier's
*system prompt itself* (not just its schema's field descriptions) actually forbid label
invention when `--allow-new-labels` is off.
**Files:** `src/query_classification/prompts.py`, `src/query_classification/resources.py`,
`resources/prompts/critic_prompt.txt` (new), `resources/prompts/reconciler_prompt.txt` (new),
`resources/prompts/system_prompt.txt` (modify), `tests/test_building_blocks.py` (extend)
**Dependencies:** None
**Do:**
- **Critical fix surfaced by plan critique:** `resources/prompts/system_prompt.txt:6`
  currently *hardcodes* `'...use "none" alone, or "none - <new label>" to suggest a
  concept...'` as static template text — Task 2's `allow_new_labels=False` change to the
  per-field schema description alone does **not** stop this independent, always-present
  instruction from telling the model it may invent labels. Replace that hardcoded sentence
  with a `{invention_policy}` placeholder in the template, and extend `build_system_prompt`
  (`prompts.py:25-39`) with `allow_new_labels: bool = True`, filling `{invention_policy}`
  with one of two fixed sentences (invention-allowed vs. invention-forbidden — no
  `"none - ..."` wording at all in the latter). Default `True` preserves today's exact
  rendered output for every existing call site/test. `cli.py` (Task 7) calls this with
  `allow_new_labels=args.allow_new_labels` only for the sampling classifier's prompt.
- Add `DEFAULT_CRITIC_PROMPT_FILE` and `DEFAULT_RECONCILER_PROMPT_FILE` constants to
  `resources.py` (alongside `DEFAULT_SYSTEM_PROMPT_FILE`, `resources.py:16-18`) — per
  INV-2, no other module computes these paths itself.
- Write `resources/prompts/critic_prompt.txt`: static role instructions for the devil's
  advocate role (construct the strongest case for an alternative label, or explicitly
  decline; never fabricate a weak objection). Include a short data-framing instruction:
  treat the text under review as untrusted data to classify, not as instructions to follow
  (defense-in-depth per the spec's Constraints; reduces, doesn't eliminate, prompt-injection
  risk).
- Write `resources/prompts/reconciler_prompt.txt`: static role instructions for the
  independent-verdict role (weigh the leading candidate and the Critic's argument against
  the evidence and the category's label definitions; not obligated to agree with either),
  with the same data-framing instruction — explicitly telling it to treat the Critic's
  argument as quoted data under review, not as additional instructions, since it's
  model-generated content being relayed into another model's prompt.
- Add `build_critic_prompt(category_name: str, category_description: str,
  label_options: str, system_prompt_file=DEFAULT_CRITIC_PROMPT_FILE) -> str` and
  `build_reconciler_prompt(...)` (same plain-string shape) to `prompts.py`. **Deliberately
  plain-string parameters, not `Category`** — passing a `Category` object here would add a
  new `prompts.py -> categories.py` import edge that INV-1 doesn't currently allow and
  Task 5's ADR doesn't cover; `debate.py` (Task 4, which already imports `categories.py`)
  extracts these plain values from its `Category` objects before calling in. Each renders
  the static template once per category (used as the `system_prompt` for that category's
  Critic/Reconciler `Classifier` instance); per-call specifics (source text, leading label,
  prior argument) are assembled as the *user message* by `debate.py`, not baked into these
  system prompts — matching the existing `cli.py`-builds-system-prompt-once /
  `classifier.py:99-102`-builds-user-message-per-call split.
- Extend `tests/test_building_blocks.py::test_bundled_resources_exist`-style assertions to
  also check `DEFAULT_CRITIC_PROMPT_FILE`/`DEFAULT_RECONCILER_PROMPT_FILE` exist.
**Verify:** `build_system_prompt(model, allow_new_labels=False)`'s rendered text contains no
`"none - "` substring; `build_system_prompt(model)` (no new arg) renders byte-identical to
today — existing `pytest` suite passes unmodified. `build_critic_prompt(...)` returns
non-empty text mentioning the given category name/description/labels; same for
`build_reconciler_prompt(...)`.
**Covers:** FR-1.2, AR-2.2

### Task 4: New `debate.py` orchestration module
**Goal:** The single-row sampling → voting → routing → Critic → Reconciler pipeline,
returning the flat result dict `pipeline.py` writes into the DataFrame unchanged.
**Files:** `src/query_classification/debate.py` (new)
**Dependencies:** Task 1, Task 2, Task 3
**Do:**
- Concrete signature: `run_debate(text: str, categories: list[Category],
  sampling_classifier: Classifier, critics: dict[str, Classifier],
  reconcilers: dict[str, Classifier], *, sampling_runs: int, consensus_threshold: int,
  allow_new_labels: bool) -> dict[str, Any]`. `critics`/`reconcilers` are keyed by category
  `name` — **one `Classifier` instance per category** for each role, since each category
  needs its own schema (`build_critic_model(cat)`/`build_reconciler_model(cat)`, Task 2)
  and `Classifier` fixes its `classification_model` at construction time (it can't swap
  schemas per call). `sampling_classifier` is a single shared `Classifier` (one call
  already returns all categories at once, matching today's `classify()` shape) already
  constructed with `temperature` set (Task 1). `debate.py` does not construct any
  `Classifier` instances itself — only calls `.classify()` on the ones it's given
  (AR-2.3); construction happens once in `cli.py` (Task 7).
- Sampling: submit `sampling_runs` concurrent `sampling_classifier.classify(text)` calls
  using a `with ThreadPoolExecutor(...) as executor:` context manager (mirroring
  `pipeline.py:97-101`'s existing pattern, so the pool is always cleaned up even if a
  future raises) — tracking each by its logical submission index 0..N-1 (not completion
  order) per AR-1.4. Catch per-sample failures individually (`# noqa: BLE001 - one bad
  sample shouldn't abort the row`, per INV-4) — a failed sample is simply excluded from
  voting (FR-1.6). **If zero samples succeed, `run_debate` raises** (propagate the last
  sample's exception, or a dedicated exception wrapping it) rather than returning a
  degraded/partial dict — this is required so `pipeline.py`'s existing per-row
  `except Exception: failed += 1` (`pipeline.py:109-110`) handling applies unchanged
  (FR-1.6); without this explicit guard the function would otherwise hit an
  `IndexError`/empty-tally error deep inside category processing instead.
- Per category (only once at least one sample succeeded): tally top labels (position 0)
  from successful samples (FR-1.2). The `allow_new_labels` prompt-level constraint itself
  is applied one level up, in how `cli.py` (Task 7) builds `sampling_classifier` (Task 2's
  `build_classification_model(..., allow_new_labels=...)` + Task 3's
  `build_system_prompt(..., allow_new_labels=...)`) — `run_debate` just tallies whatever
  top-label strings the samples actually returned. Bucketing behavior is conditional on
  `allow_new_labels` (matching FR-1.2/FR-1.5 exactly — the earlier draft of this task
  wrongly said to bucket unconditionally): when `allow_new_labels=True`, any
  `"none - ..."` value is bucketed under one shared key (e.g. `"none (new label)"`) for
  tallying purposes, while the representative sample retains its original suggested text;
  when `allow_new_labels=False`, an unexpected out-of-vocabulary top label (which
  shouldn't occur given Task 3's prompt fix, but isn't schema-enforced per INV-11) remains
  its own distinct, literal bucket — not merged with anything.
- Routing: consensus bypass (FR-1.3), escalation with leading/runner-up (FR-1.4), and the
  no-runner-up fallback (fewer than 2 distinct successful-sample labels — treat as bypass).
- Escalated categories run concurrently with each other (AR-1.4, same context-manager
  executor pattern as sampling): for each, call `critics[cat.name].classify(...)` with an
  assembled user message containing the text (clearly delimited as data under review, not
  instructions — defense-in-depth per the spec's Constraints), category name/description/
  labels, **the leading candidate, and the runner-up candidate** (the runner-up was
  computed by the routing step above but omitted from the assembled message in an earlier
  draft of this task — FR-2.1 explicitly asks the Critic to argue for the runner-up, so it
  must actually see it), parsed via that category's `build_critic_model` shape; if
  `challenges`, call `reconcilers[cat.name].classify(...)` with an assembled user message
  (text + category schema + leading candidate + critic's `proposed_label`/`argument`,
  itself framed as quoted data per Task 3's reconciler template), parsed via that
  category's `build_reconciler_model` shape.
- `{cat}_critic_argument` serialization: when challenging, a fixed format combining both
  fields, e.g. `f"Proposed: {proposed_label}. {argument}"`; when declining, just
  `argument` (the decline rationale) — Data Requirements' "proposed alternative +
  reasoning" only makes sense if both fields are actually concatenated into this one text
  column somehow, which wasn't specified concretely before.
- Failure handling (FR-2.3): wrap the Critic call and the Reconciler call each in their own
  `except Exception` (`# noqa: BLE001`, INV-4-compliant), falling back to the leading
  candidate. **Sanitization (FR-2.8), strengthened per critique:** record only the
  exception's type name plus a fixed, generic, stage-specific message (e.g.
  `f"{type(exc).__name__}: Critic call failed"` / `"...: Reconciler call failed"`) in
  `{cat}_debate_error` — **never** any part of `str(exc)` itself, not even truncated.
  Truncating raw exception text can still leak a short secret/URL/credential if it happens
  to fall within the truncated prefix; omitting the raw text entirely removes that risk
  rather than just shrinking it. Track `{cat}_challenged`/`{cat}_critic_argument`/
  `{cat}_reconciled` per the two distinct failure-state tables in spec FR-2.3.
- Assemble the final flat dict: for each category, `{cat}`, `{cat}_initial`,
  `{cat}_votes` (via `json.dumps` per AR-2.5 — not the bare dict), `{cat}_challenged`,
  `{cat}_critic_argument`, `{cat}_reconciled`, `{cat}_reconciler_reasoning`,
  `{cat}_debate_error`.
- Define the reserved suffix list as a module-level constant here (AR-2.4):
  `AUDIT_COLUMN_SUFFIXES = ("_initial", "_votes", "_challenged", "_critic_argument",
  "_reconciled", "_reconciler_reasoning", "_debate_error")` — imported by `pipeline.py`
  (Task 6) for both the collision check and the column-creation step.
**Verify:** every FR-1.x/FR-2.x Verify condition in the spec — including the all-samples-
failed case (`run_debate` raises; never a degraded/partial dict) and confirming the
Critic's assembled message actually contains the runner-up candidate, not just the leading
one — exercised directly against `run_debate` with fake `Classifier`-shaped stand-ins
(objects with a `.classify(text) -> dict` method). This is ad-hoc/local verification done
while writing this task, not yet the committed suite — Task 8 formalizes these same
scenarios into `tests/test_debate.py` and adds the remaining coverage (concurrency
ordering, CLI-level paths); this task isn't "done" until its own scenarios pass locally,
even before they're committed as formal tests.
**Covers:** FR-1.1, FR-1.2, FR-1.3, FR-1.4, FR-1.5, FR-1.6, FR-2.1, FR-2.2, FR-2.3, FR-2.8,
AR-1.2, AR-1.4, AR-2.3, AR-2.4, AR-2.5
**Notes:** the behavior when a Critic's `proposed_label` equals the leading candidate
(i.e., it "challenges" but proposes the same label) is not specified by the spec — do not
invent new rejection/decline-reinterpretation semantics for it silently; pass it through to
the Reconciler like any other challenge (the Reconciler's free-verdict framing, spec
FR-2.2, already tolerates landing back on the leading candidate). Flagged as an open edge
case in Risks below, not resolved here.

### Task 5: Architecture ADR for the revised dependency graph
**Goal:** Document the `debate.py`-shaped dependency-graph change and the new test-import
exception as an ADR, per `spec/ARCHITECTURE.md`'s own stated process — the binding doc
itself is updated later by `/spec-close`, not by this task.
**Files:** `spec/1-initial-classification-with-critics/ADR.md` (new)
**Dependencies:** None — describes a graph already fixed by this plan's design (Task 4's
concrete signature), not by its code being finished; can run in parallel with Tasks 1-3.
**Do:**
- State the INV-1 change: `pipeline.py` gains `debate.py` as an allowed import;
  `debate.py` imports `categories.py`, `classifier.py`, `schema.py`, `prompts.py`.
  Cite `spec/ARCHITECTURE.md` INV-1's current wording as the "before" state and the new
  Module Boundary Map row (`debate.py`: may import `categories.py`/`classifier.py`/
  `schema.py`/`prompts.py`; must not import `cli.py`) as the "after" state.
- State the INV-7 change: `debate.py` becomes a second accepted direct-import exception for
  tests, alongside `resources.py` (AR-1.7). Additionally — surfaced by plan critique — the
  *new* functions added to `schema.py` (`build_critic_model`, `build_reconciler_model`,
  Task 2) and `prompts.py` (`build_critic_prompt`, `build_reconciler_prompt`, Task 3) are
  also directly unit-tested per their own tasks' `Verify:` lines; declare these specific
  new functions (not the pre-existing `build_classification_model`/`build_system_prompt`,
  which remain public-API-only per today's INV-7) as accepted direct-import exceptions too.
- Reference the FR/AR IDs that require this change (AR-1.6, AR-1.7) and the spec path.
**Verify:** `ADR.md` exists, states both invariant changes precisely enough for
`/spec-close` to fold them into `spec/ARCHITECTURE.md` mechanically (exact before/after
wording for INV-1 and INV-7, exact new Module Boundary Map row).
**Covers:** AR-1.6, AR-1.7

### Task 6: `pipeline.py` wiring
**Goal:** `classify_csv` gains the `--critics`-related parameters, submits `debate.py`'s
`run_debate` instead of `classifier.classify` when enabled, validates the new config
up front, and extends the collision/restore logic to cover audit columns.
**Files:** `src/query_classification/pipeline.py`
**Dependencies:** Task 4
**Do:**
- Extend `classify_csv`'s signature: `critics: bool = False,
  critic_classifiers: dict[str, Classifier] | None = None,
  reconciler_classifiers: dict[str, Classifier] | None = None, sampling_runs: int = 5,
  consensus_threshold: int = 4, allow_new_labels: bool = False` (the existing `classifier`
  param becomes the *sampling* classifier when `critics=True` — already constructed with
  the right `temperature` by the caller). **`sampling_temperature` is deliberately not a
  `classify_csv` parameter** — surfaced by plan critique as a "dead" value otherwise:
  `pipeline.py` never uses it to build anything (the caller already baked `temperature`
  into `classifier`), so keeping it here would be an unused, misleading parameter for
  library callers. Its `>= 0`/finite validation (FR-1.7) belongs in `cli.py` (Task 7),
  at the point where `args.sampling_temperature` is actually consumed to construct the
  sampling classifier.
- At the top of `classify_csv` (alongside the existing `column in category_names` check,
  `pipeline.py:46-51`), when `critics=True`: validate `sampling_runs >= 1` and
  `1 <= consensus_threshold <= sampling_runs` (FR-1.7) — raise `ValueError` before any LLM
  call, matching the existing error style. Also validate `critic_classifiers`/
  `reconciler_classifiers` are both provided and each contains an entry for every category
  `name` — a missing role/category combination must fail loudly here, not surface as a
  `KeyError` deep inside a worker thread that would otherwise be silently counted as an
  ordinary failed row (a real gap surfaced by plan critique).
- Build one canonical helper — e.g. `generated_columns(category) -> set[str]` (using
  `debate.AUDIT_COLUMN_SUFFIXES`, imported from `debate.py`) — and use it consistently for
  every one of the following, rather than repeating ad-hoc column-name logic in each spot:
  - **Collision check (AR-1.5):** for each category, its `generated_columns(category)` must
    not collide with `column`, or with another category's `generated_columns(...)`. No
    separate check against "existing input CSV columns" is needed or correct: AR-1.5's own
    wording only prohibits collision with an existing column "that isn't itself one of
    these generated names" — a category's own audit columns already present from a prior
    `--critics` run on that same category *are* "one of these generated names," so they're
    already excluded from the prohibition by the spec's own text, not a separate case to
    special-case. This resolves plan critique's flagged ambiguity without needing
    `/spec-update`: the two checks above are the complete, unambiguous set AR-1.5 actually
    describes.
  - **Column creation** (`pipeline.py:59-62`): pre-create every column in
    `generated_columns(category)` for every category, when `critics=True`.
  - **Non-restore reset** (`pipeline.py:82-84`'s `else: df.loc[work_idx, category_names] =
    None` branch): when `critics=True`, also reset every `generated_columns(category)`
    column to `None` here, not just `category_names` — **critical fix surfaced by plan
    critique:** the original draft of this task only pre-created audit columns if absent,
    never cleared *stale* audit data already present in a fresh (non-restore) run's input
    CSV, which would leave old audit values on rows that fail sampling entirely or fall
    outside `--limit`, contradicting FR-1.6's "null result cells" behavior and corrupting
    the audit trail's internal consistency.
  - **Restore-from-separate-output copy** (`pipeline.py:64-75`): copy every
    `generated_columns(category)` column (not just the bare category value) when
    `critics=True` (FR-2.6).
  - **Completeness check** (`pipeline.py:79-81`'s `already_done`): additionally require
    `{cat}_votes` non-null per category when `critics=True` (FR-2.6) — `_votes` alone is
    sufficient as the completeness signal since it's always populated whenever a category
    was actually processed (consensus or debate), per Data Requirements.
- Change the per-row submission (`pipeline.py:99`) to
  `executor.submit(debate.run_debate, str(df.at[idx, column]), categories,
  classifier, critic_classifiers, reconciler_classifiers, sampling_runs=sampling_runs,
  consensus_threshold=consensus_threshold, allow_new_labels=allow_new_labels)` when
  `critics=True`, else today's `executor.submit(classifier.classify, ...)`.
**Verify:** with `critics=False`, `classify_csv` behaves byte-identical to today (existing
4 `pytest` tests + a new explicit "no `--critics` means no new columns" check). With
`critics=True` and fakes for the sampling `classifier` plus `critic_classifiers`/
`reconciler_classifiers`: every FR-1.7/FR-2.6/AR-1.5 Verify condition from the spec passes;
a fresh (non-restore) run with pre-existing stale audit column values in the input CSV
produces cleanly reset values, not leftover stale ones, for any row that ends up failed or
out-of-`--limit`; a missing category key in `critic_classifiers` raises `ValueError` before
any `.classify()` call happens.
**Covers:** FR-1.6, FR-1.7, FR-2.5, FR-2.6, AR-1.3, AR-1.5

### Task 7: `cli.py` wiring + README
**Goal:** New flags exist, are validated by argparse where sensible, construct the right
`Classifier` instances, and are documented.
**Files:** `src/query_classification/cli.py`, `README.md`
**Dependencies:** Task 6
**Do:**
- Add `--critics` (`store_true`), `--sampling-runs` (`type=int`, default `5`),
  `--sampling-temperature` (`type=float`, default `0.7`), `--consensus-threshold`
  (`type=int`, default `4`), `--allow-new-labels` (`store_true`), `--critic-model`
  (default `None`, help text notes it may route to a different provider than `--model` —
  spec Constraints' cross-provider disclosure requirement), `--reconciler-model` (same) to
  `build_parser` (`cli.py:24-95`).
- In `main()`, when `args.critics`: validate `args.sampling_temperature` is finite and
  `>= 0` (FR-1.7 — moved here from `pipeline.py`, Task 6, since this is where the value is
  actually consumed to construct a `Classifier`). Build the sampling classifier's
  `classification_model` via `build_classification_model(categories,
  allow_new_labels=args.allow_new_labels)` (Task 2) and its system prompt via
  `build_system_prompt(..., allow_new_labels=args.allow_new_labels)` (Task 3's extended
  signature — both halves of the no-invention constraint must use the same flag). Construct
  the sampling `Classifier` with `model_id=args.model,
  temperature=args.sampling_temperature` plus these (`max_retries`/`api_base` as today).
  Then build `critic_classifiers = {cat.name: Classifier(model_id=args.critic_model or
  args.model, system_prompt=build_critic_prompt(cat.name, cat.description,
  "; ".join(f"{l.value}: {l.description}" for l in cat.labels)),
  classification_model=build_critic_model(cat), max_retries=args.retries,
  api_base=args.api_base) for cat in categories}` (passing `build_critic_prompt` the plain
  strings Task 3 requires, not `cat` itself — the label-options string reuses the same
  `"value: description"` joining style as `schema.py`'s existing `label_docs` construction,
  `schema.py:32-34`) and the
  equivalent `reconciler_classifiers` dict using `build_reconciler_prompt`/
  `build_reconciler_model` and `args.reconciler_model or args.model` — one `Classifier`
  instance per category per role, per Task 4's resolved signature.
- Wire the new flags into the `classify_csv(...)` call (extending `cli.py:145-161`) —
  note `sampling_temperature` itself is *not* passed to `classify_csv` (Task 6), only used
  here to construct the sampling classifier.
- **Fix the bundled-resource guard's scope** (surfaced by plan critique): `cli.py:112-117`'s
  `uses_bundled_defaults` check currently only covers the main categories/system-prompt
  defaults, but the new Critic/Reconciler role prompts are *always* bundled (no
  `--critic-prompt`/`--reconciler-prompt` override exists, per spec Out of Scope) — so
  `check_default_resources_available()` must also be called whenever `args.critics` is
  `True`, independent of whether the main resources were overridden. Otherwise a wheel-mode
  user who supplies explicit `--categories`/`--system-prompt` still hits a bare
  `FileNotFoundError` building the Critic/Reconciler prompts instead of INV-3's actionable
  guard.
- Update `--workers`' help text to add the `--critics`-specific caveat (Constraints: no
  longer means zero concurrency when `--critics` is set).
- Update `README.md`: new flags in the Options table; a short `--critics` usage example;
  the concurrency-multiplier and cross-provider-data-egress disclosures from the spec's
  Constraints section; `debate.py` added to the existing "Project layout" module list;
  and — since this task's own Verify below claims "no missing flags" — **also add the two
  pre-existing undocumented flags `--api-base` and `--workers`** (a drift `spec/
  ARCHITECTURE.md`'s Stable Contracts already flags; otherwise this task's own claim would
  still be false after landing).
**Verify:** `python classify.py --help` lists all 7 new flags with accurate defaults, plus
provider-routing notes on `--critic-model`/`--reconciler-model`; `--critics --model X
--critic-model Y` constructs the Critic against model `Y` (FR-2.4); `--critics
--sampling-temperature nan` (or a negative value) exits with a clear error before any LLM
call; `--critics` with explicit `--categories`/`--system-prompt` but a simulated-missing
`resources/prompts/critic_prompt.txt` still triggers the actionable `FileNotFoundError`
guard (not a bare one); README's Options table is complete for every flag `cli.py` defines,
old and new.
**Covers:** FR-1.7, FR-2.4, FR-2.5

### Task 8: Ship unit tests
**Goal:** A `pytest`-runnable suite exercising every FR-1.x/FR-2.x Verify condition via
fakes, no live network calls — formalizing the ad-hoc verification already done locally
during Tasks 1-4/6/7 into the committed suite, plus the scenarios that only exist at the
integration level (CLI parsing, concurrency ordering).
**Files:** `tests/test_debate.py` (new)
**Dependencies:** Task 7
**Do:**
- Follow the existing `tests/test_building_blocks.py` conventions: plain `pytest`
  functions, `tmp_path` fixtures where file I/O is needed, import from the public API
  (`query_classification`) plus the accepted direct-import exceptions from Task 5's
  broadened ADR (`query_classification.resources`, `query_classification.debate`, and the
  new `schema`/`prompts` builder functions specifically).
- One test function (or a small parametrized group) per FR-1.x/FR-2.x Verify condition —
  reuse the exact scenarios already written into the spec and refined during Tasks 1-4
  (5-sample tallies, conditional `"none - ..."` bucketing per `allow_new_labels`,
  no-runner-up, all-samples-failed raising, Critic-decline vs. -challenge with the
  runner-up visible in its input, Critic-failure vs. Reconciler-failure, sanitized
  type-name-only error text, restore-from-separate-output, collision rejection, stale
  audit-column reset on fresh runs).
- Additional scenarios flagged by plan critique, not yet covered above: logical
  submission-index ordering holds even when sample futures complete out of order (e.g. a
  fake classifier that deliberately delays sample 0); `--sampling-temperature nan`/`inf`
  rejected alongside negative values; exact `.classify()` call counts on the Critic/
  Reconciler fakes (zero calls on consensus bypass, zero Reconciler calls on Critic
  decline); default vs. `--critic-model`/`--reconciler-model`-overridden model routing.
- Add an explicit `critics=False` regression test confirming `classify_csv`'s output is
  unchanged from today's baseline (same columns, same values) for a fixed fake-classifier
  scenario.
- Add a CLI-level test (via `build_parser()`/`cli.main()` with `Classifier` construction
  and `.classify()` monkeypatched, not a real LLM call) for at least one invalid-config exit
  path (e.g. `--consensus-threshold` exceeding `--sampling-runs`) confirming a clean
  non-zero exit and zero LLM calls attempted, replacing the plan's earlier vaguer "manual
  smoke test" language with an actual executable test.
**Verify:** `pytest` passes with 0 failures, including all new tests; existing 4 tests in
`tests/test_building_blocks.py` still pass unmodified.
**Covers:** FR-2.9

## Final Verification

- `.venv/bin/python -m pytest` — full suite (existing 4 + new `test_debate.py`) passes.
- `.venv/bin/python classify.py --help` shows all 7 new flags with correct defaults and
  the provider-routing notes on `--critic-model`/`--reconciler-model`.
- Task 8's CLI-level monkeypatched test (not a vague manual smoke test) covers at least: a
  consensus-bypass row, an escalated-and-reconciled row, and a Reconciler-failure row —
  confirm the printed/written columns match the Data Requirements table exactly.
- Confirm a `--critics`-off run's output CSV is column-identical to a pre-this-spec
  baseline run (no new columns leak in when the flag isn't passed).
- `git status` shows only the intended new/modified files listed in Code Impact, plus the
  pre-existing untracked files noted in Readiness (left untouched, not newly introduced).

## Documentation

- `spec/docs/` does not exist yet (per `spec/ARCHITECTURE.md`'s own header) — out of scope
  for this plan; consider `/spec-docs --full` as a separate follow-up once this and any
  other pending specs land.
- `README.md` — updated in Task 7.
- `AGENTS.md` — no change needed; its existing "Rules" already generically cover the new
  code (testing policy, noqa convention, resource-path centralization).
- `spec/ARCHITECTURE.md` — not edited directly by this plan; Task 5's `ADR.md` is folded in
  by `/spec-close` per the project's existing process.

## Spec Deviations

None identified. The spec (already refined through two critique rounds during
`/spec-write`) left `debate.run_debate`'s exact signature and the Critic/Reconciler Pydantic
field names unspecified as implementation detail — this plan makes concrete choices for
both (Task 2, Task 4) without contradicting anything the spec actually requires.

Plan critique raised three items that looked like they might need `/spec-update`; each
resolved without one, and it's worth recording why so a future critique round doesn't
re-raise the same question without this context:

- **The `prompts.py -> categories.py` import edge implied by an earlier draft's
  `build_critic_prompt(category: Category)` signature.** Resolved by redesign, not a
  spec/architecture change: Task 3 now takes plain strings (`category_name`,
  `category_description`, `label_options`) instead of a `Category` object, so no new
  import edge is introduced and AR-1.6's ADR scope didn't need to grow.
- **Whether AR-1.5's collision check must also reject pre-existing input columns that
  happen to match a category's own generated audit-column names.** Re-reading AR-1.5's
  exact wording resolved this: it only prohibits collision with an existing column "that
  isn't itself one of these generated names" — a category's own audit columns from a prior
  `--critics` run on that same category already *are* "one of these generated names," so
  they're excluded from the prohibition by the spec's own text. Task 6 implements exactly
  the two remaining, unambiguous checks (vs. `--column`, vs. another category's generated
  set); no additional "vs. arbitrary pre-existing column" check exists or is needed.
- **Whether "the schema/prompt actually discourages invention" claim in an earlier draft
  was true.** It wasn't — `resources/prompts/system_prompt.txt`'s static text
  independently instructed invention regardless of the schema's field descriptions. This
  was a genuine plan defect, fixed in Task 3 (a real prompt-template change), not a spec
  contradiction requiring `/spec-update` — FR-1.2 already required this behavior; the plan
  just hadn't correctly implemented it yet.

| # | Spec says | Plan does instead | Reason | Action required |
|---|-----------|-------------------|--------|-----------------|
| — | — | — | — | None identified |

## Risks

- **`proposed_label == leading candidate`:** the spec doesn't define what a Critic
  "challenge" that proposes the same label as the leading candidate means. Task 4
  deliberately does not invent new rejection/reinterpretation semantics for this — it's
  passed through to the Reconciler like any other challenge. Flag to the user if this comes
  up in practice; treat as a candidate for `/spec-update` rather than silently deciding
  more behavior here.
- **Rollback/checkpoint guidance:** Tasks 1-3+5 are additive/independent (no existing
  behavior changes). Task 4 (new module) carries no regression risk to existing code.
  Tasks 6-7 are the highest-risk (touch the existing `classify_csv`/`cli.py` hot path) —
  the `critics=False` regression checks in Task 6/8 are the checkpoint before considering
  those tasks done. **This plan does not mandate a commit per task** — `git revert`ing
  Tasks 6-7 "independently" is only meaningful if they were actually committed as separate,
  disjoint commits; make an explicit commit at each task boundary (or at minimum after
  Task 4, before Task 6 starts touching the hot path) if a clean revert point matters more
  than usual for this change.
- **Non-atomic CSV writes are a pre-existing, unaddressed Known Gap** (per
  `spec/ARCHITECTURE.md`), not something this spec fixes: a crash mid-write can still leave
  a truncated output file, and reverting *code* after a bad run doesn't restore a
  partially-overwritten *input* file when `--output` was omitted. Recommend always using an
  explicit `--output` (never omitted) and a small `--limit` canary run when first trying
  `--critics` against real data, rather than relying on rollback after the fact.
