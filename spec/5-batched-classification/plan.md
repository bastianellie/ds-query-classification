# Implementation Plan: 5-batched-classification

**Status:** Ready
**Date:** 2026-09-15
**Spec:** spec/5-batched-classification/spec.md
**Plan Critique:** spec/5-batched-classification/plan-critique-consolidated-v-3.md

## Overview

Third planning pass, and the first written after spec 6 (classifier-error-path) shipped and
merged to `main`. Spec 6 built, as real code, almost exactly the failure-classification
foundation this plan's first two versions were independently designing from scratch —
`classify_failure`/`FailureKind`'s two-axis classification, a generic `_DROPPABLE_PARAMS`
ladder, and unconditional truncation/empty-response/policy-refusal detection. This plan
consumes those directly rather than re-deriving them, which shrinks Task 1 and Task 3
substantially and removes an entire body of custom litellm-hierarchy logic from `batching.py`.
The remaining shape is unchanged: build `--batch` bottom-up (schema/classifier primitives,
then `batching.py`, then `pipeline.py`, then the two CLI entry points, then docs/ADR), solo,
sequential, single-direction dependencies.

## Readiness

- **Checklist:** all 10 items `[x]` in the spec.
- **Open questions:** none. Four `/spec-update` rounds resolved them in sequence: the arity cap
  default (50) and `--workers` semantics were user decisions; the v-2 spec-critique round cut
  `--critics` batching and redesigned FR-2.5; the v-1 plan-critique round's five spec-level
  defects were applied; the v-2 plan-critique round found two of those five resolved on false
  premises and required a further fix; and this round reconciles the whole spec against spec
  6's real, merged implementation, which now supplies most of what FR-2.5/FR-3.2/AR-5.1 used to
  build themselves.
- **External API/library verification:** all verified live against the installed stack
  (litellm 1.101.0 as installed in this spec's worktree setup, pydantic 2.13.5) — re-confirmed
  for this plan, not merely carried over, since a dependency version drifted between planning
  rounds and every hierarchy fact spec 6's own history shows is exactly the kind of claim that
  silently breaks on a version bump:
  - `litellm.get_model_info` — resolves the repo's real Cerebus slugs after prefix-stripping
    and chunk-down; raises bare `builtins.Exception` on failure; a large fraction of the
    bundled model map lacks `max_input_tokens`.
  - `litellm.token_counter` — returns a count for unrecognized model ids rather than raising.
  - `litellm.suppress_debug_info = True` — fully suppresses the `Provider List:` stderr noise,
    which `contextlib.redirect_stderr` cannot capture.
  - `litellm.get_supported_openai_params` — reports both `max_tokens` and
    `max_completion_tokens` supported for this repo's models.
  - `pydantic.create_model(__base__=Row, __config__=ConfigDict(extra="forbid"))` — accepted,
    yields `extra="forbid"`, leaves `Row` unmutated.
  - `type_to_response_format_param` on a positional batch model emits `strict: true` with
    exactly one `minItems` (`$defs` dedupes the nested row schema), upholding AR-2.1.
  - **New for this round:** `classifier.classify_failure`/`FailureKind`/`TruncatedResponseError`/
    `_DROPPABLE_PARAMS`/`Classifier.__init__` all confirmed present at their spec-6-shipped
    shape (see Interfaces) — `Classifier.__init__` does **not** yet have `max_tokens`, confirming
    that part of FR-2.5 is still this plan's job.
- **Repository state:** dirty, but only with two pre-existing user-owned items unrelated to
  this spec (`M experiments/ag-news/analyze_run.ipynb`, untracked
  `data/example_queries_classified.csv`). Neither is touched by this plan.
- **Test baseline:** 290 passed (229 pre-spec-6 + 61 spec 6 added; verified fresh for this
  round, since the baseline changed since the last planning pass).

## Code Impact

- **Modules:** new `src/query_classification/batching.py`. Modified: `schema.py`,
  `classifier.py`, `pipeline.py`, `cli.py`, `experiment.py`. **Not touched:** `debate.py`,
  `multi_model.py`, `induction.py`, `dataset_io.py`, `categories.py`, `prompts.py`,
  `resources.py`, `__init__.py`. Spec 6 already modified `classifier.py` and
  `tests/test_cerebus.py` in a prior, separate, merged change — this plan's edits to those two
  files are additive on top of spec 6's shipped state, not a redo of it.
- **Database/schema:** none. `run_config.json`'s `_SCHEMA_VERSION` 4 → 5 with a new `batch`
  object (FR-4.1).
- **API/interfaces:** see Interfaces below.
- **Config/scripts:** `README.md`, three `experiments/*/run_experiment.sh`, three
  `experiments/*/README.md`, `spec/ARCHITECTURE.md`, new
  `spec/5-batched-classification/ADR.md`.
- **Key files:** `pipeline.py:218-277` (the loop), `classifier.py:228-270` (`FailureKind`,
  `TruncatedResponseError`, `_DROPPABLE_PARAMS` — spec 6, read-only for this plan except the one
  edit noted in Task 1), `:428-473` (`__init__`/`_completion_kwargs` — this plan's one edit
  site), `:479-527` (`_attempt_completion`'s ladder walk — spec 6, unedited by this plan),
  `schema.py:17-63`, `experiment.py:86` (`_SCHEMA_VERSION`), `:336-360`
  (`_resolve_classifier_models`), `:412` (`_validate_args`), `:496`
  (`_resolve_induction_sizing`, the resolver precedent), `:888` (config literal), `:1012`
  (induction call), `:1054` (classification phase), `:1138` (call site), `:1213` (failure
  path), `cli.py:42` (`build_parser`), `:351-368` (model/prompt construction), `:408-459`
  (classifier construction + call site). The `experiment.py`/`cli.py`/`pipeline.py` line
  numbers are confirmed unchanged since the last planning round — spec 6 touched only
  `classifier.py`, `tests/test_cerebus.py`, and `spec/ARCHITECTURE.md`.

## Interfaces

Pinned here because Tasks 2–6 build against them. Two names change from the last planning
round's draft: `ResponseTruncatedError` never shipped under that name — spec 6's real class is
`TruncatedResponseError` — and the isolable/retryable decision is no longer this plan's own
enum; it's `classifier.FailureKind`, already real.

```python
# schema.py — new
build_batch_model(row_model: type[BaseModel], n: int) -> type[BaseModel]

# classifier.py — ALREADY SHIPPED by spec 6, this plan only reads/extends these:
class FailureKind(NamedTuple):              # (retryable: bool, isolable: bool)
    retryable: bool
    isolable: bool
class TruncatedResponseError(Exception): ...
class PolicyRefusalError(Exception): ...
class EmptyResponseError(Exception): ...
_DROPPABLE_PARAMS: tuple[str, ...] = ("temperature",)   # this plan prepends "max_tokens"
def classify_failure(exc: Exception) -> FailureKind: ...

# classifier.py — this plan's one addition:
Classifier.__init__(..., max_tokens: int | None = None)   # new parameter; _completion_kwargs
                                                            # emits it only when set, same
                                                            # conditional pattern as api_base/
                                                            # temperature/api_key/extra_headers

# batching.py — new
@dataclass(frozen=True)
class TokenBudgets:
    max_input: int | None
    max_output: int | None
    resolved_input_model: str | None             # the candidate FR-1.2's walk landed on
    resolved_output_model: str | None

resolve_token_budgets(model_id: str, input_override: int | None,
                      output_override: int | None) -> TokenBudgets

@dataclass(frozen=True)
class BatchPlan:
    arity: int
    was_trimmed: bool
    oversized_row_index: int | None       # set only when a single row alone exceeds a budget

def plan_batch(rows: list[str], *, start: int, mode: str, fixed_size: int | None,
               max_size: int, budgets: TokenBudgets, system_prompt: str,
               categories: list[Category], batch_model_for: Callable[[int], type[BaseModel]]
               ) -> BatchPlan: ...

class BatchStats:                                # lock-protected
    def record_call(self, arity: int) -> None: ...
    def record_trim(self) -> None: ...
    def record_bisection(self) -> None: ...
    def snapshot(self) -> dict[str, int | float | None]:
        # a freshly-built dict each call (never a stored reference the caller could mutate
        # back into this object's own counters): {"batched_calls": int, "min_arity": int | None,
        # "mean_arity": float | None, "max_arity": int | None, "trims": int, "bisections": int}
        # -- all-None/zero fields when no batched call has ever been recorded, matching
        # FR-4.1's null-vs-zero rule.
        ...

class BatchRunner:                               # owns the arity cache, the lock, AND batch
                                                  # formation -- plan_batch has no other caller
    def __init__(self, classifier_factory: Callable[[int], Classifier],
                 budgets: TokenBudgets, stats: BatchStats, max_size: int, mode: str,
                 fixed_size: int | None, system_prompt: str, categories: list[Category],
                 batch_model_for: Callable[[int], type[BaseModel]]): ...
    def iter_batches(self, rows: list[tuple[int, str]]) -> Iterator[list[int]]:
        # `rows` is (original_dataframe_index, text) pairs, in remaining order -- carrying the
        # true row identity through `--restore`/`limit` filtering. Yields successive lists of
        # ORIGINAL DataFrame indices (never positions into `rows`, never the row text itself),
        # each formed by one call to `plan_batch` against the remaining rows' texts. Internally
        # translates `plan_batch`'s position-based `BatchPlan.oversized_row_index` back through
        # `rows` to the real DataFrame index before FR-1.6's named-index warning is logged --
        # a warning naming a position within the filtered remaining list would be useless to a
        # user trying to find the offending row in their CSV. The sole caller of `plan_batch` --
        # pipeline.py (Task 4) never calls it directly, only this method.
        ...
    def run(self, texts: list[str]) -> list[dict[str, Any] | Exception]: ...
        # internally: classify_failure(exc) -> FailureKind, then bisect if .isolable,
        # else retry-at-original-size (bounded) if .retryable, else fail immediately.
        # No re-derivation of litellm's exception hierarchy inside this method -- the
        # branch is exhaustive by construction (three cases over two booleans), so there
        # is no "unrecognized" case to default or test for, unlike a type-enumeration design.

# pipeline.py
classify_csv(..., batch_runner: BatchRunner | None = None) -> Path
# no separate batch_mode/batch_size parameters -- BatchRunner already owns `mode`/`fixed_size`
# (it was constructed with them), so classify_csv only needs to know whether batching is on at
# all (`batch_runner is not None`). Passing mode/size again here would let the caller and the
# runner disagree about which mode is active.
```

`plan_batch` takes the model id (via `budgets`), the rendered system prompt, the categories,
and a batch-model factory — it is **not** a pure function of (rows, budgets, cap, mode).
FR-1.6 requires `token_counter` over the rendered payload *and* the serialized
`response_format` schema, both of which need those inputs.

`classify_csv`'s new parameters are keyword-with-defaults, so none of the ~24 existing call
sites in `tests/test_debate.py` and `tests/test_multi_model.py` change.

## Project Constraints

- **INV-1** (one-directional acyclic imports): `batching.py` is a new node importing only
  `categories.py`, `classifier.py`, `schema.py`, `prompts.py`. `pipeline.py`, `cli.py`, **and**
  `experiment.py` all import it. Requires an ADR amending INV-1 and the Module Boundary Map —
  both are closed per-module lists, so the ADR must edit the existing `pipeline.py`, `cli.py`,
  and `experiment.py` rows as well as adding a `batching.py` row. Precedent:
  `spec/4-multi-model-classification/ADR.md` ADR-001.
- **INV-4**: every broad `except Exception` needs `# noqa: BLE001` plus a justification
  comment. Applies to FR-1.2's per-candidate catch (AR-1.3) and FR-3.1's raising-future catch.
- **INV-6**: worker threads make only the read-only LLM call; all `df` mutation stays on the
  main thread. FR-3.1's row-at-a-time application preserves this.
- **INV-7**: the classifier-side names (`FailureKind`/`classify_failure`/the three
  response-failure exception types/`_DROPPABLE_PARAMS`/`Classifier._complete`/
  `_attempt_completion`) are **already amended in `spec/ARCHITECTURE.md` by spec 6**
  (`spec/6-classifier-error-path/ADR.md` ADR-001) — **this plan's ADR must not re-request or
  re-amend those names**, which would create a duplicate, conflicting amendment record. But
  this spec introduces its **own**, distinct new INV-7 exceptions per AR-5.1 — `batching` (all
  functions + the private `_DATA_START`/`_DATA_END` constants), `induction._DATA_START`/
  `_DATA_END`, and `schema.build_batch_model` — which are **not yet** in `spec/ARCHITECTURE.md`
  and must be added to both the `tests/` Module Boundary Map row and INV-7's own rule/evidence
  text by this spec's own work (Task 6), the same way spec 6 added its names, rather than
  deferred to `/spec-close`'s ADR-promotion step.
- **INV-8** (category name uniqueness) and **INV-11** (schema enforces cardinality/type only)
  unchanged.
- **Testing policy** (`AGENTS.md`): no live network/provider calls; LLM calls faked. Tests ship
  in the same task as the code they cover — no deferred test task.
- **PEP 604 unions + `from __future__ import annotations`** in every new/modified module under
  `src/query_classification/`.
- **Verification command**: `.venv/bin/python -m pytest`. No lint/typecheck/CI gate exists.

## Implementation Strategy

- **Size:** large (7 tasks), though Tasks 1 and 3 are now markedly smaller than the previous
  planning round sized them, since spec 6 already did most of their original work.
- **Execution mode:** solo, sequential.
- **Parallelizable work:** none across tasks. Tasks 2 and 3 both write
  `batching.py`/`tests/test_batching.py`; Tasks 5a and 5b touch disjoint file sets but are
  sequenced so 5b mirrors a known-good 5a. Within a task, tests are written alongside the code;
  the Final Verification fan-out runs concurrently.
- **Sequential blockers:** Task 2 needs Task 1's `build_batch_model` to count the
  `response_format` schema (FR-1.6). Task 3 needs Task 2's `plan_batch` and Task 1's
  `max_tokens`/ladder change (the classifier factory it builds must set `max_tokens` on
  constructed classifiers). Task 4 needs Task 3's `BatchRunner`. Task 5a needs Task 4's
  `classify_csv` signature. Task 5b mirrors 5a. Task 6's `DRY_RUN` verification needs 5a/5b's
  real flags.
- **No expected-red checkpoints.** No task changes a signature an existing caller depends on:
  `Classifier.max_tokens` defaults to unset, `build_batch_model` is new, and `classify_csv`'s
  new parameters are keyword-with-defaults. The one existing assertion that *would* go red
  (`tests/test_experiment.py`'s `schema_version == 4`) is edited inside Task 5a's `Do` list, not
  merely mentioned in Notes. Unlike the previous planning round, **no test edit is needed in
  `tests/test_cerebus.py`'s `_FakeChoice`** — spec 6 already added `finish_reason` to it.
- **Baseline capture, before Task 1 begins (not merely "before Task 6" for the scripts).** Run
  a faked-LLM classification over a fixed small CSV with today's code and save its output CSV
  byte-for-byte, plus the sequence of `completed` values at each interim `to_csv` call. This is
  the fixture Final Verification's no-batch regression check compares against; capturing it
  after any of Tasks 1–5 land would make the comparison compare the new code against itself.

## Implementation Tasks

### Task 1: Batch schema builder and `Classifier.max_tokens`
**Goal:** `schema.build_batch_model` produces a strict positional batch model, and `Classifier`
can carry an output cap that batched calls set — using spec 6's already-shipped truncation
detection and parameter-drop ladder as-is, extending neither's mechanism, only their inputs.
**Files:** `src/query_classification/schema.py`, `src/query_classification/classifier.py`,
`tests/test_batching.py` (new)
**Dependencies:** None
**Do:**
- Add `build_batch_model(row_model, n)` to `schema.py`: `ValueError` for `n < 1`; a strict
  nested copy via `create_model(__base__=row_model, __config__=ConfigDict(extra="forbid"))`
  leaving `row_model` unmutated; outer model with required `result_1..result_n` and
  `extra="forbid"`.
- In `classifier.py`: add `max_tokens: int | None = None` to `Classifier.__init__`; emit it
  from `_completion_kwargs` only when set, matching the existing conditional pattern for
  `api_base`/`temperature`/`api_key`/`extra_headers`.
- Prepend `"max_tokens"` to the module-level `_DROPPABLE_PARAMS` tuple:
  `("max_tokens", "temperature")`. `max_tokens` goes first because it is only a guard —
  proceeding without it reproduces today's unbatched behavior exactly — whereas `temperature`
  is semantically meaningful and should survive if `max_tokens` alone was rejected. Update the
  adjacent comment on `_DROPPABLE_PARAMS`/`_completion_kwargs`, which currently says a future
  output-cap parameter would be *appended* and that `temperature` is the only parameter
  `_completion_kwargs` emits — both become false once this task lands.
- **Do not** add a new exception type, a `finish_reason` inspection, or any gating logic —
  `TruncatedResponseError`/`classify_failure`/the ladder-walk mechanism already exist and
  already apply unconditionally to every call, batched or not. This task only changes what
  `Classifier` accepts and what the (already-generic) ladder iterates over.
**Verify:** `.venv/bin/python -m pytest -q` (full 290-test baseline stays green — this task is
purely additive); new tests cover FR-2.1's four arity/extras cases,
`build_classification_model`'s output staying non-strict, `build_batch_model(…, 0)` raising,
`strict: true` with exactly one `minItems`; a classifier constructed with `max_tokens` set
emits it in `_completion_kwargs`, one built without it does not; a fake rejecting both
`max_tokens` and `temperature` in turn (via `UnsupportedParamsError`) drops `max_tokens` first
and leaves `temperature` intact on the next attempt, reusing spec 6's own ladder-walk — this
task does not reimplement or retest that walk's internals, only that the new entry
participates in it correctly.
**Covers:** FR-2.1, FR-2.5, AR-2.1, AR-2.2, FR-5.1 (partial)
**Notes:** This task is far smaller than an earlier planning round sized it — that round
predated spec 6 and planned to add a dedicated truncation exception gated on whether a cap was
sent, plus rewrite `_attempt_completion`'s except-chain to generalize the parameter drop. Spec
6 already shipped both, unconditionally, for every call. Nothing here touches
`tests/test_cerebus.py`: spec 6 already added `finish_reason` to `_FakeChoice`, so there is no
fixture gap left for this task to close.

### Task 2: `batching.py` — budget resolution, sizing, payload assembly
**Goal:** resolve token budgets from a model id, decide a batch's arity under both modes, and
render the wire payload.
**Files:** `src/query_classification/batching.py` (new), `tests/test_batching.py`
**Dependencies:** Task 1
**Do:**
- `resolve_token_budgets` (FR-1.2): the four-step candidate walk, resolving input and output
  **independently**, treating a hit that lacks the sought value as a miss, recording which
  candidate each side resolved to; set `litellm.suppress_debug_info = True` at the point of
  resolution (AR-1.2) and catch broad `Exception` per candidate with the INV-4 comment (AR-1.3).
- `plan_batch` (FR-1.5–FR-1.7): greedy accumulation bounded by `max_size` **during** growth;
  additive shortlist then a rendered-payload `token_counter` measurement including the
  serialized `response_format` schema, reducing and re-measuring until it fits; floor of 1,
  returning a `BatchPlan` whose `oversized_row_index` is set (and `arity == 1`) when even a
  single row alone exceeds a budget — this is the channel FR-1.6/FR-1.7's named-index warning
  reads from, since a bare `int` arity has nowhere to carry which row was oversized; fixed-mode
  per-iteration trim that resets each iteration and never counts a short final iteration as a
  trim.
- Payload assembly (FR-2.2, AR-2.3): 1-based ordered array via `json.dumps`, wrapped in local
  `_DATA_START`/`_DATA_END` copies with a comment naming `induction.py:31-32` as the origin.
**Verify:** `.venv/bin/python -m pytest -q`; new tests cover FR-1.2's four resolution cases
plus input-known/output-missing, FR-1.5's cap **with an injected `TokenBudgets`** rather than a
real model lookup (so the test exercises this plan's logic, not litellm's bundled data file),
FR-1.6's estimate-fits-but-rendered-payload-does-not reduction and the oversized-single-row
warning, FR-1.7's trim/reset/floor-of-1 forward progress, FR-2.2's adversarial round-trip, and
AR-2.3's delimiter equality with `induction.py`.
**Covers:** FR-1.2, FR-1.5 (sizing enforcement), FR-1.6, FR-1.7, FR-2.2, AR-1.2, AR-1.3,
AR-2.3, FR-5.1 (partial)
**Notes:** `plan_batch` is not a pure function of (rows, budgets, cap, mode) — see Interfaces.
It has exactly one caller in the whole plan: `BatchRunner.iter_batches` (Task 3). Nothing in
this task or `pipeline.py` (Task 4) calls it directly. `suppress_debug_info` is set once and
deliberately **not** restored: it is scoped to batched runs by construction (a non-batched run
never calls resolution), and a `try`/`finally` restore would race across worker threads.

### Task 3: `batching.py` — classifier factory/cache, batched call via `classify_failure`, stats
**Goal:** an arity-keyed classifier cache, a batched call whose bisection policy is derived
directly from spec 6's `classify_failure`, and the thread-safe stats collector.
**Files:** `src/query_classification/batching.py`, `tests/test_batching.py`
**Dependencies:** Tasks 1, 2
**Do:**
- `BatchRunner`'s arity cache (FR-2.4): keyed on arity, run-scoped as an instance attribute
  (never a module global), with lookup-and-construction under a lock; built via the
  caller-supplied factory, which must set `max_tokens` on every classifier it constructs
  (Task 1's addition) alongside the settings already carried over.
- `BatchRunner.iter_batches(rows)` (FR-1.5–FR-1.7, giving `plan_batch` its owner): `rows` is
  `(original_dataframe_index, text)` pairs in remaining order — this is what lets the
  named-index warning survive `--restore`/`limit` filtering. Repeatedly calls `plan_batch`
  against the remaining rows' texts (using the mode/fixed-size/system-prompt/
  categories/batch-model-factory it was constructed with) and yields each resulting batch as a
  list of the covered rows' **original DataFrame indices** (translating `plan_batch`'s
  position-based result back through `rows`), advancing past however many rows the returned
  `BatchPlan` covers each time, until none remain. When `oversized_row_index` is set, resolves
  it to the corresponding original index before logging the named-index warning, and yields
  that one original index alone.
- `BatchRunner.run` (FR-3.2): for each batched call's failure, call
  `classifier.classify_failure(exc) -> FailureKind`. Branch **only** on its two booleans:
  `kind.isolable` → split into halves and recurse to size 1 (checked first, regardless of
  `kind.retryable` — splitting is tried ahead of retrying because it's more informative);
  else `kind.retryable` → retry at original size, at most 2 further attempts with exponential
  backoff from `retry_delay`, then fail every row; else → fail every row immediately, no split,
  no retry. **Do not enumerate litellm exception types here** — that hierarchy knowledge lives
  entirely in `classify_failure` (spec 6), and duplicating it in `batching.py` is exactly the
  mistake an earlier round of this spec made twice.
- `BatchStats` (FR-4.1): lock-protected counters for batched `classify` invocations, arity
  min/mean/max over all invocations including bisected ones, trims, and bisection events. Its
  `snapshot()` returns exactly the fields FR-4.1's `run_config["batch"]` block needs
  (`batched_calls`, `min_arity`, `mean_arity`, `max_arity`, `trims`, `bisections`), all `None`
  (or `0` for the counters) until the first call is recorded — matching FR-4.1's null-vs-zero
  rule at the source rather than requiring Task 5a to translate a different shape.
**Verify:** `.venv/bin/python -m pytest -q`; new tests assert exact `classify` call counts per
class against a counting fake, using real exception types to prove the derivation is correct
rather than assumed: a `pydantic.ValidationError` (isolable *and* retryable, per
`classify_failure`) in one row of 8 resolves the other 7 and bisects to isolate the failure; a
`litellm.ContextWindowExceededError` in every row of a 4-row batch bisects all the way to size
1 (proving `classify_failure` correctly classifies it isolable, and that nothing in this task
special-cases it); a `litellm.RateLimitError` retries at original size at most twice with
doubling delays, no split; a `litellm.AuthenticationError` fails immediately, no split, no
retry; a genuinely unrecognized exception type (one `classify_failure`'s chain has no entry
for, distinct from any named-fatal type) also fails immediately via the same branch, exercising
`classify_failure`'s unmatched-default `FailureKind(False, False)` path specifically rather
than only a named-fatal entry; a batch of 1 that fails increments `failed` without raising. Also: barrier-forced
concurrent misses build exactly one model per arity; Cerebus `api_key`/`extra_headers`/
`api_base`/`max_tokens` all survive the factory; `BatchStats` counters/`snapshot()` behave
under concurrent increments (tested in isolation here — the end-to-end assertion lands in
Task 5a).
**Covers:** FR-2.4, FR-3.2, AR-1.1 (module boundaries), FR-4.1 (stats object), FR-5.1 (partial)
**Notes:** This task no longer needs its own hierarchy-trap tests or its own "do not add a
handler for X" caveats — the whole point of consuming `classify_failure` directly is that
`batching.py` carries zero litellm-specific knowledge, and spec 6's own test suite (not this
one) is what pins the hierarchy facts. `BatchRunner.run`'s three-way branch is exhaustive by
construction — it covers all four combinations of `FailureKind`'s two booleans, so unlike the
pre-spec-6 design there is no "unrecognized exception" case to default or write a test for; a
prior planning round asked for exactly such a test, and it no longer applies. Bisection must
still run only after `Classifier.classify`'s own `max_retries` loop is exhausted, so the two
retry layers compose.

### Task 4: `pipeline.classify_csv` — submit batches, apply rows one at a time
**Goal:** `classify_csv` batches its work when asked, preserves today's per-row accounting and
flush cadence exactly, and refuses unsupported mode combinations at the library level.
**Files:** `src/query_classification/pipeline.py`, `tests/test_batching.py`
**Dependencies:** Task 3
**Do:**
- Add the Interfaces-pinned parameters; keep the existing per-row submission as the untouched
  code path when `batch_runner is None`.
- Form batches by calling **`batch_runner.iter_batches(remaining_rows)`** (Task 3), where
  `remaining_rows` is the `(original_dataframe_index, text)` pairs for the rows that remain
  **after** `--restore` filtering and `limit` truncation — `pipeline.py` already has each
  remaining row's DataFrame index from that filtering step, so this carries no new state, only
  passes what was previously discarded. `pipeline.py` never calls `plan_batch` itself, only
  this method; each yielded batch is a list of original DataFrame indices, which `pipeline.py`
  uses to look up texts for `batch_runner.run` and to apply outcomes back to the correct rows;
  submit each batch to the existing `ThreadPoolExecutor`; apply each future's per-row outcomes
  one at a time on the main thread — incrementing `classified`/`failed`, advancing `completed`
  by one, updating tqdm, and evaluating the existing `completed % save_every == 0` condition
  unchanged (INV-6 preserved).
- Catch a batch future that itself raises and convert it to one failure per covered row, with
  the INV-4 comment, so `classified + failed == completed == total` holds.
- Reject a batch runner together with `critics=True` or a **resolved multi-classifier**
  `models` mapping, alongside the existing `critics and have_models` check.
**Verify:** `.venv/bin/python -m pytest -q`; new tests assert `--batch 5 --workers 2` over 20
rows produces 4 batches with input row order preserved; with `--batch 7` over 200 rows the
interim-write **positions** (not merely the count) match an unbatched run, observed by
monkeypatching `pandas.DataFrame.to_csv` and recording `completed` at each call; a raising
future leaves `classified + failed == total`; `--restore` over a partially-complete batched
output classifies exactly the remaining rows (FR-3.4); `limit=3` with a `BatchRunner`
constructed at `fixed_size=10` produces one batch of 3 (both via the library-level `limit`
parameter directly, and — in Task 5a, where `--test-limit` first exists as a CLI flag — via
`experiment.py run --test-limit 3 --batch 10` end to end, so the CLI-level flag is exercised
too, not only the library parameter it resolves to); `classify_csv(batch_runner=…, critics=True)` and the multi-model equivalent each raise
`ValueError`.
**Covers:** FR-3.1, FR-3.3 (library level), FR-3.4, FR-5.1 (partial)
**Notes:** `save_every` has no CLI flag — it is a `classify_csv` parameter, so the flush test
drives it directly rather than through argv. Asserting write positions rather than the count is
what makes the test non-tautological.

### Task 5a: CLI wiring — `experiment.py`
**Goal:** `experiment.py` exposes and validates the four flags, resolves budgets before any LLM
call, builds the batch prompt and factory, and records the batch block in `run_config.json`.
**Files:** `src/query_classification/experiment.py`, `tests/test_experiment.py`
**Dependencies:** Task 4
**Do:**
- Add the four flags to the **`classification` parent group** (the pattern used by `--workers`
  at `:212` and `--sampling-runs` at `:228`), so they reach `classify`/`run` but not `induce`;
  guard every read with the `getattr(args, …, None)` pattern `_resolve_induction_sizing`
  already uses for exactly this reason.
- Add a resolver normalizing `--batch` to `(None, None)`/`("dynamic", None)`/`("fixed", n)` on
  `args`, following `_resolve_induction_sizing`/`args.resolved_induction_sizing` (`:496`).
- **Resolve token budgets inside `_validate_args` (`:412`)** and store them on `args`. This is
  load-bearing: `run_induction` fires at `:1012`, before the classification phase at `:1054`,
  so lazy resolution would let an induction run make real LLM calls before an unresolvable
  budget was detected — violating FR-1.3's zero-LLM-call condition.
- Validation matrix, all in the same fail-fast block: the three token/size flags require
  `--batch`; `--batch INT` must not exceed `--batch-max-size`; `--batch` is rejected with
  `--critics`, with a **resolved multi-classifier** configuration (`--models` with 2+ values or
  `--n-classifiers > 1` — **not** single-value `--models X`, which legally resolves to one
  classifier per `:336-360`).
- Diagnostics: FR-1.3's sentence verbatim on its own line plus a second line naming the
  unresolved side; a warning when only the output budget is unresolved; FR-4.2's comparability
  caveat exactly once.
- Prompt and factory (FR-2.3, FR-2.4): build the batch prompt from the **single-row** model
  plus the appended positional/independence/data-not-instructions text; pass a factory closing
  over the run's `gateway_kwargs` and settings, including `max_tokens`.
- Provenance (FR-4.1): bump `_SCHEMA_VERSION` to 5; add the `batch` object to the initial
  config literal with observed-outcome fields initialized to `null`; overwrite from the stats
  snapshot after `classify_csv` returns. Do **not** add `"batch"` to the failure path's
  `setdefault` list (`:1213-1222`) — that list holds only keys assigned later inside the `try`.
- **Edit `tests/test_experiment.py:1446`**, the suite's only `schema_version == 4` assertion.
**Verify:** `.venv/bin/python -m pytest -q`; new tests assert the resolver's three shapes, each
row of the validation matrix (including that single-value `--models X` with `--batch` is
**accepted**), FR-1.3's two output lines with **zero** LLM calls on a `run` invocation that
would otherwise induce, `schema_version == 5`, the `batch` object's null-vs-zero distinction, a
zero-work run (every row already complete under `--restore`), `batch == null` for a
`classify`-less run, a mid-classification failure still writing a `batch` key with null
observed fields, the caveat printing exactly once, and `--test-limit 3 --batch 10` (the CLI
flag, not only the library `limit` parameter Task 4 tested) producing exactly one batch of 3.
**Covers:** FR-1.1, FR-1.3, FR-1.4, FR-1.5 (flag validation), FR-2.3, FR-3.3 (CLI level),
FR-4.1, FR-4.2, FR-5.1 (partial)
**Notes:** Pin the persisted `batch` JSON key names here; the spec lists the fields but not
their serialized names.

### Task 5b: CLI wiring — `cli.py`
**Goal:** `cli.py` gains the same flag surface, validation, diagnostics, prompt, and factory as
`experiment.py`, with no `run_config` equivalent.
**Files:** `src/query_classification/cli.py`, `tests/test_cerebus.py`
**Dependencies:** Task 5a
**Do:**
- Mirror 5a's flags, resolver, validation matrix, and diagnostics in `build_parser` (`:42`) and
  `main`, reusing `cli.py`'s existing `task_description`/`extra_prompt` reads (`:351-368`) for
  the batch prompt and its `gateway_kwargs` (`:384-389`) for the factory.
- Pass the runner and batch parameters at the `classify_csv` call site (`:459`).
**Verify:** `.venv/bin/python -m pytest -q`; CLI-level tests in `tests/test_cerebus.py` (INV-7's
home for `cli.build_parser`/`cli.main`) assert the same validation matrix and diagnostics as
5a's, and that a Cerebus-configured run's batch classifier carries the gateway headers and
`max_tokens`.
**Covers:** FR-1.1, FR-1.3, FR-1.4, FR-1.5 (flag validation), FR-2.3, FR-3.3 (CLI level),
FR-4.2 (CLI level), FR-5.1 (partial)
**Notes:** `cli.py` and `experiment.py` deliberately share no code (INV-1: neither imports the
other), so the resolver and validation are implemented twice. Keep them behaviorally
identical — the same duplication spec 2 accepted for its own sizing flags. Implement 5b
*after* 5a so it mirrors a known-good implementation.

### Task 6: Docs, experiment scripts, ARCHITECTURE, ADR
**Goal:** the feature is documented, genuinely reachable from the experiment suites, and its
invariant amendments (INV-1, and INV-7 for this spec's own new test-import names) are recorded.
**Files:** `README.md`, `experiments/{ag-news,trec,pubmed-rct}/run_experiment.sh`,
`experiments/{ag-news,trec,pubmed-rct}/README.md`, `spec/ARCHITECTURE.md`,
`spec/5-batched-classification/ADR.md` (new)
**Dependencies:** Tasks 5a, 5b
**Do:**
- `README.md`: all four flags, both mode exclusions, the comparability caveat, and that
  `--workers` now counts batches so peak throughput scales with `workers x batch size`.
- The three scripts: add `BATCH="${BATCH:-}"` and a conditional `--batch` append matching the
  existing `INDUCTION_EXAMPLES`/`TEST_LIMIT` style. Also add `CRITICS="${CRITICS:-1}"` and lift
  the hardcoded `--critics --critic-model … --reconciler-model …` trio out of the `ARGS` array
  (`run_experiment.sh:74`) into a conditional append, so both configurations are reachable
  (FR-4.3). Setting `BATCH` while `CRITICS` is still on must **exit non-zero** naming both
  variables rather than silently dropping `--critics`.
- `spec/ARCHITECTURE.md`: add `batching.py` to the Module Boundary Map, the mermaid graph, and
  INV-1's rule/evidence. Also add this spec's own new INV-7 exception names — `batching` (all
  functions + `_DATA_START`/`_DATA_END`), `induction._DATA_START`/`_DATA_END`, and
  `schema.build_batch_model` — to the `tests/` Module Boundary Map row and INV-7's rule/evidence
  text, alongside a provenance note citing this spec's ADR. **Do not re-edit or duplicate the
  spec-6 names already there** (`FailureKind`/`classify_failure`/the three response-failure
  exception types/`_DROPPABLE_PARAMS`/`Classifier._complete`/`_attempt_completion`) — leave
  spec 6's provenance note and text exactly as-is, only appending this spec's own names and note.
- `ADR.md`: **one record**, ADR-001, amending INV-1 for `batching.py`'s three inbound edges
  (`pipeline.py`, `cli.py`, `experiment.py`) and adding the INV-7 test-import names this spec
  itself introduces (`batching` module, its private delimiter constants,
  `induction._DATA_START`/`_DATA_END`, `schema.build_batch_model`) — citing spec 6's ADR-001 as
  the prior amendment this one builds on, not duplicating it.
**Verify:** `bash -n` on all three scripts; with **no** variables set each script's `DRY_RUN`
argv is identical to today's after normalizing `--run-dir`'s embedded timestamp (captured
before Task 6 begins, including `--critics`'s position — `RUN_DIR` derives from
`$(date +%Y%m%d-%H%M%S)`, so raw byte-identity across two invocations is unachievable and the
comparison must account for that one field); `BATCH=4 CRITICS=0 DRY_RUN=1` prints `--batch 4`
and no `--critics`; `BATCH=4 DRY_RUN=1` alone exits non-zero naming `BATCH` and `CRITICS`;
`CRITICS=0 DRY_RUN=1` alone prints neither `--critics` nor `--batch`;
`CRITICS=0 BATCH=4 DRY_RUN=1 ./run_experiment.sh --critics` — a literal `--critics`
passthrough surviving `CRITICS=0` — exits non-zero via the `"$@"` scan; `grep batching.py
spec/ARCHITECTURE.md` matches in the boundary map, the graph, and INV-1; `grep
build_batch_model spec/ARCHITECTURE.md` matches in the `tests/` boundary-map row and INV-7's
evidence, alongside this spec's own provenance note (distinct from, and not replacing, spec
6's existing INV-7 provenance note).
**Covers:** FR-4.3, AR-1.1 (ADR), AR-5.1
**Notes:** Capture each script's current `DRY_RUN` argv **before** editing it, normalizing the
timestamp field, so the no-variables-set comparison has a real baseline rather than a
re-derived expectation. FR-4.3 requires the error-on-conflict behavior rather than an auto-drop
because these scripts produce the accuracy numbers compared across runs. This task's ADR is
one record, not two, unlike the previous planning round's assumption — spec 6's own merged ADR
already supplied the INV-7 amendment this spec would otherwise have needed to write a second
time.

## Final Verification

Run concurrently once Task 6 lands:

- `.venv/bin/python -m pytest -q` — full suite green (baseline: 290 passed).
- `.venv/bin/python experiment.py run --help` and `.venv/bin/python classify.py --help` — all
  four flags present on both entry points; `experiment.py induce --help` shows none of them.
- The three `BATCH=4 DRY_RUN=1` script cases plus the three omitted-`BATCH` cases.
- **No-batch regression check** — stronger than a call count, because a `classify`-level count
  cannot see behavior changes inside `_attempt_completion`: a faked-LLM run without `--batch`
  must make exactly one `Classifier.classify` call per row, produce **byte-identical output
  CSV** to the pre-Task-1 baseline (Implementation Strategy), write the same interim-save
  positions as that baseline, count failures identically, and still *ignore* an unexpected
  field in a response (FR-2.1's strictness must not leak).
- `bash -n` on all three scripts, plus the no-variables-set `DRY_RUN` argv identity check
  (post-timestamp-normalization) against the baseline captured before Task 6.

## Documentation

- Living docs: none exist under `spec/docs/` (never generated for this repo).
- `spec/ARCHITECTURE.md` — Module Boundary Map, mermaid graph, INV-1 (Task 6). INV-7 is *not*
  touched — already correct.
- `README.md` and the three `experiments/*/README.md` (Task 6).
- `spec/5-batched-classification/ADR.md` — new, **one** record (Task 6).

## Spec Deviations

None identified. (Two earlier draft rows are resolved: the test-citation-precision row was
verified unfounded during the previous plan-critique round and removed; the AR-1.1
inbound-edge-list row stopped being a deviation once the spec itself was corrected to enumerate
all three edges.)

## Risks

- **Risk: the no-batch path regresses silently.** The hardest Constraints requirement to prove
  by inspection. Mitigated by the strengthened Final Verification check (byte-identical output,
  save positions, failure counts, and the unexpected-field path) — this check already accounts
  for spec 6's own changes to `_attempt_completion`, since it was written against the merged
  code, not the pre-spec-6 assumption.
- **Risk: bisection cost is larger than FR-3.2's headline number.** `2N-1` counts
  `Classifier.classify` invocations; each carries up to `max_retries` (3) provider attempts
  with `retry_delay` (5s) sleeps, and each attempt may additionally trigger the JSON-object
  fallback. For a fully-failing 50-row batch that is on the order of 600 provider requests and
  ~1000s of sleeping. Bounded by FR-1.5's cap, but worth a documented expectation so a stalled
  run is not mistaken for a hang.
- **Risk: a future litellm upgrade changes a hierarchy fact `classify_failure` depends on.**
  This risk now belongs entirely to spec 6 — its own `FR-1.2` pins every membership assertion
  via `issubclass` against the installed namespace, so a drift fails loudly there, not silently
  inside this plan's `batching.py`. This plan carries no independent exposure to it, which is
  the point of Task 3's design.
- **Risk: the arity cache's lock becomes a contention point** under `--workers 8`. Low: at most
  `--batch-max-size` distinct arities per run and the lock is held only for construction.
- **Rollback/checkpoint guidance:** each task is a commit on the spec worktree branch. Tasks
  1–3 are additive (a new module plus unset-by-default parameters) and revert without touching
  behavior — Task 1 is now lower-risk than the previous planning round's version, since it no
  longer touches `_attempt_completion`'s control flow at all, only `__init__`/
  `_completion_kwargs`. Task 4 is the first task that alters an existing code path — checkpoint
  before it. Task 5a's `_SCHEMA_VERSION` bump is the only change affecting persisted artifacts;
  reverting leaves already-written `run_config.json` files at version 5, which no code reads
  back, so the revert is safe.
