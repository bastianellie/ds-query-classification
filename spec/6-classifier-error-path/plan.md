# Implementation Plan: 6-classifier-error-path

**Status:** Ready
**Date:** 2026-09-15
**Spec:** spec/6-classifier-error-path/spec.md
**Plan Critique:** spec/6-classifier-error-path/plan-critique-consolidated-v-1.md

## Overview

Everything this spec touches lives in one ~355-line file (`src/query_classification/
classifier.py`) plus its direct test file (`tests/test_cerebus.py`) and `spec/ARCHITECTURE.md`.
The plan builds bottom-up within that file: classification primitives first (pure additions),
then the response-parsing tail (additive, changes what an existing return path does on bad
input), then the fallback control-flow rewrite (the actual behavior change), then `classify`'s
retry loop (consumes everything above), then docs/ADR. Tasks are sequenced by genuine code
dependency, not artificial splitting — no task leaves the suite red for the next one to fix.

## Readiness

- **Checklist:** all 10 items `[x]` in the spec, confirmed current as of the v-2
  `/spec-update`.
- **Open questions:** none. One stale cross-reference was found and fixed while reading the
  spec for this plan: FR-2.2's heading still said "re-raises the original exception" after the
  v-2 update changed its body to "raises the fallback's own exception, chained from the
  original" — corrected directly in `spec.md` (a one-line title fix, not a behavior change) so
  this plan is built against a spec that is internally consistent.
- **External API/library verification:** all of FR-1.2's hierarchy claims, `finish_reason`'s
  nine literals, `map_finish_reason`'s normalization, `Message`/`ModelResponse`'s field
  annotations, and `pydantic.ValidationError`'s identity with `pydantic_core.ValidationError`
  were verified live during spec authoring (two rounds, after two hierarchy mistakes were
  caught by critique). Re-verified for this plan: no litellm/pydantic version pin exists in
  `pyproject.toml`/`requirements.txt` (both list `litellm` and `pydantic` unpinned), so the
  spec's `issubclass`-based pinning tests (FR-1.2's Verify) are what will catch a future
  version drift — there is no version constraint to add here.
- **Repository state:** dirty, but only with the two pre-existing user-owned items unrelated to
  this spec (`M experiments/ag-news/analyze_run.ipynb`, untracked
  `data/example_queries_classified.csv`) plus the untracked `spec/5-batched-classification/` and
  `spec/6-classifier-error-path/` directories. None are touched by this plan.
- **Test baseline:** 229 passed (unchanged since spec 5's planning; this spec has not yet
  touched any code).

## Code Impact

- **Modules:** `src/query_classification/classifier.py` only. **Not touched:** every other
  module — `pipeline.py`, `debate.py`, `multi_model.py`, `induction.py`, `cli.py`,
  `experiment.py` all call `Classifier.classify` through a bare `except Exception`, verified by
  reading all six call sites (Integration Points, and confirmed fresh for this plan — see
  Interfaces; a plan-critique round found a sixth site the original codebase exploration missed,
  `debate.py`'s sampling loop, also a bare `except Exception`).
- **Tests:** new `tests/test_classifier_errors.py`; `tests/test_cerebus.py` modified (one
  fixture gains a field, one test is replaced by two, per FR-2.3/FR-4.1).
- **Docs:** `spec/ARCHITECTURE.md` (INV-7's contradicted sentence and the `tests/` Module
  Boundary Map row corrected; the "Fallback-trigger scope" Known Gap closed); new
  `spec/6-classifier-error-path/ADR.md` (one record, amending INV-7 — INV-1 is unaffected, so no
  second record is needed here, unlike spec 5).
- **API/interfaces:** see below.

## Interfaces

Pinned here so Tasks 2–4 build against the same shapes. Names below (`FailureKind`,
`classify_failure`) are the spec's own literal names, verified by grep against `spec.md`; the
rest are this plan's concrete choices for underspecified "WHAT" requirements.

```python
# classifier.py — new, module-level

class FailureKind(NamedTuple):          # or an equivalent frozen dataclass; either needs
    retryable: bool                     # one new stdlib import (typing.NamedTuple or
    isolable: bool                      # dataclasses.dataclass) — neither is imported today

class TruncatedResponseError(Exception): ...
class PolicyRefusalError(Exception): ...
class EmptyResponseError(Exception): ...

_DROPPABLE_PARAMS: tuple[str, ...] = ("temperature",)   # FR-2.5's patchable ladder constant;
                                                          # spec 5 appends "max_tokens" later

def classify_failure(exc: Exception) -> FailureKind: ...   # FR-1.1/FR-1.2, public, total function
def _classify_failure_verbose(exc: Exception) -> tuple[FailureKind, bool]: ...
    # private; the bool is `matched` — True if exc hit a named table entry, False if it fell
    # through to FR-1.1's default. `classify_failure` is `_classify_failure_verbose(exc)[0]`.
    # Needed because FR-3.2's Verify requires a `RuntimeError` log line to *identify itself as
    # the default branch* — distinct from BudgetExceededError/AuthenticationError, which are
    # ALSO (False, False) by an explicit table entry, not by falling through.

def _log_classified_failure(exc: Exception, kind: FailureKind, matched: bool, will_retry: bool) -> None: ...
    # FR-3.2's SOLE call site is classify()'s loop (Task 4) — see Task 3's Notes for why
    # _attempt_completion's fallback guard must NOT also call this (double-logging).

# classifier.py — modified signatures
def _attempt_completion(self, kwargs: dict[str, Any]) -> str: ...
    # drops the `*, allow_temperature_drop: bool` parameter entirely — the ladder constant
    # replaces it; no caller passes this kwarg today (verified: only `_complete` calls
    # `_attempt_completion`, and no test calls `_attempt_completion` directly), so this is not
    # a breaking signature change for any existing caller
```

`pydantic.ValidationError` needs `from pydantic import ValidationError` added (today only
`BaseModel` is imported from `pydantic`); `pydantic.ValidationError` and
`pydantic_core.ValidationError` are verified to be the same object, so no `pydantic_core` import
is needed.

**Call-site rule (resolves a double-logging defect found in plan critique v-1):** exactly one
place calls `_log_classified_failure` — `classify()`'s loop (Task 4). `_attempt_completion`'s
fallback guard (Task 3) calls `classify_failure` only, to decide what exception to chain and
raise; it never logs. Without this rule, a fallback failure would be classified-and-logged once
inside `_attempt_completion` and again when the (chained) exception propagates to `classify()`'s
loop — contradicting FR-2.2's own Verify line, which expects exactly one log line for that
scenario.

## Project Constraints

- **INV-1** is unaffected — no new module, no new import edge, `classifier.py` stays a leaf.
  Confirmed by the file's own import block: only `logging`, `os`, `time`, `typing.Any`,
  `litellm`, `pydantic.BaseModel` today; this spec adds only stdlib (`typing.NamedTuple` or
  `dataclasses.dataclass`) and one more `pydantic` name.
- **INV-4**: every broad `except Exception` needs its `# noqa: BLE001` justification kept
  current. `classify`'s existing comment ("retried and re-raised below") must be updated to
  reflect that the exception is now classified before either happening (AR-3.1); the new
  fallback guard in `_attempt_completion` needs its own such comment.
- **INV-5** (`max_retries >= 1`): unchanged, no task touches the constructor guard.
- **INV-7**: requires an ADR amendment (AR-1.2) that does two things, not one — adds the new
  accepted-exception names (`FailureKind`, `classify_failure`, the three response-failure
  exceptions, `Classifier._complete`/`_attempt_completion`) **and** deletes the sentence
  "`Classifier` itself is still tested only via the public API", which is already false today
  (`tests/test_cerebus.py` calls `clf._complete(...)` in three places, added by spec 3 without
  amending that sentence). Must also update the `tests/` row of the Module Boundary Map
  (`spec/ARCHITECTURE.md:84`), which restates the same permitted-import set in prose.
- **Testing policy** (`AGENTS.md`): no live network/provider calls; every task ships its own
  tests, no deferred test task.
- **`from __future__ import annotations` + PEP 604 unions**: already present in `classifier.py`;
  no change needed, but new code must keep following it.
- **Verification command**: `.venv/bin/python -m pytest`. No lint/typecheck/CI gate exists.

## Implementation Strategy

- **Size:** medium (5 tasks).
- **Execution mode:** solo, sequential. Everything is one file's internal control flow; there
  is no file-conflict axis to parallelize across.
- **Parallelizable work:** none across tasks. Within a task, tests are written alongside the
  code they cover.
- **Sequential blockers:** Task 2 needs Task 1's three exception classes to exist (so
  `classify_failure` can recognize instances of them, and so Task 2 can raise them) but not
  `classify_failure` itself. Task 3 needs Task 1's `classify_failure`/`_log_classified_failure`
  (FR-2.2 logs through the shared helper) and Task 2's exception classes and ordered checks (the
  fallback's successful response must still pass through them). Task 4 needs the full completion
  path from Tasks 1–3 to test end-to-end per-class retry counts. Task 5 needs nothing further
  from `classifier.py` but documents what Tasks 1–4 shipped.
- **No expected-red checkpoints.** `_attempt_completion`'s signature change
  (`allow_temperature_drop` removed) breaks no existing caller — verified only `_complete` calls
  it, and no test calls it directly. The three existing `tests/test_cerebus.py` tests that call
  `_complete` are handled explicitly in Task 2/3 (one fixture edit, one test replaced), not left
  to accidentally break and get noticed later.

## Implementation Tasks

### Task 1: Classification primitives
**Goal:** `FailureKind`, `classify_failure`, the three response-failure exception classes, the
`_DROPPABLE_PARAMS` ladder constant, and the shared logging helper exist as pure additions —
nothing in `_attempt_completion`/`classify` calls them yet.
**Files:** `src/query_classification/classifier.py`, `tests/test_classifier_errors.py` (new)
**Dependencies:** None
**Note on INV-7 sequencing:** this task's new test file imports directly from
`query_classification.classifier` in ways the *current, unamended* INV-7 text doesn't yet list
as accepted. The ADR/INV-7 amendment doesn't land until Task 5. This is a documented,
intentional ordering — the same pattern spec 3 followed (using new `classifier.py` Cerebus
functions directly for two commits before amending INV-7 to match) — not an oversight; nothing
mechanically enforces INV-7, so no test or tool breaks in the interim.
`tests/test_classifier_errors.py` may import `test_cerebus.py`'s `_FakeMessage`/`_FakeChoice`/
`_FakeResponse` rather than duplicating them (no `conftest.py` exists anywhere in this repo to
share fixtures otherwise) — an implementer's choice, not a requirement.
**Do:**
- Add `from pydantic import ValidationError` alongside the existing `BaseModel` import, and one
  stdlib import for `FailureKind` (`typing.NamedTuple` or `dataclasses.dataclass` — either is
  fine, the spec does not mandate a mechanism).
- Declare `TruncatedResponseError`, `PolicyRefusalError`, `EmptyResponseError` as plain
  module-level `Exception` subclasses (no behavior yet — Task 2 raises them).
- Declare `_DROPPABLE_PARAMS: tuple[str, ...] = ("temperature",)` at module level, read by
  attribute lookup wherever it's consumed later (Task 3), so a test can `monkeypatch` it.
- Implement `_classify_failure_verbose(exc) -> tuple[FailureKind, bool]` as a **flat ordered
  chain** (a list of `(exception_type, FailureKind)` pairs tested top-to-bottom with
  `isinstance`, falling through to `(FailureKind(False, False), matched=False)` if nothing
  matches) over every named type in FR-1.2's table, in the order FR-1.2 specifies: the five
  `BadRequestError` subclasses before `BadRequestError` itself; no base-class grouping anywhere
  (AR-2.1's rule applies to this dispatch too, even though AR-2.1 itself is Task 3's).
  `classify_failure(exc)` is a one-line wrapper returning just the `FailureKind`.
- **Explicitly enumerate all three of FR-2.4's exception types in the chain**, using the values
  FR-2.4 itself already states — FR-1.2's table only tabulates two of the three (truncation and
  content-policy rows cite FR-2.4; the empty-response case does not get its own row, though its
  value is given in FR-2.4's own text): `TruncatedResponseError` → `(False, True)`,
  `PolicyRefusalError` → `(False, False)`, `EmptyResponseError` → `(True, False)`. This is
  completing an already-decided value, not introducing a new one — Task 2 raises these three
  types and its own Verify already assumes all three are classifiable.
- Implement `_log_classified_failure(exc, kind, matched, will_retry)`: one `logger.warning` call
  naming the exception type, both axis values, whether the match was a named table entry or the
  default (`matched`), and whether it will retry — reusing the existing
  `logger = logging.getLogger(__name__)` module logger. No response body, credential, header, or
  endpoint value is ever passed to it (FR-3.2) — it only ever receives what its four parameters
  already are. **This is the only place in the codebase that calls it** — see Interfaces'
  call-site rule.
**Verify:** `.venv/bin/python -m pytest -q` (full 229-test baseline stays green — nothing
existing is touched yet); new tests assert `classify_failure`'s pair for **one instance of every
named type** in FR-1.2's table (not one representative per row — several rows list 2–4 types,
e.g. the rate-limit row alone has three; FR-4.1 requires every type its own assertion) using
`issubclass`/`isinstance` against the actual installed `litellm` namespace (not `__mro__`
names), plus the three FR-2.4 exception types; `BudgetExceededError` classifies despite deriving
from `Exception`, **with `matched=True`** (an explicit table entry, not the default);
`BadGatewayError` and `Timeout` each classify retryable and each is asserted *not* to be an
instance-match for the type it could plausibly be folded into (`InternalServerError`,
`APIConnectionError` respectively); `AuthenticationError` and `InvalidRequestError` are asserted
not to match the `BadRequestError` branch; a bare `RuntimeError` classifies `(False, False)`
**with `matched=False`** — the distinction that lets FR-3.2's default-branch log requirement be
satisfied later, in Task 4.
**Covers:** FR-1.1, FR-1.2, AR-1.1, AR-2.1 (the dispatch mechanism, exercised again by Task 3's
own dispatch), FR-3.2 (mechanism only — wired into `classify`'s loop by Task 4), FR-4.1 (partial)

### Task 2: Ordered response checks and the honest return contract
**Goal:** the tail of `_attempt_completion` — the part that turns a successful
`litellm.completion()` response into the `str` `classify()` expects — checks for truncation,
policy refusal, and emptiness in the spec's mandated order before falling through to a normal
return.
**Files:** `src/query_classification/classifier.py`, `tests/test_cerebus.py`,
`tests/test_classifier_errors.py`
**Dependencies:** Task 1
**Do:**
- Replace the bare `return response.choices[0].message.content` tail with the four ordered
  checks FR-2.4 specifies: empty `choices` first (only because every later check indexes
  `choices[0]`), then `finish_reason == "length"` (raise `TruncatedResponseError`), then
  `finish_reason == "content_filter"` (raise `PolicyRefusalError`), then `content is None`
  (raise `EmptyResponseError`); otherwise return the content.
- **Edit `tests/test_cerebus.py`'s `_FakeChoice`** to accept and store a `finish_reason`
  parameter, defaulting to `"stop"` so every existing call site that doesn't pass one keeps
  working unchanged. This is required, not optional: without it, the three existing
  `clf._complete(...)` tests will `AttributeError` the moment this task's code reads
  `choice.finish_reason`.
**Verify:** `.venv/bin/python -m pytest -q` full suite green, including all three existing
`test_cerebus.py` "temperature drop" tests unmodified (they now implicitly exercise
`finish_reason="stop"` via the fixture's new default and are otherwise unaffected — see Task 3's
Notes for why one of the three later needs replacing, not this task); new tests assert:
`finish_reason="length"` raises `TruncatedResponseError` classifying `(False, True)`;
`"content_filter"` raises `PolicyRefusalError` classifying `(False, False)`; `"stop"` with valid
content returns normally; `choices=[]` and (`"stop"` with `content=None`) each raise
`EmptyResponseError` classifying `(True, False)`; and the two overlap cases resolve by order —
`finish_reason="length"` with `content=None` raises `TruncatedResponseError` (not
`EmptyResponseError`), and `"content_filter"` with `content=None` raises `PolicyRefusalError`
(not `EmptyResponseError`).
**Covers:** FR-2.4, AR-2.2, FR-4.1 (partial)
**Notes:** This task only changes the *tail* of `_attempt_completion` (after a `litellm.
completion()` call already succeeded). It does not touch the `except` handlers above it — that
is Task 3 — so at the end of this task the fallback logic is still the old, unguarded,
over-broad version, just feeding into the new tail. That is an intentional intermediate state,
not a bug: both the structured-output success path and the (still old) fallback success path
already flow through the same return statement today, so this task's tests can exercise the new
checks via the structured-output path alone without needing Task 3's rewrite yet.

### Task 3: Fallback narrowing, guarding, ladder, and reachability
**Goal:** rewrite `_attempt_completion`'s `except` chain: the JSON-mode fallback triggers only
where it can help, is itself guarded, is reached after ladder exhaustion, and drops parameters
from the ordered `_DROPPABLE_PARAMS` ladder instead of a single boolean.
**Files:** `src/query_classification/classifier.py`, `tests/test_cerebus.py`,
`tests/test_classifier_errors.py`
**Dependencies:** Tasks 1, 2
**Do:**
- Remove the `*, allow_temperature_drop: bool` parameter from `_attempt_completion`; drop
  parameters by walking `_DROPPABLE_PARAMS` from the constant, reading it by attribute lookup at
  call time (so a test can `monkeypatch` it), skipping any entry absent from `kwargs` without an
  extra completion call, and never re-dropping an entry already removed.
- Order the `except` handlers so the five `BadRequestError` subclasses are tested before
  `BadRequestError` itself (AR-2.1); `ContextWindowExceededError`, `ContentPolicyViolationError`,
  `ImageFetchError`, and `LiteLLMUnknownProvider` propagate directly (FR-2.1 — no fallback
  attempt); a plain `BadRequestError`, or an `UnsupportedParamsError` whose ladder is exhausted,
  proceeds to the JSON-mode fallback using the **most-reduced** kwargs (every ladder entry that
  was present has been removed).
- Wrap the fallback `litellm.completion()` call. On failure: call `classify_failure` and
  `_log_classified_failure` (Task 1), then raise the fallback's own exception chained `from` the
  first attempt's exception (FR-2.2) — never the original object, and never a wrapper.
- **Replace** `test_complete_reraises_unsupported_params_error_when_no_temperature_to_drop`
  (the test whose docstring says "there's nothing to drop and retry — this must not loop or
  swallow the real error") with two tests per FR-2.3's Verify: (a) a fake that raises
  `UnsupportedParamsError` on the structured-output call with no temperature set, and *succeeds*
  on the JSON-mode fallback — asserting the fallback is reached and its content returned, where
  today the call raises after one attempt; (b) a fake where the fallback call *also* fails —
  asserting that failure surfaces (chained per FR-2.2) without a third attempt.
**Verify:** `.venv/bin/python -m pytest -q` full suite green; new tests assert:
`ContextWindowExceededError` and `ContentPolicyViolationError` from the structured-output call
each produce exactly one `litellm.completion` call with no fallback attempt; a plain
`BadRequestError` produces two calls, the second in JSON-object mode; with the ladder patched to
two entries and both parameters actually present in the classifier's kwargs, a fake rejecting
the first and accepting the second drops them in order, one per attempt, without re-dropping;
an entry absent from kwargs is skipped with no additional call; the fallback's kwargs after
exhaustion contain none of the ladder's parameters; with the first call raising a plain
`BadRequestError` and the second raising `RateLimitError`, the call raises the `RateLimitError`
with `__cause__` set to the `BadRequestError`, logging it as retryable.
**Covers:** FR-2.1, FR-2.2, FR-2.3, FR-2.5, AR-2.1, FR-4.1 (partial)
**Notes:** Of the three existing `test_cerebus.py` tests that call `_complete` directly, two
survive this task **unmodified** and one is replaced (above). Trace why, since it is easy to
second-guess while implementing:
  - `test_complete_drops_temperature_once_on_unsupported_params_error` — fails only when
    `"temperature" in kwargs`, succeeds otherwise. The ladder's one entry is dropped on attempt
    1, attempt 2 succeeds before the ladder is ever exhausted, so the fallback path is never
    reached. Unaffected.
  - `test_complete_does_not_retry_temperature_drop_twice` — fails unconditionally. Attempt 1
    (with temperature) fails, the ladder drops it, attempt 2 (without temperature) fails again,
    the ladder is now exhausted, and — under the *new* behavior — this reaches the JSON-mode
    fallback (attempt 3) rather than re-raising immediately as it does today. That fallback
    attempt fails too (same unconditional fake), so the exception still propagates as
    `UnsupportedParamsError`, chained. The test has no call-count assertion, so it stays green
    without modification — but it now exercises 3 calls where it used to exercise 2. Do not
    "fix" this by adding a call-count assertion; that would encode a coincidence of this
    particular fake (it happens to raise the same exception type from both the ladder-drop
    path and the fallback) as a requirement, which it is not.
  - `test_complete_reraises_unsupported_params_error_when_no_temperature_to_drop` — replaced,
    per FR-2.3. Not because it would fail mechanically: under the exact fake it uses
    (unconditional `UnsupportedParamsError` regardless of kwargs), it would very likely stay
    green too, the same as the survivor above, since the fallback attempt fails with the same
    exception type the original immediate raise would have. It is replaced because its
    **premise** ("there's nothing to drop and retry — this must not loop or swallow the real
    error") is no longer the whole story once ladder exhaustion routes to a fallback attempt
    instead of an immediate raise — a test whose docstring asserts something no longer true is
    worse than no test, regardless of whether its assertion happens to still pass.

### Task 4: Retry policy and logging
**Goal:** `classify`'s loop consults `classify_failure`'s `retryable` axis only, retrying exactly
as today for retryable failures and failing immediately — no sleep, no further attempts — for
everything else; every decision is logged via Task 1's shared helper.
**Files:** `src/query_classification/classifier.py`, `tests/test_classifier_errors.py`
**Dependencies:** Tasks 1, 3
**Do:**
- In `classify`'s `except Exception as e` branch, call `_classify_failure_verbose(e)` to get
  `(kind, matched)`, then call `_log_classified_failure(e, kind, matched, will_retry=...)` — this
  is the **sole** call site for logging in the whole feature, per the Interfaces call-site rule —
  in place of the two existing `logger.warning` call sites; re-raise immediately (no sleep, no
  further loop iteration) when `not kind.retryable`; otherwise keep today's exact
  retry/sleep/warning behavior.
- Update the `except Exception as e:  # noqa: BLE001 - retried and re-raised below` comment to
  say the exception is now classified before either retrying or being re-raised (AR-3.1); apply
  the same `# noqa: BLE001` treatment to Task 3's new fallback guard if not already covered.
- Update `classify`'s docstring — "Raises the last exception if all retry attempts fail" is no
  longer accurate for a non-retryable failure, which raises on the first attempt with zero
  retries.
**Verify:** `.venv/bin/python -m pytest -q` full suite green; new tests assert: a fake raising
`RateLimitError` is attempted `max_retries` times with `time.sleep` called between attempts; a
fake raising `pydantic.ValidationError` is likewise attempted `max_retries` times — an explicit
regression guard, since this is the retry the whole redesign exists to preserve; a fake raising
`AuthenticationError` is attempted exactly once with `time.sleep` never called (patched and
asserted, using `max_retries >= 2` so the assertion cannot pass trivially at `max_retries=1`); a
fake raising Task 2's `TruncatedResponseError` is likewise attempted exactly once. **FR-3.2's
three named assertions, exercised end to end through `classify` (not `_log_classified_failure`
in isolation)**: the `AuthenticationError` case's log line names the type, both axes as false,
and states it will not retry; a fake raising a bare `RuntimeError` emits a log line that
identifies it as the default branch (`matched=False`) — distinct from the `AuthenticationError`
case, which is also `(False, False)` but via an explicit table entry; and for a classifier built
with a known API key, header value, and endpoint (e.g. via `build_cerebus_completion_kwargs`-
shaped kwargs), none of those three values appears in any log line emitted during a failing
call. **This is also where the fallback-guard scenario from FR-2.2 is finally verified as
producing exactly one log line** — a plain `BadRequestError` first call and a `RateLimitError`
fallback failure, run through the real `classify()` path end to end (not `_complete` in
isolation, which no longer logs at all per Task 3), asserting exactly one `_log_classified_
failure` call.
**Covers:** FR-3.1, FR-3.2 (wiring), AR-3.1, FR-4.1 (partial)

### Task 5: Documentation and ADR
**Goal:** `spec/ARCHITECTURE.md` reflects the corrected INV-7 and the closed Known Gap; one ADR
record documents the INV-7 amendment.
**Files:** `spec/ARCHITECTURE.md`, `spec/6-classifier-error-path/ADR.md` (new)
**Dependencies:** Tasks 1–4
**Do:**
- INV-7: delete "`Classifier` itself is still tested only via the public API" and add
  `FailureKind`, `classify_failure`, `TruncatedResponseError`, `PolicyRefusalError`,
  `EmptyResponseError`, and `Classifier._complete`/`_attempt_completion` to the accepted
  direct-import list, citing this spec.
- Update the `tests/` row of the Module Boundary Map (`:84`) to match the amended INV-7 text —
  it currently restates the same permitted-import set in prose and would otherwise contradict
  the correction.
- Remove (or mark resolved, citing this spec) the Known Gap "Fallback-trigger scope is broader
  than 'schema unsupported'" — FR-2.1 closes it.
- Write `ADR.md` with one record: amending INV-7 for the reasons above. INV-1 needs no record —
  confirmed unaffected (Project Constraints).
**Verify:** `spec/ARCHITECTURE.md`'s INV-7 section no longer contains the contradicted sentence;
its `tests/` row and INV-7's exception list agree; the Known Gap entry is gone or marked
resolved; `.venv/bin/python -m pytest -q` — the full suite, one final time, end to end.
**Covers:** AR-1.2

## Final Verification

- `.venv/bin/python -m pytest -q` — full suite green (baseline: 229 passed; expect roughly +25
  new tests across `test_classifier_errors.py` and the `test_cerebus.py` edits, no regressions).
- `grep -rn "allow_temperature_drop" src/` — zero matches, confirming the old parameter is fully
  removed rather than left as dead code alongside the new ladder.
- Re-run the exact `issubclass`-based hierarchy assertions from Task 1 one more time after all
  five tasks land, as a sanity check that no later task's edits silently reintroduced a
  base-class grouping (AR-2.1's rule).
- Manually trace one full example end to end (not a new automated test, a final human check):
  an `UnsupportedParamsError` on `response_format` with no temperature set → ladder skip → JSON
  fallback → success — confirming Tasks 1–3 compose as the spec describes, not just each in
  isolation.

## Documentation

- `spec/ARCHITECTURE.md` — INV-7, the `tests/` Boundary Map row, the closed Known Gap (Task 5).
- `spec/6-classifier-error-path/ADR.md` — new, one record (Task 5).
- No `README.md` change — this spec adds no CLI flag, no new public API, no persisted artifact.

## Spec Deviations

None identified. (A prior draft of this plan recorded a row here claiming the spec's cited test
line numbers — `:361`, `:397`, `:368-381` — didn't match a fresh read of `tests/test_cerebus.py`.
A plan-critique round verified directly: they match exactly. The apparent mismatch was this
plan's own error — conflating each test's `def` line with the specific call line the spec cites.
No deviation exists; the row was removed rather than left as a false one.)

## Risks

- **Risk: Task 3's fallback-exhaustion path is the one piece of new control flow with no direct
  precedent elsewhere in the codebase** (ladder-then-fallback, as opposed to spec 5's simpler
  bisection). Mitigated by Task 3's Notes tracing all three existing temperature tests' new
  behavior explicitly, so the implementer isn't surprised by `test_complete_does_not_retry_
  temperature_drop_twice` now making 3 calls instead of 2 without a call-count assertion to catch
  a mistake — the new tests in Task 3 are what actually pin the call counts.
- **Risk: a seventh call site is discovered late.** Six call sites were verified
  (`pipeline.py`, `debate.py` ×3 — including the sampling loop found during plan critique,
  distinct from the critic/reconciler calls — `multi_model.py`, `induction.py`), all bare
  `except Exception`. If implementation turns up a seventh, it should be checked against the
  same bare-catch pattern before Task 4 is considered complete — a caller matching on a specific
  exception type or a retry count would be a genuine (currently unforeseen) behavior-change
  risk.
- **Risk: `FailureKind` as `NamedTuple` vs `dataclass` is left to the implementer.** Both satisfy
  the spec's "two independent booleans" requirement identically; recorded as a free choice
  rather than a deviation because the spec's own wording never mandates a mechanism.
- **Rollback/checkpoint guidance:** each task is a commit on the spec's worktree branch. Tasks
  1–2 are additive or tail-only and revert cleanly. Task 3 is the one task that changes existing
  control flow and existing test behavior (per its Notes) — checkpoint before it. Task 4's
  docstring/comment edits and Task 5's documentation-only changes carry no behavioral risk to
  revert.
