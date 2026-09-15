# Spec 6: Classifier Error-Path Cleanup

## Overview

`Classifier`'s ~40-line completion/retry chain (`classifier.py:282-355`) treats every failure
the same way: one over-broad fallback, one hard-coded parameter drop, and a `except Exception`
loop that retries everything `max_retries` times with `retry_delay` sleeps. Seven defects were
verified live against this chain — two already recorded as Known Gaps, one a live regression
that makes the documented JSON-mode fallback unreachable for most roles, and one a return-type
contract that was never true. This spec replaces the uniform handling with an explicit failure
classification: decide what a failure permits, then retry, fall back, or fail fast accordingly.

It is deliberately sequenced **before** spec 5 (batched classification), whose plan-critique
rounds traced three separate blocking defects back to this chain.

## Goals

- Classify every failure that can leave a `litellm.completion` call along two independent axes
  — may the same request be retried, and may a caller reshape the input — as a total function
  with an explicit default.
- Stop retrying failures that retrying provably cannot fix, without removing the retry that
  the loop was actually built for.
- Stop the JSON-mode fallback from being used as a catch-all for errors unrelated to the
  response schema, and make it reachable for the case it was built for.
- Make a length-truncated, policy-blocked, or empty response distinguishable from a malformed
  one.
- Give callers (spec 5's batching layer, and any future one) one importable source of truth
  instead of re-deriving litellm's subclass ordering.
- Leave every classification decision observable in logs.

---

## Feature 1: A failure classification

**Who & why:** Anyone writing a caller that must react to a failure — spec 5's batching layer
has to decide whether to split a batch, back off, or give up — currently has to re-derive
litellm's exception hierarchy themselves. That derivation is genuinely tricky (five distinct
error types are all `BadRequestError` subclasses, one common error doesn't descend from
`APIError` at all, and a 502 doesn't descend from `InternalServerError`), and spec 5's plan got
it wrong twice across two critique rounds. One importable classifier removes the guesswork.

### Functional Requirements

#### FR-1.1: Two independent axes, assigned by a total function
`classifier.py` gains `FailureKind`, carrying exactly two booleans, and a
`classify_failure(exc)` helper that maps **any** exception to one:

- **`retryable`** — repeating the *same* request may succeed.
- **`isolable`** — the request is answerable, but this one was not answered usably; a caller
  that shrinks or splits its input, or attributes the failure to a subset of it, may succeed.

The two axes are independent, and collapsing them into one enum is the error this requirement
exists to avoid. Most failures set exactly one, but `pydantic.ValidationError` sets **both**,
and that case is load-bearing: `classifier.py:333`'s `model_validate_json` runs inside
`classify`'s `try`, so a malformed or schema-violating response is retried today — and that is
the most valuable retry in the system, because LLM output is stochastic and the same prompt
very often validates on the next attempt. A single-class taxonomy that called it "isolable"
would delete that retry. It is also isolable, because if it keeps recurring a caller with a
batch can find which input provoked it.

`classify_failure` is **total**: an exception matching nothing known is assigned
`retryable=False, isolable=False`, which is the safe default — it neither retries nor invites a
caller to reshape and re-send. The default must be a deliberate branch with a log line
(FR-3.2), not an accident of `except` ordering.
**Verify:** `classify_failure` returns the documented pair for one instance of **every row** of
FR-1.2's table, returns `(True, True)` for a `pydantic.ValidationError`, and returns
`(False, False)` for a bare `RuntimeError`.

#### FR-1.2: The classification table, and the four hierarchy traps
The assignment is exhaustive over every exception on the `litellm` namespace, plus
`pydantic.ValidationError`. It is expressed as a **flat, ordered chain over named types**, never
as base-class grouping — see trap 3 for why no usable grouping exists.

| Exception | `retryable` | `isolable` | Why |
|---|---|---|---|
| `RateLimitError`, `RouterRateLimitError`, `RouterRateLimitErrorBasic` | yes | no | throttling; the same request is accepted later |
| `InternalServerError`, `ServiceUnavailableError`, `BadGatewayError` | yes | no | transient 5xx server/gateway faults |
| `APIConnectionError` | yes | no | transport-level; nothing about the request is wrong |
| `Timeout` | yes | no | transport-level. **Its own row, not folded into `APIConnectionError`** — see trap 3 |
| `pydantic.ValidationError`, `APIResponseValidationError`, `JSONSchemaValidationError` | **yes** | **yes** | the response did not parse or validate; stochastic output often validates next attempt, and a persistent failure is attributable to an input |
| `ContextWindowExceededError` | no | yes | identical request will not fit; a smaller one will |
| truncation (FR-2.4) | no | yes | identical request and identical cap will truncate again |
| `AuthenticationError`, `PermissionDeniedError`, `NotFoundError`, `LiteLLMUnknownProvider` | no | no | configuration; no retry or reshaping helps |
| `ContentPolicyViolationError`, policy `finish_reason` (FR-2.4) | no | no | the provider refused the content itself |
| `BudgetExceededError` | no | no | repeating cannot un-exceed a budget; reshaping does not either |
| `UnsupportedParamsError` surviving FR-2.5's ladder | no | no | a parameter the caller cannot drop |
| `BadRequestError` (plain), `InvalidRequestError`, `UnprocessableEntityError` | no | no | the request itself was rejected as malformed |
| `ImageFetchError` | no | no | unreachable in this text-only repo, but classified so the table stays exhaustive |
| `OpenAIError`, `BaseLLMException`, `MockException` | no | no | base classes and a test double; not raised by a `completion()` call, classified for exhaustiveness |

Four hierarchy facts must be handled. Each was verified by running `issubclass` against the
`litellm` namespace on the installed version, and each is a trap a plausible implementation
falls into:

1. **Exactly five types are `BadRequestError` subclasses** — `ContentPolicyViolationError`,
   `ContextWindowExceededError`, `ImageFetchError`, `LiteLLMUnknownProvider`, and
   `UnsupportedParamsError` — so any dispatch on `BadRequestError` must test those five
   **first**. This generalizes the lesson in `classifier.py:295-305`'s existing comment. Note
   that `AuthenticationError` and `InvalidRequestError` are **not** among them.
2. **`BudgetExceededError` derives directly from `Exception`** — not from `APIError` — so a
   catch-all rooted anywhere in the API hierarchy silently misses it.
3. **No usable base-class grouping exists, because litellm's concrete errors descend from the
   *openai SDK*, not from litellm's own base classes.** `litellm.APIError` has **zero**
   subclasses, `litellm.APIStatusError` does **not exist**, and
   `issubclass(litellm.Timeout, litellm.APIConnectionError)` is **`False`** — the two are
   siblings under *openai*'s `APIConnectionError`. Reading a class's `__mro__` shows names like
   `APIStatusError` and `APIConnectionError` that belong to openai, which is exactly what makes
   this trap easy to fall into. Grouping 5xx faults under `InternalServerError`, transport
   faults under `APIConnectionError`, or anything under `APIError` would silently drop the most
   common failures to FR-1.1's default.
4. **`InvalidRequestError` is not a `BadRequestError` subclass** on the installed version,
   despite the name, so it needs its own entry rather than inheriting one.

**Verify:** the pinning tests assert every table row's membership with **`issubclass` against
the `litellm` namespace**, never by reading `__mro__` names or assuming a base-class
relationship — so a future litellm upgrade that re-parents a type fails loudly rather than
silently re-defaulting it. Specifically: each of the five `BadRequestError` subclasses gets its
own row's pair rather than collapsing; `BudgetExceededError` classifies despite deriving from
`Exception`; `BadGatewayError` and `Timeout` each classify as retryable **and** are each
asserted *not* to be subclasses of the type they might plausibly be folded into; and
`AuthenticationError` and `InvalidRequestError` are each asserted not to match the
`BadRequestError` branch.

### Architectural Requirements

#### AR-1.1: The classification lives in `classifier.py` and adds no new module
`classifier.py` already owns the LLM call and its retries, and INV-1 keeps it a leaf (stdlib +
litellm + pydantic, no sibling imports). `FailureKind` and `classify_failure` have no
dependency beyond those, so they belong there rather than in a new module — which would add an
INV-1 node and an import edge for one dataclass and one function.

Callers import them directly from `query_classification.classifier`. They are **not** added to
`__init__.py`'s eight re-exported names: that list is the supported *library* surface, and this
exists for internal orchestration modules (`pipeline.py`, spec 5's batching layer), matching how
`debate.py` and `multi_model.py` are used without being re-exported.

#### AR-1.2: Tests may import the classification and the private completion helpers — extends INV-7
`tests/` gains direct imports of `FailureKind`, `classify_failure`, FR-2.4's truncation
exception, and `Classifier._complete`/`_attempt_completion`. The private helpers are required
because FR-2.1–FR-2.5 all describe behavior *below* `classify`, unreachable through it without
a live provider.

INV-7's text currently asserts "`Classifier` itself is still tested only via the public API".
That is **already false** — `tests/test_cerebus.py:361`, `:381`, and `:397` call
`clf._complete(...)` today, added by spec 3 without amending the sentence. The implementation's
ADR must drop the contradicted claim, not merely append to the exception list, and must also
update the `tests/` row of `spec/ARCHITECTURE.md`'s Module Boundary Map (`:84`), which lists the
same permitted-import set and would otherwise disagree with the amended INV-7.

---

## Feature 2: The completion path

**Who & why:** The current path has one `except litellm.BadRequestError` doing three unrelated
jobs — schema fallback, error surfacing, and (accidentally) context-length handling — plus a
fallback call that can fail and escape unclassified, and a return line that can hand back
`None`. A run against a model that genuinely lacks `json_schema` support gets no fallback at
all, despite the module docstring promising one.

### Functional Requirements

#### FR-2.1: The JSON-mode fallback triggers only where a schema fallback can help
The fallback to `response_format={"type": "json_object"}` runs only when the structured-output
attempt failed in a way a schema change could plausibly fix: a plain `BadRequestError`, or an
`UnsupportedParamsError` whose drop ladder (FR-2.5) is exhausted. It must **not** run for
`ContextWindowExceededError`, `ContentPolicyViolationError`, `ImageFetchError`, or
`LiteLLMUnknownProvider` — none is about the response schema, and re-sending identical content
cannot succeed.

Narrowing is expressed by **excluding specific subclasses**, not by a positive capability test.
A positive test does exist — `litellm.supports_response_schema(model=...)` — and it is
deliberately rejected: verified against this repo's real model ids, it returns `False` for
**every** Cerebus gateway slug, because it is a static map lookup and gateway slugs are absent
from the map. Since `DEFAULT_LLM_PROVIDER=cerebus` is the normal configuration, gating on it
would silently degrade every real run to JSON-object mode and lose structured output entirely.
Message-text parsing is rejected for a different reason: it is provider- and version-specific,
and nothing in this repo validates it.
**Verify:** a `ContextWindowExceededError` and a `ContentPolicyViolationError` from the
structured-output call each produce exactly one `litellm.completion` call and raise without a
fallback attempt; a plain `BadRequestError` produces two calls, the second in JSON-object mode.

#### FR-2.2: The fallback call is guarded, and its own failure is chained and re-raised
The fallback `litellm.completion` call is wrapped. Any failure it raises is passed through
`classify_failure`, logged per FR-3.2, and then **that same exception is re-raised, chained
`from` the first attempt's exception**.

Which exception propagates is the fallback's own, not the first attempt's: the fallback is the
call that most recently and most specifically failed, and a caller reacting to it — spec 5's
batching layer deciding whether to split — needs *its* type. The first attempt's exception is
preserved as `__cause__`, so the reason the fallback was entered at all stays diagnosable. An
earlier draft said "the original exception object is re-raised", which contradicted this
requirement's own Verify line and justified itself with a rationale that has since lapsed (it
cited spec 5 needing to catch `ContextWindowExceededError` by type, which FR-2.1 now prevents
from reaching the fallback at all).

Note `classify_failure` returns a `FailureKind`, not an exception, so "raise the classified
failure" would not be a coherent instruction — the guard's job is to ensure the failure is
classified, logged, and chained, not to substitute a different object.

This closes a verified escape: today the fallback at `classifier.py:314-316` sits outside any
`try`, so when it fails, that exception propagates out of `_complete` unlogged and
unclassified — from a code path that looks like it handled errors.
**Verify:** with the first `litellm.completion` call raising a plain `BadRequestError` (which
FR-2.1 *does* route to the fallback) and the second raising `RateLimitError`, the call raises
the `RateLimitError`, its `__cause__` is the `BadRequestError`, and one FR-3.2 log line
classifies it as retryable.

#### FR-2.3: A model that rejects `response_format` reaches the fallback
When the structured-output attempt fails with `UnsupportedParamsError` and FR-2.5's drop ladder
is exhausted, the call proceeds to the JSON-mode fallback instead of raising.

Ladder exhaustion **is** the mechanism that distinguishes the two cases, since both arrive as
the same exception type: a rejection caused by a parameter the caller controls is fixed by
dropping it, and whatever survives that is treated as a possible `response_format` rejection and
given the fallback. No message inspection is involved.

This is a live regression, not a hypothetical. `classifier.py`'s module docstring promises "a
graceful fallback to plain JSON mode for models that don't support `json_schema`", but spec 3's
`except litellm.UnsupportedParamsError` branch re-raises whenever `temperature` is absent from
kwargs — which is every role except `--critics` sampling. Reproduced live: a fake model
rejecting `response_format` with `UnsupportedParamsError` and no temperature set raised after
exactly one call; the fallback never ran.

**`tests/test_cerebus.py:368-381` is superseded by this requirement.** That test
(`test_complete_reraises_unsupported_params_error_when_no_temperature_to_drop`) asserts today's
re-raise as intended, and its reasoning was right for its own case — "some OTHER param we don't
control" — but it cannot distinguish that case from a `response_format` rejection, so it
encodes the regression. Its replacement asserts that ladder exhaustion reaches the fallback,
and that a fallback which *also* fails surfaces the original error rather than looping
(preserving the "must not loop or swallow the real error" intent that test was protecting).
**Verify:** a fake raising `UnsupportedParamsError` on the structured-output call with no
temperature set produces a second call in JSON-object mode and returns its content — where
today it raises after one call; and when that second call also fails, the failure surfaces
without a third attempt.

#### FR-2.4: Unusable responses are distinct, classified failures
Before returning content, the response is checked in this **exact order**, each condition
raising a dedicated exception type declared in `classifier.py`. The ordering is normative, not
stylistic — the conditions overlap on real responses and carry opposite classifications:

1. **Empty `choices`** — an empty-response error, `retryable=True, isolable=False`. Checked
   first only because every later condition indexes `choices[0]`.
2. **`finish_reason == "length"`** — a truncation error, `retryable=False, isolable=True`.
3. **`finish_reason == "content_filter"`** — a policy refusal,
   `retryable=False, isolable=False`.
4. **`message.content is None`** — an empty-response error, `retryable=True, isolable=False`.

Checks 2 and 3 must precede check 4, because a response can satisfy both a `finish_reason`
condition and the `content is None` condition simultaneously, and the two answers are
opposites:

- A **reasoning model that spends its whole output allowance on reasoning tokens** returns
  `finish_reason="length"` with `content=None`. Classifying it as an empty response
  (`retryable`) would re-send an identical request that truncates again, burning `max_retries`
  on a deterministic failure — the exact waste this spec exists to eliminate. This is not an
  edge case here: `Message` declares `reasoning_content`, `reasoning_items`, and
  `thinking_blocks` fields, spec 3's temperature handling exists because this repo targets
  reasoning models, and spec 5's output cap makes the shape more likely still.
- A **content-filtered response** likewise often carries no content. Classifying it as retryable
  would re-send policy-violating content `max_retries` times, contradicting this spec's own
  security rationale.

Only `length`, `content_filter`, `tool_calls`, and `function_call` reach this code as
themselves: `litellm`'s `map_finish_reason` normalizes `guardrail_intervened` to
`content_filter`, and normalizes `eos`, `finish_reason_unspecified`,
`malformed_function_call`, **and any value it does not recognize** to `stop` (all verified). So
the policy check needs only `content_filter`, and a future provider-specific reason will arrive
as `stop` rather than as an unhandled literal — which means `finish_reason` can never be relied
on to signal a failure litellm does not already know about, and the content checks are what
catch those.

The last condition is the seventh defect and it fixes a contract that was never true:
`_complete` is annotated `-> str`, but `ModelResponse.choices` is a `list` that can be empty
and `Message.content` is annotated `str | None` (both verified), so
`return response.choices[0].message.content` can raise `IndexError` or return `None` today —
after which `model_validate_json(None)` produces a confusing `ValidationError` that looks like
a model formatting problem.

All three checks apply to both the structured-output and JSON-mode responses, since both return
through the same line.
**Verify:** `finish_reason="length"` raises the truncation type and classifies `(False, True)`;
`"content_filter"` raises the policy type and classifies `(False, False)`; `"stop"` with valid
content returns normally; `choices=[]` and (`"stop"` with `content=None`) each raise the
empty-response type and classify `(True, False)`; **and the two overlap cases resolve by
order** — `finish_reason="length"` with `content=None` raises the *truncation* type, and
`"content_filter"` with `content=None` raises the *policy* type, neither as an empty response;
`_complete` never returns `None`.

#### FR-2.5: Parameter dropping is an ordered ladder, not a single boolean
The `UnsupportedParamsError` handler drops parameters from an **ordered sequence declared as a
single module-level constant** rather than handling one hard-coded parameter. Each attempt
removes at most one parameter, never re-drops one already removed, and the ladder terminates
when the sequence is exhausted — at which point FR-2.3's fallback applies. The fallback then
sends the **most-reduced** kwargs, not the original: re-sending a parameter the model already
rejected would reintroduce exactly the failure `classifier.py:295-305`'s comment documents.

The constant's only entry today is `temperature`, which is the sole droppable parameter
`_completion_kwargs` emits; spec 5 adds `max_tokens` by appending to it rather than adding a
second boolean. Three mechanics make it testable now, before a second real parameter exists,
and each must be specified rather than left to the implementer:

- **The constant is read by module-attribute lookup at call time**, not bound as a default
  argument or copied at import, so a test can patch it.
- **A ladder entry absent from `kwargs` is skipped without a completion call** — dropping
  nothing and re-sending an identical request would waste a round-trip and could loop.
- **"Most-reduced" means the kwargs of the last attempt made**, i.e. every ladder entry that
  was present has been removed. With today's single-entry ladder that is simply "without
  `temperature`".

Today the mechanism is `_attempt_completion(kwargs, *, allow_temperature_drop: bool)` — one
boolean for one parameter. Spec 5's plan critique flagged that signature as an ambiguity two
implementers would resolve differently.
**Verify:** with the ladder constant patched to two entries **and both parameters injected into
the classifier's kwargs** (patching the constant alone is insufficient, since
`_completion_kwargs` would not emit a synthetic parameter), against a fake that rejects the
first and accepts the second: the parameters are dropped in declared order, one per attempt,
without re-dropping, and the call succeeds. A ladder entry absent from kwargs is skipped with no
additional completion call. With the unpatched one-entry constant, the **drop mechanics** for
`temperature` are unchanged from today — the post-exhaustion outcome deliberately is not, per
FR-2.3. After exhaustion the fallback's kwargs contain none of the ladder's parameters.

### Architectural Requirements

#### AR-2.1: Dispatch is a flat ordered chain over named types
Every dispatch in this feature is a flat chain over the specific types named in FR-1.2's table,
ordered so the five `BadRequestError` subclasses precede `BadRequestError` itself. It must not
group types by a shared base class. This is not style — it follows from FR-1.2's trap 3:
`litellm.APIError` has no subclasses, `litellm.APIStatusError` does not exist, and
`litellm.Timeout` is not a `litellm.APIConnectionError`, so any grouping attempt silently drops
the most common failures to FR-1.1's default. `classifier.py:294-312`'s existing ordering
comment is the narrow precedent for the subclass-first half of this rule.

#### AR-2.2: `_complete`'s return type becomes honest, not merely preserved
`_complete`/`_attempt_completion` keep the `-> str` annotation, but FR-2.4 is what makes it
true for the first time by raising on the `None`/empty cases instead of returning them. The
annotation is unchanged; the behavior is corrected to match it. `classify`'s body changes only
by FR-3.1's classification branch, and the two surviving `tests/test_cerebus.py` `_complete`
tests (`:361`, `:397`) stay valid — the third is superseded per FR-2.3.

---

## Feature 3: Retry policy

**Who & why:** A run against expired credentials currently spends `max_retries` attempts and
`retry_delay` sleeps per row — 3 attempts and 10s of sleeping by default — on a failure that
cannot succeed, across every row of a 29.6k-row split. The operator waits a long time for an
error that was knowable on the first attempt, and the log gives no indication that retrying was
pointless.

### Functional Requirements

#### FR-3.1: `classify` retries on the `retryable` axis only
`classify`'s loop classifies each caught failure via `classify_failure` and consults
**`retryable` alone**. A retryable failure retries exactly as today — same attempt count, same
`retry_delay`, same warning. A non-retryable failure is re-raised immediately, without sleeping
and without consuming further attempts. The `isolable` axis is never consulted here; it exists
for callers.

Because `pydantic.ValidationError` is retryable (FR-1.1), the retry the loop was actually built
for is preserved exactly. The observable change is confined to failures that are genuinely
non-retryable: a credential error now surfaces on the first attempt instead of the third.
`max_retries >= 1` (INV-5) still holds, and the trailing `assert last_exc is not None` stays
reachable only on the retryable path.
**Verify:** a fake raising `RateLimitError` is attempted `max_retries` times with sleeps
between; a fake raising `pydantic.ValidationError` is likewise attempted `max_retries` times
(today's behavior, explicitly asserted as a regression guard); a fake raising
`AuthenticationError` is attempted exactly once with no sleep; a fake raising FR-2.4's
truncation error is attempted exactly once.

#### FR-3.2: Every classification decision is logged
Each caught failure logs, at warning level, the exception type, both axis values, and whether
the call will retry or give up — including FR-1.1's default branch, so an unrecognized
exception being treated as non-retryable is visible rather than silent. Log messages carry no
response body, credential, header, or endpoint value, matching the existing sanitization
convention in `debate.py`'s `_sanitize_error` and `classifier.py`'s
`_cerebus_key_help_message` (which deliberately logs only `type(underlying).__name__`).
**Verify:** classifying an `AuthenticationError` emits one warning naming the type, both axes
as false, and that it will not retry; an unrecognized `RuntimeError` emits a warning
identifying it as the default branch; and for a classifier constructed with a known API key,
header, and endpoint, none of those three values appears in any emitted log line.

### Architectural Requirements

#### AR-3.1: Broad catches keep their INV-4 justification
`classify`'s `except Exception` remains — it must still catch anything, since
`classify_failure` is total — and keeps its `# noqa: BLE001` plus a comment updated to state
that the exception is now classified rather than uniformly retried. FR-2.2's fallback guard
carries the same annotation.

---

## Feature 4: Tests

**Who & why:** Every requirement here is about which of several branches a failure takes, which
is cheap to assert exactly against a fake and nearly impossible to observe in a real run — a
misclassification shows up as a slow run or a confusing error days later. The project also
forbids live provider calls in tests, so fakes are the only option.

### Functional Requirements

#### FR-4.1: Unit tests ship with the implementation
A `pytest` suite ships alongside the code, in a new `tests/test_classifier_errors.py`, runnable
with `.venv/bin/python -m pytest`. It covers every FR's **Verify:** condition above, and
specifically:

- FR-1.1's axis pair for one instance of every type in FR-1.2's table, `(True, True)` for
  `pydantic.ValidationError`, and the `(False, False)` default for an unrecognized type.
- FR-1.2's four hierarchy traps, every assertion made with `issubclass` against the `litellm`
  namespace rather than by reading `__mro__` names: the five `BadRequestError` subclasses not
  collapsing; `BudgetExceededError` classifying despite deriving from `Exception`;
  `BadGatewayError` and `Timeout` each classifying as retryable **and** each asserted *not* to
  subclass the type it might plausibly be folded into; and `AuthenticationError` and
  `InvalidRequestError` each asserted not to match the `BadRequestError` branch.
- FR-2.1's no-fallback cases asserted by **call count**, not only by the raised type.
- FR-2.2's guarded fallback, using a plain `BadRequestError` first call so the fallback is
  actually reached, asserting the raised exception is the fallback's own and its `__cause__` is
  the first attempt's.
- FR-2.3's regression case, plus its replacement of the superseded
  `test_complete_reraises_unsupported_params_error_when_no_temperature_to_drop`.
- FR-2.4's four conditions **plus both overlap cases** — `length` with `content=None` must
  raise the truncation type, and `content_filter` with `content=None` must raise the policy
  type. A test that only covers the non-overlapping shapes would pass against the wrong
  ordering.
- FR-2.5's ladder via the patched constant **with matching kwargs injection**, the
  skip-absent-entry case, the unpatched single-entry drop mechanics, and the
  most-reduced-kwargs assertion.
- FR-3.1's per-class attempt counts. The no-sleep assertion is made against a **multi-attempt**
  configuration (`max_retries >= 2`) so it cannot pass trivially, by patching `time.sleep` and
  asserting it is not called for the non-retryable cases and is called for the retryable ones.
- FR-3.2's log content, with the secret-absence assertion scoped to the specific key, header,
  and endpoint values the test itself sets on the classifier.

**No test may invoke a real `litellm.completion`** — the LLM call is faked, following the
existing `_FakeResponse`/`_FakeChoice` fixtures in `tests/test_cerebus.py:328-336`. Those
fixtures must gain a `finish_reason` attribute, which they currently lack, or FR-2.4's
inspection breaks the two surviving tests that use them. Tests written alongside the code, not
deferred.
**Verify:** `.venv/bin/python -m pytest` passes with the new tests present, the pre-existing
suite still passes apart from the one test FR-2.3 supersedes, and no test triggers a network
call.

---

## Integration Points

- **`classifier.py`** — the only module whose logic changes. Gains `FailureKind`,
  `classify_failure`, three response-failure exception types, the narrowed and guarded
  fallback, the declared drop ladder, and `classify`'s classification branch.
- **`tests/test_cerebus.py`** — `_FakeChoice` gains `finish_reason`;
  `test_complete_reraises_unsupported_params_error_when_no_temperature_to_drop` (`:368-381`) is
  superseded per FR-2.3; the other two `_complete` tests (`:361`, `:397`) are the regression
  guard for AR-2.2.
- **`pipeline.py` / `debate.py` / `multi_model.py` / `induction.py`** — all call
  `Classifier.classify` and all catch broadly. None require changes, verified by tracing all
  five call sites: FR-3.1 changes *when* an exception arrives, never its type or the fact that
  one arrives. This spec deliberately does not make them react to the classification.
- **`spec/ARCHITECTURE.md`** — INV-7's contradicted sentence (AR-1.2), and the Known Gap
  "Fallback-trigger scope is broader than 'schema unsupported'", which FR-2.1 closes.

## Related Specs

| Spec | Relationship | Affected Requirements |
|------|-------------|---------------------|
| Spec 5: batched-classification | **Depends on** — its error classification, output cap/truncation handling, and ordered parameter drop all consume this spec's `isolable` axis and mechanics; three of its plan-critique blocking items were defects in this chain | FR-1.1, FR-1.2, FR-2.4, FR-2.5 |
| Spec 3: cerebus-gateway | **Modifies** — generalizes the `UnsupportedParamsError` temperature drop it introduced into a declared ladder, fixes the fallback-shadowing regression that change caused, and supersedes one of its tests | FR-2.3, FR-2.5 |
| Spec 1: initial-classification-with-critics | **References** — `debate.py`'s sampling/Critic/Reconciler calls all go through `Classifier.classify`, so they see FR-3.1's earlier failures; no change required of them | FR-3.1 |
| Spec 4: multi-model-classification | **References** — `multi_model.py`'s per-model calls likewise see earlier failures; its per-model error columns record the same exception types | FR-3.1 |

## Constraints

- **Behavior change is confined to failure timing, fallback reach, and unusable responses.** A
  successful call is byte-for-byte unaffected, and a retryable failure — including
  `pydantic.ValidationError`, the common case — retries exactly as today. The changes are:
  non-retryable failures surface sooner (FR-3.1), a truncated, policy-blocked, or empty response
  raises instead of returning unusable content (FR-2.4), the fallback no longer runs for four
  error classes (FR-2.1), and it now runs for one case where it was unreachable (FR-2.3).
- **No new runtime dependency.** Every litellm exception type, `Choices.finish_reason`'s nine
  literals, `ModelResponse.choices`' list type, and `Message.content`'s `str | None` annotation
  were all enumerated live from the installed version.
- **No live provider call in any test** (project rule, `AGENTS.md`).
- **INV-1** is unaffected — no new module, no new import edge; `classifier.py` stays a leaf.
  **INV-5** (`max_retries >= 1`) and **INV-11** (schema enforces cardinality/type only) are
  unchanged. **INV-4** applies to the new broad catches (AR-3.1). **INV-7** requires the
  amendment in AR-1.2.
- **Classification is by exception type and `finish_reason` only.** Message-text parsing and
  `supports_response_schema` are both rejected, for the distinct reasons given in FR-2.1.
- The two axes describe what a caller may *do* about a failure, not what caused it. This is why
  they are independent booleans rather than one enum — spec 5's plan critique found that naming
  a single class after its cause produced a requirement that contradicted its own contents.

## Out of Scope

- **Making any caller react to the classification.** `pipeline.py`, `debate.py`,
  `multi_model.py`, and `induction.py` keep their current broad catches. Spec 5 is the first
  intended consumer of the `isolable` axis, and it is specced separately.
- **Gating structured output on `litellm.supports_response_schema`** — considered and rejected;
  it returns `False` for every Cerebus gateway slug, so it would degrade every real run (FR-2.1).
- **Persisting or surfacing per-row failure detail.** The Known Gap that `pipeline.py:89-90`
  discards the exception and only increments a counter is real and adjacent, but closing it
  means a new output-CSV column or log sink — a second module and an observable artifact change.
  FR-3.2's logging is inside `classifier.py` only.
- **Retry backoff strategy.** `retry_delay` stays a fixed sleep; exponential backoff and jitter
  are a separate concern from classification, and spec 5 defines its own bounded backoff at the
  batch level.
- **The unguarded incremental CSV write** (the other `pipeline.py` Known Gap) — unrelated.
- **Making `Classifier` thread-safe or documenting its concurrency properties.** INV-6 already
  records that this is a LiteLLM property not verified here.
- **Adding the classification to the public API** (`__init__.py`'s eight names) — see AR-1.1.

## Spec Completeness Checklist

- [x] **Scope & acceptance criteria** — seven verified defects enumerated in the Overview and
  each assigned to an FR (FR-2.1/2.2/2.3/2.4/2.5, FR-3.1, and FR-2.4's empty-response case); the
  behavior change is bounded explicitly in Constraints; seven exclusions with reasons in Out of
  Scope; every FR has a **Verify:** line.
- [x] **Testing strategy** — FR-4.1 requires a `pytest` suite shipped alongside the code, names
  the runner and the new file, maps each FR's **Verify:** condition to a test, enumerates nine
  specific groups including all four hierarchy traps and the two regressions, restates the
  no-live-call rule, names the fixture that must gain `finish_reason` and the test that is
  superseded, and pins the no-sleep assertion to a multi-attempt configuration so it cannot pass
  trivially; AR-1.2 records the INV-7 amendment.
- [x] **Existing patterns** — builds on `classifier.py:294-312`'s own subclass-ordering lesson
  (AR-2.1), reuses the sanitization convention from `debate.py`'s `_sanitize_error` and
  `_cerebus_key_help_message` (FR-3.2), the not-re-exported-but-importable pattern from
  `debate.py`/`multi_model.py` (AR-1.1), and the INV-4 annotation convention (AR-3.1).
- [x] **Dependencies** — none added. Every litellm type, the nine `finish_reason` literals, and
  the `choices`/`content` annotations were enumerated live (Constraints).
- [x] **Architecture & interfaces** — the classification's home and visibility settled (AR-1.1),
  the `-> str` annotation made honest rather than merely claimed (AR-2.2), dispatch ordering
  constrained (AR-2.1), the INV-7 amendment specified (AR-1.2), and all touched, superseded, and
  deliberately-untouched call sites enumerated (Integration Points).
- [x] **Error handling & failure modes** — this spec *is* the error handling. FR-1.1 makes
  classification total along two axes with a stated default; FR-1.2 tabulates every type on the
  `litellm` namespace and the four hierarchy traps, and requires membership to be pinned by
  `issubclass` rather than by base-class names; FR-2.4's checks are a normative ordered sequence
  with both overlap cases specified; FR-2.1/2.2 close the over-broad and unguarded fallback; FR-2.3 restores
  an unreachable fallback and names the test it supersedes; FR-2.4 separates truncation, policy
  refusal, and empty responses from malformation; FR-2.5 bounds the ladder and fixes the
  post-exhaustion kwargs; FR-3.1 defines retry behavior on one axis.
- [x] **Security review** — FR-3.2 is the only new output channel and is constrained to
  exception type plus axis values, with no response body, credential, header, or endpoint,
  following the two existing sanitization precedents, and its Verify line is scoped to values the
  test itself sets. FR-2.1 has a modest security-adjacent benefit: a narrowed fallback stops
  re-sending content-policy-violating and oversized payloads a second time. (An earlier draft
  claimed the same for credential errors; that was **false** — `AuthenticationError` is not a
  `BadRequestError`, verified, so it never reached the fallback.) No authn/authz surface is added
  and no new input is parsed.
- [x] **Performance impact** — positive and bounded for non-retryable failures: 1 attempt
  instead of `max_retries` (default 3) and 0 sleeps instead of 2 × `retry_delay` (default 5s),
  per affected call. On a 29.6k-row run against bad credentials that is the difference between
  one wasted call per row and three plus 10s of sleeping. Crucially **not** a saving on
  malformed responses — those stay retryable by design (FR-1.1), since removing that retry
  would cost accuracy rather than save time. Successful calls are unaffected.
- [x] **Rollout & migration** — no persisted artifact, schema, or CLI flag changes, so there is
  nothing to migrate and no version to bump. The behavior change is confined per Constraints, and
  the rollback is a revert of one module plus one test: callers are untouched (Integration
  Points), so nothing depends on the new surface until spec 5 does.
- [x] **Assumptions & risks** — the axis assignments are a judgement about what callers can do,
  and a misassignment is the main risk. Two realized instances are recorded rather than
  hypothesised. First: an earlier draft marked `pydantic.ValidationError` non-retryable, which
  would have deleted the system's most valuable retry — FR-1.1 now carries that case as its
  worked example and FR-3.1's Verify asserts it as a regression guard. Second, and the reason
  FR-1.2's Verify is worded as it is: two earlier drafts got the litellm hierarchy wrong by
  reading `__mro__` *names*, which belong to the openai SDK — first missing `BadGatewayError`,
  then wrongly folding `Timeout` under `APIConnectionError` and adding an `APIError` row that
  could never match. Both would have silently stopped retrying a common transient failure.
  Residual mitigations: the default is the no-action direction, FR-3.2 logs every decision
  including the default branch, FR-4.1 asserts each row individually, and FR-1.2 requires every
  membership assertion to use `issubclass` against the `litellm` namespace so a future upgrade
  that re-parents a type fails loudly instead of silently re-defaulting it.

---

## Change Log

### Update from critique-consolidated-v-1.md (2026-09-15)

All eight blocking items applied, plus every non-blocking improvement. Three of the critique's
findings were corrections to factual claims I had made in the spec's own supporting text; all
three are verified and fixed rather than reworded.

**Applied — the central design fix:**
- **FR-1.1: the taxonomy is now two independent booleans (`retryable`, `isolable`), not three
  mutually-exclusive classes.** The single-enum design assigned `pydantic.ValidationError` to
  `ISOLABLE`, and FR-3.1 failed fast on it — which would have deleted the retry the loop was
  actually built for. `classifier.py:333`'s `model_validate_json` runs inside `classify`'s
  `try`, so a malformed response is retried today, and LLM output is stochastic enough that the
  retry usually works. `ValidationError` is genuinely both retryable and isolable, and one enum
  cannot say so. FR-3.1 now consults `retryable` only; the `isolable` axis exists purely for
  callers. The Constraints paragraph that defended the single-class naming is replaced — its own
  argument was papering over this defect.
- **FR-3.1's Verify now asserts the `ValidationError` retry explicitly as a regression guard**,
  and the performance claim is corrected: it is a saving on non-retryable failures only, and
  deliberately not on malformed responses.

**Applied — taxonomy completeness (FR-1.2 rewritten as a table):**
- `BadGatewayError` (a 502 that descends from `APIStatusError`, **not**
  `InternalServerError`) and bare `APIError` are now named retryable. Both previously fell to
  the default and would have stopped being retried — for this repo's Cerebus gateway a 502 is
  the likeliest transient failure.
- `BudgetExceededError` reclassified to neither axis: repeating cannot un-exceed a budget, which
  contradicted the old `RETRYABLE` assignment's own definition.
- `ImageFetchError` and `LiteLLMUnknownProvider` classified, so the table is exhaustive.
- **"Four `BadRequestError` subclasses" corrected to five** (verified: `ContentPolicyViolationError`,
  `ContextWindowExceededError`, `ImageFetchError`, `LiteLLMUnknownProvider`,
  `UnsupportedParamsError`), and a third hierarchy trap added for `BadGatewayError`.

**Applied — the completion path:**
- **FR-2.2 now says the original exception object is re-raised**, resolving an incoherence all
  three critics flagged: `classify_failure` returns a `FailureKind`, so "re-raise as the
  classified failure" had no meaning, and the old Verify line demanded the raised object *not*
  be the original. Re-raising the original is also required for spec 5 to catch
  `ContextWindowExceededError` by type.
- **FR-2.2's Verify no longer uses `ContextWindowExceededError`** — FR-2.1 stops that error from
  reaching the fallback at all, so the old reproduction was unreachable under this spec's own
  rules. It now uses a plain `BadRequestError` first call.
- **FR-2.3 declares `tests/test_cerebus.py:368-381` superseded** and specifies its replacement.
  That test asserts today's re-raise as intended, so the old spec's claim that all three
  `_complete` tests "keep passing unchanged" was false. FR-2.3 also now states *how* the two
  `UnsupportedParamsError` cases are distinguished — ladder exhaustion, not message inspection.
- **FR-2.4 gained two conditions beyond `length`:** the policy `finish_reason` values
  (`content_filter`, `guardrail_intervened` — two of the nine literals litellm declares), and
  the **seventh defect**: `ModelResponse.choices` is a list that can be empty and
  `Message.content` is `str | None` (both verified), so `_complete`'s `-> str` annotation was
  never true. AR-2.2 no longer claims the contract "stays" — FR-2.4 is what makes it true.
- **FR-2.5 declares the ladder as a patchable module-level constant** (one entry today,
  `temperature`), which makes its Verify implementable without importing spec 5's `max_tokens`,
  and now specifies that the post-exhaustion fallback sends the **most-reduced** kwargs — the
  alternative would re-send a parameter the model already rejected, reintroducing the bug
  `classifier.py:295-305` documents.

**Applied — factual corrections to the spec's own text:**
- **The Security-review section's auth-error benefit was false and is deleted.** It claimed an
  `AuthenticationError` "currently triggers a second identical request carrying the same
  credentials"; verified that `AuthenticationError` is **not** a `BadRequestError` subclass, so
  it never reached the fallback. Replaced with the smaller accurate benefit.
- **FR-2.1's rationale was wrong.** It claimed no positive capability signal exists.
  `litellm.supports_response_schema(model=...)` does exist — but verified to return `False` for
  **every** Cerebus gateway slug, since it is a static map lookup, so gating on it would degrade
  every real run to JSON-object mode. Now recorded as a considered-and-rejected alternative in
  both FR-2.1 and Out of Scope, so a future reader who finds the predicate does not "improve"
  the code into that regression.
- FR-2.4 now states which of `finish_reason`'s nine literals are treated as success and why.

**Applied — non-blocking:** FR-3.2's secret-absence assertion scoped to values the test sets;
FR-4.1's no-sleep assertion pinned to a multi-attempt configuration so it cannot pass trivially.

**Rejected:** none. Every finding was verified against the code and accepted.

**Reorganized:** FR-1.2 restructured from prose into a classification table, since the two-axis
model makes per-type assignment the requirement's substance. No FRs or ARs added, removed, or
renumbered — totals remain 10 FRs and 5 ARs. Feature 1's title changed from "A failure taxonomy"
to "A failure classification", since "taxonomy" implied the single-enum shape that was the
central defect.

### Update from critique-consolidated-v-2.md (2026-09-15)

All five blocking items applied, plus every non-blocking improvement. The v-1 round's central
design fix (the two-axis model) was confirmed resolved by both critics and is untouched here;
this round is entirely about the mechanical verification underneath it.

**Applied — the litellm hierarchy, rebuilt from `issubclass` rather than `__mro__` names:**
- **`Timeout` is now its own retryable row**, and the false parenthetical "(which `Timeout`
  subclasses)" is deleted. Verified `issubclass(litellm.Timeout, litellm.APIConnectionError)` is
  **`False`** — they are siblings under *openai*'s `APIConnectionError`. As previously specced,
  one of the most commonly raised errors in any deployment fell to the `(False, False)` default
  and stopped being retried.
- **The inert `APIError` row and the `APIStatusError` trap are deleted.** Verified
  `litellm.APIError` has **zero** subclasses and `litellm.APIStatusError` does **not exist** —
  every concrete litellm error descends from the openai SDK, so the row could never match and
  the trap named a nonexistent class.
- **FR-1.2 is now a flat ordered chain over named types**, and ten previously-unclassified
  namespace types are assigned: `Timeout`, `InvalidRequestError` (verified **not** a
  `BadRequestError` subclass despite the name — now trap 4), `UnprocessableEntityError`,
  `APIResponseValidationError`, `JSONSchemaValidationError`, `RouterRateLimitError`,
  `RouterRateLimitErrorBasic`, `OpenAIError`, `BaseLLMException`, `MockException`.
- **FR-1.2's Verify now requires every membership assertion to use `issubclass` against the
  `litellm` namespace**, never `__mro__` names or an assumed base-class relationship. This is
  the procedural fix for the actual root cause: two consecutive drafts got the hierarchy wrong
  the same way, because `__mro__` prints openai's class names and they look like litellm's.
  AR-2.1 is retitled and rewritten around the same rule.

**Applied — contradictions between a requirement and its own Verify:**
- **FR-2.2 now raises the fallback's own exception, chained `from` the first attempt's.** The
  previous wording ("the original exception object is re-raised") contradicted its own Verify
  line, which asserted the *fallback's* `RateLimitError` propagates. The lapsed rationale is
  removed too — it cited spec 5 needing to catch `ContextWindowExceededError` by type, which
  FR-2.1 now prevents from reaching the fallback at all. Chaining keeps the first failure
  diagnosable via `__cause__`.
- **FR-2.4's checks are now a normative ordered sequence** (empty `choices` → `length` →
  `content_filter` → `content is None`) because the conditions overlap with opposite
  classifications. Both critics found this independently with different worked examples: a
  reasoning model exhausting its output allowance returns `length` **and** `content=None`, and a
  content-filtered response often carries no content — under the old unordered list, the
  content-first reading would re-send policy-violating content `max_retries` times, contradicting
  this spec's own security rationale.
- **FR-2.5's ladder Verify is now implementable**: the test must inject the synthetic parameter
  into kwargs as well as patching the constant (patching alone is insufficient, since
  `_completion_kwargs` never emits it), the constant is read by module-attribute lookup at call
  time, an absent entry is skipped without a completion call, "most-reduced" is defined as the
  last attempt's kwargs, and the "unchanged behavior" clause is narrowed to the drop *mechanics*
  so it stops contradicting FR-2.3's deliberate change to the post-exhaustion outcome.

**Applied — non-blocking:**
- FR-2.4 now records that `map_finish_reason` normalizes `guardrail_intervened` to
  `content_filter`, and normalizes `eos`, `finish_reason_unspecified`, `malformed_function_call`,
  **and any unrecognized value** to `stop` (all verified) — so only four literals ever arrive as
  themselves, and `finish_reason` can never signal a failure litellm does not already know
  about. The policy check needs only `content_filter`.
- AR-1.2 now requires the ADR to update `spec/ARCHITECTURE.md:84`'s `tests/` row as well as
  INV-7's prose, since both list the same permitted-import set.
- FR-4.1's test list updated for the four traps, both FR-2.4 overlap cases, the ladder's kwargs
  injection and skip-absent-entry step, and FR-2.2's `__cause__` assertion.
- The Assumptions & risks entry now records the hierarchy mistake as a second realized risk
  alongside the `ValidationError` one, and names FR-1.2's `issubclass` requirement as its
  mitigation.

**Rejected:**
- **The claim that there are six `BadRequestError` subclasses, including `RejectedRequestError`.**
  Verified against the installed version: there are exactly **five**
  (`ContentPolicyViolationError`, `ContextWindowExceededError`, `ImageFetchError`,
  `LiteLLMUnknownProvider`, `UnsupportedParamsError`), and `RejectedRequestError` is not present
  on the `litellm` namespace at all. The spec's existing count was correct and is unchanged.

**Reorganized:** FR-1.2's table rebuilt around a flat type chain and its trap list extended from
three to four; AR-2.1 retitled from "Subclass tests must precede base-class tests" to "Dispatch
is a flat ordered chain over named types", since the subclass-ordering rule is now the narrower
half of a broader constraint. No FRs or ARs added, removed, or renumbered — totals remain 10 FRs
and 5 ARs.
