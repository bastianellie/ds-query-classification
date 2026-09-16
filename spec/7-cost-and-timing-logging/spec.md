# Spec 7: Cost and Timing Logging

> **Status: CLOSED** — Implemented and verified on 2026-09-16.
> Implementation summary: `spec/7-cost-and-timing-logging/implementation-summary.md`
> Merged from worktree branch `spec/7-cost-and-timing-logging` on 2026-09-16.

## Overview

Every run of `experiment.py` or `classify.py`/`cli.py` currently reports whether it succeeded
and how many rows failed, but not what it cost or how long it took. This spec adds a small,
always-on cost/timing report, written as a separate JSON artifact alongside each run's other
outputs: total experiment cost, category-induction cost, average/standard-deviation cost per
query, average/standard-deviation cost per batch (`--batch` only), and total wall-clock time.

Cost is an **estimate** derived from `litellm.get_model_info`'s per-token pricing and real
per-call token usage — never a billing-system-accurate figure, and explicitly labeled as such.

## Goals

- Report five figures per run: total cost, induction cost, per-query cost (mean + stddev),
  per-batch cost (mean + stddev, `--batch` only), and total wall-clock time.
- Reuse spec 5/6's established patterns (candidate-walk model-id resolution, thread-safe stats
  collectors, artifact-file conventions) rather than inventing new ones.
- Never change any existing public function's behavior for a caller that doesn't opt in to the
  new cost-tracking parameter — every existing test that monkeypatches `Classifier.classify`
  wholesale must keep passing unmodified.

---

## Feature 1: Per-call token usage capture in `Classifier`

**Who & why:** Every cost figure in this spec is derived from real per-call token usage, which
`Classifier` currently discards. `_attempt_completion` (`classifier.py:489-537`) returns
`_extract_content(response)` — a bare string — throwing away `response.usage` entirely. Nothing
above `Classifier` can compute a real cost without this data, so it has to be captured at the
one place a genuine `litellm.completion()` response object is available.

### Functional Requirements

#### FR-1.1: Every real completion call's token usage is captured, not discarded
Each `litellm.completion()` response `_attempt_completion` obtains — whether from the
preferred structured-output attempt or the JSON-mode fallback (`_fallback_completion`) — has its
`response.usage.prompt_tokens`/`response.usage.completion_tokens` captured before the response
is reduced to its content string. Only calls that actually return a response object contribute:
an attempt that raises before ever obtaining one (e.g. `UnsupportedParamsError`, a network
error) contributes nothing, since no usage exists to read — this is the common case for a
parameter-rejection error, which providers typically reject before generating any billable
output. A response that does exist but has no `usage` attribute (a fake/test double that
doesn't set one) likewise contributes nothing, not an error.
**Verify:** a fake completion response with `usage.prompt_tokens=100, completion_tokens=50`
results in exactly those two numbers being attributed to that call; a fake response lacking a
`usage` attribute entirely does not raise; an attempt that raises `UnsupportedParamsError`
before any response is obtained contributes nothing, and only the subsequent, successful
retry's usage is captured.

#### FR-1.2: `classify()`'s own retry attempts are summed into one total per call
`Classifier.classify(text)`'s existing retry loop (`classifier.py:564-598`) may call
`_complete`/`_attempt_completion` more than once for a single logical classification (each
attempt that reaches a real response is itself a real, billed API call). All of a single
`classify()` invocation's real, response-producing attempts — including a JSON-mode fallback
attempt inside `_attempt_completion` that itself returns a response — are summed into one
`(prompt_tokens, completion_tokens)` total for that invocation, whether it ultimately succeeds
or raises.
**Verify:** a classify() call whose first attempt raises `UnsupportedParamsError` (no usage) and
whose second (parameter-dropped) attempt succeeds reports only the second attempt's usage; a
classify() call that exhausts `max_retries`, where every attempt reached a real response before
failing content validation, reports the summed usage of every one of those responses.

#### FR-1.3: A context manager exposes one call's own usage, without changing `classify()`'s signature
`Classifier` gains a context manager, `classifier.track_usage()`, used by wrapping (not
changing the arguments to) a call: `with classifier.track_usage() as sink: classifier.classify(text)`,
then reading `sink.usage` afterward. This is the *only* mechanism Feature 3's exact-attribution
call sites (FR-3.1, FR-3.2) use — **`classify(text)`'s own call signature and arguments are
never touched, at any call site, under any circumstance.** This is deliberate: every existing
test that monkeypatches `Classifier.classify` wholesale (`tests/test_experiment.py`'s
`fake_classify`/`fake_batch_classify`, `tests/test_cerebus.py`, `tests/test_debate.py`,
`tests/test_multi_model.py` — all with a narrow `(self, text)` signature) would raise
`TypeError` on a new keyword argument the fake doesn't accept, the instant *any* code path those
tests exercise passed one — including `pipeline.py`'s own plain-mode submission and
`BatchRunner`'s own call, which those exact fixtures cover. A context manager sidesteps this
completely: it wraps the call from *outside*, so a wholesale-mocked `classify` runs exactly as
it does today (and simply produces no tracked usage inside the `with` block — acceptable, since
those tests aren't testing cost). Internally, `track_usage()` pushes a thread-local accumulator
that `_attempt_completion` checks and updates (when present) at the same point FR-1.1 captures
usage, and pops it back on `__exit__`. This is race-free for concurrent calls on a shared
instance because thread-local state is per-thread by construction, and — critically — is only
ever used by call sites where `classify()` runs synchronously start-to-finish on the *same*
thread that entered the `with` block (true for `pipeline.py`'s plain-mode submission and
`BatchRunner`'s top-level call; **not** used for `debate.py`/`multi_model.py`, whose own nested
`ThreadPoolExecutor`s would break thread-local propagation — see FR-3.3).
**Verify:** two threads each running `with classifier.track_usage() as sink: classifier.classify(text)`
concurrently on the same `Classifier` instance each observe only their own call's usage in
`sink.usage`, never the other's; a `classify()` call made with no `track_usage()` context active
behaves identically to today, with no new argument passed anywhere; a test that monkeypatches
`Classifier.classify` wholesale continues to pass unmodified, and code wrapping that mocked call
in `track_usage()` observes `sink.usage` as unknown/empty rather than raising.

#### FR-1.4: A per-instance running total, independent of `track_usage()`
Independent of whether a caller uses `track_usage()`, every real completion call updates a
thread-safe running total on the `Classifier` instance itself (mirroring `BatchStats`'s
lock-protected-counter pattern from spec 5), exposed via a public accessor (e.g.
`classifier.usage_totals() -> (prompt_tokens, completion_tokens)`). This is what Feature 3 reads
for aggregate figures (total cost, induction cost, and the `--critics`/`--models` approximation)
where per-call attribution isn't needed — it works correctly under concurrent, shared-instance
use because it's a single atomic accumulation, not a diff-before/after-a-call read, and it
requires no thread-locality assumption (unlike `track_usage()`), so it is exactly what
`debate.py`/`multi_model.py`'s nested-executor call patterns can still rely on despite FR-1.3's
limitation.
**Verify:** ten concurrent `classify()` calls on one `Classifier` instance (real or faked
completion) leave the instance's running total equal to the sum of all ten calls' usage,
regardless of interleaving; a `Classifier` instance whose `classify` is wholesale-mocked reports
an all-zero/unknown running total, never an error.

### Architectural Requirements

#### AR-1.1: `classify()`'s existing contract and every current caller are unchanged
`debate.py`, `multi_model.py`, `induction.py`, `batching.py`, and `pipeline.py`'s existing
plain-mode call site continue calling `classify(text)` with no arguments changed and no
different return value or exception behavior — this is now a structural guarantee of FR-1.3's
design (a context manager can never require its wrapped call to change), not merely a stated
intention. `tests/test_experiment.py`'s `fake_classify`/`fake_batch_classify`,
`tests/test_cerebus.py`, `tests/test_debate.py`, and `tests/test_multi_model.py`'s
monkeypatches of `Classifier.classify` (which replace the method wholesale, never invoking the
real implementation at all) keep working exactly as today — this spec must not require them to
be updated.
**Verify:** the full existing test suite passes unmodified except for the new tests this spec
adds; grepping every production call site of `.classify(` shows no call site anywhere passes
any argument beyond the existing `text`.

#### AR-1.2: `classifier.py` stays a leaf module
Per INV-1, `classifier.py` never imports any sibling module. The usage-capture machinery
(FR-1.1–FR-1.4) is implemented entirely with stdlib primitives (`threading.Lock`, plain
tuples/dataclasses) — it does not import `batching.py`, `cost.py` (Feature 2, new in this
spec), or anything else under `query_classification`.

---

## Feature 2: Cost-per-token resolution and computation

**Who & why:** `litellm.completion_cost()` was verified live against this repo's installed
litellm (1.101.0) and does not reliably work for the Cerebus workspace-slug model ids this
project actually runs with (e.g. `@sciencedirect-global-openai/gpt-5.4-mini-2026-03-17`): it
either raises `BadRequestError` ("LLM Provider NOT provided") when the slug is passed as
`model`, or silently resolves to the wrong model's pricing when passed inconsistently via
`completion_response.model` vs the `model=` override — verified live, both failure modes
reproduced. Cost must instead be computed manually from `get_model_info`'s
`input_cost_per_token`/`output_cost_per_token` fields, resolved the same way spec 5's FR-1.2
resolves token budgets.

### Functional Requirements

#### FR-2.1: Per-token cost is resolved via the same candidate walk as spec 5's token budgets
A new function resolves a model id's `(input_cost_per_token, output_cost_per_token)` using
the identical candidate sequence as `batching.py`'s FR-1.2 walk (the id as given, `openai/`
prefix stripped, `@workspace/` segment stripped, then progressively-shorter `-`-delimited
chunks) — independently per field, exactly like `resolve_token_budgets`, since a candidate
that resolves one field may not report the other. Verified live: `gpt-4o-mini`'s
`input_cost_per_token`/`output_cost_per_token` are `1.5e-07`/`6e-07`. Broad `except Exception`
per candidate (mirroring `resolve_token_budgets`'s own `# noqa: BLE001` — `get_model_info`
raises a bare `Exception` for an unmapped id, per spec 5's AR-1.3) carries the same comment.
Resolved once per distinct model id per run, not once per classifier instance — a `--critics`
run can construct one critic and one reconciler classifier per category, all potentially
sharing the same model id, and re-walking the same candidate sequence for each would multiply
lookup work by category count for no benefit.
**Verify:** the same three FR-1.2 chunk-down cases spec 5 already verifies (workspace slug
with/without `openai/` prefix, chunk-down to a real model, total failure) produce the expected
cost-per-token pair or `(None, None)`; resolving the same model id twice in one run performs
the candidate walk only once.

#### FR-2.2: Cost is computed from usage and pricing, never estimated when both are known
Given a `(prompt_tokens, completion_tokens)` total and a resolved `(input_cost_per_token,
output_cost_per_token)` pair, cost in USD is
`prompt_tokens * input_cost_per_token + completion_tokens * output_cost_per_token`. When
either side of the pricing pair is unresolved, the cost for that call is `None` (not `0`,
not silently dropped) — a `None` in a sum poisons the whole aggregate to `None` rather than
silently under-reporting, per FR-4.1's null-vs-zero handling.
**Verify:** known usage + known pricing produces the exact arithmetic result; an unresolved
price on either side produces `None` for that call, and a total that includes even one `None`
call is itself `None`.

#### FR-2.3: Deliberate duplication of the candidate walk, not a new import edge
`batching.py` already owns an equivalent candidate-walk (`_candidate_ids`, for token budgets).
This spec's cost-resolution module does not import `batching.py` to reuse it — cost tracking
must work identically whether or not `--batch` is active, so depending on the batching module
for a name-parsing utility would tie two orthogonal concerns together. The walk is duplicated,
following the same "small, cheap-to-duplicate helper, not worth a new import edge" precedent
spec 5's AR-2.3 already established for the `<<<DATA>>>` delimiters.
**Verify:** a test asserts the two modules' candidate-walk outputs stay identical for the same
inputs (mirroring AR-2.3's delimiter-equality test).

### Architectural Requirements

#### AR-2.1: A new `cost.py` module owns pricing resolution and computation
`src/query_classification/cost.py` (new) owns FR-2.1's candidate walk, FR-2.2's computation,
and Feature 4's timing helper. It may import `categories.py`/`classifier.py` (to read
`Classifier`'s usage totals via its public accessors); it must not import `batching.py`,
`pipeline.py`, `debate.py`, `multi_model.py`, `cli.py`, or `experiment.py`. `pipeline.py`,
`cli.py`, and `experiment.py` each gain an import of it. This mirrors `batching.py`'s own
role/placement from spec 5 (AR-1.1) — a new, independently-testable building block, not a
composed-in extension of an existing module.
**Verify:** `cost.py`'s own import block contains only `categories`/`classifier` from this
package.

---

## Feature 3: Cost attribution — exact per-row, per-batch, or approximated

**Who & why:** A user reading this report needs to know not just "the average cost" but
whether that average is exact (one call really did cover one row) or a necessary
approximation (one call covered several rows, or a row's true cost was scattered across
several classifiers with no cheap way to reassemble it). Silently presenting an approximation
as if it were exact would mislead exactly the kind of cost-conscious decision this report
exists to support.

### Functional Requirements

#### FR-3.1: Plain mode (no `--batch`, `--critics`, or `--models`) reports exact per-row cost
`pipeline.classify_csv` gains an optional `cost_collector` parameter (a run-scoped object the
caller — `cli.py`/`experiment.py` — creates and passes in, default `None`, following the exact
opt-in convention `batch_runner`/`critic_classifiers` already use: omitted, nothing changes for
any existing caller/test). When given, the existing plain-mode submission
(`executor.submit(classifier.classify, str(df.at[idx, column])): idx`, `pipeline.py:294`) is
wrapped — not altered in its call to `classify` itself — with `classifier.track_usage()`
(FR-1.3) around that one call, and the resulting per-row cost (via FR-2.2) is recorded into
`cost_collector`. Because exactly one `classify()` call is made per row in this mode, the
resulting per-row cost values are exact, real per-row samples — not derived by dividing
anything.
**Verify:** a faked-LLM plain run over 10 rows with distinct, known usage per row, given a
`cost_collector`, produces a per-query mean/stddev matching the exact arithmetic mean/(sample)
standard deviation of those 10 real values; a plain run with `cost_collector=None` (or an
existing test never passing one) behaves exactly as today.

#### FR-3.2: `--batch` reports exact per-formed-batch cost, and estimated per-row cost
`BatchRunner` (spec 5) gains a per-formed-batch cost figure: the cost of **only the first
attempt of a top-level formed batch** (the call made directly from `iter_batches`'s yielded
batch, before any of spec 5's FR-3.2 retry-at-original-size or bisection recursion) — not the
sum including retries/splits; `total_cost_usd` (FR-3.4) still includes those, only this
distribution figure excludes them. This is recorded once per formed batch via a new
`BatchStats` method (mirroring `record_call`/`record_trim`/`record_bisection`'s existing
lock-protected-counter pattern), wrapping that one top-level `classifier.classify()` invocation
in `classifier.track_usage()` (FR-1.3) — never changing what's passed to `classify` itself.
Per-query cost under `--batch` is an **estimate**: each formed batch's own cost (as recorded
above) is divided evenly across the rows it covers, and those per-row shares (which will differ
across batches of different arity/cost, so the resulting distribution is not degenerate)
become the samples for the per-query mean/stddev. The per-batch figure itself is **not** an
estimate — it is the real, measured cost of that one real call.
**Verify:** a `--batch 5` run over 20 rows (4 formed batches) whose batches have known,
distinct costs produces: a per-batch mean/stddev over exactly those 4 values, `batch_cost`
marked `estimated: false`; a per-query mean/stddev over 20 values, each being its batch's cost
divided by 5, `query_cost` marked `estimated: true`; a batch that gets bisected due to failure
still contributes only its first attempt's cost to the per-batch figure, not the bisection's
additional calls' cost (which are still counted in `total_cost_usd`, per FR-3.4).

#### FR-3.3: `--critics`/`--models` report an approximated, uniform per-query cost
`debate.run_debate` spawns not one but **two** of its own nested `ThreadPoolExecutor` instances
(`debate.py:174` for the sampling runs, `debate.py:221` for escalated categories' critic/
reconciler calls) — meaning a row's calls run on worker threads distinct from the one
pipeline.py assigned to that row, and distinct from each other. Verified live that thread-local
state and `contextvars` do not propagate across a `ThreadPoolExecutor` submission boundary, so
exact per-row attribution here would require restructuring `debate.py`/`multi_model.py`'s
return contracts, which is out of scope for this spec (see Out of Scope).
`multi_model.run_multi_model` has the identical obstacle: its own nested `ThreadPoolExecutor`
(`multi_model.py:70`) calls every resolved model concurrently, on worker threads distinct from
pipeline.py's. Instead, per-query cost under `--critics` or `--models` is approximated uniformly:
mean = (sum of every real call's cost across the sampling classifier, every critic classifier,
and every reconciler classifier — or every multi-model classifier — via FR-1.4's aggregate
totals) divided by the number of rows the run attempted to classify (`pipeline.py`'s `total`,
i.e. `len(work_idx)` — every row scheduled for this invocation, whether it ultimately succeeded
or failed, since a failed row's calls still cost real money and must not silently vanish from
the denominator); every row is assigned this same value as its sample, so the reported standard
deviation is `0`. Both the mean and the (trivial) stddev are marked `"estimated": true` in the
report (FR-5.1), same as `--batch`'s per-query figure.
**Verify:** a `--critics` run's per-query mean equals total classification-phase cost divided
by the number of rows attempted (including any that failed) exactly; its per-query stddev is
`0`; the report marks both as estimated.

#### FR-3.4: Induction and total cost are always exact aggregates
Induction cost is the induction `Classifier` instance's FR-1.4 running total (a single
instance, used for exactly one call — see `experiment.py`'s `_construct_classifiers`,
`will_induce` branch). Because `experiment.py`'s `main()` reassigns its own `classifiers`
local variable to the classification-phase dict immediately after induction
(`experiment.py:1239` then `:1349` — verified, the same name is reused, not preserved), the
induction classifier's usage **must be read and stored before that reassignment** — e.g. into a
separate `induction_classifier` reference kept alongside `classifiers`, not recovered from
`classifiers` after the fact. Total experiment cost is the sum of induction cost (if any) and
every classification-phase classifier's running total (the plain classifier, the
critic/reconciler dicts, the multi-model dict, or the batch runner's arity-keyed cache —
whichever the run actually used), regardless of mode. Under `--batch`, this total includes
every real call the arity-keyed cache made — top-level formed batches *and* every
retry-at-original-size and bisection call FR-3.2 explicitly excludes from the *per-batch
distribution* figure — since FR-1.4's running total accumulates unconditionally; only the
per-batch/per-query *distribution* statistics apply that exclusion, never the total.
`BatchRunner` exposes a public method summing FR-1.4's running total across every classifier in
its arity-keyed cache (never requiring a caller to reach into the private cache itself), for
this purpose. Neither the induction nor total figure is ever approximated; only the per-query
and per-batch *distribution* statistics (FR-3.2/FR-3.3) involve estimation. A classifier
instance that was constructed but never actually called (e.g. a critic/reconciler role bypassed
because consensus was reached without debate) contributes an exact `0`, never `None` — `None`
is reserved for unresolved *pricing*, not for the absence of calls.
**Verify:** total cost equals induction cost plus classification-phase cost exactly, summed
from the real per-classifier running totals (including retry/bisection calls under `--batch`),
for every mode this spec covers; a `run` invocation (induce + classify in one process) reports
a nonzero `induction_cost_usd` distinct from the classification-phase total, not lost to
variable reassignment; a critic/reconciler instance that consensus never escalated to
contributes exactly `0` to the total, not `null`.

### Architectural Requirements

#### AR-3.1: `BatchStats` gains cost tracking; `BatchRunner.run` gains one internal-only parameter
`BatchStats` (spec 5, `batching.py`) gains a new method for recording one formed batch's cost
and a corresponding field in `snapshot()`'s returned dict, following the exact null-vs-zero
convention `record_call`/`min_arity`/etc. already established. `BatchRunner.run`'s arity,
bisection policy, and retry policy are unchanged; it gains one new keyword-only parameter,
private and defaulted (e.g. `_top_level: bool = True`), so every existing caller — `pipeline.py`'s
top-level `executor.submit(batch_runner.run, texts)` and every test that calls `run(texts)`
directly, including tests that monkeypatch `Classifier.classify` wholesale — is unaffected.
`run`'s own recursive calls (retry-at-original-size and bisection) pass this parameter as
`False`; only a `_top_level=True` invocation's first attempt is wrapped in
`classifier.track_usage()` (FR-1.3 — not a new argument to `classify` itself) and recorded into
`BatchStats`.
**Verify:** `BatchStats.snapshot()`'s existing fields (`batched_calls`, `min_arity`, etc.) are
unchanged in shape; a new field for cost is present and null before any batch completes; a
direct call to `run(texts)` with no keyword argument behaves exactly as it does today.

---

## Feature 4: Wall-clock timing

**Who & why:** Alongside cost, a user wants to know how long a run actually took, to compare
`--batch`'s throughput claims against reality or to budget future runs.

### Functional Requirements

#### FR-4.1: Total wall-clock time is measured for the whole run
`experiment.py`'s `main()` has two distinct `try` blocks: an early one (`experiment.py:1065-1108`)
covering validation/preflight, whose failure exits before any run-directory work or artifact is
written; and the main one (`experiment.py:1162` onward) covering dataset loading, induction,
classification, and verification, whose broad failure handler (`:1456`) still writes
`run_config.json`. Timing starts right after the **first** `try` block succeeds — at
`stage = "starting"` (`experiment.py:1112`, immediately before `config` is first built) — and
stops immediately before the terminal `run_config.json`/`cost_report.json` write, on **either**
the success path or the failure handler. A failure in the *first*, early `try` block (before
timing starts) produces no `cost_report.json` at all — consistent with it also producing no
`run_config.json` today. `cli.py`'s `main()` records the same kind of window around its
**entire** working body (from just after its own argument/resource validation through writing
the cost report), not narrowly around the `classify_csv` call alone. Measured with
`time.monotonic()` (immune to wall-clock adjustments), never `time.time()`.
**Verify:** a faked-LLM run with an injected `time.sleep` of a known duration reports a wall
time at least that long; a run that fails during the main `try` block still reports a (shorter,
non-zero) wall time in `cost_report.json` rather than omitting the field; a run that fails
during the *early* validation `try` block produces no `cost_report.json` at all, matching
`run_config.json`'s existing behavior for the same failure class.

---

## Feature 5: Report artifact and CLI integration

**Who & why:** This data needs a stable, discoverable location so a user (or a script
comparing runs) can find it without re-deriving it from `run_config.json`'s unrelated fields.

### Functional Requirements

#### FR-5.1: `experiment.py` writes `cost_report.json` into the run directory
A new artifact, `{run_dir}/cost_report.json`, added to `_ARTIFACT_FILENAMES`
(`experiment.py:75-83`) so `--overwrite`'s existing cleanup handles it identically to
`run_config.json`/`test_classified.csv`. Written once, after the run reaches its terminal
state within the *main* `try` block (success or failure — see FR-4.1; a failure in the early
validation `try` block produces no report at all), using the shared atomic-write helper
(AR-5.2). Shape:
```json
{
  "schema_version": 1,
  "total_cost_usd": 0.1234,
  "induction_cost_usd": 0.0050,
  "query_cost": {"mean_usd": 0.0012, "stddev_usd": 0.0003, "count": 100, "estimated": false},
  "batch_cost": {"mean_usd": 0.0080, "stddev_usd": 0.0015, "count": 20, "estimated": false},
  "pricing": {"resolved_input_model": "gpt-4o-mini", "resolved_output_model": "gpt-4o-mini"},
  "wall_clock_seconds": 42.7
}
```
`total_cost_usd` already includes `induction_cost_usd` (per FR-3.4) — a reader must not add
them together. `induction_cost_usd` is `null` for a `classify`-only run (no induction phase).
`batch_cost` is `null` entirely when `--batch` was not used (matching spec 5's `batch` block's
own null-when-absent convention in `run_config.json`). `query_cost.estimated` is `true` exactly
under the conditions FR-3.2/FR-3.3 describe (`--batch`, `--critics`, or `--models`), `false` for
exact plain-mode figures; `batch_cost.estimated` is **always `false`** when present — the
per-batch figure is a real, measured cost, never a division-derived estimate (only the
per-*query* figure divides a batch's cost across its rows). `"count"` is the number of samples
the mean/stddev were computed over — rows *attempted* (not merely succeeded, per FR-3.3) for
`query_cost`, formed batches for `batch_cost`. `"pricing"` records the same kind of
diagnostic candidate names spec 5's `run_config.json` `batch` block already records for token
budgets (`resolved_input_model`/`resolved_output_model`) — here, whichever model id FR-2.1's
candidate walk actually resolved pricing from, so a wrong chunk-down is diagnosable the same
way; `null` for either side that never resolved.

Only *monetary* fields (`total_cost_usd`, `induction_cost_usd`, `query_cost`/`batch_cost`'s
`mean_usd`/`stddev_usd`) become `null` when pricing is unresolved (FR-2.2) — `count` and
`estimated` stay populated and meaningful regardless, since they describe the run's shape, not
its cost. A run with `query_cost.count == 0` (e.g. every row already complete under
`--restore`, or an `induce`-only run) reports `mean_usd`/`stddev_usd` both `null` — never a
division-by-zero error, never a fabricated `0`. A run with exactly one sample
(`query_cost.count == 1` or `batch_cost.count == 1`) reports that one value as `mean_usd` and
`stddev_usd` as `null` — sample standard deviation is mathematically undefined for a single
observation, and `0` would misleadingly imply certainty about variance rather than an absence
of enough data to compute it. All numeric fields are finite (never `NaN`/`Infinity`, which
`json.dump` would otherwise happily emit as non-standard JSON tokens) — an internal
non-finite result is treated as unresolved (`null`), not serialized literally.
**Verify:** a classifying run's `cost_report.json` has all seven top-level shapes present; an
`induce`-only run has `induction_cost_usd` set, `query_cost.count == 0` with both stats `null`,
and `batch_cost` `null`; a `--batch` run has `batch_cost` populated with `estimated: false` and
`query_cost` with `estimated: true`; a run with an unresolvable model has every *monetary* field
`null` while `query_cost.count`/`estimated` and `wall_clock_seconds` remain populated; a run
with exactly one classified row reports that row's exact cost as `mean_usd` and `null` for
`stddev_usd`.

#### FR-5.2: `cli.py` writes a sibling cost report file, on success only
`cli.py`'s `main()` writes `{output_stem}.cost_report.json` next to the CSV it wrote, using
`Path(csv_path).with_suffix(".cost_report.json")` semantics (so `out.csv` → `out.cost_report.json`,
and a path with multiple suffixes like `out.csv.gz` or no suffix at all still produces exactly
one well-formed sibling, replacing only the final suffix) — deriving `csv_path` from `--output`
when given, or from `--input` when `--output` is omitted and the input is overwritten in place,
matching how `classify_csv` itself already resolves the effective output path
(`pipeline.py:73-74`). Unlike `experiment.py`, `cli.py` has no existing "write an artifact even
on failure" convention (its current exception handler, `cli.py:651`, catches only
`ValueError`/`FileNotFoundError`/`RuntimeError` and writes nothing) — this spec does not extend
`cli.py` with new failure-artifact infrastructure. The cost report is written **only after
`classify_csv` returns successfully**; a run that raises writes no `cost_report.json`, exactly
as it writes no other artifact today. Same JSON shape as FR-5.1 with `induction_cost_usd`
always `null` (no induction phase exists on this entry point) and `batch_cost` following the
same null-when-`--batch`-absent rule. Under `--restore` (a `cli.py`-only flag —
`experiment.py` has no equivalent, verified: it never passes `restore=` to `classify_csv`), the
report reflects only the rows *this invocation* classified, not rows skipped because they were
already complete — matching `--restore`'s own existing scope (it doesn't re-read or
re-attribute cost to work a prior invocation already paid for).
**Verify:** `python classify.py --input in.csv --output out.csv` produces
`out.cost_report.json` alongside `out.csv`; omitting `--output` produces
`in.cost_report.json` alongside the overwritten `in.csv`; `--output out.csv.gz` produces
`out.cost_report.json` (only the final suffix replaced); a run that raises before
`classify_csv` returns writes no cost report.

#### FR-5.3: A one-line stdout summary on both entry points
After writing the report, both entry points print exactly one **additional** line (alongside
whatever existing warnings/progress output they already print — `tqdm`'s progress bar,
`--batch`'s comparability caveat, etc.) summarizing total cost and wall-clock time, in a format
such as `Cost: $0.1234 estimated | Time: 42.7s` — with `Cost: unavailable` in place of a dollar
figure when `total_cost_usd` is `null`. Printed once, after the terminal `cost_report.json`
write — not per row, not per batch. `cli.py` prints nothing further on a failed run (FR-5.2:
no report is written in that case either).
**Verify:** stdout contains exactly one line matching this summary's format per successful run;
a run with unresolvable pricing prints `Cost: unavailable` rather than `$None` or a crash.

### Architectural Requirements

#### AR-5.1: `pipeline.py`, `cli.py`, and `experiment.py` each import `cost.py`
This adds three inbound edges to INV-1's dependency graph: `cost.py --> pipeline.py`,
`cost.py --> cli.py`, `cost.py --> experiment.py` — the same shape as `batching.py`'s own
three edges from spec 5. The implementation ships an ADR citing INV-1, following that
precedent, editing the existing `pipeline.py`/`cli.py`/`experiment.py` Module Boundary Map rows
plus adding a `cost.py` row.

#### AR-5.2: Both entry points write the report atomically; `cli.py` has no existing JSON writer
`experiment.py` already has `_write_json_atomic` (temp file + `os.replace`), used for
`run_config.json`/`categories.json`. `cli.py` currently writes no JSON at all — verified, it
has no `json.` usage anywhere today. `cost.py` owns one shared `write_json_atomic(path, data)`
helper (the same temp-file-plus-`os.replace` pattern) that both `cli.py` and `experiment.py`
call for their respective report writes, rather than `cli.py` duplicating `experiment.py`'s
private helper or writing non-atomically.
**Verify:** an interrupted write (simulated) never leaves a partially-written
`cost_report.json`/`{stem}.cost_report.json` on disk.

---

## Feature 6: Tests

### Functional Requirements

#### FR-6.1: Unit tests ship with the implementation
A `pytest` suite covers every FR's **Verify:** condition above, in a new `tests/test_cost.py`
plus additions to `tests/test_batching.py` (for `BatchStats`'s new field and `BatchRunner`'s
per-batch cost recording) and `tests/test_experiment.py`/`tests/test_cerebus.py` (for the
`cost_report.json` shape and the stdout summary line). No test makes a live provider call —
`litellm.completion` is faked, following `tests/test_batching.py`'s existing pattern. Tests
asserting a specific cost-per-token or a specific arithmetic result monkeypatch
`litellm.get_model_info` to return fixed, known values (`litellm` is unpinned in
`requirements.txt`/`pyproject.toml`, so its bundled pricing table can change independently of
this repo — a test asserting today's live `gpt-4o-mini` price would be brittle to a routine
dependency reinstall); at most one smoke test, mirroring spec 5's own precedent, may assert
against the real installed table's current shape (fields exist, are numeric) without pinning
exact values.
**Verify:** `.venv/bin/python -m pytest` passes with the new tests present, and no test
triggers a network call.

---

## Data Requirements

| Artifact | Change |
|---|---|
| `{run_dir}/cost_report.json` (new, `experiment.py`) | New artifact per FR-5.1. Not part of `run_config.json`; independently versioned (`schema_version: 1`). |
| `{output_stem}.cost_report.json` (new, `cli.py`) | New artifact per FR-5.2, sibling to the classified output CSV. |

## Integration Points

- **`classifier.py`** — gains usage-capture and `track_usage()` (FR-1.1–FR-1.4), entirely
  additive and backward-compatible (AR-1.1) — `classify()`'s own call signature never changes.
- **`batching.py`** — `BatchStats` gains one new method/field (AR-3.1); `BatchRunner.run`
  gains a private `_top_level` parameter and wraps its top-level call in `track_usage()`, no
  other behavior change.
- **`pipeline.py`** — plain-mode submission wraps its existing `classify(text)` call in
  `track_usage()` when an opt-in `cost_collector` is given (FR-3.1); imports `cost.py`.
- **`experiment.py`** / **`cli.py`** — each times the run (FR-4.1), reads every constructed
  classifier's usage after the run, computes and writes the report (FR-5.1/FR-5.2), and prints
  the summary line (FR-5.3).

## Related Specs

| Spec | Relationship | Affected Requirements |
|------|-------------|---------------------|
| Spec 5: batched-classification | **Extends** — `BatchStats`/`BatchRunner` gain cost tracking; reuses the FR-1.2 candidate-walk pattern (duplicated, per FR-2.3) | AR-3.1, FR-2.1 |
| Spec 6: classifier-error-path | **References** — the usage-capture hook (Feature 1) sits alongside, and must not disturb, spec 6's failure-classification/retry logic in the same methods | FR-1.1, FR-1.2 |
| Spec 3: cerebus-gateway | **References** — the Cerebus workspace-slug model id format is exactly what makes FR-2.1's candidate walk necessary; the same models this spec must price correctly | FR-2.1 |

## Constraints

- Cost figures are **estimates**, derived from `litellm.get_model_info`'s bundled pricing table,
  never a real billing reconciliation. The report and any documentation referencing it must say
  so.
- No new CLI flags — this feature is always on for both entry points, with no way to disable
  it (a user who doesn't want the file can ignore it; there is no cost to producing it beyond
  a small amount of extra bookkeeping already amortized across existing per-call work).
- `classify()`'s public contract must never change, for any caller, under any circumstance
  (AR-1.1) — this is the hard backward-compatibility line the whole design is built around;
  all cost-tracking attribution happens by wrapping calls from outside (`track_usage()`,
  FR-1.3), never by adding arguments to `classify` itself.

## Out of Scope

- Exact (non-approximated) per-row cost attribution under `--critics`/`--models`. This would
  require restructuring `debate.py`/`multi_model.py`'s return contracts to thread per-row usage
  back through concurrently-running nested thread pools — a substantially larger, separate
  change. FR-3.3's uniform approximation is the accepted behavior for this spec.
- Per-role cost breakdown in the report (e.g. separate `critics_cost_usd`/`multi_model_cost_usd`
  subtotals). Only the five figures the user asked for are in scope; a future spec can add
  finer-grained breakdowns if needed.
- Any new CLI flag to disable, redirect, or configure this feature.
- Real-time/streaming cost display during a run (e.g. a running total in the progress bar).
- Cost tracking for `induction.py`'s sampled-examples preflight or any non-LLM work.

## Spec Completeness Checklist

- [x] **Scope & acceptance criteria** — five figures, two entry points, exact-vs-estimated
  rules per mode are all explicit (FR-3.1–FR-3.4); Out of Scope lists what's deliberately
  excluded.
- [x] **Testing strategy** — FR-6.1 requires a `pytest` suite mapping every FR's Verify
  condition to a test, in `tests/test_cost.py` plus additions to existing files.
- [x] **Existing patterns** — candidate-walk duplication mirrors spec 5's own AR-2.3 precedent
  (FR-2.3); `BatchStats` extension mirrors its existing lock-protected-counter methods
  (AR-3.1); artifact-file handling mirrors `run_config.json`'s `_write_json_atomic`/
  `_ARTIFACT_FILENAMES` pattern (FR-5.1).
- [x] **Dependencies** — no new third-party libraries; only `litellm.get_model_info` (already
  used by spec 5) and stdlib (`time`, `threading`, `dataclasses`, `json`).
- [x] **Architecture & interfaces** — new `cost.py` module and its import boundaries specified
  (AR-2.1); `Classifier`'s new `track_usage()` context manager fully specified, with an explicit
  guarantee that `classify()`'s own call signature never changes at any call site (FR-1.3,
  AR-1.1); `classify_csv`'s new opt-in `cost_collector` parameter (FR-3.1); `BatchStats`'s new
  method and `BatchRunner.run`'s new private `_top_level` parameter (AR-3.1).
- [x] **Error handling & failure modes** — unresolvable pricing produces `null`, never a
  fabricated `0` (FR-2.2); zero- and one-sample distributions are explicitly defined (FR-5.1);
  a failed run still writes a `cost_report.json` with whatever was spent before failing plus
  wall-clock time, but only for failures inside `experiment.py`'s main work `try` block — an
  early-validation failure produces no report, matching `run_config.json`'s own behavior
  (FR-4.1); `cli.py` writes a report only on success, matching its existing (thinner) failure
  handling rather than inventing new artifact-on-failure infrastructure (FR-5.2).
- [x] **Security review** — no new external calls or auth surface; the new artifacts contain
  only derived numeric data (costs, counts, resolved model-id strings for pricing provenance),
  never raw prompts, completions, API keys, or endpoints; `get_model_info` calls are local
  library lookups, not network calls (verified in spec 5). All serialized numeric fields are
  constrained to finite values (FR-5.1) so a malformed usage/pricing value can't produce
  non-standard JSON.
- [x] **Performance impact** — the per-call lock-protected accumulation (FR-1.4) is O(1) per
  call; pricing is resolved once per distinct model id per run, not per classifier instance
  (FR-2.1), avoiding a multiplicative cost under `--critics`' per-category classifier
  construction. Not separately load-tested, since spec 5's own `BatchStats` already established
  this lock-protected-counter pattern's negligible overhead under concurrency.
- [x] **Rollout & migration** — purely additive; no existing artifact's shape changes
  (`run_config.json` is untouched by this spec), no schema version bump needed there. The new
  `cost_report.json`/`{stem}.cost_report.json` files are new artifacts with their own
  `schema_version: 1`.
- [x] **Assumptions & risks** — the biggest risk (exact per-row cost under `--critics`/
  `--models`) is explicitly called out and descoped (Out of Scope) rather than silently
  under-delivered; the `litellm.completion_cost()` unreliability finding (Feature 2's Who & why)
  is the load-bearing technical justification for building manual cost computation instead of
  using that API directly; the monkeypatch-compatibility risk (an earlier `usage_sink`-parameter
  design would have broken every test that wholesale-mocks `Classifier.classify`, verified
  against `tests/test_experiment.py`'s `fake_classify`/`fake_batch_classify` fixtures) is what
  drove FR-1.3's context-manager design instead.
