# Implementation Plan: 7-cost-and-timing-logging

**Status:** Ready
**Date:** 2026-09-16
**Spec:** spec/7-cost-and-timing-logging/spec.md
**Plan Critique:** spec/7-cost-and-timing-logging/plan-critique-consolidated-v-1.md

## Overview

Build bottom-up: `classifier.py`'s usage-capture hook and `track_usage()` context manager
first (Feature 1 — the foundation everything else reads from, and the piece with the hard
backward-compatibility constraint), then the new `cost.py` module (Feature 2's pricing/
computation primitives plus Feature 4's timer and AR-5.2's shared atomic-write helper), then
`pipeline.py`'s plain-mode wiring and `batching.py`'s `BatchStats`/`BatchRunner` extension
(Feature 3, independent of each other — different files, no shared state), then the two CLI
entry points (which depend on everything above), then docs/ADR. Solo, sequential, matching
spec 5's own precedent at a comparable size.

## Readiness

- **Checklist:** all 10 items `[x]` in the spec.
- **Open questions:** none — the spec's own critique round (`critique-consolidated-v-1.md`)
  already resolved the one genuinely blocking design question (the `usage_sink` keyword-argument
  approach would have broken every test that monkeypatches `Classifier.classify` wholesale;
  replaced with the `track_usage()` context-manager design, verified compatible).
- **External API/library verification:** re-confirmed live in this session, matching the
  spec's own citations:
  - `litellm.get_model_info('gpt-4o-mini')['input_cost_per_token'/'output_cost_per_token']` =
    `1.5e-07`/`6e-07`.
  - `litellm.completion_cost()` does not reliably resolve Cerebus workspace-slug ids (spec's
    Feature 2 "Who & why" — re-verified as still true against the installed litellm 1.101.0).
  - `_attempt_completion`'s only response-producing exit points are the primary attempt, the
    parameter-drop ladder's own recursive re-entry, and `_fallback_completion`'s return value —
    all three funnel through the single `return _extract_content(response)` line
    (`classifier.py:537`), which is the one hook point Task 1 needs.
- **Repository state:** clean except the same two pre-existing, unrelated items noted in
  spec 5/6's own plans (`M experiments/ag-news/analyze_run.ipynb`, untracked
  `data/example_queries_classified.csv`) — neither touched by this plan.
- **Test baseline:** re-run fresh for this plan — currently **34 failed, 342 passed**, every
  single failure a `botocore.exceptions.TokenRetrievalError`/`InvalidGrantException` from an
  expired local AWS SSO session attempting credential refresh (verified via full tracebacks on
  several failures) — the identical environmental issue already documented in spec 5's
  `implementation-summary.md` (there: 21 failures, self-resolved mid-session without any code
  change). Not a regression from any spec 5/6 work and not blocking for this plan; whoever
  implements Task 5/6's tests should expect this baseline to fluctuate with local SSO session
  state, independent of this feature.

## Code Impact

- **Modules:** new `src/query_classification/cost.py`. Modified: `classifier.py`,
  `batching.py`, `pipeline.py`, `cli.py`, `experiment.py`.
- **Database/schema:** none. Two brand-new artifacts (`cost_report.json`,
  `{stem}.cost_report.json`), each independently versioned (`schema_version: 1`);
  `run_config.json`'s own shape is untouched.
- **API/interfaces:** see Interfaces below.
- **Config/scripts:** `spec/ARCHITECTURE.md`, new `spec/7-cost-and-timing-logging/ADR.md`.
  `README.md` (a spec-deviation addition — see Spec Deviations).
- **Key files:** `classifier.py:428-460` (`__init__` — this plan's new lock/thread-local
  state), `:489-537` (`_attempt_completion` — the one usage-capture hook point), `:564-598`
  (`classify` — unchanged, per AR-1.1); `batching.py:229-296` (`BatchStats`), `:300-419`
  (`BatchRunner`, `run` in particular — `:380-419`); `pipeline.py:227-259` (the batch branch,
  untouched by this plan), `:292-311` (the plain-mode branch — this plan's edit site);
  `experiment.py:1065-1108`/`:1112`/`:1162`/`:1239`/`:1349`/`:1456` (the two `try` blocks, the
  induction/classification `_construct_classifiers` calls, the failure handler);
  `cli.py:385-655` (the one `try` block, classifier construction, `classify_csv` call site).

## Interfaces

Pinned here because Tasks 3–6 build against them.

```python
# classifier.py — additive
class Classifier:
    def __init__(self, ...):
        ...
        self._usage_lock = threading.Lock()        # new
        self._total_prompt_tokens = 0               # new
        self._total_completion_tokens = 0           # new

    def usage_totals(self) -> tuple[int, int]:      # new, public, thread-safe read
        """(prompt_tokens, completion_tokens) accumulated across every real
        completion call this instance has made, regardless of track_usage()."""

    def track_usage(self) -> "_UsageTrackingContext":  # new
        """Context manager: `with classifier.track_usage() as sink: classifier.classify(text)`
        then read `sink.usage -> tuple[int, int] | None`. Backed by a MODULE-level
        threading.local() stack (not per-instance) -- correct because only one
        track_usage() scope is ever active per thread at a time in this codebase's
        actual call sites (pipeline.py's plain-mode wrapper, BatchRunner's top-level
        call), and thread-local state is inherently per-thread, so no cross-instance
        state is needed. classify()'s own signature/behavior is completely unchanged;
        _attempt_completion's existing hook point (see below) simply also checks
        "is a track_usage() context active on this thread?" and feeds it if so."""

# classifier.py — internal, not part of the public surface. Task 1's tests exercise
# it directly, which is a NEW INV-7 exception this plan's own critique round caught:
# spec 6's ADR-001 named specific existing methods (_complete/_attempt_completion),
# not a blanket grant for every future private method -- Task 7's ADR explicitly
# adds `_capture_usage_from_response` (and `usage_totals`/`track_usage`, though
# those are public and need no exception) to INV-7's list, alongside (not
# replacing) spec 6's own entry.
def _capture_usage_from_response(self, response: Any) -> None:
    """Called once, at classifier.py:537, immediately before `_extract_content`.
    Reads `getattr(response, "usage", None)`; if both `.prompt_tokens`/
    `.completion_tokens` are present, adds them to (a) this instance's running
    total under `self._usage_lock`, and (b) the current thread's active
    track_usage() sink, if any. A response with no usable usage contributes
    nothing to either -- never an error."""

# cost.py — new module
@dataclass(frozen=True)
class CostPerToken:
    input: float | None
    output: float | None
    resolved_input_model: str | None    # NEW -- which candidate resolved the input side
    resolved_output_model: str | None   # NEW -- which candidate resolved the output side
    # FR-5.1's `pricing` block needs this provenance (mirroring spec 5's
    # resolved_input_model/resolved_output_model in run_config.json's `batch`
    # block) -- a rates-only type cannot populate that field at all, a gap this
    # plan's own critique caught.

class PricingCache:                                  # run-scoped, constructed once per run
    def resolve(self, model_id: str) -> CostPerToken: ...  # lock-protected dict cache,
                                                            # keyed on model_id, walks the
                                                            # candidate sequence only on a
                                                            # cache miss (FR-2.1); resolution
                                                            # itself happens under the lock
                                                            # (simple correctness over a
                                                            # single-flight optimization --
                                                            # a redundant walk on a rare
                                                            # concurrent cache miss costs a
                                                            # few local dict lookups, not a
                                                            # network call)

def compute_cost(prompt_tokens: int, completion_tokens: int, cost_per_token: CostPerToken
                  ) -> float | None: ...               # FR-2.2 -- None if either price unknown

def stats_from_samples(samples: list[float | None]) -> dict:
    """{"mean_usd": ..., "stddev_usd": ..., "count": int}. For REAL, independently-
    observed samples (plain mode's exact per-row costs; --batch's per-batch costs;
    --batch's per-row *shares*, which genuinely vary batch-to-batch): count=0 ->
    both None. count=1 -> mean=that value, stddev=None (sample stddev is
    mathematically undefined for a single real observation). Any None in `samples`
    poisons mean/stddev to None but does not change count. Sample (not population)
    standard deviation for count>=2. Do NOT use this for --critics/--models'
    uniform estimate (see uniform_estimate below) -- feeding a uniform,
    by-construction-identical list through this function would incorrectly report
    stddev=None at count=1, when FR-3.3 requires stddev=0 there (the value isn't
    undersampled, it's a defined estimate, even for a single row).

def uniform_estimate(total_cost: float | None, count: int) -> dict:
    """FR-3.3's --critics/--models case only: {"mean_usd": total_cost/count if
    total_cost is not None and count > 0 else None, "stddev_usd": 0.0 if
    mean_usd is not None else None, "count": count}. stddev is always exactly
    0.0 (never None), including at count == 1 -- every row is assigned the
    identical estimate by construction, so there is no variance to be
    undefined about, unlike stats_from_samples' real-sample case."""

def expand_batch_costs_to_query_samples(
    batch_costs: list[tuple[int, float | None]]
) -> list[float | None]:
    """Each (arity, cost) pair becomes `arity` copies of (cost/arity if cost is not
    None else None) in the returned list -- the per-row estimate FR-3.2 describes."""

class Timer:                                          # FR-4.1, NOT a contextmanager --
    def elapsed_seconds(self) -> float: ...            # a live method/property, readable
                                                        # at ANY point after construction
                                                        # (time.monotonic() - self._start),
                                                        # including from inside a broad
                                                        # except handler before sys.exit,
                                                        # not only after some `with` block
                                                        # closes. Constructed once at the
                                                        # start of the timed window
                                                        # (`Timer()` starts it immediately,
                                                        # no separate .start() call needed).

def write_json_atomic(path: Path, data: Any) -> None:  # AR-5.2, temp-file + os.replace,
                                                        # identical pattern to
                                                        # experiment.py's existing
                                                        # (private) _write_json_atomic

# batching.py -- additive
class BatchStats:
    def record_cost(self, arity: int, cost: float | None) -> None: ...  # new
    def snapshot(self) -> dict[str, ...]:
        ...  # existing fields unchanged, PLUS:
        # "batch_costs": list[tuple[int, float | None]]  -- a fresh copy each call,
        # one (arity, cost) entry per TOP-LEVEL formed batch's first attempt only
        # (never a retry-at-original-size or bisection call's cost)

class BatchRunner:
    def __init__(self, ..., cost_fn: Callable[[int, int], float | None] | None = None
                 ) -> None: ...
        # new, keyword, default None (existing direct-construction tests that don't
        # pass it simply never record a cost -- record_cost(arity, None) is skipped
        # entirely rather than raising). Deliberately a plain CALLABLE
        # (prompt_tokens, completion_tokens) -> cost | None, NOT a `cost.CostPerToken`
        # value and NOT a call to `cost.compute_cost` from inside batching.py --
        # this plan's own critique round caught that the original design would have
        # required `batching.py` to import `cost.py`, a FOURTH inbound edge AR-5.1
        # never declared (only pipeline.py/cli.py/experiment.py do) and INV-1 does
        # not authorize. Injecting a closure keeps `batching.py`'s import list
        # exactly as it is today (`categories.py`+`classifier.py` only). The caller
        # (experiment.py/cli.py, which DOES import `cost.py`) constructs it as
        # `lambda p, c: cost.compute_cost(p, c, pricing_cache.resolve(model_id))` --
        # one resolution suffices for the whole run, since `--batch` requires
        # exactly one resolved model (spec 5's own validation), so every arity in
        # the cache shares the identical price.
    def run(self, texts: list[str], *, _top_level: bool = True
            ) -> list[dict[str, Any] | Exception]: ...
        # UNCHANGED behavior/return; new: when _top_level and self._cost_fn is not
        # None, the first `_attempt()` call (only that one -- not the
        # retry-at-original-size loop's subsequent attempts, not the recursive
        # bisection calls, both of which pass _top_level=False / stay within the
        # same top-level call respectively) is wrapped in track_usage(), its cost
        # computed via `self._cost_fn(*sink.usage)`, and recorded via
        # `self.stats.record_cost(arity, cost)`.
    def total_usage(self) -> tuple[int, int]: ...
        # new, public: sums usage_totals() across every classifier in the private
        # arity-keyed cache, so FR-3.4's total-cost computation never reaches into
        # `_cache` directly

# pipeline.py -- additive
def classify_csv(..., cost_collector: "QueryCostCollector | None" = None,
                  cost_per_token: "cost.CostPerToken | None" = None) -> Path: ...
    # both new, keyword-only, default None -- existing behavior unchanged when
    # omitted (every existing call site in tests/test_debate.py/test_multi_model.py
    # passes neither). `cost_collector`, when given, has `set_rows_attempted(total)`
    # called unconditionally in EVERY mode (plain/critics/have_models/have_batch) --
    # that one call is the only thing critics/have_models/have_batch modes do with
    # it. Only the plain-mode branch additionally uses it for real per-row cost
    # samples via `.record(cost)` (per-call cost data via BatchStats / FR-1.4
    # aggregate reads instead for the other modes, per FR-3.2/FR-3.3). When
    # cost_collector is given AND the plain-mode branch is taken, that branch's
    # submission wraps its existing `classifier.classify(text)` call
    # (unchanged arguments) in `track_usage()` from INSIDE the submitted callable
    # (on the worker thread, mirroring INV-6: only the read-only LLM call happens
    # there), returns (result, usage) instead of just result, and the MAIN thread
    # -- in the same new conditional branch inside the existing result-application
    # loop, not a separate pass -- computes that row's cost via
    # `cost.compute_cost(*usage, cost_per_token)` and calls
    # `cost_collector.record(cost)`.

# cost.py -- additive
class QueryCostCollector:                              # new, thread-safe (lock + list),
    def record(self, cost: float | None) -> None: ...   # mirrors BatchStats' pattern;
                                                          # plain mode only (FR-3.1)
    def samples(self) -> list[float | None]: ...         # fresh copy each call
    def set_rows_attempted(self, n: int) -> None: ...    # NEW -- see note below
    def rows_attempted(self) -> int | None: ...          # NEW
```

`classify_csv` calls `cost_collector.set_rows_attempted(total)` (where `total = len(work_idx)`,
already an existing local variable used for `tqdm`'s progress bar) **once, unconditionally,
right after `total` is computed and before branching into `critics`/`have_models`/`have_batch`/
plain** -- regardless of which mode the run takes. This is the fix for a gap this plan's own
codebase check caught: `classify_csv` returns only a `Path` today, and while `experiment.py`
could get away with assuming its own already-loaded `test_df`'s row count equals
`classify_csv`'s internal `work_idx` count (true only because it always passes
`restore=False, limit=None`), `cli.py` cannot make that assumption at all -- `--restore`/
`--limit` are real, user-facing flags there, and `cli.py` has no equivalent pre-filtered
dataframe of its own to count. Both entry points therefore always construct a `QueryCostCollector`
(even under `--batch`/`--critics`/`--models`, where `.samples()` stays empty but
`.rows_attempted()` is still populated) and read `.rows_attempted()` back after the call, rather
than each entry point deriving the count a different, mode-fragile way.

`classify_csv`'s new parameter is keyword-with-default, so none of the ~24 existing call sites
in `tests/test_debate.py`/`tests/test_multi_model.py` change, matching spec 5's own precedent
for `batch_runner`.

## Project Constraints

- **INV-1** (one-directional acyclic imports): `cost.py` is a new node importing only
  `categories.py`/`classifier.py` (to read `Classifier.usage_totals()`). `pipeline.py`,
  `cli.py`, and `experiment.py` all import it — three new inbound edges, the same shape as
  `batching.py`'s own from spec 5. ADR amends INV-1 and the Module Boundary Map (adds a
  `cost.py` row, edits the three importer rows).
- **INV-4**: every broad `except Exception` needs `# noqa: BLE001` plus a justification
  comment. Applies to `PricingCache.resolve`'s per-candidate catch (mirroring
  `batching.resolve_token_budgets`'s existing one) and to `write_json_atomic`'s cleanup-on-
  failure path (mirroring `experiment.py`'s existing `_write_json_atomic`).
- **INV-6**: worker threads make only the read-only LLM call; all stateful recording that
  isn't itself thread-safe-by-design stays off the critical "only the call" path. `pipeline.py`'s
  plain-mode wrapper keeps this by returning `(result, usage)` from the worker-thread callable
  and recording into `cost_collector` on the main thread, in the same place existing per-row
  outcomes are already applied. `BatchStats.record_cost` is itself lock-protected (like every
  other `BatchStats` method), so `BatchRunner.run` calling it directly from whichever thread
  is running `run()` is unchanged from spec 5's existing pattern (`record_call` already works
  this way).
- **INV-7**: `cost.py`'s new functions/classes are not part of the public API (not re-exported
  by `__init__.py`) and `tests/test_cost.py` needs to import them directly, exactly the same
  shape of exception spec 5's AR-5.1 requested for `batching.py`. The spec's own Feature 6 has
  no AR making this explicit (unlike spec 5's dedicated AR-5.1) — this plan's ADR requests it
  anyway, since FR-6.1's testing requirement is unsatisfiable without it. See Spec Deviations.
- **Testing policy** (`AGENTS.md`): no live network/provider calls; LLM calls faked. Tests ship
  in the same task as the code they cover — no deferred test task.
- **PEP 604 unions + `from __future__ import annotations`** in every new/modified module under
  `src/query_classification/`.
- **Verification command**: `.venv/bin/python -m pytest`. No lint/typecheck/CI gate exists.

## Implementation Strategy

- **Size:** large (7 tasks).
- **Execution mode:** solo, sequential.
- **Parallelizable work:** none across tasks. Task 3 (`pipeline.py`) and Task 4 (`batching.py`)
  touch disjoint production files, but both add tests to the **same** `tests/test_batching.py`
  file (per spec 5's own precedent of collecting `classify_csv`-level and `BatchRunner`-level
  tests there together) — a real, shared-file dependency, not a false-parallel opportunity.
  Solo sequential execution avoids that conflict entirely and matches this project's established
  convention (spec 5/6) regardless.
- **Sequential blockers:** Task 2 needs Task 1's `usage_totals()`/`track_usage()` to exist to
  build/test `PricingCache`/`compute_cost` against real `Classifier` instances. Tasks 3 and 4
  each need Task 1 (usage capture) and Task 2 (`cost.py`'s primitives). Task 5 needs Tasks 3
  and 4 (`cost_collector`, `BatchStats.record_cost`/`BatchRunner.total_usage`). Task 6 mirrors
  Task 5. Task 7 (docs/ADR) needs the real, shipped shape from Tasks 1–6.
- **No expected-red checkpoints.** No task changes a signature an existing caller depends on:
  `classify()`'s own signature never changes (the whole point of `track_usage()`);
  `classify_csv`'s new `cost_collector` parameter and `BatchRunner.run`'s new `_top_level`
  parameter are both keyword-with-defaults.

## Implementation Tasks

### Task 1: `Classifier` usage capture and `track_usage()`
**Goal:** every real completion call's token usage is captured at the one point a response
object exists, exposed both as a per-instance running total and via an opt-in context manager
that never changes `classify()`'s own call signature.
**Files:** `src/query_classification/classifier.py`, `tests/test_cost.py` (new)
**Dependencies:** None
**Do:**
- Add `import threading` and a module-level `_thread_local = threading.local()`.
- In `Classifier.__init__`: add `self._usage_lock = threading.Lock()`,
  `self._total_prompt_tokens = 0`, `self._total_completion_tokens = 0`.
- Add `_capture_usage_from_response(self, response)`: reads `getattr(response, "usage", None)`;
  if both `prompt_tokens`/`completion_tokens` are present and not `None`, updates the instance
  running total under `self._usage_lock`, then checks `getattr(_thread_local, "sink", None)` —
  if set, adds to it too (no lock needed there; a thread-local sink is only ever touched by its
  own thread).
- Call `self._capture_usage_from_response(response)` at exactly one point:
  `classifier.py:537`, immediately before `return _extract_content(response)` in
  `_attempt_completion`. This single point is reached by the primary structured attempt, the
  parameter-drop ladder's own recursive re-entry (which independently reaches this same line
  for its own `response`), and `_fallback_completion`'s successful return (which flows back
  into this same `response` variable) — covering every response-producing path with one hook.
- Add `usage_totals(self) -> tuple[int, int]` (lock-protected read) and `track_usage(self)`
  (a `@contextmanager` method): creates a small sink object, pushes it onto
  `_thread_local.sink` (saving/restoring any previous value, though nesting isn't expected in
  practice), yields it, restores on exit.
**Verify:** `.venv/bin/python -m pytest -q` (full 376-test baseline stays green — additive
only); new tests in `tests/test_cost.py` cover: a fake response with
`usage.prompt_tokens=100, completion_tokens=50` is captured exactly; a fake response with no
`usage` attribute contributes nothing (no error); a `classify()` call whose first attempt
raises `UnsupportedParamsError` (no response, no usage) then succeeds on the parameter-dropped
retry reports only the second attempt's usage; a `classify()` call that exhausts `max_retries`
across several response-producing-but-validation-failing attempts sums all of them; two
threads each running `with classifier.track_usage() as sink: classifier.classify(text)`
concurrently on one shared `Classifier` instance each see only their own call's usage in
`sink.usage`; `classify()` called with no active `track_usage()` context behaves identically to
today (return value, exceptions, call signature); a monkeypatched `Classifier.classify`
(matching `tests/test_experiment.py`'s existing `fake_classify` shape) wrapped in
`track_usage()` observes `sink.usage is None` rather than raising.
**Covers:** FR-1.1, FR-1.2, FR-1.3, FR-1.4, AR-1.1, AR-1.2, FR-6.1 (partial)
**Notes:** This is the one task where a subtle bug would silently corrupt every other task's
numbers, so its own test list is intentionally the most exhaustive of the seven tasks — every
other task can trust `usage_totals()`/`track_usage()` are correct once this lands.

### Task 2: `cost.py` — pricing, computation, stats, timer, atomic write
**Goal:** the new module owning every pure/reusable primitive the two entry points need,
independent of `--batch`/`--critics`/`--models`.
**Files:** `src/query_classification/cost.py` (new), `tests/test_cost.py`
**Dependencies:** Task 1
**Do:**
- `CostPerToken` (frozen dataclass), `PricingCache` (lock-protected `dict[str, CostPerToken]`
  cache; `resolve(model_id)` walks the identical candidate sequence as
  `batching._candidate_ids` — duplicated, not imported, per FR-2.3 — catching broad
  `Exception` per candidate with the INV-4 comment, exactly mirroring
  `batching.resolve_token_budgets`'s existing walk).
- `compute_cost(prompt_tokens, completion_tokens, cost_per_token)`: the FR-2.2 arithmetic;
  `None` if either side of `cost_per_token` is `None`.
- `stats_from_samples(samples)`: count/mean/(sample, ddof=1)stddev over the list, `None`s for
  count 0/1 and for any aggregate touched by a `None` sample, per the Interfaces docstring.
  The non-finite-to-`None` guarantee is not this function's alone to keep: `compute_cost`,
  `uniform_estimate`, and `stats_from_samples` each convert a non-finite result (`NaN`/`±Infinity`
  — realistically only reachable via a corrupted/zero pricing table, not normal arithmetic) to
  `None` at the point they produce it, and `write_json_atomic` itself calls
  `json.dumps(data, allow_nan=False)` as the final defense-in-depth boundary, so a non-finite
  value can never reach the on-disk JSON regardless of which upstream function let one slip
  through.
- `expand_batch_costs_to_query_samples(batch_costs)`: the per-row-share expansion FR-3.2
  describes.
- `Timer` (plain object, not a contextmanager): records `time.monotonic()` at construction;
  `elapsed_seconds()` is a live read (`time.monotonic() - self._start`) callable at any later
  point, including from inside a broad `except` handler before `sys.exit` — never `time.time()`.
  A contextmanager would only expose the final elapsed time after its `with` block closes,
  which is too late for `experiment.py`'s failure handler (Task 5), which needs to read
  "elapsed so far" *before* deciding to exit.
- `write_json_atomic(path, data)`: temp file (same directory, `tempfile.mkstemp`) +
  `Path.replace`, identical pattern to `experiment.py`'s existing private
  `_write_json_atomic` — `experiment.py`'s own callers switch to this shared one in Task 5 (its
  private copy is removed, not left duplicated). Serialize with
  `json.dumps(data, indent=2, allow_nan=False)` (`_write_json_atomic`'s existing call has no
  `allow_nan` argument today, i.e. it currently defaults to permissive `True` — this is a
  behavior tightening for every caller, not just `cost_report.json`'s own writer).
**Verify:** `.venv/bin/python -m pytest -q`; new tests cover: the three FR-1.2-style
chunk-down cases (workspace slug with/without `openai/` prefix, chunk-down to a real model,
total failure) against a **monkeypatched** `litellm.get_model_info` (never the live table, per
FR-6.1's brittleness note) produce the expected `CostPerToken` or `(None, None)`; resolving the
same model id twice through one `PricingCache` performs the underlying walk only once (assert
via a call-counting monkeypatch); `compute_cost` arithmetic and its `None`-propagation;
`stats_from_samples` for count 0, 1, and N>=2, including a sample list containing one `None`;
`uniform_estimate` at count 0, 1, and N>=2 all report `stddev_usd == 0.0` (never `None`) whenever
`mean_usd` is not `None`, distinguishing it from `stats_from_samples`' count==1 behavior;
`expand_batch_costs_to_query_samples` on a mixed list including one `(arity, None)` entry;
`Timer` reports at least a known injected `time.sleep` duration; `write_json_atomic` leaves no
partial file on a simulated mid-write failure; at most one smoke test asserts the real
installed `litellm.get_model_info('gpt-4o-mini')` table has numeric, non-`None`
`input_cost_per_token`/`output_cost_per_token` (shape only, no pinned value).
**Covers:** FR-2.1, FR-2.2, FR-2.3, AR-2.1, FR-4.1 (timer), AR-5.2 (atomic write), FR-6.1 (partial)
**Notes:** `cost.py` imports only `categories.py`/`classifier.py` — verify via its own import
block, matching AR-2.1's Verify line. `QueryCostCollector` (used by Task 3) also lands here
since it's a small, reusable, `cost.py`-owned primitive, mirroring `BatchStats`'s own
lock-protected-list pattern.

### Task 3: `pipeline.py` — rows-attempted count in every mode, exact per-row cost in plain mode
**Goal:** an opt-in `cost_collector` parameter that (a), in every mode, reports how many rows
this invocation actually attempted — the piece `cli.py` has no other way to learn, since
`--restore`/`--limit` filtering happens inside `classify_csv` itself — and (b), in plain mode
specifically, records each row's exact real cost, all without changing `classify()`'s call or
any existing caller's behavior.
**Files:** `src/query_classification/pipeline.py`, `tests/test_batching.py` (existing
`classify_csv`-level tests live there per spec 5's precedent; new plain-mode cost tests join
them)
**Dependencies:** Tasks 1, 2
**Do:**
- Add `cost_collector: QueryCostCollector | None = None` and `cost_per_token:
  "cost.CostPerToken | None" = None` to `classify_csv`'s signature (keyword-only-by-convention,
  matching `batch_runner`'s own placement; `cost_per_token` resolved once by the caller before
  the run, the same precomputed-value pattern as `BatchRunner`'s new constructor parameter in
  Task 4).
- Right after `total = len(work_idx)` is computed (the existing local variable already feeding
  `tqdm(total=total, ...)`), and **before** branching into `critics`/`have_models`/`have_batch`/
  plain: if `cost_collector is not None`, call `cost_collector.set_rows_attempted(total)`
  unconditionally — this is the *only* thing critics/`have_models`/`have_batch` modes ever do
  with `cost_collector`; it costs nothing extra since `total` is already computed at this point
  regardless.
- **The shared `for future in as_completed(future_to_idx): ... classification: dict[str, Any]
  = future.result()` loop (`pipeline.py:297-311`) currently assumes every branch's future
  resolves to a plain `dict` — `critics`/`have_models`/plain-without-tracking all still must.**
  Only the plain-mode branch, and only when `cost_collector is not None`, changes what its
  *own* submitted callable returns; every other branch (`critics`, `have_models`, and plain
  mode when `cost_collector is None`) is untouched. Concretely: in the plain-mode branch, when
  `cost_collector is not None`, submit a small wrapper callable instead of `classifier.classify`
  directly:
  ```python
  def _classify_tracked(classifier, text):
      with classifier.track_usage() as sink:
          try:
              result = classifier.classify(text)
          except Exception as exc:
              exc._tracked_usage = sink.usage  # may be None (no billable attempt at all)
              raise
          return result, sink.usage
  ```
  entirely on the worker thread (preserving INV-6: only the read-only call happens there). This
  is this plan's fix for a gap its own critique round caught: `track_usage()` is explicitly
  designed (FR-1.2) to retain usage from response-producing attempts even when `classify()`
  ultimately raises, and FR-5.1 defines `query_cost.count` as rows *attempted*, not merely
  succeeded — a design that silently dropped a failed row's real, billable spend from the
  report would both discard real cost data and undercount. The shared result loop's existing
  `except Exception:` clause becomes `except Exception as e:` so it can read
  `getattr(e, "_tracked_usage", None)`; a new conditional branch, gated on `cost_collector is
  not None` (a single run-scoped flag known before the loop starts, not per-future), unpacks
  the success tuple or reads the exception attribute; every other combination (`critics`,
  `have_models`, plain mode with `cost_collector is None`) keeps today's exact code path.
- In that same new conditional branch: if usage (from either the success tuple or the caught
  exception's `_tracked_usage`) is not `None`, compute the row's cost via
  `cost.compute_cost(*usage, cost_per_token)` and call `cost_collector.record(cost)` — in the
  same place `classified`/`failed`/`completed` are already updated, not a separate pass. If
  usage genuinely is `None` (the row never produced even one billable response — e.g. an
  immediate credential failure before any provider round-trip), **do not call `record` at
  all** for that row: a `None` cost sample would poison `query_cost.mean_usd` for the *entire*
  run to `None` via `stats_from_samples`' pricing-unresolved poisoning rule (FR-2.2), which is
  correct when pricing itself is unknown for every call but wrong for one isolated row that
  never touched the API at all — that row is already visible via the existing `failed` counter,
  and excluding it from the cost *sample set* (rather than recording a poisoning `None`) is what
  keeps the rest of the run's genuine cost data meaningful. `query_cost.count`, in practice,
  therefore reflects rows with at least one billable attempt — which equals `len(work_idx)`
  except in the rare case of a failure that never reaches the provider at all.
**Verify:** `.venv/bin/python -m pytest -q`; a faked-LLM plain run over 10 rows with distinct,
known usage per row, given a `cost_collector`, ends with `cost_collector.samples()` containing
exactly those 10 real per-row costs (order not required to match row order, since aggregate
stats are order-independent); the identical run with `cost_collector=None` produces
byte-identical output CSV/flush behavior to a pre-Task-3 baseline (no behavior change for the
default case); a row whose `classify()` call raises *after* a response-producing attempt (e.g.
a validation failure that exhausts retries) still contributes that attempt's real cost to
`cost_collector.samples()`; a row that fails with zero billable attempts (a fake raising before
ever returning a response) contributes nothing to `.samples()` and does not poison the other
9 rows' `mean_usd`/`stddev_usd` to `None`; `cost_collector.set_rows_attempted(total)` is called
exactly once for a `--critics` run and once for a `--models` run (not only plain/`--batch`),
with the value matching `len(work_idx)` after `--restore`/`--limit` filtering in each case.
**Covers:** FR-1.2 (consuming it correctly on the failure path), FR-3.1, FR-3.3 (partial —
`rows_attempted` plumbing), FR-6.1 (partial)

### Task 4: `batching.py` — per-batch cost via `BatchStats`/`BatchRunner`
**Goal:** the cost of only each top-level formed batch's first attempt, recorded distinctly
from retry-at-original-size/bisection calls, plus a way to sum a run's whole arity-keyed
cache's usage for the total-cost figure.
**Files:** `src/query_classification/batching.py`, `tests/test_batching.py`
**Dependencies:** Tasks 1, 2
**Do:**
- `BatchStats`: add `self._batch_costs: list[tuple[int, float | None]] = []` alongside the
  existing counters (empty list before any top-level batch completes — the natural "no data
  yet" representation for a list-typed field, distinct from the scalar `min_arity`/etc. fields'
  own `None`-when-empty convention, since a list can be safely iterated/measured with `len()`
  either way); `record_cost(arity, cost)` appends under the existing lock; `snapshot()` gains
  one new key, `"batch_costs"` (a fresh copy of the list), alongside the unchanged existing keys.
- `BatchRunner.__init__`: add `cost_fn: Callable[[int, int], float | None] | None = None`
  (keyword, default `None` — every existing direct-construction test/call site that doesn't
  pass it is unaffected; `record_cost` is simply skipped when it's `None`). A plain callable,
  not a `cost.CostPerToken`/`cost.py` import — see the Interfaces section's note on why this
  avoids a fourth, undeclared `batching.py -> cost.py` edge. The caller (`experiment.py`/
  `cli.py`) builds it as a closure over one `PricingCache.resolve(model_id)` result, since
  `--batch` requires exactly one resolved model per run (spec 5's own validation), so every
  cached arity shares the identical price.
- `BatchRunner.run`: add the keyword-only `_top_level: bool = True` parameter. The two
  recursive call sites (bisection's `self.run(texts[:mid])`/`self.run(texts[mid:])`) pass
  `_top_level=False`. Only when `_top_level` is `True` **and** `self._cost_fn` is not `None`,
  wrap the very first `_attempt()` call (line 391 in the current file — the one before the
  retry-at-original-size loop and before any bisection decision) in `classifier.track_usage()`;
  compute its cost via `self._cost_fn(*sink.usage)` and call `self.stats.record_cost(arity,
  cost)` — regardless of whether that first attempt ultimately succeeds or fails (the money was
  spent either way; `sink.usage` is still populated from a failed-but-response-producing
  attempt, per FR-1.2). The retry-at-original-size loop's subsequent attempts and any bisection
  recursion's own calls are never wrapped or recorded here.
- `BatchRunner`: add `total_usage(self) -> tuple[int, int]`, summing `usage_totals()` across
  every classifier currently in `self._cache` (a public method — no caller reaches into
  `_cache` directly).
**Verify:** `.venv/bin/python -m pytest -q`; a `--batch 5`-style run over 20 rows (4 formed
batches) whose 4 top-level batches have known, distinct costs produces
`BatchStats.snapshot()["batch_costs"]` with exactly those 4 `(arity, cost)` pairs; a batch that
gets bisected due to failure still has only its first attempt's cost in `batch_costs` — the
bisection's own additional calls' costs do not appear there (though they do count toward
`BatchRunner.total_usage()`'s aggregate); `BatchRunner.total_usage()` after a run with 3
distinct arities used equals the sum of those 3 cached classifiers' own `usage_totals()`; a
direct call to `run(texts)` with no `_top_level` argument behaves exactly as it does today
(existing spec-5 tests for `BatchRunner.run` pass unmodified).
**Covers:** FR-3.2, FR-3.4 (partial — `BatchRunner.total_usage()`), AR-3.1, FR-6.1 (partial)

### Task 5: `experiment.py` — timing, report, induction-cost preservation, stdout summary
**Goal:** `experiment.py` times the run, preserves the induction classifier's cost before it's
overwritten, assembles and writes `cost_report.json`, and prints the one-line summary.
**Files:** `src/query_classification/experiment.py`, `tests/test_experiment.py`
**Dependencies:** Tasks 2, 3, 4
**Do:**
- Right after the early validation `try` block succeeds, at `stage = "starting"`
  (`experiment.py:1112`, before `config` is first built), initialize **every** local variable
  the report-assembly step (below) or the broad failure handler at `:1456` might reference,
  so none of them can ever raise `NameError`/`UnboundLocalError` regardless of which phase a
  failure occurs in — this plan's own critique round caught that initializing only
  `induction_classifier` was insufficient, since a failure early in the main `try` block (before
  `_construct_classifiers` is ever called) would leave `classifiers`, `cost_collector`, and the
  per-run cost/timing helpers unbound too:
  - `pricing_cache = cost.PricingCache()`
  - `timer = cost.Timer()`
  - `induction_classifier = None`
  - `classifiers: dict[str, Any] = {}` (empty dict; the *existing* code that populates and
    reads it via direct indexing, e.g. `classifiers["induction"]`, is unaffected and unchanged
    — this initialization only matters for *this task's own new* failure-handler report code,
    which must read from `classifiers` via `.get(...)` specifically so a failure before
    `_construct_classifiers` ever ran degrades gracefully rather than raising `KeyError`)
  - `query_cost_collector = cost.QueryCostCollector()` (constructed once, unconditionally,
    since it's cheap and always safe to pass — see the next bullet)
- **Immediately after `run_induction(...)` returns** (still inside the `if will_induce:`
  branch, before the later `classifiers, models = _construct_classifiers(...)` call at
  `:1349` reassigns `classifiers`), capture `induction_classifier = classifiers["induction"]`
  into its own local variable — this is the one fix that prevents the reference from being
  silently lost to the reassignment.
- Pass the already-constructed `query_cost_collector` as `cost_collector=...` at the
  `classify_csv` call site **regardless of mode** (plain/`--critics`/`--models`/`--batch`) — not
  only in plain mode. Every mode benefits from `.rows_attempted()` (Task 3) for FR-3.3's
  denominator; only plain mode also gets real per-row samples from `.samples()`.
- After `classify_csv` returns (or in the failure handler, using whatever partial state
  exists — every local it needs was already initialized above, so this is always safe), assemble
  the report. **`total_cost_usd`/`induction_cost_usd`: each classifier instance is priced
  individually with its own `model_id`'s resolved rate before summing dollar amounts — never
  by summing raw token counts across classifiers and applying one rate.** This matters
  concretely for `--critics` (the sampling classifier, `--critic-model`, and
  `--reconciler-model` can be three distinct ids) and `--models` (each entry in `multi_model`
  can be a distinct id): `induction_cost_usd = cost.compute_cost(*induction_classifier.usage_totals(),
  pricing_cache.resolve(induction_classifier.model_id))`; the classification-phase total is the
  `None`-poisoning **sum of each individual classifier's own `compute_cost(...)` result**
  (the plain classifier; every classifier in the `critics` dict plus every classifier in the
  `reconcilers` dict, each priced by its own `.model_id`; every classifier in the `multi_model`
  dict, each priced by its own `.model_id`; or `batch_runner.total_usage()` priced by the
  single model `--batch` requires — whichever `models`/mode was active); `total_cost_usd` is
  `induction_cost_usd` plus that classification-phase sum. `query_cost` via
  **`cost.stats_from_samples`** fed
  either the plain-mode collector's real samples, or (under `--batch`)
  `cost.expand_batch_costs_to_query_samples(classifiers["batch_stats"].snapshot()["batch_costs"])`
  (both genuinely-varying real/derived samples), **or, under `--critics`/`--models`, via
  `cost.uniform_estimate`** (not `stats_from_samples` — a uniform, by-construction-identical
  estimate needs `stddev_usd == 0.0` even at a row count of 1, which `stats_from_samples`'
  real-sample semantics would incorrectly report as `None`) with the classification-phase total
  cost and `cost_collector.rows_attempted()` (Task 3 — not a locally-recomputed row count);
  `batch_cost` from
  `cost.stats_from_samples` over `batch_stats.snapshot()["batch_costs"]`'s cost values directly,
  or `None` entirely when `--batch` wasn't used; `pricing` is the single `CostPerToken`'s
  `resolved_input_model`/`resolved_output_model` when every classifier priced above resolved to
  the *same* `model_id` (the common case: plain, `--batch`, or `--critics`/`--models` all using
  one model), and `None` entirely when two or more distinct `model_id`s were priced this run
  (Spec Deviation row 4 — a single pair can't truthfully represent several distinct
  resolutions). Write via `cost.write_json_atomic` to `run_dir / "cost_report.json"`
  (added to `_ARTIFACT_FILENAMES`), only reachable from inside the *main* `try` block (an early
  validation failure produces no report, matching `run_config.json`'s own behavior) — including
  from the broad failure handler at `:1456`, using whatever `induction_classifier`/`classifiers`
  state exists at that point (may be partially constructed; the report should include whatever
  usage was actually captured, never crash on a partially-built state).
- Print the one-line stdout summary after a successful write.
- Remove `experiment.py`'s own private `_write_json_atomic` in favor of `cost.write_json_atomic`
  (used by every existing `run_config.json`/`categories.json`/`sampled_examples.json` write site
  too — one helper, not two). This is a behavior-preserving swap for those three pre-existing
  files (same temp-file-plus-replace shape, `indent=2` unchanged) plus the new `allow_nan=False`
  tightening (Task 2) — existing tests asserting `run_config.json`/`categories.json`'s exact
  content (e.g. `tests/test_experiment.py`'s config-shape assertions) must still pass unmodified,
  which is the regression check for this swap; no test targets `_write_json_atomic` by name, so
  nothing needs renaming at the test level, only re-running.
**Verify:** `.venv/bin/python -m pytest -q`; a `run` invocation (induce + classify) produces a
`cost_report.json` with a nonzero `induction_cost_usd` distinct from the classification-phase
portion of `total_cost_usd`; an `induce`-only invocation has `query_cost.count == 0`, both
`query_cost` stats `None`, `batch_cost` `None`; a `--batch` invocation has `batch_cost`
populated with `estimated: false` and `query_cost` with `estimated: true`; a `--critics`
invocation's `query_cost.mean_usd` equals total classification-phase cost divided by rows
attempted, `stddev_usd == 0`, both marked `estimated: true`; a run with an unresolvable model
has every monetary field `null` while `query_cost.count`/`estimated`/`wall_clock_seconds`
remain populated; a run that fails inside the main `try` block still writes a
`cost_report.json` with a real (non-zero) `wall_clock_seconds`; a run that fails during early
validation writes no `cost_report.json` at all; stdout contains exactly one
`Cost: ... | Time: ...`-shaped line per successful run.
**Covers:** FR-3.3, FR-3.4 (remaining), FR-4.1 (experiment.py half), FR-5.1, FR-5.3
(experiment.py half), AR-5.1 (experiment.py's import edge), FR-6.1 (partial)

### Task 6: `cli.py` — timing, report, stdout summary
**Goal:** the same behavior as Task 5, mirroring `experiment.py`'s wiring for `cli.py`'s
simpler, single-`try`-block structure and success-only report semantics.
**Files:** `src/query_classification/cli.py`, `tests/test_cerebus.py` (existing CLI-level
tests live there per spec 5's precedent)
**Dependencies:** Tasks 2, 3, 4
**Do:**
- Construct one `cost.PricingCache()` and start a `cost.Timer()` right after `categories =
  load_categories(args.categories)` (`cli.py:467`) — **not** at the top of `main()`'s `try`
  block (`:385`), which this plan's own critique round caught as too early: everything between
  `:385` and `:467` is still argument/resource validation (bundled-resource checks,
  `--sampling-temperature` checks, `--model`/`--models` resolution, the entire `--batch`
  validation-and-token-budget-resolution block), matching FR-4.1's "just after
  argument/resource validation" boundary, not the point classifier construction/classification
  (the actual timed work) begins. `cli.py` has no single clean boundary the way
  `experiment.py`'s `stage = "starting"` is; `load_categories`'s call site is the last point
  before classifier construction begins and is used consistently as the anchor.
- Construct one `cost.QueryCostCollector()` and pass it as `cost_collector=...` at the
  `classify_csv` call site **regardless of mode** (mirroring Task 5's own fix) — `cli.py` has
  no pre-filtered dataframe of its own to count rows from (unlike `experiment.py`'s `test_df`),
  since `--restore`/`--limit` are real, user-facing flags here that `classify_csv` applies
  internally; `.rows_attempted()` (Task 3) is the *only* way `cli.py` can learn this count at
  all, in every mode, not only plain.
- After `classify_csv` returns successfully (never on failure — `cli.py` has no existing
  "artifact even on failure" convention to extend, per FR-5.2), assemble the same report shape
  Task 5 builds (`induction_cost_usd` always `null` here) from `cli.py`'s own local classifier
  variables (`classifier`, `critic_classifiers`, `reconciler_classifiers`, `models_classifiers`,
  `batch_runner` — distinct names from `experiment.py`'s `classifiers["..."]` dict, per
  `cli.py:524-526`/`:593-594`; unlike Task 5, no `NameError` risk here since the report is only
  ever assembled after `classify_csv` has already returned, meaning every classifier that would
  be used was already constructed successfully), using `cost_collector.rows_attempted()` for
  FR-3.3's `--critics`/`--models` denominator, derive the sibling path via
  `Path(csv_path).with_suffix(".cost_report.json")` where `csv_path = Path(args.output) if
  args.output else Path(args.input)`, write via `cost.write_json_atomic`, and print the
  stdout summary line.
**Verify:** `.venv/bin/python -m pytest -q`; `python classify.py --input in.csv --output
out.csv` produces `out.cost_report.json` alongside `out.csv`; omitting `--output` produces
`in.cost_report.json` alongside the overwritten `in.csv`; `--output out.csv.gz` produces
`out.cost_report.json` (only the final suffix replaced); a run that raises before
`classify_csv` returns writes no cost report and prints no summary line; a `--batch`-configured
run's report has `batch_cost.estimated == false`/`query_cost.estimated == true`, matching
Task 5's `experiment.py` behavior; a Cerebus-configured run's resolved pricing reflects the
same gateway-routed model id the classifier itself used.
**Covers:** FR-3.3 (cli.py's own critics/models paths), FR-4.1 (cli.py half), FR-5.2, FR-5.3
(cli.py half), AR-5.1 (cli.py's import edge), FR-6.1 (remaining)
**Notes:** `cli.py` and `experiment.py` deliberately share no code (INV-1) — the report-shape
assembly is implemented twice, following the same duplication precedent
`_resolve_classifier_models`/`_build_classifier_dict` already established. Implement after
Task 5 so it mirrors a known-good implementation.

### Task 7: Docs, ARCHITECTURE, ADR
**Goal:** the feature's invariant amendments are recorded, and the new artifact is documented.
**Files:** `spec/ARCHITECTURE.md`, `spec/7-cost-and-timing-logging/ADR.md` (new), `README.md`
**Dependencies:** Tasks 5, 6
**Do:**
- `spec/ARCHITECTURE.md`: add `cost.py` to the mermaid graph, the Module Boundary Map (new row;
  edit `pipeline.py`/`cli.py`/`experiment.py` rows' "May import from" lists — **not**
  `batching.py`, which gains no new import per the Interfaces section's `cost_fn`-callable
  design), and INV-1's rule/evidence text (three new inbound edges, mirroring `batching.py`'s
  own entry). Add `cost`'s new functions/classes, **and** `classifier._capture_usage_from_response`
  specifically (a new private name spec 6's own ADR-001 does not already cover — it named
  specific existing methods, not a blanket grant), to INV-7's `tests/` exception list and the
  `tests/` Module Boundary Map row, alongside (not replacing) every prior spec's own provenance
  note.
- `ADR.md`: one record amending INV-1 (new module + three inbound edges) and INV-7 (new
  test-import names for `cost.py`, plus `_capture_usage_from_response`), citing spec 5/6's own
  ADRs as the precedent being followed for both.
- `README.md`: a short "Cost and timing" note (not a full new section — this feature has no
  new CLI flags to document) stating that every run now writes `cost_report.json`/
  `{stem}.cost_report.json`, that figures are estimates, and that `--critics`/`--models`/
  `--batch` per-query figures are approximated (cross-reference the spec's own framing).
**Verify:** `grep cost.py spec/ARCHITECTURE.md` matches in the boundary map, the graph, and
INV-1; `grep` for the new `cost.py` names matches in the `tests/` boundary-map row and INV-7's
evidence, alongside a provenance note distinct from every prior spec's own note; `README.md`
mentions `cost_report.json`.
**Covers:** AR-1.2 (documentation), AR-2.1 (documentation), AR-5.1

## Final Verification

- `.venv/bin/python -m pytest -q` — no *new* failures beyond whatever this run's local AWS SSO
  session state already contributes (see Readiness); this plan's new tests add to the passing
  count, and no existing test is removed or weakened.
- `.venv/bin/python experiment.py run --help` / `.venv/bin/python classify.py --help` — no new
  flags appear (this spec adds none); both entry points still parse and run.
- A faked-LLM end-to-end `experiment.py run` (plain mode) produces a `cost_report.json` whose
  `total_cost_usd` matches hand-computed arithmetic from the fake's own known token counts and
  a monkeypatched pricing table.
- A faked-LLM end-to-end `--batch` run's `cost_report.json` has `batch_cost`/`query_cost` with
  the correct `estimated` flags and internally consistent counts (`query_cost.count` equals the
  number of rows classified; `batch_cost.count` equals the number of formed batches).
- `grep -rn "usage_sink"` across `src/`/`tests/` — no matches (confirms the rejected design
  never leaked into the implementation).
- `grep -n "def classify(" src/query_classification/classifier.py` plus every call site
  (`grep -rn "\.classify("` across `src/`) — argument lists are byte-identical to the
  pre-implementation baseline (AR-1.1's compatibility promise, checked mechanically rather than
  only by test pass/fail).
- A `PricingCache.resolve()` unit test asserts the exact candidate-id sequence walked for a
  Cerebus workspace-slug id matches `batching._candidate_ids`'s own sequence for the same input
  (FR-2.3's "identical candidate walk" requirement, checked by direct comparison rather than
  independently re-derived logic that could silently drift from `batching.py`'s).
- `tests/test_cost.py` and every other new/modified test module import only
  `unittest.mock`/pytest monkeypatching for `litellm.get_model_info` and `litellm.completion` —
  `grep -rn "litellm.completion(" tests/` (excluding mock/monkeypatch setup lines) has no
  matches, confirming the `AGENTS.md` "no live network/provider calls in tests" rule holds for
  every file this plan touches, not only the ones its own Verify lines mention.
- Independently spot-check one non-Cerebus, non-workspace-slug model id (e.g. `gpt-4o-mini`)
  resolves through `PricingCache` to the same `input_cost_per_token`/`output_cost_per_token`
  `litellm.get_model_info` itself reports — a check on the "simple" path, not only the
  chunk-down/candidate-walk path Task 2's own Verify line already covers.

## Documentation

- Living docs: none exist under `spec/docs/` (never generated for this repo, same as spec 5).
- `spec/ARCHITECTURE.md` — Module Boundary Map, mermaid graph, INV-1, INV-7 (Task 7).
- `README.md` — brief cost/timing note (Task 7; see Spec Deviations).
- `spec/7-cost-and-timing-logging/ADR.md` — new, one record (Task 7).

## Spec Deviations

| # | Spec says | Plan does instead | Reason | Action required |
|---|-----------|-------------------|--------|-----------------|
| 1 | Feature 6 (Tests) has only FR-6.1, no AR for INV-7's test-import boundary (unlike spec 5's dedicated AR-5.1) | Task 7's ADR amends INV-7 for `cost.py`'s new test-import names anyway | FR-6.1 requires `tests/test_cost.py` to exercise `cost.py`'s internals directly, which is unsatisfiable without an INV-7 amendment — the spec's own testing requirement can't be met otherwise; this is filling an omission, not changing behavior | None |
| 2 | No FR/AR requires `README.md` changes (unlike spec 5's FR-4.3) | Task 7 adds a short README note about the new artifact | This spec adds no new CLI flag, so there's no Options-table row to add, but a permanent new artifact users will discover deserves at least a pointer, matching every prior spec's documentation discipline | None |
| 3 | FR-5.1 says `query_cost.count` is "rows attempted (not merely succeeded)"; FR-2.2 says a `None` cost "poisons the whole aggregate to `None`" | Task 3 records a failed plain-mode row's real cost when *some* billable attempt happened (per FR-1.2), but skips recording anything at all for a row that failed with zero billable attempts, rather than recording `None` for it | Applying FR-2.2's poisoning rule literally to a single isolated row-level failure would zero out `mean_usd`/`stddev_usd` for an *entire* run over one row that never reached the provider — clearly not the rule's intent, which is about a run-wide unresolvable *price*, not a per-row zero-attempt failure. This keeps `count` equal to `len(work_idx)` in the overwhelmingly common case (any real failure that reaches the provider produces usage) while avoiding a single edge case corrupting the whole report | None (a reasonable reading of the spec's evident intent, not a behavior change any test currently exercises) |
| 4 | FR-5.1's `pricing` field names one `resolved_input_model`/`resolved_output_model` pair | Plan reports `pricing` as `null` when a run's classifiers resolve to more than one distinct model id (`--critics` with a different `--critic-model`/`--reconciler-model`, or `--models` with 2+ distinct ids) — populated only for the common single-model case (plain, `--batch`, or `--critics`/`--models` all resolving to one id) | `pricing` was this plan's own addition (mirroring spec 5's `run_config.json` diagnostic precedent), not spec-mandated — the spec's own Out of Scope already excludes per-role cost breakdowns, and a single model-id pair cannot truthfully represent several actually-used models. Reporting `null` rather than an arbitrary "whichever" candidate avoids fabricating a misleading value | None (the field itself is a plan-level addition; a future spec could request a full per-role pricing breakdown if needed) |

## Risks

- **Risk: the single usage-capture hook point (`classifier.py:537`) is bypassed by some future
  code path.** Mitigated by Task 1's own exhaustive test list, which pins every currently-known
  response-producing path (primary attempt, ladder recursion, fallback) explicitly — a future
  change adding a new response-producing branch to `_attempt_completion` without routing through
  this same line would need to be caught in review, not by this plan.
- **Risk: `BatchRunner.run`'s `_top_level` gating is subtly wrong for a future spec-5-style
  change to the retry/bisection structure.** Low: Task 4's tests assert the exact call-count/
  cost-attribution behavior spec 5's own bisection tests already pin, so a future change to that
  structure would already need to touch (and re-verify) this same test file.
- **Risk: `--critics`/`--models`'s uniform per-query approximation could be mistaken for exact
  data by a downstream consumer that doesn't check `"estimated"`.** Accepted per the spec's own
  Out of Scope — the `estimated` flag is the documented mitigation; no further plan-level
  mitigation is in scope.
- **Rollback/checkpoint guidance:** each task is a commit on the spec worktree branch. Tasks 1–4
  are purely additive (new module, new opt-in parameters, new methods) and revert without
  touching existing behavior. Tasks 5–6 are the first to change file-write behavior
  (`experiment.py`'s `_write_json_atomic` removal in favor of `cost.write_json_atomic`) —
  checkpoint before Task 5. Task 7 touches only docs/ADR, trivially revertible.
