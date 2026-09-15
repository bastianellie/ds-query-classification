# Spec 5: Batched Classification (`--batch`)

> **Status: CLOSED** — Implemented and verified on 2026-09-15.
> Implementation summary: `spec/5-batched-classification/implementation-summary.md`
> Merged from worktree branch `spec/5-batched-classification` on 2026-09-15.

## Overview

Today every row costs at least one LLM call: `pipeline.classify_csv` submits one
`classifier.classify(text)` per row across `--workers` threads. This spec adds a `--batch` mode
that packs several queries into a single LLM call — sized either automatically from the model's
token budget (`--batch dynamic`) or to a fixed count (`--batch INT`), in both cases bounded by
a hard maximum arity — cutting total call volume on the large test splits
(`experiments/pubmed-rct` alone is ~29.6k rows) without changing behavior when the flag is
absent.

Batching applies to plain single-shot classification only. It is mutually exclusive with both
`--critics` and `--models`. Because packing N queries into one prompt lets them influence each
other, a batched run's accuracy is not strictly comparable to an unbatched one — this spec
requires that fact be recorded and surfaced, not hidden.

**Depends on spec 6 (classifier-error-path).** Spec 6 shipped, ahead of this one, the exact
failure-classification foundation this spec was originally going to build itself:
`classify_failure`/`FailureKind`'s two-axis (`retryable`, `isolable`) classification, an already
generalized parameter-drop ladder (`_DROPPABLE_PARAMS`), and truncation/empty-response/
policy-refusal detection. This spec does not re-derive any of that — FR-2.5 and FR-3.2 consume
it directly. Spec 6 must be merged before this spec is implemented.

## Goals

- Reduce total LLM calls for a plain classification run by roughly the batch factor, on both
  `classify.py`/`cli.py` and `experiment.py`'s `classify`/`run` subcommands.
- Make batch sizing automatic where the model's token budget is knowable, and explicit
  (with an actionable error) where it isn't — while never letting an automatic size grow
  large enough to break failure isolation or save granularity.
- Keep the no-`--batch` path byte-for-byte identical to today's behavior.
- Preserve today's "one bad row loses one row" failure granularity despite the larger
  failure unit, via bisection — without amplifying failures that bisection cannot fix.
- Keep batched runs auditable and distinguishable from unbatched runs in `run_config.json`.

---

## Feature 1: Batch sizing and token budgeting

**Who & why:** Someone running the ag-news / TREC / PubMed-RCT experiment suites is paying
one LLM call per test row — tens of thousands of calls for a single run, which is both slow
and expensive. They want to trade some per-call prompt overhead for a large reduction in
call count, and they want the tool to figure out how many queries fit rather than making
them guess and hit mid-run `BadRequestError`s.

### Functional Requirements

#### FR-1.1: `--batch` flag surface and its resolved shape
Both `cli.py`'s parser and `experiment.py`'s `classify`/`run` subcommand parsers gain a
`--batch` flag accepting either the literal string `dynamic` or a positive integer. Its
argparse default is `None`, meaning "not given" — and when it is not given, the
classification path must be the existing per-row path with no batching code in the call
chain at all.

Because the flag is a `str | int` union it cannot be validated by `experiment.py`'s numeric
`>= 1` loop. Instead a dedicated resolver normalizes it once, at validation time, into an
explicit pair stored on `args` — `(None, None)` when the flag is absent, `("dynamic", None)`,
or `("fixed", n)` — and both `run_classification` and the FR-4.1 config record read only that
resolved pair, so the two can never disagree. This mirrors spec 2's
`_resolve_induction_sizing` / `args.resolved_induction_sizing` one-authority pattern. An
integer `< 1` and any other string value (e.g. `--batch auto`, `--batch 3.5`) are each
rejected by the resolver with a message naming `--batch` and stating the two accepted forms.

`--batch 1` is legal and takes the batching path with an arity-1 batch model (not the
unbatched path). It is the intended control condition for measuring batching's prompt
overhead in isolation from cross-row effects.
**Verify:** `--batch dynamic`, `--batch 8`, and `--batch 1` parse and produce resolved pairs
`("dynamic", None)`, `("fixed", 8)`, `("fixed", 1)`; `--batch 0`, `--batch -1`, and
`--batch auto` each exit non-zero with a message naming `--batch`; omitting `--batch`
resolves to `(None, None)` and produces a run whose per-row call count equals the row count.

#### FR-1.2: Independent input and output token-budget resolution
Batch sizing needs the model's maximum input and output token counts, resolved
**independently** — either may be available while the other is not. Each is resolved by
calling `litellm.get_model_info` against a sequence of candidate ids derived from the
configured model id, in this order, stopping at the first candidate that resolves and reports
the value being sought:

1. The model id as given.
2. The id with a leading `openai/` prefix stripped.
3. The result of (2) with a leading `@workspace/` segment stripped (everything up to and
   including the first `/`, when the id starts with `@`).
4. The result of (3) with trailing `-`-delimited segments progressively dropped, longest
   candidate first, down to the first segment.

A candidate that resolves but reports no value for the side being sought is treated as
unresolved for that side and the walk continues — 625 of the 3957 entries in litellm's model
map have no `max_input_tokens`, so this is a live case. Resolution happens once per distinct
model id per run and both results are reused for every batch.

The walk is a **best-effort heuristic, not an identity mapping**: chunking down can land on a
sibling model with a *different* (including larger) context window, which would overestimate
the budget. FR-1.5's hard arity cap is the primary guard against that, and FR-4.1 records the
candidate actually resolved so a wrong mapping is diagnosable after the fact.
**Verify:** `openai/@sciencedirect-global-openai/gpt-5.4-mini-2026-03-17` resolves to
`gpt-5.4-mini-2026-03-17` (272000/128000), `@sciencedirect-global-openai/gpt-5.4-2026-03-05`
to `gpt-5.4-2026-03-05` (1050000/128000), `@ws/gpt-4o-mini-2099-99-99` to `gpt-4o-mini`
(128000/16384) by chunk-down, `@ws/totally-made-up-model` to neither side, and a model
reporting only `max_input_tokens` resolves the input side while leaving the output side
unresolved.

#### FR-1.3: An unresolvable input budget is fatal under `dynamic`
`--batch dynamic` requires a resolvable **input** budget. When it is unavailable from both
FR-1.2's walk and FR-1.4's override, the run fails before any LLM call, printing this line
verbatim with the model id substituted for `model-XXX`:

```
Impossible to dynamically calculate number of queries: no max token value known for model-XXX. Please use `--batch INT` to set a pre-definite numbers of queries.
```

followed by a second line naming the unresolved side. The two are separate lines precisely so
the quoted sentence stays verbatim while the diagnostic detail still reaches the user — an
earlier revision required the quoted text to itself name the side, which it cannot do.

An unresolvable **output** budget is not fatal in either mode: FR-2.5's cap is a runaway guard,
and its absence leaves the request shaped exactly as today's unbatched calls. Dynamic mode
proceeds, sizing on the input budget alone, and warns once that no output guard is in effect.

`--batch INT` does not require either budget. It enforces every side that is known or
overridden and skips the trim check for only the unknown side, printing a one-line warning
naming precisely which side could not be checked. The no-`--batch` path never attempts
resolution at all.
**Verify:** `--batch dynamic` with an unresolvable input budget exits non-zero, prints that
exact sentence on its own line plus a second line naming the input side, and makes zero LLM
calls; `--batch dynamic` with a resolvable input budget but no output budget runs and warns;
`--batch 8` with an unknown output side enforces the input trim check and warns that only the
output side is unchecked.

#### FR-1.4: Manual token-budget override flags
Two flags, `--batch-max-input-tokens N` and `--batch-max-output-tokens N`, override the
corresponding value from FR-1.2. Each defaults to `None` and satisfies **only its own side**,
so setting one leaves the other to be resolved (or to remain unresolved per FR-1.3). A value
`< 1` is rejected like FR-1.1's. Setting either flag without `--batch` is rejected with a
message saying the flag has no effect without `--batch`, rather than being silently ignored.
When a flag is set, it — not the resolved value — is what FR-1.6/FR-1.7 budget against, what
FR-2.5 caps to, and what FR-4.1 records.
**Verify:** `--batch dynamic --batch-max-input-tokens 4000` against a model whose resolved
input window is 272000 produces batches sized against 4000; `--batch-max-input-tokens 4000`
without `--batch` exits non-zero; setting only `--batch-max-output-tokens` leaves the input
budget at its resolved value.

#### FR-1.5: Hard maximum arity, enforced during accumulation
No batch may exceed `--batch-max-size` rows, default **50**, regardless of what the token
budget would permit. The cap **bounds growth while a batch is being accumulated** — FR-1.6
stops admitting rows the moment either the cap or the budget is reached, whichever comes
first. It is deliberately not a post-hoc clamp applied after sizing: measuring a candidate
payload first and capping afterwards would mean running `litellm.token_counter` over a
19,000-row payload to then discard all but 50 rows of it.

A value `< 1` is rejected like FR-1.1's, as is `--batch-max-size` without `--batch` (matching
FR-1.4's treatment of its sibling flags). A `--batch INT` greater than the cap is rejected at
validation time with a message naming both flags, rather than being silently clamped — so by
the time sizing runs, `INT <= cap` always holds.

The cap exists because the token budget alone permits arities that break three other
requirements at once. Measured against this repo's own configured models: `gpt-5.4-mini`'s
272k input window admits roughly 4,900 PubMed sentences per batch, and
`gpt-5.4-2026-03-05`'s 1,050,000-token window — the model all three `run_experiment.sh`
scripts use for the critic/reconciler roles — admits roughly **19,000**, i.e. most of the
29.6k test split in a single call. At that size FR-3.2's bisection costs tens of thousands of
calls, FR-3.1's flush cadence degenerates to one save per run, and FR-4.2's comparability
caveat becomes a severe understatement. The default of 50 is chosen on diminishing returns:
on a 29.6k-row split, arity 50 cuts calls by 98% (29,600 → 592), while arity 5,000 reaches
only 99.98% — two further percentage points bought at the cost of all three properties above.
**Verify:** `--batch dynamic` against a 1,050,000-token model produces batches of at most 50
rows and never renders a payload larger than 50 rows for measurement; `--batch-max-size 10
--batch dynamic` produces batches of at most 10; `--batch 100` with the default cap exits
non-zero naming `--batch` and `--batch-max-size`; `--batch-max-size 10` without `--batch`
exits non-zero.

#### FR-1.6: `--batch dynamic` sizes every iteration from the fully rendered payload
For each iteration, the batch is grown greedily from the next unclassified rows in order,
admitting one row at a time while the projected cost stays within the input budget, and
stopping at the first row that would exceed it or at FR-1.5's cap, whichever comes first.

**Input — measured, not summed.** Token counting is not additive across string boundaries,
and `json.dumps` escaping changes the content being counted, so an additive per-row estimate
may be used to *shortlist* a candidate arity but the chosen arity must then be validated by
counting the **complete rendered payload** for that arity — the system prompt plus the fully
serialized user message — with `litellm.token_counter`, plus the serialized `response_format`
JSON Schema, which is charged as input and grows with arity (it carries one required field per
position plus the nested row schema). If the measured total exceeds the input budget, the arity
is reduced and re-measured until it fits or reaches 1.

**Output — an estimate used for sizing only.** Per admitted row, for every category, the
maximum label count the schema permits (3, per `build_classification_model`'s `max_length=3`)
times the token count of that category's **longest configured** label value, plus the nested
`result_i` object's JSON overhead; the per-batch total is multiplied by a safety factor of 2
and compared against the output budget as a second admission test when that budget is known.

This figure is explicitly **not** an upper bound, and is never used as the request's output
cap (FR-2.5). INV-11 records that `build_classification_model` validates cardinality and type
only, never label values, so a model may legally return an arbitrarily long string — verified
live, a 5,007-character label validates even with `allow_new_labels=False`. Reasoning tokens
also count against real output while being invisible to this calculation. Both are reasons the
estimate can only ever inform *how many rows to send*, never *how many tokens to allow*.

A batch always contains at least one row: if a single row's measured cost alone exceeds either
budget, that row is still sent alone and a warning naming the row index is printed.
**Verify:** with a small `--batch-max-input-tokens`, batch sizes shrink as row texts grow and
the measured rendered payload never exceeds the budget; a single oversized row is submitted
alone with a warning naming its index; an arity whose additive estimate fits but whose
rendered payload does not is reduced before sending.

#### FR-1.7: `--batch INT` uses INT, trimmed only where INT would not fit
`--batch INT` uses exactly `INT` rows per iteration. For each iteration the same measurements
as FR-1.6 are applied to those `INT` rows; if a known budget is exceeded, the count is trimmed
down — one row at a time — to the largest count that fits, **with a floor of 1**: if even a
single row exceeds a budget, that row is sent alone with FR-1.6's named-index warning rather
than trimming to zero. A trim applies to **that iteration only**; every subsequent iteration
starts again from `INT`. The final iteration of a run is naturally short when fewer than `INT`
rows remain; that is not a trim and must not be recorded as one.

The floor is load-bearing, not defensive: without it, "the largest count that fits" is zero
when the head row alone overflows, which would either raise from `build_batch_model(…, 0)`
(AR-2.2) or produce an iteration that consumes no rows and livelocks — reachable with exactly
the `--batch-max-input-tokens 4000` configuration FR-1.4's Verify line uses.
**Verify:** with `--batch 10` and one long row in the middle of the input, the iteration
containing it is trimmed below 10 while the iterations before and after it are exactly 10; a
`--batch 10` run whose first remaining row alone exceeds the budget sends that row alone and
makes forward progress rather than looping; the last iteration's size equals the remainder and
is not counted as a trim.

### Architectural Requirements

#### AR-1.1: Sizing and batched-call logic lives in a new `batching.py` module
A new `src/query_classification/batching.py` owns token-budget resolution (FR-1.2), batch
sizing (FR-1.5–FR-1.7), the per-arity schema/classifier cache (FR-2.4), payload assembly
(FR-2.2), and the batched-call-with-bisection routine (FR-3.2). It may import
`categories.py`, `classifier.py`, `schema.py`, and `prompts.py`; it must not import
`pipeline.py`, `debate.py`, `multi_model.py`, `cli.py`, or `experiment.py`. `pipeline.py`
gains an import of `batching.py`; `debate.py` is not touched by this spec at all.

This adds a new node and **three** inbound edges to INV-1's dependency graph:
`batching.py --> pipeline.py`, `batching.py --> cli.py`, and `batching.py --> experiment.py`.
All three are required — `pipeline.py` runs the batches, and both entry points must construct
the batch prompt (FR-2.3), the classifier factory (FR-2.4), and the stats object (FR-4.1). The
implementation must ship an ADR citing INV-1, following the precedent of
`spec/4-multi-model-classification/ADR.md` ADR-001 for `multi_model.py`. Because INV-1's rule
text and the Module Boundary Map are **closed per-module lists**, the ADR must edit the
existing `pipeline.py`, `cli.py`, and `experiment.py` rows as well as adding a `batching.py`
row — enumerating only one edge would leave the invariant contradicting the shipped import
graph. The graph stays acyclic: `batching.py` has no inbound imports from any module it
imports.

#### AR-1.2: litellm's model-lookup noise must be suppressed
`litellm.get_model_info` writes a colored `Provider List: https://docs.litellm.ai/docs/providers`
block to the real stderr on every failed lookup, and it is **not** capturable via
`contextlib.redirect_stderr` (verified live). FR-1.2's walk tries up to ~6 candidates per side
per model, so without suppression a normal batched run prints several noise blocks per
classifier role. Setting `litellm.suppress_debug_info = True` before the walk suppresses it
completely (verified live). This must be set by `batching.py` at the point of resolution, not
globally at import time, so it does not silently change diagnostics for the non-batched path.
It is set once and deliberately **not** restored afterwards: a non-batched run never reaches
resolution, so the flag is already scoped to batched runs by construction, and a
`try`/`finally` restore would race against other worker threads resolving concurrently.

#### AR-1.3: `get_model_info` raises a bare `Exception`
The failure mode for an unknown model id is a bare `builtins.Exception`, not a litellm-specific
type (verified live against the installed litellm). The resolution walk must therefore catch
broad `Exception` per candidate, carrying the `# noqa: BLE001` plus justification comment
required by INV-4.

---

## Feature 2: The batched request/response contract

**Who & why:** The implementer needs an unambiguous wire contract for a batched call. Spec 2's
induction redesign already paid the cost of learning this the hard way: a response schema that
lets the model choose how many items to return, or that re-states identifiers the runner
already knows, produces failures that are impossible to fix after the fact. The batch contract
must make a wrong-arity or misaligned response structurally impossible instead.

### Functional Requirements

#### FR-2.1: Positional batch response schema, with extras forbidden at both levels
The batch response model is built dynamically for a given arity `N`: one **required** field per
position, `result_1` through `result_N`, each typed as a copy of the single-row
`ClassificationResult` model that `build_classification_model` already produces for the run's
categories. It is not a length-bounded `list` field, and it carries no row identifiers —
position is the only linkage. `extra="forbid"` is set on **both** the outer batch model and the
nested row copy.

All three parts of the arity bound are load-bearing and were verified live:
- Required fields reject a **short** response (`result_3` missing from a 3-arity model →
  Pydantic `missing`).
- `extra="forbid"` on the outer model rejects a **long** response. Without it, Pydantic's
  default `extra="ignore"` silently accepts a 4th `result_4` against a 3-arity model — the
  shape a model produces when it mis-splits one input into two, which would leave every
  position after the split misaligned with the wrong row.
- Outer `extra="forbid"` does **not** protect the nested row model: a bogus field inside
  `result_1` is silently ignored unless the nested copy is also strict.

The nested strictness applies to the **batch model's copy only**, never to
`build_classification_model`'s own output. Tightening it globally would change how the
unbatched path handles an unexpected field (today: ignored; then: a validation failure burning
all three retries), which this spec's Constraints forbid.

INV-11 is otherwise unchanged: this tightens *shape and arity*, never label validity. An
invented, duplicated, or overlong label value inside a `result_i` is still accepted exactly as
it is today.
**Verify:** a 3-arity model accepts exactly 3 results, raises `missing` for 2, raises
`extra_forbidden` for 4, and raises `extra_forbidden` for an unexpected field nested inside
`result_1`; `build_classification_model`'s own output still ignores unexpected fields.

#### FR-2.2: Ordered, JSON-encoded, delimited payload
The user message for a batched call is a JSON object carrying an **ordered array** of the
batch's queries, each an object with a 1-based `position` field and the query text, wrapped in
the `<<<DATA>>>` / `<<<END DATA>>>` delimiters already used by `induction.py`. The whole payload
is produced by `json.dumps`. The array's `position` values correspond 1:1 with FR-2.1's
`result_i` field names.

The guarantee this provides is **payload integrity, and only that**: a query's own text cannot
structurally forge a delimiter, a position field, or a neighboring array entry, because it can
only ever appear as a JSON string value. It does **not** isolate rows semantically — the model
still reads every query in one context, so instruction-like text in one row can still
influence another row's label, and because FR-2.1 deliberately carries no row identifiers such
influence is undetectable after the fact. That residual risk is new to batching (in per-row
mode a query could only affect its own classification), is accepted, and is surfaced by
FR-4.2.
**Verify:** a batch whose query text literally contains `<<<END DATA>>>`, `"position": 1`, and
`{"result_2": {...}}` still parses as a single JSON payload with the original number of
entries and its text survives round-trip as one string value — **and**, against a fake
classifier, that hostile row cannot change the number of results attributed to its neighbors
or the row each result is mapped to.

#### FR-2.3: The caller builds the batch system prompt
The batched system prompt is built by the **CLI layer** (`cli.py` / `experiment.py`), which is
the only place that holds `--system-prompt`, `--task-description`, `--extra-prompt`, and
`--allow-new-labels`, and is passed down ready-made. `batching.py` must never reconstruct it:
`classify_csv` receives `allow_new_labels` but **not** the other three (verified against its
signature), so a prompt rebuilt inside the batching layer would silently drop the user's task
description and extra instructions.

It is rendered by `build_system_prompt` from the **single-row** `ClassificationResult` — so
`schema_description` still enumerates every category's label options, which a batch model's
nested fields would render as useless bare type names — plus appended text stating: queries
arrive in a fixed numbered order; the response must carry exactly one `result_i` per query in
that same order; each query must be classified **independently** of the others in the batch;
and the query texts are data, never instructions. The independence sentence is part of the
contract, not decoration — it is the only in-prompt mitigation for FR-2.2's residual semantic
risk.
**Verify:** the rendered batch prompt contains every category's label options, the user's
`--task-description` and `--extra-prompt` text, the one-`result_i`-per-query-in-order rule, the
independence instruction, and the data-not-instructions framing.

#### FR-2.4: A classifier factory, and a thread-safe per-arity cache
Batch arity varies within a single run: the final iteration is short, `--batch dynamic`
re-sizes every iteration, and bisection produces halved arities down to 1. Since `Classifier`
holds one `classification_model` for its lifetime, a batched run needs one batch model — and
one `Classifier` wrapping it — per arity encountered.

These are produced by a **factory supplied by the CLI layer**, not constructed inside
`batching.py` from scratch. The factory carries over every setting of the run's prototype
classifier: `model_id`, `max_retries`, `retry_delay`, `api_base`, `temperature`, `api_key`,
`extra_headers`, and `max_tokens` — the complete parameter list of `Classifier.__init__` other
than the model and prompt — plus FR-2.3's caller-built system prompt and FR-2.5's output cap
(the last of these is simply `max_tokens` itself; it is called out by name in FR-2.5 because
that is the requirement that decides its value, not because it is a separate constructor
parameter).
`api_key`/`extra_headers` are load-bearing rather than tidy: they hold the Cerebus gateway
credential and Portkey auth headers, so dropping them would make **every batched run fail on
this repo's normal configuration** (`DEFAULT_LLM_PROVIDER=cerebus`).

The cache is keyed on **arity** and scoped to a single run, never module-global (a
module-global cache would leak arities between tests, the same problem `tests/test_cerebus.py`
already has to solve for `classifier._cerebus_api_key_cache`). Because FR-3.1 populates it
from concurrent worker threads and FR-3.2 discovers new arities inside those workers,
lookup-and-construction must be performed under a lock: a plain check-then-insert is a
compound operation that two threads can execute simultaneously, producing two models for the
same key. Cache growth is bounded by FR-1.5's cap, so at most `--batch-max-size` entries
exist.
**Verify:** a run over 25 rows with `--batch 10` uses exactly 2 distinct arities (10 and 5);
under concurrent misses forced by a barrier, exactly one model is constructed per arity; a
batch classifier derived from a Cerebus prototype carries the same `api_key`, `extra_headers`,
and `api_base`.

#### FR-2.5: A generous output cap as a runaway guard, reusing spec 6's classification
`Classifier` gains an optional `max_tokens: int | None = None` parameter, added by spec 6 to
`__init__`'s call-through pattern but not yet to its parameter list — `_completion_kwargs`
includes it only when set, the same conditional pattern already used for `api_base`,
`temperature`, `api_key`, and `extra_headers`. Batched calls set it to the **resolved (or
overridden) output budget**, and send nothing when that budget is unknown.

**It is deliberately not set to FR-1.6's output estimate.** The estimate's job is choosing how
many rows to send; the cap's job is stopping a runaway response. Sizing the cap to the estimate
would make it tight by construction, and because the estimate excludes both unbounded label
values (INV-11) and reasoning tokens, a tight cap can sit *below a reasoning model's floor* —
making truncation deterministic at every arity, so FR-3.2 bisects to size 1, the size-1 call
truncates too, and every row fails while the run still exits `"completed"`. Measured: on
ag-news's four short labels the estimate at the bisection endpoint is ~38 tokens. A cap that is
generous by construction cannot misfire this way, and on the unknown-budget path sending no cap
leaves the request shaped exactly as today's unbatched calls.

**Truncation detection and the parameter-drop ladder are not new work — spec 6 already built
both, generically, for every call.** This spec was originally written to build them itself; by
the time it reaches implementation, spec 6 (`classifier.py`) has already shipped:

- `TruncatedResponseError`, raised unconditionally whenever a response's `finish_reason` is
  `"length"` (`classifier.py`'s `_extract_content`), and classified `(retryable=False,
  isolable=True)` by `classify_failure`. It is **not** gated on whether a cap was sent — spec 6
  made this check universal, for every call, batched or not, on the grounds that a provider can
  truncate a response even when the caller set no cap of its own. This spec inherits that
  behavior as-is: adding `max_tokens` only changes how *likely* truncation is to occur (a
  tighter cap truncates sooner), never how it is detected or classified. There is nothing left
  for this spec to gate, build, or test at the `Classifier` level — FR-3.2 simply consumes
  `classify_failure`'s result, below.
- `_DROPPABLE_PARAMS`, an ordered, module-level tuple that `_attempt_completion`'s
  `except litellm.UnsupportedParamsError` handler walks generically, dropping at most one
  entry per attempt and never re-dropping one already removed. Its only entry today is
  `"temperature"`. This spec's sole remaining action here is to **prepend `"max_tokens"`**,
  giving `_DROPPABLE_PARAMS = ("max_tokens", "temperature")`. `max_tokens` goes first because it
  is only a guard — proceeding without it reproduces today's unbatched behavior exactly —
  whereas `temperature` is semantically meaningful and should survive if `max_tokens` alone was
  the problem. This preserves the guarantee `classifier.py`'s existing
  `UnsupportedParamsError` handling comment was written for (a model rejecting `max_tokens`
  no longer misdiagnoses the failure as a temperature problem or vice versa), and makes the
  `max_tokens` vs `max_completion_tokens` question non-load-bearing — litellm translates for
  this repo's models (verified via `get_supported_openai_params`), and any model where it
  cannot simply loses the guard.

**Verify:** an unbatched call (no `max_tokens` set) sends none and behaves exactly as before
spec 6's own changes; a batched call with a known output budget sends `max_tokens` equal to it,
and one with an unknown budget sends none; appending `"max_tokens"` to `_DROPPABLE_PARAMS`
and constructing a classifier with both `max_tokens` and `temperature` set, against a fake that
rejects both in turn, drops `max_tokens` first and leaves `temperature` intact on the next
attempt.

### Architectural Requirements

#### AR-2.1: Bounded lists are not an option — same strict-mode constraint as spec 2
`litellm.utils.type_to_response_format_param` serializes a Pydantic `response_format` with
`"strict": true`, and strict structured output does not support `minItems`/`maxItems`. Measured
live on the installed stack: today's single-row `ClassificationResult` already emits **one**
`minItems` (from `build_classification_model`'s `Field(min_length=1, max_length=3)` on the
label list); FR-2.1's positional design emits **one** — unchanged; a
`results: list[RowResult]` field pinned with `min_length=max_length=N` emits **two**. The
positional design therefore keeps the payload exactly as strict-compatible as the current
unbatched path, while a bounded list would make it strictly worse and risk silently degrading
every batched call into `Classifier._attempt_completion`'s `except litellm.BadRequestError`
JSON-object fallback. This is the same finding that drove `build_induction_model`'s positional
design (`schema.py`, spec 2 AR-2.1) and it is re-verified here for the nested-model case,
including that strict serialization still emits `additionalProperties: false` on both the outer
and nested objects with `extra="forbid"` set.

#### AR-2.2: The batch model builder belongs in `schema.py`
`build_batch_model(row_model: type[BaseModel], n: int) -> type[BaseModel]` is added to
`schema.py` alongside `build_classification_model` and `build_induction_model`, using the same
`create_model` mechanism. It takes the already-built row model rather than the category list, so
it stays agnostic to how the row model was constructed and does not duplicate
`build_classification_model`'s duplicate-name validation (INV-8). It produces the strict nested
copy required by FR-2.1 without mutating the row model it was given. It raises `ValueError` for
`n < 1`, mirroring `build_induction_model`.

#### AR-2.3: The `<<<DATA>>>` delimiters are duplicated deliberately
`_DATA_START`/`_DATA_END` are module-private to `induction.py` (`induction.py:31-32`), and
AR-1.1 forbids `batching.py` from importing `induction.py`. `batching.py` therefore declares
its own copies of the two literals, with a comment naming `induction.py` as the origin of the
convention, and a test asserts the two modules' values stay identical. Promoting two string
constants into a shared module is not worth a new import edge; silent divergence is the only
real risk, and a test closes it.

---

## Feature 3: Batched execution, failure isolation, and mode exclusions

**Who & why:** A 29.6k-row run cannot be allowed to lose 10 rows because one row in a batch
provoked a malformed response, and it cannot lose its resumability. The batching layer has to
degrade into today's per-row guarantees precisely when things go wrong, while keeping the
concurrency that makes long runs bearable.

### Functional Requirements

#### FR-3.1: Batches are contiguous, run concurrently, and are applied row-at-a-time
Batches are formed from `work_idx` in order, as contiguous runs of row positions, after
`--restore` filtering and `--limit`/`--test-limit` truncation have already been applied.
`pipeline.classify_csv` submits **batches** rather than rows to its existing
`ThreadPoolExecutor(max_workers=max(1, workers))`, so `--workers` now means concurrent
in-flight batches.

A completed batch future returns **per-row outcomes** (a result or an error per row, in input
order), and the main thread applies them **one row at a time** — incrementing `classified` or
`failed`, advancing `completed` by one, updating the progress bar, and evaluating the flush
condition — exactly as it does for a single row today. A batch future that itself **raises**
(from the batching layer's own code rather than the LLM call) is caught and converted to a
failure for every row that batch covered, so the accounting invariant
`classified + failed == completed == total` holds and the progress bar always reaches 100%.
INV-6 is unchanged: worker threads still only make the read-only LLM call, and all `df`
mutation stays on the main thread, so output row order still follows input order regardless of
completion order.

Applying results row-at-a-time is required, not incidental: it is what preserves
`pipeline.py`'s existing `completed % save_every == 0` flush arithmetic unchanged. Advancing
`completed` by the batch size instead would make the modulo miss its multiples for any batch
size that does not divide `save_every` (default 20, no CLI flag exposes it) — measured over
200 rows, batch size 7 fires 2 flushes and batch size 32 fires 2, against ~10 expected — and
trimmed and bisected batches make sizes irregular even when the nominal size does divide.
Failed singleton leaves from bisection count as completed rows, as failures do today.
**Verify:** with `--batch 5 --workers 2`, a 20-row input produces 4 batches over 2 threads and
the output CSV's row order matches the input's; with `--batch 7` over 200 rows the number of
interim `to_csv` writes equals what the same row count produces unbatched; a batch future that
raises records one failure per covered row and leaves `classified + failed == total`.

#### FR-3.2: Failures are classified via spec 6's `classify_failure`, not re-derived
When a batched call fails, `batching.py` classifies it with spec 6's `classifier.
classify_failure(exc) -> FailureKind(retryable, isolable)` — the same total classification
`Classifier.classify` itself now uses — rather than deriving its own litellm-hierarchy
knowledge. This is not a convenience: an earlier draft of this spec built its own three-way
taxonomy from scratch and, across two rounds of review, got the litellm hierarchy wrong twice
in exactly the ways spec 6's implementation was independently critiqued into correctness for
(folding `Timeout` under `APIConnectionError`, missing that `BudgetExceededError` doesn't
derive from `APIError`). Reusing one already-verified classifier removes an entire class of
defect this spec would otherwise be exposed to a third time.

The three-way bisection policy is derived from `FailureKind`'s two axes as follows:

- **Isolable** (`kind.isolable` is `True`, regardless of `kind.retryable`) — the batch is split
  into two halves (the first taking the extra row when the size is odd) and each half retried
  as its own batched call, recursing until size 1. Splitting is tried first, ahead of retrying,
  because attributing a failure to a subset of the batch is more informative than repeating the
  whole thing unchanged. This covers `pydantic.ValidationError` (isolable *and* retryable, per
  spec 6 — a malformed response may validate on a plain retry, but bisection additionally
  narrows down which row is responsible if it keeps recurring), spec 6's `TruncatedResponseError`,
  and `litellm.ContextWindowExceededError`. The last of these was, in an earlier draft, believed
  unreachable — `classifier.py`'s old fallback handler treated it as a generic
  `BadRequestError` and silently retried in JSON-object mode. Spec 6 closed that: its
  `_attempt_completion` now excludes `ContextWindowExceededError` (along with
  `ContentPolicyViolationError`/`ImageFetchError`/`LiteLLMUnknownProvider`) from the fallback
  entirely, so it propagates as itself on the very first attempt — making it not only reachable
  but the cleanest possible bisection trigger, since an oversized batch is exactly what
  splitting is for.
- **Retryable, not isolable** (`kind.retryable` is `True`, `kind.isolable` is `False`) — e.g.
  `litellm.RateLimitError`, transient transport errors, spec 6's `EmptyResponseError`. The
  batch is **not** split; splitting cannot fix a limit or a transient hiccup that isn't about
  size, and would issue up to `2N-1` additional calls, raising request rate at precisely the
  moment it must fall. It is retried at its original size at most **2 further times**, with
  exponential backoff starting at the classifier's own `retry_delay` and doubling. This budget
  is the batching layer's own and runs *after* `Classifier.classify`'s `max_retries` loop is
  already exhausted, so the two compose rather than nest ambiguously. On exhaustion every row in
  the batch is recorded as failed, without bisection.
- **Neither** (`kind.retryable` and `kind.isolable` both `False`) — e.g.
  `litellm.AuthenticationError`, `litellm.ContentPolicyViolationError`,
  `litellm.BudgetExceededError`: not split, not retried, and recorded as a failure for every row
  in the batch. A run-wide credential problem would otherwise trigger a full bisection on every
  batch, multiplying a deterministic failure.

A failing size-1 call is recorded exactly as today's per-row failure: the `failed` counter is
incremented, the exception is not re-raised, and the run continues (matching `pipeline.py`'s
existing `except Exception` behavior and the Known Gap that per-row failures are counted, not
persisted). Bisection happens **after** `Classifier.classify`'s own `max_retries` loop has
already been exhausted for that call.

The worst case is bounded by FR-1.5: a fully-isolable-failing batch of size N costs `2N-1`
batched `classify` calls instead of today's `N`, so at the default cap of 50 that is at most 99
`classify` calls per batch. Note this counts `classify` invocations, not provider requests —
each carries its own `max_retries` attempts and may additionally trigger the JSON-object
fallback, so the provider-request ceiling is several times higher.
**Verify:** a batch of 8 in which exactly one row always fails (raising a `ValidationError`)
resolves the other 7 and records exactly 1 failure; an all-failing batch of 4 makes exactly 7
batched `classify` calls and records 4 failures; a `ContextWindowExceededError` from every row
in a batch bisects down to size 1 exactly like a validation failure, with no fallback attempt
in between (verifying it is reachable, not swallowed); a `RateLimitError` retries the batch at
its original size at most twice with doubling delays, then fails all its rows without
splitting; an authentication error fails the batch's rows immediately without splitting or
retrying; a batch of 1 that fails increments `failed` without raising.

#### FR-3.3: `--batch` is mutually exclusive with `--critics` and `--models`
`--batch` combined with `--critics`, with `--models` (2+ values), or with `--n-classifiers > 1`
is rejected at validation time with a `ValueError` naming both conflicting flags, in both
`cli.py` and `experiment.py`. This mirrors the existing `--models`/`--critics` exclusion
(`experiment.py`'s `_resolve_classifier_models`, `pipeline.py`'s `critics and have_models`
check) and is enforced in `classify_csv` itself as well as at the CLI layer, so a library
caller cannot reach an unsupported combination. `debate.py` and `multi_model.py` are therefore
untouched by this spec.
**Verify:** `--batch 4 --critics`, `--batch 4 --models a b`, and
`--batch 4 --model x --n-classifiers 3` each exit non-zero naming both flags;
`classify_csv(..., batch=..., critics=True)` and `classify_csv(..., batch=..., models={...})`
each raise `ValueError`.

#### FR-3.4: `--restore` remains correct, with batch boundaries recomputed
`--restore`'s completeness check is unchanged: it operates on row-level category columns,
before batching. A resumed run forms batches from only the remaining rows, so batch
composition after a resume generally differs from the original run's. This is acceptable —
batch composition is not persisted state — but it is a second-order source of the
non-comparability in FR-4.2 and must be documented as such rather than treated as a bug.
**Verify:** a batched run interrupted after some rows are saved, then resumed with `--restore`,
classifies exactly the remaining rows and leaves the already-complete ones untouched.

---

## Feature 4: Provenance and the comparability caveat

**Who & why:** These runs exist to produce accuracy numbers that get compared across
configurations in `experiments/*/analyze_run.ipynb`. If a batched run's numbers silently land
next to unbatched ones with nothing distinguishing them, the comparison is wrong and no one can
tell. Whoever reads a `run_config.json` six months from now needs to know a run was batched,
how it was sized, and that its numbers carry a caveat.

### Functional Requirements

#### FR-4.1: `run_config.json` records the batch configuration and outcome
`experiment.py`'s `_SCHEMA_VERSION` is bumped from 4 to 5 and its config dict gains a `batch`
object, `None` when the run does not classify — mirroring how `classifier_config` is `None`
when `not will_classify` (`experiment.py:899-912`). For a classifying run the object is always
present in the initial config literal, with `mode` set to `"dynamic"`, `"fixed"`, or `null`
when `--batch` was absent.

It records the requested configuration — mode, the requested integer for fixed mode, the arity
cap, the resolved and overridden input/output token budgets, and the candidate model id
FR-1.2's walk actually resolved to for each side (so a wrong chunk-down is diagnosable) — plus
the observed outcome: total batched calls, the min/mean/max arity actually sent, the number of
iterations trimmed by FR-1.7, and the number of bisection events from FR-3.2.

Every observed-outcome field is **initialized to `null` in the same literal** and overwritten
only from the stats snapshot after `classify_csv` returns. That initialization is what makes a
mid-classification failure record `null`s rather than a missing key — `batch` must *not* be
added to the failure path's `setdefault` list (`experiment.py:1213-1222`), which contains
exactly the keys assigned later inside the `try` block and correctly excludes init-time keys
like `classifier_config` and `cerebus`. `null` also distinguishes "no batching" from "batching
that did nothing", so the fields are never `0` when `--batch` was absent.

"Batched calls" counts `Classifier.classify` invocations made by the batching layer — not
litellm attempts, internal retries, or JSON-object fallback calls — and the arity statistics
cover every such invocation, including bisected ones. Summary statistics only, not a
per-iteration log, so the file stays small on a 29.6k-row run. As with the existing `cerebus`
block, no URL, key, or header value is recorded.

Because `config` is assembled before classification runs (`experiment.py:888`) and
`classify_csv` returns only a `Path` (`pipeline.py:277`), the observed-outcome fields need a
transport: the caller creates a thread-safe stats object, passes it into `classify_csv`, and
reads an immutable snapshot from it after the call returns. `classify_csv`'s return type is
unchanged.
**Verify:** a batched run's `run_config.json` has `schema_version == 5` and a `batch` object
whose resolved model ids, call count, and arity range match the run; a classifying run without
`--batch` has `batch.mode == null` and every observed-outcome field `null`, not `0`; a
`classify`-less run has `batch == null`; a run that fails mid-classification still writes a
`batch` key whose observed-outcome fields are `null`.

#### FR-4.2: A batched run warns that its results are not directly comparable
When `--batch` is active, both entry points print exactly one line, before classification
begins, stating that batching places multiple queries in a shared prompt and that results are
therefore not directly comparable to an unbatched run. The warning is printed once per run,
not per batch. This is the honest framing of a real effect — position within the batch, the
label distribution of neighboring queries, adversarial or instruction-like text in a
neighboring row (FR-2.2), and, after a resume, batch composition itself (FR-3.4) can all shift
an individual row's label — and FR-2.3's independence instruction reduces but does not
eliminate it.
**Verify:** a batched run prints the caveat exactly once on stdout; an unbatched run prints
nothing of the sort.

#### FR-4.3: Documentation and experiment scripts reflect the new flags
`README.md`'s options documentation covers `--batch`, `--batch-max-size`,
`--batch-max-input-tokens`, and `--batch-max-output-tokens`, including the
`--critics`/`--models` exclusions, the comparability caveat, and the fact that `--workers` now
counts concurrent batches so peak token throughput scales with `workers x batch size`.

The three `experiments/{ag-news,trec,pubmed-rct}/run_experiment.sh` scripts gain a `BATCH`
env-var override plumbed through in the same conditional-append style as the existing
`INDUCTION_EXAMPLES`/`TEST_LIMIT` handling, using `${BATCH:-}` to match those two. (The
no-colon `${VAR-}` form is only meaningful when there is a non-empty default to protect, as
with `EXAMPLES_PER_LABEL-20`; for an empty default the two forms are provably identical, so
matching the neighboring empty-default variables is the right call.)

All three scripts currently hardcode `--critics` inside their `ARGS` array
(`run_experiment.sh:74`), which FR-3.3 rejects outright in combination with `--batch` — so a
`BATCH` override alone would make every real run fail while a `DRY_RUN` check still passed,
because `DRY_RUN` never invokes the CLI. The scripts therefore also gain a `CRITICS`
env-var, defaulting to on, that makes the `--critics`/`--critic-model`/`--reconciler-model`
trio conditional. Setting `BATCH` **without** `CRITICS=0` is rejected by the script itself with
a message naming both variables, rather than silently dropping `--critics`: these scripts
produce the accuracy numbers compared across configurations, and silently switching the
experimental condition from a critics debate to plain batched classification is precisely the
class of error FR-4.2 exists to prevent. Defaulting `CRITICS` to on keeps the committed
configuration byte-identical when neither variable is set.

`spec/ARCHITECTURE.md`'s Module Boundary Map and INV-1 gain `batching.py`.
**Verify:** `README.md` documents all four flags and the `CRITICS` variable; with no variables
set, each script's `DRY_RUN` argv is **identical to today's after normalizing `--run-dir`'s
embedded timestamp** (each script's `RUN_DIR` derives from `$(date +%Y%m%d-%H%M%S)`, so
byte-identical argv across two separate invocations is unachievable — the comparison must strip
or replace that one field before comparing, and every other token, including `--critics`'s
position, must match exactly); `BATCH=4 CRITICS=0 DRY_RUN=1 run_experiment.sh` shows `--batch 4`
and no `--critics`; `BATCH=4 DRY_RUN=1` alone exits non-zero naming `BATCH` and `CRITICS`;
`bash -n` passes on all three scripts.

---

## Feature 5: Tests

**Who & why:** This feature's correctness lives almost entirely in arithmetic (sizing,
trimming, bisection) and in a wire contract — both of which are cheap to test exactly and
expensive to debug in a 29.6k-row live run. The project's rule is that no test may make a live
provider call, so all of it must be exercised against fakes.

### Functional Requirements

#### FR-5.1: Unit tests ship with the implementation
A `pytest` suite ships alongside the code, in a new `tests/test_batching.py` plus additions to
the existing `tests/test_experiment.py`, runnable with `.venv/bin/python -m pytest`. It covers
every FR's **Verify:** condition above. Beyond the happy paths, it must specifically cover:

- FR-1.2's resolution cases including chunk-down, total failure, and the asymmetric
  input-known/output-missing case; FR-1.3's exact error line, its second diagnostic line, and
  the unknown-output-budget path that proceeds with a warning.
- FR-1.5's cap against a large-window model, `--batch` exceeding the cap, and
  `--batch-max-size` without `--batch`.
- FR-1.6's rendered-payload re-measurement (an arity whose additive estimate fits but whose
  rendered payload does not) and the oversized-single-row case.
- FR-1.7's trim arithmetic and its floor-of-1 forward-progress case.
- FR-2.1's four arity/extras cases, including the nested-extras case and the assertion that
  `build_classification_model`'s own output is unchanged.
- FR-2.2's adversarial payload round-trip **and** the neighbor-attribution assertion.
- FR-2.4's concurrent-miss single-construction test (forced with a barrier) and the
  Cerebus credential/header preservation assertions.
- FR-2.5's `max_tokens` conditional-kwarg behavior (unset by default, sent when a batched
  call has a known output budget, absent when the budget is unknown) and the ordered-drop test
  asserting `_DROPPABLE_PARAMS = ("max_tokens", "temperature")` drops `max_tokens` first and
  leaves `temperature` intact. Truncation detection and classification are **not** retested
  here — they are spec 6's `TruncatedResponseError`/`classify_failure`, already covered by
  spec 6's own test suite; this spec's tests exercise only what it actually adds.
- FR-3.1's flush-count equivalence for a batch size that does not divide `save_every`, and the
  raising-future case preserving `classified + failed == total`.
- FR-3.2's exact `classify` call counts per error class, via `classify_failure` — one-bad-row
  (`ValidationError`, isolable), all-bad, a `ContextWindowExceededError` bisecting all the way to
  size 1 with no fallback attempt in between (proving reachability, not just classification),
  rate-limit (no split, bounded retries), auth (no split, no retry) — asserted against a
  counting fake.
- FR-3.3's three exclusion errors at both the CLI and `classify_csv` layers.
- FR-4.1's null-vs-zero distinction and the failed-run `batch`-key case.
- AR-2.3's delimiter-equality assertion between `induction.py` and `batching.py`.

No test may invoke `litellm.completion` — LLM calls are faked, following the existing
`FakeInductionClassifier`/`fake_classify` fixture patterns in `tests/test_experiment.py`. Tests
written alongside the code, not deferred.
**Verify:** `.venv/bin/python -m pytest` passes with the new tests present, and no test
triggers a network call.

### Architectural Requirements

#### AR-5.1: Test import boundaries extend INV-7 — mostly already granted by spec 6
`tests/` gains direct imports of `query_classification.batching` (all functions and the
module-private `_DATA_START`/`_DATA_END` constants that AR-2.3's equality test requires — the
same kind of exception INV-7 already grants for `classifier._cerebus_api_key_cache`),
`induction._DATA_START`/`_DATA_END` (the other half of that test), and
`schema.build_batch_model`. These are the only names this spec's ADR needs to add.

`classifier.FailureKind`/`classify_failure`/`TruncatedResponseError`/`PolicyRefusalError`/
`EmptyResponseError`/`_DROPPABLE_PARAMS` and `Classifier._complete`/`_attempt_completion` are
**already** accepted INV-7 exceptions as of spec 6's `ADR.md` (ADR-001) — including the
correction of INV-7's former claim that "`Classifier` itself is still tested only via the
public API", which spec 6 already fixed. This spec's own ADR must **not** re-request or
re-amend any of these; doing so would create a second, redundant amendment record for a
permission that already exists. It cites spec 6's amendment rather than duplicating it.

This extends INV-7 and must be recorded in the implementation's ADR alongside AR-1.1's INV-1
amendment, following the precedent of `spec/4-multi-model-classification/ADR.md` ADR-002.

---

## Data Requirements

| Artifact | Change |
|---|---|
| `run_config.json` | `schema_version` 4 → 5. New `batch` object per FR-4.1 (`null` when the run does not classify; `batch.mode` and every observed-outcome field `null` when `--batch` was absent). Present in the initial config literal, so the failure path needs no `setdefault` entry. This is a real shape change, not an additive one, hence the bump. |
| `test_classified.csv` | Unchanged — same columns, same per-row cell format. Batching changes how cells are produced, never the output schema. |
| Batch payload / response | Internal wire format only (FR-2.1/FR-2.2); never persisted to disk. Unlike induction, there is no `sampled_examples.json`-style snapshot of batch composition — FR-4.1's summary statistics are the audit record. |

## Integration Points

- **`pipeline.classify_csv`** — gains `batch`-related parameters and a stats object, submits
  batches instead of rows, and applies per-row outcomes one at a time (FR-3.1). This is the
  first modification to `classify_csv`'s loop structure since spec 4 added `--models`; the
  no-`--batch` path must remain the existing code path. Its return type is unchanged.
- **`classifier.Classifier`** — gains an optional `max_tokens` parameter, added to
  `__init__`/`_completion_kwargs` and prepended to `_DROPPABLE_PARAMS` (FR-2.5). `max_tokens` is
  unset by default so no existing call site changes behavior. The `finish_reason` inspection,
  its dedicated truncation exception, and the ordered parameter-drop mechanism itself are spec
  6's, already shipped and unmodified by this spec — see FR-2.5's note on what is and isn't new
  work here.
- **`experiment.py`** — new flags on `classify`/`run`, the `--batch` resolver, the
  `_SCHEMA_VERSION` bump, the `batch` config block, the batch prompt and classifier factory it
  now builds, and the stats object it creates. Its `classify_csv` call site
  (`experiment.py:1138`) gains the batch arguments.
- **`cli.py`** — the same new flags, validation, prompt/factory construction, and call-site
  arguments.
- **`debate.py` / `multi_model.py`** — **not touched.** FR-3.3 excludes both modes from
  batching, so neither module changes.
- **Cerebus gateway (spec 3)** — the `@workspace/model` slug format is exactly what makes
  FR-1.2's stripping walk necessary; `DEFAULT_LLM_PROVIDER=cerebus` is the repo's normal
  configuration, so the walk is the common path, and FR-2.4's credential carry-over is what
  keeps batched runs working at all under it.

## Related Specs

| Spec | Relationship | Affected Requirements |
|------|-------------|---------------------|
| Spec 1: initial-classification-with-critics | **References** — `--critics` is explicitly excluded from batching; `debate.py` is untouched | FR-3.3 |
| Spec 2: experiment-runner | **Extends** — adds `--batch` to the `classify`/`run` subcommands and bumps `run_config.json`'s `schema_version`; **References** — reuses the positional-schema and strict-mode findings from its induction redesign, the `<<<DATA>>>` payload convention, and its one-authority-resolver pattern | FR-1.1, FR-4.1, AR-2.1, AR-2.3 |
| Spec 3: cerebus-gateway | **References** — the gateway's `@workspace/model` slug format drives the token-budget resolution walk, and its credential/header fields are what FR-2.4 must carry over | FR-1.2, FR-2.4 |
| Spec 4: multi-model-classification | **References** — `--models`/`--n-classifiers` are explicitly excluded, mirroring that spec's own `--models`/`--critics` exclusion | FR-3.3 |
| Spec 6: classifier-error-path | **Depends on** — this spec's bisection policy (FR-3.2) and truncation/parameter-drop mechanics (FR-2.5) are built entirely on spec 6's already-shipped `classify_failure`/`FailureKind`/`TruncatedResponseError`/`_DROPPABLE_PARAMS`, rather than re-deriving equivalent logic; spec 6's INV-7 amendment already covers those classifier-side names, but this spec still needs its own INV-7 amendment for `batching`/`induction`'s delimiter constants/`schema.build_batch_model` (AR-5.1) | FR-2.5, FR-3.2, AR-5.1 |

## Constraints

- The no-`--batch` path must be behaviorally identical to today, including call counts, flush
  cadence, failure counting, unexpected-field handling, and output CSV bytes. This is what
  forbids tightening `build_classification_model` globally (FR-2.1) and what requires
  `Classifier.max_tokens` to default to unset (FR-2.5).
- No new runtime dependency. Everything needed (`litellm.get_model_info`,
  `litellm.token_counter`, `litellm.suppress_debug_info`, `litellm.get_supported_openai_params`,
  `pydantic.create_model`, `ConfigDict`) exists in the installed stack and was verified live.
- No live provider call in any test (project rule, `AGENTS.md`).
- INV-6 (main-thread-only DataFrame mutation), INV-8 (category name uniqueness / column
  collision), and INV-11 (the schema enforces cardinality and type only, never label validity)
  are all unchanged and must stay satisfied.
- Token counting is an estimate of the provider's own tokenizer, not a guarantee.
  `litellm.token_counter` falls back to a default tokenizer for unrecognized model ids
  (verified: it returns a count rather than raising). The input side is therefore measured on
  the rendered payload and still treated as approximate; the output side is a sizing estimate
  only, never the request cap (FR-1.6, FR-2.5).
- Reasoning tokens count against real output while being invisible to FR-1.6's label-based
  estimate. This repo actively targets reasoning models — spec 3's temperature-drop handling,
  now generalized by spec 6 into `classifier.py`'s `_DROPPABLE_PARAMS` ladder, exists because
  `gpt-5.4-mini` variants reject an explicit temperature while reasoning. This is precisely why
  FR-2.5's cap is the *output budget* rather than the estimate: a tight cap would sit below the
  reasoning floor and truncate every call.
- Peak token throughput scales with `workers x batch size`. With the default `--workers 8` and
  the default arity cap of 50, up to ~400 rows' worth of tokens are in flight at once — a real
  increase over per-row mode, bounded by FR-1.5. Rate-limit errors are retried at batch size
  with a bounded backoff rather than bisected (FR-3.2) precisely so throttling does not
  amplify itself.

## Out of Scope

- **Batching `--critics`** — excluded by FR-3.3, having been specified and then deliberately
  cut. Three reasons: `CriticVerdict.argument` and `ReconcilerVerdict.reasoning` are free-text
  `str` fields with no configured maximum, so not even FR-1.6's estimate-with-a-safety-factor
  approach applies to them; `run_debate`'s post-sampling tally/escalation block is inline
  (`debate.py:191-244`) rather than a reusable unit, so a batched sampling stage would require
  refactoring it or duplicating the vote logic; and batching only the sampling stage would
  serialize up to `--batch-max-size` rows' Critic/Reconciler calls inside a single batch
  worker, where per-row mode spreads them across all `--workers` — plausibly a wall-clock
  *regression* despite the call-count reduction. Revisiting this needs its own spec.
- **Batching `--models`** — excluded by FR-3.3. Batching N rows across M models multiplies
  arity and complicates the three audit columns plurality voting depends on
  (`{category}_by_model`, `{category}_votes`, `{category}_model_errors`).
- **Bounding label values to the configured set** — this would make FR-1.6's output figure a
  true upper bound, but it changes INV-11 and the `"none - <new label>"` invention convention
  for the unbatched path too. A separate spec, not a side effect of batching.
- **Batching the induction call** — already a single call covering all labels (spec 2); nothing
  to batch.
- **Per-batch composition snapshots** — FR-4.1 records summary statistics only. A full
  per-iteration log of which rows shared a prompt would make batch effects fully
  reconstructible, but is not justified at 29.6k-row scale.
- **A token-weighted in-flight concurrency limiter** — considered and excluded; FR-1.5's arity
  cap plus FR-3.2's rate-limit backoff bound the exposure with far less machinery.
- **Making batched and unbatched runs comparable** — out of scope by decision; FR-4.2 records
  and surfaces the caveat rather than attempting to eliminate it. Seeded intra-batch shuffling
  to decorrelate position bias from dataset order was considered and deliberately excluded.
- **Provider-side batch APIs** (OpenAI Batch API and equivalents) — a different mechanism with
  different latency and polling semantics; this spec is about packing a single synchronous call.

## Spec Completeness Checklist

- [x] **Scope & acceptance criteria** — three sizing modes plus a hard arity cap defined
  (FR-1.1/FR-1.5/FR-1.6/FR-1.7), scope narrowed to plain classification only (FR-3.3), both
  mode exclusions explicit with reasons (FR-3.3, Out of Scope); every FR carries a **Verify:**
  line.
- [x] **Testing strategy** — FR-5.1 requires a `pytest` suite shipped alongside the code, names
  the runner, maps each FR's **Verify:** condition to a test, and enumerates thirteen specific
  error/edge cases including concurrency races, non-divisor save thresholds, the ordered
  parameter drop, asymmetric budget metadata, nested extras, and the raising-future accounting
  case — explicitly scoped to what this spec adds, not re-testing spec 6's own truncation/
  classification test suite; AR-5.1 records the (now much smaller) INV-7 extension this spec
  still needs beyond what spec 6 already granted.
- [x] **Existing patterns** — the positional-schema design is lifted from spec 2's
  `build_induction_model` (AR-2.1), payload delimiting from `induction.py` (FR-2.2, AR-2.3),
  the one-authority resolver from spec 2's `_resolve_induction_sizing` (FR-1.1), the
  conditional-kwarg pattern from `_completion_kwargs`'s existing optional params and the
  ordered-drop ladder spec 6 already generalized, onto which this spec only prepends one entry
  (FR-2.5), the
  mode-exclusion pattern from spec 4 (FR-3.3), the `None`-when-not-classifying pattern from
  `classifier_config` (FR-4.1), the env-var plumbing style from spec 2's `INDUCTION_EXAMPLES`
  (FR-4.3), and the new-module + ADR-amending-INV-1 pattern from `multi_model.py` (AR-1.1).
- [x] **Dependencies** — none added; every API relied on was verified live against the installed
  litellm/pydantic (Constraints).
- [x] **Architecture & interfaces** — new `batching.py` with explicit import boundaries, all
  three inbound edges enumerated, and a required ADR that must edit INV-1's and the Boundary
  Map's existing closed per-module lists (AR-1.1); `build_batch_model` placed in `schema.py`
  (AR-2.2); the classifier factory and its carried settings specified, including `max_tokens`
  (FR-2.4); the stats transport specified (FR-4.1); `Classifier`'s one remaining change
  (`max_tokens` plus the one-entry ladder prepend) specified against spec 6's already-shipped
  primitives, which this spec explicitly does not re-derive (FR-2.5); AR-5.1 identifies exactly
  what spec 6 already granted and what little this spec still needs; all touched call sites
  enumerated, and `debate.py`/`multi_model.py` explicitly named as untouched (Integration
  Points).
- [x] **Error handling & failure modes** — FR-1.3 (unresolvable input budget, exact message
  plus diagnostic line; unresolvable output budget proceeds with a warning), FR-1.6/FR-1.7
  (oversized single row, floor of 1, livelock prevention), FR-2.5 (`max_tokens` conditionally
  sent, dropped first on the ladder — truncation raising/classification is spec 6's, inherited
  as-is), FR-3.1 (raising batch future preserves the accounting invariant), FR-3.2 (bisection
  policy derived cleanly from `classify_failure`'s two axes, with `ContextWindowExceededError`
  now confirmed reachable and bisectable rather than swallowed by the old unguarded fallback,
  bounded rate-limit retry with a defined exhaustion outcome, bisection bounded by the arity
  cap), FR-4.1 (failed-run `batch` key via null initialization), AR-1.3 (bare `Exception`).
- [x] **Security review** — FR-2.2 identifies the genuinely new risk class batching introduces:
  cross-row influence, where one row's untrusted text can affect *other* rows' labels. Its claim
  is scoped precisely to what is actually guaranteed — payload integrity via `json.dumps`, which
  blocks structural forgery of delimiters, positions, and neighboring entries — and it states
  plainly that semantic isolation is **not** provided and that FR-2.1's identifier-free design
  makes such influence undetectable after the fact. That residual risk is accepted and surfaced
  by FR-4.2. The Verify line tests the attribution property, not just a `json.dumps` round-trip.
  FR-4.1 keeps the existing rule that no URL, key, or header value is recorded. No authn/authz
  surface is added.
- [x] **Performance impact** — the point of the spec: call count drops by roughly the batch
  factor, 98% at the default cap on a 29.6k-row split (FR-1.5); concurrency preserved with
  `--workers` re-scoped to batches (FR-3.1); peak `workers x batch size` token throughput
  quantified and bounded (Constraints); bisection amplification bounded at 99 `classify` calls
  per batch by the arity cap, with the higher provider-request ceiling stated, and kept off
  rate-limit errors (FR-3.2); `run_config` records realized call count and arity range so the
  actual saving is measurable (FR-4.1). The `--critics` wall-clock regression risk was the
  deciding reason that mode is out of scope.
- [x] **Rollout & migration** — the flag is opt-in and absent by default, the no-`--batch` path
  is required to be unchanged (Constraints), and the experiment scripts stay unbatched by
  default so committed accuracy configurations do not change meaning (FR-4.3). The one breaking
  edge is `run_config.json`'s `schema_version` 4 → 5 (Data Requirements); older run directories
  are not migrated, matching how spec 2's 3 → 4 bump was handled.
- [x] **Assumptions & risks** — the output figure is stated as a sizing estimate rather than a
  bound, with INV-11 and reasoning tokens cited as the two reasons, and is deliberately not used
  as the request cap (FR-1.6, FR-2.5, Constraints); token counts are approximate and the input
  side is measured on the rendered payload (Constraints); the chunk-down walk is best-effort and
  can land on a different-window sibling, guarded by the arity cap and diagnosable via FR-4.1's
  recorded candidate (FR-1.2); batched accuracy is not comparable to unbatched and is recorded
  rather than fixed (FR-4.2); batch composition is not reconstructible after the fact, so an
  anomalous row's label can never be traced to who shared its prompt (Out of Scope, FR-3.4).

---

## Change Log

### Update from critique-consolidated-v-1.md (2026-09-15)

Applied all 15 blocking items and every non-blocking improvement except one rejection. Two
user decisions were taken during this round: the arity cap's default (50, raisable via
`--batch-max-size`) and keeping `--workers` as concurrent batches rather than adding a separate
batch-concurrency knob.

**Applied — new requirements:** FR-1.5 (hard maximum arity, default 50 — the token budget alone
admitted ~19,000 rows per batch on the model the experiment scripts use); FR-2.5 (request-level
output cap and truncation detection); FR-3.5 (the batched-sampling interface); AR-2.3 (the
`<<<DATA>>>` duplication made deliberate and test-enforced).

**Applied — corrections to requirements that could not have been satisfied as written:**
FR-1.6's "worst case output" claim was false (verified: a 5,007-character label validates even
with `allow_new_labels=False`, per INV-11), recast as a budgeted estimate; FR-2.3's batch prompt
moved to the caller (verified: `classify_csv` never receives
`system_prompt`/`task_description`/`extra_prompt`); FR-2.4 replaced in-layer classifier
construction with a factory (nothing previously carried `api_key`/`extra_headers`, which would
have failed every run under `DEFAULT_LLM_PROVIDER=cerebus`); FR-2.1 required extras forbidden at
both levels (verified: outer `extra="forbid"` does not protect the nested model); FR-3.1 applies
results row-at-a-time (measured: the existing modulo fires 2 times instead of ~10 at batch size
7 or 32); FR-3.2 classifies failures before bisecting; FR-4.1 gained a stats transport;
FR-1.2/FR-1.3 resolve the two budgets independently; FR-1.7 gained the floor of 1; FR-1.1
specified `args.batch`'s resolved shape; FR-1.6 measures the rendered payload rather than an
additive sum.

**Applied — non-blocking:** FR-2.2's security claim narrowed to payload integrity with its
Verify line replaced (the original tested the standard library and would have passed
unconditionally); `workers x batch size` throughput quantified; reasoning tokens noted; INV-11
restated as unchanged; `--batch 1` declared legal and purposeful; batch-composition
unreconstructibility moved into Assumptions; the "would fail identically in the unbatched path"
claim dropped; FR-1.2's larger-window-sibling risk stated; "bounding label values" added to Out
of Scope.

**Applied — with the critique's reasoning corrected:** FR-4.3's `${BATCH-}` form. The finding
that the justification was bogus is right, but for a different reason than stated: verified that
`${BATCH-}` and `${BATCH:-}` are provably identical when the default is empty, so the original
rationale described a protection that cannot apply. Changed to `${BATCH:-}`.

**Rejected:** the claim that `pipeline.py` importing `batching.py` creates a forbidden
transitive dependency on `schema.py`/`prompts.py`. Dismissed — `pipeline.py` already imports
`debate.py`, which imports both, so that path exists in shipped code today and INV-1 governs
direct imports only.

### Update from critique-consolidated-v-2.md (2026-09-15)

The v-2 round confirmed 11 of v-1's 15 items as solidly resolved but found 14 new blocking
items, concentrated in the four requirements v-1 had introduced and which had therefore never
been critiqued. Two user decisions shaped this round: **drop `--critics` batching entirely**,
and **fix FR-2.5**.

**Applied — scope reduction (removes v-2 blocking items 6, 7, 8):**
- **`--critics` batching cut.** FR-3.4 (batched sampling behavior) and FR-3.5 (the
  `run_debate_batch` interface) are deleted; the old FR-3.6 becomes FR-3.4. FR-3.3 now excludes
  `--critics` alongside `--models`. The three reasons are recorded in Out of Scope: the
  Critic/Reconciler free-text fields have no configured maximum; `run_debate`'s post-sampling
  tally/escalation is inline at `debate.py:191-244` rather than a reusable unit, so batched
  sampling needed an unspecified refactor; and serializing up to 50 rows' Critic/Reconciler
  calls inside one batch worker was plausibly a wall-clock *regression* against per-row mode's
  spread across `--workers`. Consequent edits: AR-1.1 drops the `batching.py --> debate.py`
  edge, Integration Points names `debate.py`/`multi_model.py` as untouched, Related Specs
  changes Spec 1 from **Modifies** to **References**, FR-5.1 drops its `tests/test_debate.py`
  additions, Constraints drops the `sampling_runs` compounding, and FR-4.3 now notes that the
  experiment scripts pass `--critics` today so `BATCH` requires dropping it.

**Applied — FR-2.5 redesigned (dissolves v-2 blocking items 1, 3, 4, 5 and specifies 2):**
- **The cap is now the resolved output budget, not FR-1.6's estimate.** This was the round's
  most serious finding and was self-inflicted: Constraints already stated that reasoning tokens
  are invisible to the label-based estimate, and FR-2.5 then used that same estimate as a hard
  cap. On a reasoning model the cap could sit below the reasoning floor, making truncation
  deterministic at every arity — bisection to size 1, size 1 truncates too, every row fails, and
  the run exits `"completed"`. Measured: ~38 tokens at the bisection endpoint for ag-news.
  Separating the two roles (estimate sizes the batch, budget caps the request) removes the
  failure mode by construction and gives the unknown-output-budget path an obvious answer —
  send no cap, which is exactly today's unbatched request shape.
- **Truncation is raised, not returned.** `_attempt_completion` discards `finish_reason` at
  `classifier.py:317`, so the old wording was unreachable. It now inspects `finish_reason` and
  raises a dedicated exception, keeping the `-> str` contract, and `classify` must let it
  propagate rather than retrying a deterministic failure three times with sleeps.
- **An unsupported `max_tokens` is dropped in a defined order.** The existing
  `UnsupportedParamsError` branch drops only `temperature`; adding `max_tokens` without
  touching it would have meant either no fallback, or dropping *temperature* instead and
  misreporting the cause — reintroducing the pattern `classifier.py:295-305`'s comment exists
  to prevent. The branch now drops from `("max_tokens", "temperature")`, at most one per
  attempt, never repeating — which also makes the `max_tokens` vs `max_completion_tokens`
  question non-load-bearing.

**Applied — remaining blocking items:**
- **FR-3.2's error class renamed** from "size-related" to **isolable**, resolving the
  contradiction between the rule's title and its inclusion of generic Pydantic validation
  failures. The class is now named for what splitting achieves (attributing a failure to
  specific rows) rather than for the failure's cause.
- **FR-3.2's rate-limit retry bounded:** at most 2 further attempts at the original size, with
  exponential backoff from `retry_delay`, explicitly after and independent of
  `Classifier.max_retries`, and every row failed on exhaustion. The previous wording admitted a
  retry-forever reading that could hang a run under sustained throttling.
- **FR-3.1 defines the raising-future path:** caught and converted to a per-row failure for
  every covered row, preserving `classified + failed == completed == total` and letting the
  progress bar reach 100%.
- **FR-1.3's self-contradiction fixed:** it required the message to be *exactly* the quoted
  string *and* to name the missing side, which that string cannot do. The side is now named on a
  second line. FR-1.3 was also narrowed to make only the **input** budget fatal under dynamic
  mode — with the cap no longer derived from the output estimate, an unknown output budget just
  means no runaway guard, which is today's behavior.
- **FR-1.5's enforcement point settled:** the cap bounds growth *during* accumulation, not as a
  post-hoc clamp — the difference between running `token_counter` over a 50-row payload and a
  19,000-row one. FR-1.7's "subject to FR-1.5's cap" was deleted as dead text, since an
  over-cap `--batch INT` is rejected at validation time.
- **FR-4.1's `setdefault` requirement reverted.** This was a v-1 **false positive** that two
  critics asserted and this spec applied, creating a contradiction with FR-4.1's own
  "always present" sentence. Verified by extracting both key sets: the failure path's
  `setdefault` list contains exactly the eight keys assigned inside the `try` block and zero
  init-time keys — `classifier_config` and `cerebus` are both init-time and both correctly
  absent. The real requirement was null-initializing `batch`'s observed-outcome fields in the
  literal, which is now stated; the Verify line was already correct and is kept.

**Applied — non-blocking:** AR-5.1 now authorizes the module-private delimiter constants
AR-2.3's test needs (INV-7 precedent: `classifier._cerebus_api_key_cache`) plus the new
truncation exception type; FR-1.5 rejects `--batch-max-size` without `--batch`, matching
FR-1.4; FR-3.2's "99 calls" figure is now labelled as `classify` invocations with the higher
provider-request ceiling stated; FR-2.4's cache key simplified from `(role, arity)` to `arity`,
since dropping `--critics` leaves exactly one classifier role.

**Reorganized:**
- Feature 3 shrank from 6 FRs to 4 (FR-3.4 and FR-3.5 deleted; old FR-3.6 → FR-3.4).
- Totals: 22 → 20 FRs, 7 ARs unchanged. All cross-references updated.

### Update from plan-critique-consolidated-v-1.md (2026-09-15)

The plan-critique round found five **spec**-level defects (alongside ten plan-level ones fixed
in `plan.md`). All five are applied here. Three were found by the external critics against my
own judgment, and two of those I had originally classified as harmless.

**Applied — contradictory acceptance criteria:**
- **FR-2.5: the truncation raise is now gated on `max_tokens` having been sent.**
  `_attempt_completion` is shared with every unbatched call, so an unconditional raise would
  surface a length-truncated *unbatched* response as a hard error where today `classify`
  retries it — changing the no-`--batch` path's failure counting, which this spec's own
  Constraints forbid. The gate confines the new semantics to calls that asked for a cap. FR-5.1
  gains an explicitly-named regression test for the uncapped path, because a
  `Classifier.classify` call count cannot detect a behavior change *inside*
  `_attempt_completion` — the originally-planned no-batch check would have passed while the
  regression shipped.
- **FR-3.2: the context-length clause is removed as unreachable.** Verified that
  `litellm.ContextWindowExceededError` is a `BadRequestError` subclass, so
  `classifier.py:312`'s `except litellm.BadRequestError` catches it first and retries in
  JSON-object mode — it never reaches the batching layer as itself. An oversized batch still
  surfaces as a truncation or validation failure, both already isolable, so nothing is lost.

**Applied — incomplete requirements:**
- **AR-1.1 now enumerates all three inbound edges** (`pipeline.py`, `cli.py`,
  `experiment.py`) and states that the ADR must edit INV-1's and the Module Boundary Map's
  existing rows, since both are closed per-module lists. I had classified this as a
  plan-level deviation with "widens no permission" reasoning; two critics independently
  showed that was a misreading — the permission question is separate from the enumeration,
  and leaving one edge listed would leave the invariant contradicting the shipped graph.
- **AR-5.1 now covers `Classifier._complete`/`_attempt_completion`** and requires the ADR to
  **correct** INV-7's claim that "`Classifier` itself is still tested only via the public
  API". That claim is already false: `tests/test_cerebus.py:361/:381/:397` call
  `clf._complete(...)` today, added by spec 3 without amending the sentence.
- **FR-4.3 gains a `CRITICS` env-var** so the `BATCH` override is actually usable. All three
  scripts hardcode `--critics` at `run_experiment.sh:74`, which FR-3.3 rejects with
  `--batch` — so `BATCH` alone would fail every real run while the `DRY_RUN` check passed,
  since `DRY_RUN` never invokes the CLI. `BATCH` without `CRITICS=0` is now rejected by the
  script rather than silently dropping `--critics`: these scripts produce the compared accuracy
  numbers, and silently switching the experimental condition is the error class FR-4.2 exists
  to prevent. `CRITICS` defaults to on, so the committed configuration is unchanged.

**Applied — non-blocking:**
- AR-1.2 now states that `litellm.suppress_debug_info` is set once and deliberately not
  restored: a non-batched run never reaches resolution, so the flag is already scoped by
  construction, and a `try`/`finally` restore would race across worker threads.

**Rejected:** none. Every plan-critique finding classified as spec-level was verified against
the code and accepted.

**Reorganized:** no requirements added, removed, or renumbered — all five fixes are edits
within existing FR/AR bodies. Totals remain 20 FRs and 7 ARs.

### Update from plan-critique-consolidated-v-1.md, reconciled against spec 6 (2026-09-15)

Applied both items the consolidated critique flagged as requiring `/spec-update`, plus a much
larger reconciliation the critique could not have anticipated: **spec 6
(classifier-error-path) was written, critiqued through two rounds, implemented, and merged to
`main` between this spec's last update and this one.** It shipped, as real code, almost exactly
the failure-classification foundation this spec's Feature 2/3 had been independently
(re-)deriving from scratch. Left unreconciled, this spec would have specified requirements that
actively contradict the merged codebase — including asking an implementer to build a truncation
exception, a parameter-drop ladder, and an INV-7 amendment that already exist.

**Applied — the two originally-flagged items:**
- **FR-3.2's context-length clause is restored**, and upgraded from "isolable" to explicitly
  reachable and correctly bisectable: spec 6's `_attempt_completion` now excludes
  `ContextWindowExceededError` from its JSON-mode fallback entirely (verified in the merged
  code), so it propagates as itself on the very first attempt — no longer merely "not swallowed
  by an unguarded fallback" as the critique's fix would have required, but structurally
  guaranteed never to reach the fallback at all.
- **FR-4.3's Verify line no longer claims byte-identical `DRY_RUN` argv.** Verified against the
  scripts: `RUN_DIR` embeds `$(date +%Y%m%d-%H%M%S)`, so two separate invocations can never
  produce byte-identical output. The Verify line now requires identity after normalizing that
  one field.

**Applied — reconciliation against spec 6's shipped implementation:**
- **FR-2.5 rewritten.** Its truncation-detection and gating design is now inert: spec 6 already
  ships `TruncatedResponseError`, raised **unconditionally** on `finish_reason == "length"`
  (never gated on whether a cap was sent) and classified `(retryable=False, isolable=True)` by
  `classify_failure`. This spec no longer builds or gates a truncation exception — it inherits
  spec 6's universal behavior as-is. Its parameter-drop-ladder design is likewise now just an
  append: spec 6's `_DROPPABLE_PARAMS` is already a generic, ordered, module-level tuple this
  spec extends with `"max_tokens"` (prepended, before `"temperature"`) rather than building the
  ladder mechanism itself. The only genuinely new `Classifier` work left is adding the
  `max_tokens` constructor parameter and conditional-kwarg emission.
- **FR-3.2 rewritten** to classify failures via spec 6's `classify_failure` directly, deriving
  its three-way bisection policy from `FailureKind`'s two axes (`isolable` → bisect;
  `retryable` alone → retry at original size; neither → fail immediately) instead of
  re-enumerating litellm exception types. This closes a real risk, not just a redundancy: an
  earlier draft of this spec's own hand-rolled taxonomy was independently caught getting the
  litellm hierarchy wrong **twice** across spec 6's own two critique rounds (folding `Timeout`
  under `APIConnectionError`; missing that `BudgetExceededError` doesn't derive from
  `APIError`). Reusing spec 6's already-corrected classification removes this spec's exposure
  to the same class of mistake a third time.
- **AR-5.1 rewritten.** Spec 6's own `ADR.md` (ADR-001) already amended INV-7 for
  `FailureKind`/`classify_failure`/the three response-failure exception types/
  `_DROPPABLE_PARAMS`/`Classifier._complete`/`_attempt_completion`, including the exact INV-7
  text correction this spec's prior draft asked for a second time. This spec's own ADR now adds
  only what it genuinely introduces (`batching.py`, the two `_DATA_START`/`_DATA_END` pairs,
  `schema.build_batch_model`) and explicitly does not re-request or re-amend anything spec 6
  already granted, to avoid a duplicate, conflicting amendment record.
- **A stale line-number citation corrected**: the reasoning-model/temperature-drop comment
  Constraints cited at `classifier.py:294-311` now lives elsewhere after spec 6's rewrite;
  that line range is now `_classify_failure_verbose`'s docstring. The citation is now described
  by name (`_DROPPABLE_PARAMS`) rather than by a line number that will keep drifting.
- **Related Specs gains a row for spec 6**, relationship **Depends on** — the first spec in
  this project to depend on spec 6 rather than merely referencing it — and the Overview states
  the dependency plainly: spec 6 must be merged before this spec is implemented. (It already
  has been, as of this update.)
- **FR-5.1's test enumeration corrected** to stop asking for tests of behavior spec 6 already
  owns and already tests (the gated-truncation regression test, the no-cap-when-unknown-budget
  case under the old framing) and to add the one test this reconciliation newly requires: that
  `ContextWindowExceededError` bisects all the way to size 1 with no fallback attempt observed
  in between, proving reachability rather than only classification.
- Several checklist notes (Testing strategy, Existing patterns, Architecture & interfaces,
  Error handling & failure modes) updated to match the above rather than describing designs
  that no longer correspond to any task this spec's implementation would need to perform.

**Rejected:** none. Every reconciliation was a correction against verified, already-merged code,
not a new product decision.

**Reorganized:** no FRs or ARs added, removed, or renumbered — totals remain 20 FRs and 7 ARs.
The rewrites are substantial in content (FR-2.5, FR-3.2, AR-5.1) but preserve every requirement's
identity and position.
