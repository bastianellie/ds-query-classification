# Spec 4: Multi-Model Voting Classification

## Overview

Adds a new `--models` classification mode to `classify.py` (`cli.py`) and `experiment.py`'s
`classify`/`run` subcommands: a standalone alternative to `--critics` where 2 or more
distinct models each classify the same row once, independently, and their answers are
merged into a single result per category by plurality vote. It reuses `--critics`'
audit-column and vote-tallying conventions (`debate.py`) without invoking any of its
self-consistency sampling or Critic/Reconciler debate machinery.

## Goals

- Let a user compare/ensemble several distinct models against the same test set in one
  run, without paying for `--critics`' sampling + debate overhead.
- Keep the merge auditable: every model's own raw answer, the vote tally, and any
  per-model failures are recorded alongside the final merged answer.
- Reuse existing categories/schema/prompt/restore/incremental-save infrastructure — this
  is a lightweight sibling of `--critics`, not a fork of the pipeline.

---

## Feature 1: Multi-model voting classification mode

**Who & why:** A data scientist wants to know whether several distinct frontier models
agree on a classification task, or ensembling them beats any single model — for example,
running the AG News experiment with `--models gpt-5.4-mini gpt-4o-mini gemini-2.5-pro`
instead of picking one model, to see where they agree/disagree per row. `--critics` isn't
the right tool for this: its self-consistency sampling and Critic/Reconciler debate are
built around resolving *one* model's uncertainty, not comparing genuinely different
models, and multiply LLM calls even further than a plain N-model vote would.

### Functional Requirements

#### FR-1.1: `--models` flag — 2+ distinct models, validated before any LLM call
A new `--models` flag (`nargs="+"`) accepts 2 or more space-separated, non-empty, **unique**
LiteLLM model ids, e.g. `--models azure/gpt-5-chat gpt-4o-mini gemini/gemini-2.5-pro`. It is
added to the same shared classification-flags group that already carries `--critics` on
both `cli.py` and `experiment.py`'s `classify`/`run` subcommands (not `induce`, which has
no classification role). All of the following are validation errors raised before any LLM
call, in the same style as the existing
`"--hf-dataset and --train-file/--test-file are mutually exclusive"` check
(`experiment.py:272-275`) and the conditional-range checks for `--consensus-threshold`
(`experiment.py:313-317`, `cli.py:207-215`):
- `--models` together with `--critics`.
- `--models` with fewer than 2 values, or any empty/whitespace-only value.
- `--models` with a duplicate id (each id must be distinct — see Constraints; this
  supersedes any notion of intentionally repeating an id to "weight" it).
- `--models` together with `CEREBUS_MODE=azure` (see AR-1.6 for why).

Every other CLI flag that only makes sense under `--critics` (`--sampling-runs`,
`--sampling-temperature`, `--consensus-threshold`, `--critic-model`, `--reconciler-model`)
is silently unused when `--models` is given without `--critics` — exactly like today's
existing behavior when `--critics` is already `False`. No new validation is added for
these; they simply have no effect.
**Verify:** `--models a b` and `--models a b c` both parse successfully on `classify.py`,
`experiment.py classify`, and `experiment.py run`; `--models a` (one value),
`--models a b --critics`, `--models a a b` (duplicate), `--models a ""` (empty value), and
`--models a b` combined with `CEREBUS_MODE=azure` each exit non-zero with a clear message,
before any network call, mirroring
`test_cli_invalid_sampling_temperature_exits_before_any_llm_call`'s test pattern;
`experiment.py induce --help` does not list `--models`.

#### FR-1.2: Independent single classification per model — no sampling, identical shared config
Each of the N models classifies the row's text exactly once — one logical
`Classifier.classify(...)` invocation per model per row, not necessarily one network
request (that single logical call may still internally retry or fall back to JSON mode,
exactly as it does today). There is no `--sampling-runs`-style repeated sampling of any
individual model. All N models share the identical system prompt, category schema, and
temperature (`temperature=None`, i.e. each model's own provider default — the same value
plain single-model mode already uses today; not `--critics`' `--sampling-temperature`
default). `--allow-new-labels` is **honored** for every model in `--models` mode on both
entry points, exactly as it already is under `--critics` — this is a deliberate carve-out
from `cli.py`'s existing plain-mode behavior, which instead *forces* label invention on
regardless of the flag (`cli.py:221`); `--models` is a `--critics` sibling, not a
plain-mode variant, so it follows `--critics`' convention here. No per-model
prompt/temperature override exists in this spec (see Out of Scope).
**Verify:** a test that mocks/counts `Classifier.classify` confirms exactly 3 calls per
row (one per model, no repeats) when `--models a b c` is used; a `--models`-mode
classifier's `system_prompt` mentions label invention only when `--allow-new-labels` is
passed, on both `classify.py` and `experiment.py`, regardless of `--critics`.

#### FR-1.3: Plurality vote merge, ties broken by `--models` list order
For each category, the final label list written to the `{category}` output column is the
full ordered label list from whichever model's *top* label (the first element of its 1-3
list) wins a plurality vote across the N models — the same "vote on the top label, use
the winning model's full list as the answer" convention `debate.py`'s consensus step
already uses (`debate.py:194-207`), just voting across distinct models instead of samples
of one model. There is no consensus threshold or escalation: whichever label gets the
most votes wins, even without a strict majority. Votes are tallied by each model's
**position in the `--models` list**, not by which model's LLM call happens to complete
first — this is what makes the tie-break deterministic under concurrent execution (see
AR-1.5 for the mechanism). A tie for the most-voted label is broken in favor of whichever
tied label the earliest-listed model picked. `"none"`/`"none - <suggestion>"` labels are
bucketed for voting using the same `_vote_bucket` merge convention `debate.py` already
uses (`debate.py:35-47`) when `--allow-new-labels` is set.
**Verify:** with `--models a b c` where `a`/`b` agree on a category's top label and `c`
disagrees, the `a`/`b` label is the final answer; with `--models a b` (2 models)
disagreeing entirely on a category, `a`'s (first-listed) full label list is the final
answer, **regardless of which model's mocked call is made to complete first** (a test
should assert this by completing `b`'s call before `a`'s and confirming the result is
unchanged).

#### FR-1.4: Per-model and vote-tally audit columns
Two new audit columns are added per category — following the existing "add only if not
already present" convention (`pipeline.py:22-25,126-131`): `{category}_votes` — a JSON
object mapping each voted-on label (or none-bucket) to its vote count across the N models,
in the same shape as `--critics`' existing `{category}_votes` column; and
`{category}_by_model` — a JSON object mapping each supplied model id to that model's own
validated, parsed 1-3 label list for that category (the same value `Classifier.classify`
returns for that field — not raw provider completion text), so a user can see exactly
where the models agreed or disagreed, not just the final tally. If a model's response for
the row fails schema validation for *any* category (the classification schema validates
all categories as one atomic unit — see `schema.py:17-63`), that model is absent from
`{category}_by_model` and the vote tally for **every** category in that row, not just the
category that failed validation — this is the same "whole-row" failure granularity as
FR-1.6.
**Verify:** after classifying a row with `--models a b c`, `{category}_by_model` parses as
JSON with exactly the 3 model ids as keys, and `{category}_votes` parses as JSON whose
counts sum to the number of models that succeeded for that row; a model whose response
fails schema validation for one category is absent from `{category}_by_model` for every
category in that row, not just the failing one.

#### FR-1.5: `--restore` integrates the new audit columns (`classify.py`/library only)
This requirement applies to `classify.py` and direct `classify_csv` library usage —
`experiment.py` has no `--restore` flag today and is unaffected. Under `--models`, a row
counts as "already classified" for `--restore` purposes only when `{category}`,
`{category}_votes`, `{category}_by_model`, and `{category}_model_errors` (FR-1.7) are all
non-null for every category — mirroring the existing critics-mode rule that also requires
`{category}_votes` non-null in addition to `{category}` itself.
**Verify:** a partially-written CSV with `{category}` filled but `{category}_by_model`
empty is re-classified (not skipped) on `--restore`, matching the existing critics-mode
test pattern (`test_debate.py:364-385`'s equivalent for `_votes`); a separate-output
`--restore` (`test_debate.py:388-443`'s pattern) correctly seeds all four multi-model
audit columns from the prior output file, not just `{category}`.

#### FR-1.6: Partial and total model-failure handling
If one or more (but not all) of the N models fails to classify a row (an LLM/network
error, after that model's own internal retries are exhausted, or a schema-validation
failure per FR-1.4), the row still produces a merged result using only the models that
succeeded: `{category}_by_model` includes only the successful models' keys, and the vote
tally reflects only successful responses. If **every** model fails for a row, the
multi-model orchestration still returns a result rather than raising — a deliberate
divergence from `debate.run_debate`'s precedent (which raises, and the row is silently
discarded, when every sample fails): `{category}` is `None` for every category (no answer
is possible), `{category}_votes`/`{category}_by_model` are empty JSON objects, and
`{category}_model_errors` (FR-1.7) records every one of the N models' sanitized failures.
This keeps a systemic failure (bad credential, quota exhaustion, provider outage) visible
in the output CSV instead of silently vanishing into `pipeline.py`'s in-memory `failed`
counter — `experiment.py`'s existing post-run completeness check (AR-1.3f) already treats
a null `{category}` as an incomplete row, so a total-failure row is still correctly
flagged there.
**Verify:** mocking 1 of 3 models to always raise still produces a merged `{category}`
answer from the other 2, with `{category}_by_model` containing exactly 2 keys; mocking all
3 to raise produces a row where every `{category}` is `None`, `{category}_by_model == "{}"`,
and `{category}_model_errors` contains all 3 model ids with sanitized failure notes — the
row is not silently discarded, and `pipeline.py`'s `classified` counter (not `failed`)
increments for it, since the orchestration itself did not raise.

#### FR-1.7: Sanitized per-model failure audit
A third per-category audit column, `{category}_model_errors`, is added alongside
`{category}_votes`/`{category}_by_model` (same "add only if not present" convention): a
JSON object mapping each model id that failed for that row to a sanitized failure note, in
the exact format `f"{type(exc).__name__}: Classification call failed"` — exception type
plus the fixed literal stage text `"Classification"`, never the raw exception text —
following `debate.py`'s existing `_sanitize_error` convention (`debate.py:66-70`, whose
existing callers use stage values `"Critic"`/`"Reconciler"`; this feature's one and only
stage value is `"Classification"`). Per FR-1.6, this column is always present for every
row `--models` processes, including total-failure rows (an empty JSON object `{}` when no
model failed), so its non-null-ness participates in the FR-1.5 restore-completeness check.
This exists so a systemic failure (bad credential, quota exhaustion, provider outage —
which could affect every model on every row) is distinguishable, from the output alone,
from ordinary cross-model disagreement, in both the partial- and total-failure cases.
**Verify:** mocking one model to always raise a specific exception type produces
`{category}_model_errors == {"<that model id>": "<ExceptionType>: Classification call
failed"}` for every category in that row, and the raw exception message text never
appears in any output cell; mocking all N models to fail (FR-1.6) produces this same
format for every model id, in the same column.

#### FR-1.8: Available on both entry points
`--models` is available on `classify.py` and on `experiment.py`'s `classify`/`run`
subcommands, exactly like `--critics` is today.
**Verify:** `python classify.py --help`, `python experiment.py classify --help`, and
`python experiment.py run --help` all list `--models`.

#### FR-1.9: `run_config.json` records the model list and per-model failure counts
`experiment.py`'s `run_config.json` `models` field gains a new key,
`classification_models` — an ordered list of the model ids from `--models` — which is
always present, and is `null` when `--models` was not used (a single, consistent
absent-state representation, not sometimes omitted and sometimes `null`). The existing
`model` key is `null` when `--models` is active, since no single model id drove the
classification role in that run. A second new key, `model_failure_counts`, is a JSON
object mapping each `--models` id to the number of rows in which that model failed (per
FR-1.6/FR-1.7); it is `null` under the same condition as `classification_models`. This
requires updating `_construct_classifiers`'s return type (currently
`tuple[dict[str, Any], dict[str, str | None]]`, `experiment.py:454`) to allow the `models`
dict's values to also hold a list or nested dict, and updating the `run` subcommand's
existing induction-then-classification merge step (`experiment.py:745-780,844-851`, which
already carries `critic_model`/`reconciler_model` across the two `_construct_classifiers`
calls) to carry `classification_models`/`model_failure_counts` across the same way.
`_SCHEMA_VERSION` (`experiment.py:85`) is bumped from `1` to `2` to reflect this additive
shape change.
**Verify:** a `run_config.json` from a `--models a b` run has `models.model == null`,
`models.classification_models == ["a", "b"]`, and `models.model_failure_counts` present
with keys `"a"`/`"b"`; a run without `--models` has `models.model` holding the single
model id and both new keys `null`; `models.schema_version == 2` (via the top-level
`schema_version` field) in both cases.

#### FR-1.10: Ship unit tests
A new `tests/test_multi_model.py` module ships alongside the new `multi_model.py`,
following `tests/test_debate.py`'s structure and style: vote-tally/tie-break logic tested
in isolation (including the index-order-not-completion-order case from FR-1.3),
audit-column contents tested end-to-end via `classify_csv`, and partial/whole-row failure
handling (FR-1.4/FR-1.6/FR-1.7). `cli.py`-level validation and end-to-end tests (the
`--models`+`--critics`, fewer-than-2, duplicate-id, empty-id, and Cerebus-Azure-mode cases
from FR-1.1) are added to `tests/test_debate.py`, matching where `cli.py`'s existing
`--critics` validation tests already live (there is no `tests/test_cli.py` in this
codebase today). `experiment.py`-level validation, end-to-end, and `run_config.json`
tests (FR-1.5 restore scoping N/A here, FR-1.9) are added to `tests/test_experiment.py`,
matching where `experiment.py`'s existing `--critics` tests already live. Together these
cover every Verify condition in FR-1.1 through FR-1.9.
**Verify:** `pytest` passes, including `tests/test_multi_model.py` and the new tests added
to `tests/test_debate.py`/`tests/test_experiment.py`.

### Architectural Requirements

#### AR-1.1: New `multi_model.py` module, mirroring `debate.py`'s role
A new `src/query_classification/multi_model.py` module holds the vote-tally/tie-break/
audit-column logic (FR-1.3/FR-1.4/FR-1.7), analogous to `debate.py`'s role for `--critics`.
Its allowed imports are `categories.py`, `classifier.py`, `schema.py`, `prompts.py`, and
`debate.py` itself (to reuse `debate.py`'s existing `_vote_bucket` convention,
`debate.py:35-47`, rather than duplicating the none-label merge logic) — it must not
import `pipeline.py`, `cli.py`, or `experiment.py`. **This requires an ADR amending
`spec/ARCHITECTURE.md`'s INV-1** (add `multi_model.py`'s Module Boundary Map row,
including `debate.py` as a new one-directional dependency — `debate.py` itself has zero
outbound imports of `pipeline.py`/`cli.py`/`experiment.py`/`multi_model.py` today, so a
new `debate.py → multi_model.py` edge doesn't create a cycle, even though `pipeline.py`
and `experiment.py` already import `debate.py` themselves; add `multi_model.py` to
`pipeline.py`'s and `experiment.py`'s "May import from" lists), mirroring how
`spec/1-initial-classification-with-critics/ADR.md` and
`spec/2-experiment-runner/ADR.md` already amended INV-1 for `debate.py`/`induction.py`/
`experiment.py`.

#### AR-1.2: `pipeline.classify_csv` gains a `models` parameter; `classifier` becomes nullable (not optional)
`classify_csv`'s existing `classifier` parameter changes type from `Classifier` to
`Classifier | None` — it remains a **required, non-default positional parameter** (its
position and required-ness are unchanged, so existing positional callers, e.g.
`tests/test_debate.py`'s `classify_csv(input_csv, "text", SimpleClassifier(), categories,
...)`, are unaffected, and `categories`'s own lack of a default stays valid Python: giving
`classifier` a *default* value here would raise `SyntaxError: non-default argument
follows default argument` on the very next parameter, `categories`). A caller using
`models=` mode passes `classifier=None` explicitly. A new optional keyword parameter is
added (`models: dict[str, Classifier] | None = None`, keyed by model id — safe now that
FR-1.1 guarantees unique ids), alongside the existing `critics`/`critic_classifiers`/
`reconciler_classifiers` parameters. Because `classify_csv` is part of the re-exported
public API (`spec/ARCHITECTURE.md`'s Stable Contracts list it as 1 of 8 public names),
`classify_csv` itself — not only the CLIs — must validate, before any CSV read/write:
exactly one of {`classifier` is not `None` (plain or critics mode), `models` is given} is
configured; `critics=True` requires `classifier` (as the sampling classifier); and when
`models` is given, it has at least 2 entries with non-empty, unique model ids, mirroring
the location/style of the existing critics-mode validation block (`pipeline.py:75-95`).
Per-row dispatch: `models` given → `multi_model`'s orchestration function; `critics` given
→ `debate.run_debate` (unchanged); neither → today's single `classifier.classify(...)`
call (unchanged).

#### AR-1.3: Every `critics`-gated branch in `pipeline.py`, plus `experiment.py`'s own completeness check, gets a `models` counterpart
Reading `classify_csv` end-to-end, `critics` gates five separate places, all of which need
a `models`-aware equivalent: (a) the validation + inter-category/text-column collision
check (`pipeline.py:75-118`, including the `_audit_columns` helper at `pipeline.py:22-25`,
generalized to cover 3 audit-column suffixes instead of `debate.py`'s 7); (b) initial
column creation (`pipeline.py:126-131`); (c) restore-seeding from a prior, separate
`--output` file (`pipeline.py:136-147`); (d) the restore-completeness check
(`pipeline.py:151-156`); and (e) the non-restore column reset, including the
dtype-coercion-to-object step (`pipeline.py:157-168`). Additionally, (f)
`experiment.py`'s own separate post-run completeness/status check (`experiment.py:869-881`,
which currently checks `{category}` + `{category}_votes` under `--critics`) must also
check `{category}_by_model` and `{category}_model_errors` under `--models`, so a `run`/
`classify` invocation can't report `status: "completed"` despite silent per-row audit
gaps.

#### AR-1.4: `experiment.py`'s pre-existing source-column collision check extended
`experiment.py`'s pre-existing-column collision check (`experiment.py:830`, which already
checks the critics audit suffixes against source-CSV columns when `args.critics`) is
extended to also check `multi_model`'s three audit suffixes (`_votes`, `_by_model`,
`_model_errors`) against source-CSV columns when `args.models` is set.

#### AR-1.5: Concurrency, index-based tallying, and never-raising failure handling
The N models are called concurrently per row via a nested
`ThreadPoolExecutor(max_workers=max(1, len(models)))`, mirroring `debate.py`'s sampling
concurrency pattern (`debate.py:174`). Results **must** be collected into a
pre-allocated, index-ordered structure (mirroring `debate.py:172`'s
`samples: list[...] = [None] * sampling_runs`, indexed by each model's position in
`--models`) before any tallying happens — never tallied in `as_completed()` arrival order
— so that FR-1.3's tie-break-by-list-order guarantee holds regardless of which model's
call finishes first. Unlike `debate.run_debate` (which raises when every sample fails),
`multi_model`'s orchestration function never raises for ordinary model-call failures — it
always returns a result dict, using FR-1.6's total-failure shape when 0 of N models
succeed. `pipeline.py`'s existing per-row `try`/`except` around `future.result()`
therefore takes the success branch for every `--models` row regardless of how many
individual models failed; that `except` branch (and the `failed` counter) is reached only
by a genuinely unexpected error in the orchestration itself, not by ordinary model
failures — a deliberate, `--models`-specific divergence from how `critics`/plain mode use
that counter today, made so failure information survives into the output CSV (FR-1.7)
instead of being silently discarded.

#### AR-1.6: Shared configuration across all N models
`--allow-new-labels` and the default (`None`) temperature apply identically to every one
of the N models' `Classifier` instances — no per-model override of either exists in this
spec (implements FR-1.2). In direct-provider mode, `--cerebus`'s gateway routing and
`--api-base`/env-var-priority resolution likewise apply identically to all N models,
exactly like today's existing cross-provider support for `--critic-model`/
`--reconciler-model` (`README.md`'s existing guidance: "make sure that's intentional
before pointing them at a different provider") — mixing native providers within
`--models` is supported on this same "user's responsibility" basis, not specially
validated. `--models` combined with `CEREBUS_MODE=azure` is rejected outright (FR-1.1),
because `CEREBUS_CONFIG_ID` is a single, workspace/model-specific value (Spec 3;
`classifier.py:190-205`) and applying one config id to N distinct models is not supported.

#### AR-1.7: `--model`'s existing behavior is fully preserved
`--model`'s existing meaning (its default value, its role in plain/`--critics` mode, and
`experiment.py`'s induction-model fallback default) is completely unaffected by this
feature. When `--models` is supplied, `--model`'s value is simply not used to construct
the classification-role classifier — no error, no special interaction, no mutual
exclusion with `--model` itself (only with `--critics`/`CEREBUS_MODE=azure`, per FR-1.1).

#### AR-1.8: Documentation stays in sync
`README.md`'s Options table and "Experiment runner" section document `--models`
analogously to how `--critics` is documented today: flag description; a precise
call-volume note (`--models` multiplies per-row LLM call volume by `len(models)` before
any internal retries/fallback, and total concurrent requests are approximately
`--workers × len(models)` — size `--workers` down accordingly, mirroring the existing
`--critics` guidance); an extension of the existing cross-provider data-egress note
(`README.md:214-222`) to cover N models instead of one; and an acknowledgment that the
existing CSV/spreadsheet-formula-injection caveat (`README.md:219-222`) applies to
`--models`' output the same as it does today. It also notes that
`experiments/ag-news/analyze_run.ipynb` reads critics-specific columns
(`_initial`/`_votes`/`_challenged`/`_reconciled`) and cannot analyze a `--models` run's
output as-is (see Out of Scope).

---

## Data Requirements

`run_config.json`'s existing `models` dict (`experiment.py:461-466`,
`{"model", "induction_model", "critic_model", "reconciler_model"}`) gains two new keys,
always present:

```json
{
  "model": null,
  "classification_models": ["azure/gpt-5-chat", "gpt-4o-mini", "gemini/gemini-2.5-pro"],
  "model_failure_counts": {"azure/gpt-5-chat": 0, "gpt-4o-mini": 2, "gemini/gemini-2.5-pro": 0},
  "induction_model": "...",
  "critic_model": null,
  "reconciler_model": null
}
```

`classification_models`/`model_failure_counts` are both `null` when `--models` was not
used; `model` is `null` only when `--models` was used (otherwise unchanged from today).
`schema_version` (top-level) is `2`.

## Integration Points

- `src/query_classification/multi_model.py` (new) — vote-tally/audit logic (AR-1.1)
- `src/query_classification/pipeline.py` — new `models` parameter, optional `classifier`,
  its own validation, dispatch, and all six branch points (AR-1.2, AR-1.3)
- `src/query_classification/cli.py` — new `--models` flag, validation, classifier
  construction, `--allow-new-labels` carve-out (FR-1.1, FR-1.2)
- `src/query_classification/experiment.py` — new `--models` flag on `classify`/`run`,
  validation, classifier construction, `run_config.json`/`_construct_classifiers` typing,
  post-run completeness check, source-column collision check (FR-1.1, FR-1.8, FR-1.9,
  AR-1.3, AR-1.4)
- `spec/ARCHITECTURE.md` — INV-1 (Module Boundary Map) and INV-7 (tests direct-import
  exceptions) both require an ADR amendment (AR-1.1)
- `README.md` — documentation sync (AR-1.8)

## Related Specs

| Spec | Relationship | Affected Requirements |
|------|-------------|---------------------|
| Spec 1: Initial classification with critics | **References** — reuses `debate.py`'s vote-bucket/consensus/audit-column/error-sanitization conventions without modifying `debate.py` itself | FR-1.3, FR-1.4, FR-1.7, AR-1.1 |
| Spec 2: Experiment runner | **Extends** — wires `--models` into `experiment.py`'s `classify`/`run` subcommands and `run_config.json` schema | FR-1.1, FR-1.8, FR-1.9, AR-1.3, AR-1.4 |
| Spec 3: Cerebus gateway | **References/Constrains** — `--cerebus` direct-mode routing applies uniformly to all N models; Azure mode's single, workspace/model-specific `CEREBUS_CONFIG_ID` is incompatible with N distinct models and is rejected | AR-1.6 |

## Constraints

- `--models` requires at least 2 model ids, all non-empty and mutually distinct —
  duplicate ids are rejected at validation time.
- `--models` and `--critics` are mutually exclusive; `--models` and `CEREBUS_MODE=azure`
  are mutually exclusive.
- All N models share identical prompt/categories/temperature — no per-model overrides in
  this spec. `--allow-new-labels` is shared and honored on both entry points (see FR-1.2's
  `cli.py` carve-out).
- `--model` is unaffected when `--models` is not given, and simply unused (not an error)
  for the classification role when `--models` is given.
- Mixing native (non-Cerebus) providers within `--models` is supported on the same
  "user's responsibility" basis as today's existing `--critic-model`/`--reconciler-model`
  cross-provider support — not separately validated.
- Larger per-row audit data (three JSON columns instead of one plain column) amplifies two
  pre-existing, accepted characteristics of this codebase rather than introducing new
  ones: incremental CSV writes are non-atomic (`spec/ARCHITECTURE.md` Known Gaps), and
  output size grows with rows × categories × models. Neither is addressed by this spec.
- No per-model or per-row timeout/cancellation policy is introduced; a hung model call
  relies on whatever timeout LiteLLM/the provider itself enforces, exactly as today's
  single-classifier and `--critics` calls already do.
- This feature requires an ADR amending `spec/ARCHITECTURE.md`'s INV-1 and INV-7 before
  `/spec-close`, mirroring how Specs 1-3 each amended these same invariants for their own
  new modules.

## Out of Scope

- Per-model prompt/temperature/category overrides.
- Weighted voting — all N models count equally; no per-model weight configuration.
- Combining `--models` with `--critics`' self-consistency sampling or Critic/Reconciler
  debate in any way.
- Per-model Cerebus Azure-mode configuration (multiple `CEREBUS_CONFIG_ID` values) — this
  spec rejects `--models` + `CEREBUS_MODE=azure` outright rather than supporting it.
- Updating `experiments/ag-news/analyze_run.ipynb` (or any other analysis notebook) to
  visualize the new per-model audit columns — it will not be able to analyze a `--models`
  run's output as-is (see AR-1.8).
- Any dashboard/UI for visualizing cross-model disagreement beyond the raw CSV audit
  columns.
- A global request-concurrency cap beyond `--workers` — `--workers × len(models)`
  concurrent calls is documented (AR-1.8) but not newly bounded, matching today's
  unenforced `--critics`/`--sampling-runs` precedent.

## Spec Completeness Checklist

- [x] **Scope & acceptance criteria** — FR-1.1 through FR-1.10 define scope; Out of Scope
      lists non-scope explicitly.
- [x] **Testing strategy** — FR-1.10 requires `tests/test_multi_model.py` plus new tests
      in `tests/test_debate.py`/`tests/test_experiment.py`, mapped to every FR's Verify
      condition.
- [x] **Existing patterns** — every FR/AR is explicitly compared to `debate.py`'s
      vote-tally/audit-column/error-sanitization conventions and `experiment.py`'s/
      `cli.py`'s existing validation-error style; deviations from precedent (e.g. the
      `cli.py` `--allow-new-labels` carve-out, the Cerebus-Azure rejection) are called out
      explicitly rather than left implicit.
- [N/A] **Dependencies** — no new third-party library; reuses the existing `Classifier`/
      `ThreadPoolExecutor` machinery. Operational dependencies (each additional model
      brings its own provider credentials/quotas/structured-output support) are the
      user's responsibility, consistent with how `--critic-model`/`--reconciler-model`
      already work.
- [x] **Architecture & interfaces** — AR-1.1 through AR-1.4 define the new module and all
      integration points; Constraints flags the required INV-1/INV-7 ADR amendment
      explicitly (an expected step for this project — see Specs 1-3's own ADRs).
- [x] **Error handling & failure modes** — FR-1.6/FR-1.7/AR-1.5 define partial- and
      total-model-failure behavior, including a sanitized per-model failure audit so
      systemic failures are distinguishable from ordinary disagreement.
- [x] **Security review** — `--models` is a new external input surface that deliberately
      routes the same row text to N potentially different providers; AR-1.8 requires
      extending the existing cross-provider data-egress documentation (`README.md:
      214-222`) to cover this. The existing CSV/spreadsheet-injection caveat is
      acknowledged (also AR-1.8) rather than newly fixed. Failure notes are sanitized by
      construction (FR-1.7, following `debate._sanitize_error`) — never raw exception
      text, which could contain credentials or internal endpoint URLs.
- [x] **Performance impact** — AR-1.5/AR-1.8 document the exact concurrency formula
      (`workers × len(models)`, before retries/fallback) and the audit-data size growth;
      neither is newly bounded, matching existing `--critics` precedent (see Out of
      Scope).
- [x] **Rollout & migration** — purely additive CLI flag + output columns; no migration of
      existing data; `_SCHEMA_VERSION` bump to `2` is explicit (FR-1.9), with one
      consistent absent-state representation (`null`, never omitted).
- [x] **Assumptions & risks** — assumes the N models behave independently (no shared
      caching that would make distinct model ids silently return identical answers) and
      that the N-fold call-volume cost is acceptable to a user who opts into `--models`;
      also assumes existing LiteLLM/provider-level timeouts are sufficient (no new timeout
      mechanism is introduced) — all stated here since none are enforced in code.

---

## Change Log

### Update from critique-consolidated-v-1.md

**Applied:**
- Required `--models` ids to be unique and non-empty (rejecting the previous
  "duplicates allowed, double-weights" design, which was incompatible with the
  dict/JSON-keyed audit columns) — FR-1.1, Constraints.
- Added an explicit Cerebus Azure-mode rejection (`--models` + `CEREBUS_MODE=azure` is a
  validation error) rather than leaving heterogeneous-model gateway routing undefined —
  FR-1.1, AR-1.6.
- Specified mixed native-provider support as following the existing
  `--critic-model`/`--reconciler-model` precedent (user's responsibility, not separately
  validated) — AR-1.6, Constraints.
- Made `classify_csv`'s `classifier` parameter explicitly optional and required
  `classify_csv` to validate the `models`/`critics` exclusivity and minimum-count/
  uniqueness invariants itself, since it's public library API — AR-1.2.
- Added `debate.py` to `multi_model.py`'s allowed-imports list, resolving the
  self-contradiction between AR-1.1's import ceiling and its own instruction to reuse
  `debate.py`'s `_vote_bucket` — AR-1.1.
- Required index-based (not completion-order-based) vote tallying so FR-1.3's tie-break
  guarantee actually holds under concurrent execution — FR-1.3, AR-1.5.
- Broadened AR-1.3 from "the restore completeness check" to all five `critics`-gated
  branches in `pipeline.py` plus `experiment.py`'s own separate post-run completeness
  check.
- Scoped FR-1.5 (`--restore`) explicitly to `classify.py`/library usage, since
  `experiment.py` has no `--restore` flag.
- Resolved the `--allow-new-labels` baseline ambiguity: `--models` honors the flag on
  `cli.py` (like `--critics`), rather than inheriting plain mode's forced `True` — FR-1.2.
- Fully specified `run_config.json`/`_construct_classifiers` typing, the `run`
  subcommand's merge behavior, a single absent-state representation (`null`, always
  present), and the literal new schema version (`2`) — FR-1.9.
- Added a new requirement (FR-1.7) for a sanitized per-model failure audit column
  (`{category}_model_errors`), so a systemic failure is distinguishable from ordinary
  model disagreement, following `debate.py`'s existing `_sanitize_error` convention.
- Clarified "exactly once" means one logical `.classify()` call, not necessarily one
  network request — FR-1.2.
- Clarified `{category}_by_model` holds each model's validated/parsed label list, not raw
  provider text, and that a malformed category invalidates that model's whole row, not
  just the malformed category — FR-1.4.
- Named the actual test files unit tests belong in (`tests/test_multi_model.py`,
  `tests/test_debate.py`, `tests/test_experiment.py`), correcting a reference to a
  nonexistent `tests/test_cli.py` — FR-1.10.
- Documented the precise `workers × len(models)` concurrency formula, the cross-provider
  data-egress extension, the CSV-injection acknowledgment, and the `analyze_run.ipynb`
  incompatibility, all in AR-1.8; changed the Security checklist item from `[N/A]` to
  `[x]` with content.
- Added Constraints/checklist acknowledgments of amplified (but pre-existing, unaddressed)
  incremental-write non-atomicity and output-size growth, and of relying on existing
  timeout behavior with no new mechanism — no new code required, just stated explicitly.

**Rejected:**
- A hard global concurrency cap beyond `--workers × len(models)` — would be new scope
  beyond what `--critics`/`--sampling-runs` already does today (also unbounded); documenting
  the exact formula (accepted above) is a targeted, sufficient fix.
- A shared/single `ThreadPoolExecutor` across rows instead of AR-1.5's per-row nested pool
  — would deviate from `debate.py`'s own precedented per-row nested-pool pattern for no
  stated benefit; not adopted.
- Rewriting every code citation to include function names alongside line numbers — both
  critique passes independently re-verified this spec's line numbers as accurate; a
  wholesale rewrite for future-proofing alone isn't worth the churn right now.
- Per-model weighted voting, per-model prompt/temperature overrides, and occurrence-keyed
  duplicate-id support — all remain Out of Scope; requiring uniqueness (accepted above)
  removes the need for occurrence-keyed records entirely.

**Reorganized:**
- Added FR-1.7 (sanitized per-model failure audit) as its own requirement rather than
  folding it into FR-1.6, since it's independently testable and adds a new audit column.
- Split AR-1.3 into an explicit six-point enumeration (five in `pipeline.py`, one in
  `experiment.py`) instead of naming only the restore-completeness check.
- Renumbered FR-1.7/FR-1.8 (old) to FR-1.8/FR-1.9, and old FR-1.9 (testing) to FR-1.10, to
  make room for the new FR-1.7.

### Update from critique-v-2 (Codex + Claude delta round)

Both v2 critics independently verified all 11 v1 Blocking Items resolved, but Codex found
two of the v1 fixes were still broken in ways that only surface on close reading — both
accepted and fixed here without another full round, since they're targeted corrections to
requirements just written, not new design work:

**Applied:**
- Fixed AR-1.2: giving `classifier` a *default* value (`= None`) is invalid Python here,
  since the very next parameter, `categories`, has no default
  (`SyntaxError: non-default argument follows default argument`). `classifier` now stays a
  required, non-default positional parameter with a widened type
  (`Classifier | None`) — callers using `models=` mode pass `classifier=None` explicitly.
  This preserves the exact existing call signature/order for every current positional
  caller.
- Fixed a real contradiction between FR-1.6 (total model failure "propagates ... silently
  counted, not logged") and FR-1.7 (the failure-audit column is "always present") — as
  written, a total-failure row would have produced *no* `_model_errors` data at all,
  defeating FR-1.7's own stated purpose for exactly the worst case, and leaving
  FR-1.9's `model_failure_counts` uncountable for those rows. FR-1.6/AR-1.5 now specify
  that the multi-model orchestration never raises (a deliberate divergence from
  `debate.run_debate`'s raise-on-total-failure precedent): a total-failure row gets null
  `{category}` values but a fully-populated `{category}_model_errors`, so it's counted as
  `classified` (not `failed`) by `pipeline.py`, while `experiment.py`'s existing
  completeness check still correctly flags it as incomplete via the null `{category}`.
- Specified FR-1.7's exact fixed stage text (`"Classification"`, giving the literal format
  `f"{type(exc).__name__}: Classification call failed"`) — the spec previously said "a
  fixed stage description" without naming it, which would have produced different CSV
  contracts depending on which literal an implementer picked.

**Rejected:** none this round — both findings were straightforward corrections with a
single clear fix each.

**Reorganized:** none — all three fixes are localized edits to AR-1.2/FR-1.6/FR-1.7/AR-1.5,
no renumbering needed.

### Correction found during /spec-plan's plan critique (2026-09-10)

**Applied:** fixed a factual error in AR-1.1's justification — it claimed "`debate.py` has
no existing dependents," but `pipeline.py` and `experiment.py` already import `debate.py`
today (per `spec/ARCHITECTURE.md`'s own Mermaid graph and Module Boundary Map). The
correct, and originally intended, justification is that `debate.py` itself has zero
*outbound* imports of `pipeline.py`/`cli.py`/`experiment.py`/`multi_model.py` — so a new
`debate.py → multi_model.py` edge doesn't create a cycle, regardless of how many existing
modules already import `debate.py`. This doesn't change the conclusion (the new edge is
still acyclic and still requires the same ADR) — only the stated reasoning was wrong.
