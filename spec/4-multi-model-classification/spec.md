# Spec 4: Multi-Model Voting Classification

> **Status: CLOSED** — Implemented and verified on 2026-09-10.
> Implementation summary: `spec/4-multi-model-classification/implementation-summary.md`
> Merged from worktree branch `spec/4-multi-model-classification` on 2026-09-10.
> **Reopened 2026-09-10** for a follow-up CLI-surface redesign (`--model`/`--models`/
> `--n-classifiers`) — see the Change Log's final entry. The status block above records
> the *original* feature's close; this reopening does not un-close it, it amends it.
> **Re-closed 2026-09-11** — the redesign is implemented and verified; see
> `implementation-summary.md`'s "Supersedes" note at the top. Re-merged from worktree
> branch `spec/4-multi-model-classification` on 2026-09-11.

## Overview

Adds a multi-classifier voting mode to `classify.py` (`cli.py`) and `experiment.py`'s
`classify`/`run` subcommands: a standalone alternative to `--critics` where N classifier
instances each classify the same row once, independently, and their answers are merged
into a single result per category by plurality vote. The N instances may run distinct
models (`--models model-a model-b model-c`) or repeat the same model N times
(`--model gpt-4o-mini --n-classifiers 5`, or `--models gpt-4o-mini --n-classifiers 5`) —
both are first-class, and mixing is intentional: this is what makes "5 classifiers, same
model, plurality vote, no `--critics` debate" possible without `--critics`' sampling +
debate machinery at all. It reuses `--critics`' audit-column and vote-tallying
conventions (`debate.py`) without invoking any of its self-consistency sampling or
Critic/Reconciler debate machinery.

## Goals

- Let a user compare/ensemble several distinct models against the same test set in one
  run, without paying for `--critics`' sampling + debate overhead.
- Let a user run several independent instances of the *same* model and take a plurality
  vote, without paying for `--critics`' Critic/Reconciler debate escalation either — a
  simpler, cheaper alternative to `--critics --sampling-runs N --consensus-threshold 1`.
- Keep the merge auditable: every classifier instance's own raw answer, the vote tally,
  and any per-instance failures are recorded alongside the final merged answer — including
  when several instances share the same underlying model, via a disambiguated audit-trail
  key.
- Reuse existing categories/schema/prompt/restore/incremental-save infrastructure — this
  is a lightweight sibling of `--critics`, not a fork of the pipeline.

---

## Feature 1: Multi-classifier voting mode

**Who & why:** A data scientist wants to know whether several distinct frontier models
agree on a classification task, or ensembling them beats any single model — for example,
running the AG News experiment with `--models gpt-5.4-mini gpt-4o-mini gemini-2.5-pro`
instead of picking one model, to see where they agree/disagree per row. The same user
also sometimes wants a simpler question answered: "does *one* model, asked 5 times
independently, converge on the same answer?" — without paying for `--critics`' Critic/
Reconciler debate escalation, which is built to resolve one model's *uncertainty*, not to
run a plain plurality vote. Both needs are really the same mechanism (N classifier
instances, merged by vote) with a different classifier-list shape feeding it — this spec
lets one flag set (`--model`/`--models`/`--n-classifiers`) serve both.

### Functional Requirements

#### FR-1.1: `--model`/`--models`/`--n-classifiers` — resolving the classifier list, validated before any LLM call
Three flags jointly determine how many classifier instances are constructed and which
model(s) they use:
- `--model` (`nargs="+"` — changed from a plain single-value flag; see AR-1.8 for why)
  still means "the single model to use" everywhere it already did (plain mode, `--critics`'
  sampling classifier, `experiment.py`'s induction-model fallback default) — it accepts
  only **one** value; passing 2+ is a validation error (see below), not a silent list.
- `--models` (`nargs="+"`, unchanged flag name) accepts 1 or more space-separated LiteLLM
  model ids. **Duplicates are now allowed** — this supersedes the original version's
  "must be unique" rule (see the Change Log for why that reversal is safe: AR-1.3's
  occurrence-suffixed audit-trail keys resolve what previously made duplicates unsafe).
- `--n-classifiers` (**new**, `type=int`, default `1`) sets how many classifier instances
  to construct.

All of the following are validation errors raised before any LLM call, in the same style
as the existing `"--hf-dataset and --train-file/--test-file are mutually exclusive"` check
(`experiment.py:272-275`) and the conditional-range checks for `--consensus-threshold`
(`experiment.py:313-317`, `cli.py:207-215`):
1. `--n-classifiers` less than `1`.
2. `--model` given more than one value — the message points at `--models` instead.
3. `--model` explicitly given (i.e. not left at its default) together with `--models`
   given — **except** when `--models` has exactly one value, which silently overrides
   `--model` instead of erroring: `--models gpt-4o-mini` (with or without `--model` also
   present) behaves identically to `--model gpt-4o-mini`.
4. `--models` given with 2+ values together with `--critics` — always mutually exclusive,
   regardless of `--n-classifiers` (see point 8 below for the case where `--models` isn't
   given with 2+ values).
5. `--n-classifiers` greater than `1` together with `--critics`, when `--models` was
   **not** separately given with 2+ values (i.e. this is what actually drives the resolved
   list — point 9 below) — `--critics` keeps its own independent
   `--sampling-runs`/`--consensus-threshold` mechanism for "same model, N samples" and is
   untouched by this spec; a resolved single classifier combined with `--critics` is just
   today's ordinary `--critics` + `--model`/`--critic-model` usage — still fine.
6. Any resolved model id that is empty or whitespace-only.
7. `CEREBUS_MODE=azure` together with a resolved classifier list containing **2 or more
   distinct** model ids (not merely 2+ classifier instances — see point 9): a single
   `CEREBUS_CONFIG_ID` is workspace/model-specific and can't serve two different models.
   N instances that all resolve to the *same* model id are fine under Azure mode, since
   one config id correctly serves all of them.

Once validated, the **resolved classifier list** is built:
8. If `--models` was given with 2+ values, the resolved list is **exactly those values, in
   order, duplicates preserved — its length is `len(--models)`, full stop.
   `--n-classifiers` plays no role and is not validated against it in this case**:
   `--models a b c` alone (the common case, no `--n-classifiers` needed) resolves to 3
   classifiers; `--models a b c --n-classifiers 5` *also* resolves to 3 classifiers (the
   `--n-classifiers 5` is simply not applicable here, not an error — see the Change Log for
   why an earlier draft of this rule required them to match and was reverted).
9. Otherwise (i.e. `--models` was not given with 2+ values) there is exactly one resolved
   model id — from `--model`, from a single-value `--models` (point 3's override), or, if
   neither flag was given, the existing default (`azure/gpt-5-chat`) — and the resolved
   list is that one id repeated `--n-classifiers` times. This is the only case where
   `--n-classifiers` has any effect.

A resolved list of length 1 is today's existing plain-classifier (or `--critics`-sampling-
classifier) behavior, entirely unchanged. A resolved list of length 2+ engages the
multi-classifier voting mode described in the rest of this spec, regardless of whether the
underlying model ids are all distinct, all identical, or a mix. `--models`/`--n-classifiers`
are added to the same shared classification-flags group that already carries `--critics`
on both `cli.py` and `experiment.py`'s `classify`/`run` subcommands (not `induce`, which
has no classification role); `--model` is unaffected in placement (`cli.py`'s single flat
parser; `experiment.py`'s `common` group, shared with `induce`).
**Verify:** `--models a b c` alone (no `--n-classifiers`) constructs 3 distinct classifiers
— the primary, common-case usage, requiring no second flag; `--n-classifiers 5` alone
constructs 5 classifiers using the default model; `--model gpt-4o-mini --n-classifiers 5`
constructs 5 using `gpt-4o-mini`; `--models a a b --n-classifiers 3` (or with
`--n-classifiers` omitted entirely) constructs 3 classifiers, two of which use model `a`;
`--models a b c --n-classifiers 99` still constructs exactly 3 classifiers (`--n-classifiers`
has no effect here — not an error); `--models a` (no `--n-classifiers`) is equivalent to
`--model a`; `--model a --models a` (both given, single-value `--models`) does not error;
`--model a b` (two values), `--model a --models b c` (both given, multi-value `--models`),
`--n-classifiers 0`, and `--models a b --critics` (2+ values, any `--n-classifiers`) each
exit non-zero with a clear message before any network call, mirroring
`test_cli_invalid_sampling_temperature_exits_before_any_llm_call`'s test pattern;
`--n-classifiers 3 --critics` (no `--models`, or `--models` with 1 value) also exits
non-zero; `--models a b --n-classifiers 2` combined with `CEREBUS_MODE=azure` exits
non-zero, but `--model a --n-classifiers 3` combined with `CEREBUS_MODE=azure` does **not**
error (one
distinct model, three instances); `experiment.py induce --help` does not list `--models`/
`--n-classifiers`.

#### FR-1.2: Independent single classification per classifier instance — no sampling, identical shared config
Each of the N resolved classifier instances classifies the row's text exactly once — one
logical `Classifier.classify(...)` invocation per instance per row, not necessarily one
network request (may still internally retry or fall back to JSON mode, exactly as today).
There is no `--sampling-runs`-style repeated sampling of any individual instance, even
when several instances share the same model id. All N instances share the identical
system prompt, category schema, and temperature (`temperature=None`, each instance's own
provider default — not `--critics`' `--sampling-temperature`). `--allow-new-labels` is
**honored** for every instance on both entry points, exactly as it already is under
`--critics` — a deliberate carve-out from `cli.py`'s plain-mode behavior, which instead
forces label invention on regardless of the flag (`cli.py:264`); this mode is a `--critics`
sibling, not a plain-mode variant. No per-instance prompt/temperature override exists in
this spec, even between instances that happen to share a model id (see Out of Scope).
**Verify:** a test that mocks/counts `Classifier.classify` confirms exactly N calls for N
resolved classifier instances (no repeats), including when some instances share a model
id; a multi-classifier system prompt mentions label invention only when
`--allow-new-labels` is passed, on both entry points, regardless of `--critics`.

#### FR-1.3: Plurality vote merge, ties broken by classifier position
For each category, the final label list written to the `{category}` output column is the
full ordered label list from whichever classifier instance's *top* label (the first
element of its 1-3 list) wins a plurality vote across the N instances — the same "vote on
the top label, use the winning instance's full list as the answer" convention `debate.py`'s
consensus step already uses (`debate.py:194-207`), just voting across classifier instances
instead of samples of one model. There is no consensus threshold or escalation: whichever
label gets the most votes wins, even without a strict majority. Votes are tallied by each
instance's **position in the resolved classifier list** (FR-1.1 points 8/9 — `--models`'
own order when given with 2+ values, or construction order when expanded from a single
resolved model), not by which instance's LLM call happens to complete first. A tie for the
most-voted label is broken in favor of whichever tied label the earliest-positioned
instance picked — this holds even when the tied instances share the same underlying model
id. `"none"`/`"none - <suggestion>"` labels are bucketed for voting using the same
`_vote_bucket` merge convention `debate.py` already uses (`debate.py:35-47`) when
`--allow-new-labels` is set.
**Verify:** with a 3-instance resolved list where the 1st/2nd agree on a category's top
label and the 3rd disagrees, the 1st/2nd label is the final answer; with a 2-instance
resolved list disagreeing entirely, the 1st instance's full label list is the final
answer, **regardless of which instance's mocked call completes first**; this holds
identically when two of the tied instances resolve to the same underlying model id.

#### FR-1.4: Per-classifier-instance and vote-tally audit columns, disambiguated when a model repeats
Two new audit columns are added per category — following the existing "add only if not
already present" convention (`pipeline.py:22-25,126-131`, unaffected by this update):
`{category}_votes` — a JSON object mapping each voted-on label (or none-bucket) to its
vote count across the N instances; and `{category}_by_model` — a JSON object mapping each
classifier instance to its own validated, parsed 1-3 label list for that category (the
same value `Classifier.classify` returns — not raw provider completion text). Each
instance's key is its model id, **unless that model id occurs more than once among the
resolved N instances** — in that case, every instance sharing that id gets an
occurrence-suffixed key instead: `"<model id>#1"`, `"<model id>#2"`, ... (1-indexed, in
classifier-position order). Model ids that occur exactly once keep their bare id as the
key — **this preserves the feature's originally-shipped, already-tested behavior for the
all-distinct case exactly** (no suffix ever appears unless it's actually needed to
disambiguate). If an instance's response for the row fails schema validation for *any*
category (the classification schema validates all categories as one atomic unit —
`schema.py:17-63`), that instance is absent from `{category}_by_model` and the vote tally
for **every** category in that row, not just the category that failed — the same
"whole-row" failure granularity as FR-1.6.
**Verify:** a resolved list `[a, a, b]` produces `{category}_by_model` keys `"a#1"`,
`"a#2"`, `"b"` (not `"a"` twice, and `b` unsuffixed since it's the only instance of that
model); a resolved list of 3 distinct models produces bare `"a"`/`"b"`/`"c"` keys exactly
as before this update; `{category}_votes` parses as JSON whose counts sum to the number of
instances that succeeded for that row; an instance whose response fails schema validation
for one category is absent from `{category}_by_model` for every category in that row.

#### FR-1.5: `--restore` integrates the new audit columns (`classify.py`/library only)
This requirement applies to `classify.py` and direct `classify_csv` library usage —
`experiment.py` has no `--restore` flag today and is unaffected. Under multi-classifier
mode, a row counts as "already classified" for `--restore` purposes only when
`{category}`, `{category}_votes`, `{category}_by_model`, and `{category}_model_errors`
(FR-1.7) are all non-null for every category — mirroring the existing critics-mode rule
that also requires `{category}_votes` non-null in addition to `{category}` itself. This
requirement is agnostic to whether audit-column keys are bare or occurrence-suffixed
(FR-1.4) — it only checks column non-nullness, never key content.
**Verify:** a partially-written CSV with `{category}` filled but `{category}_by_model`
empty is re-classified (not skipped) on `--restore`, matching the existing critics-mode
test pattern (`test_debate.py:364-385`'s equivalent for `_votes`); a separate-output
`--restore` (`test_debate.py:388-443`'s pattern) correctly seeds all four multi-classifier
audit columns from the prior output file, not just `{category}`.

#### FR-1.6: Partial and total classifier-failure handling
If one or more (but not all) of the N classifier instances fails to classify a row (an
LLM/network error, after that instance's own internal retries are exhausted, or a
schema-validation failure per FR-1.4), the row still produces a merged result using only
the instances that succeeded: `{category}_by_model` includes only the successful
instances' keys (occurrence-suffixed per FR-1.4 when applicable), and the vote tally
reflects only successful responses. If **every** instance fails for a row, the
multi-classifier orchestration still returns a result rather than raising — a deliberate
divergence from `debate.run_debate`'s precedent (which raises, and the row is silently
discarded, when every sample fails): `{category}` is `None` for every category, `{category}
_votes`/`{category}_by_model` are empty JSON objects, and `{category}_model_errors`
(FR-1.7) records every one of the N instances' sanitized failures. This keeps a systemic
failure (bad credential, quota exhaustion, provider outage) visible in the output CSV
instead of silently vanishing into `pipeline.py`'s in-memory `failed` counter —
`experiment.py`'s existing post-run completeness check (AR-1.4f) already treats a null
`{category}` as an incomplete row, so a total-failure row is still correctly flagged there.
**Verify:** mocking 1 of 3 instances to always raise still produces a merged `{category}`
answer from the other 2, with `{category}_by_model` containing exactly 2 keys; mocking all
3 to raise produces a row where every `{category}` is `None`, `{category}_by_model ==
"{}"`, and `{category}_model_errors` contains all 3 instances' keys with sanitized failure
notes — the row is not silently discarded, and `pipeline.py`'s `classified` counter (not
`failed`) increments for it.

#### FR-1.7: Sanitized per-classifier-instance failure audit
A third per-category audit column, `{category}_model_errors`, is added alongside
`{category}_votes`/`{category}_by_model` (same "add only if not present" convention): a
JSON object mapping each classifier instance that failed for that row to a sanitized
failure note, in the exact format `f"{type(exc).__name__}: Classification call failed"` —
exception type plus the fixed literal stage text `"Classification"`, never the raw
exception text — following `debate.py`'s existing `_sanitize_error` convention
(`debate.py:66-70`). Keys use the same bare-id-unless-repeated / occurrence-suffixed
scheme as `{category}_by_model` (FR-1.4). This column is always present for every row
processed in multi-classifier mode, including total-failure rows (an empty JSON object
`{}` when no instance failed), so its non-null-ness participates in the FR-1.5
restore-completeness check.
**Verify:** mocking one instance to always raise a specific exception type produces
`{category}_model_errors == {"<that instance's key>": "<ExceptionType>: Classification
call failed"}` for every category in that row, and the raw exception message text never
appears in any output cell; mocking all N instances to fail (FR-1.6) produces this same
format for every instance's key, in the same column; when two instances share a model id,
their keys in this column are occurrence-suffixed exactly as in `{category}_by_model`.

#### FR-1.8: Available on both entry points
`--models`/`--n-classifiers` are available on `classify.py` and on `experiment.py`'s
`classify`/`run` subcommands, exactly like `--critics` is today (`--model` already was
available everywhere, unaffected).
**Verify:** `python classify.py --help`, `python experiment.py classify --help`, and
`python experiment.py run --help` all list `--models` and `--n-classifiers`.

#### FR-1.9: `run_config.json` records the classifier list and per-instance failure counts
`experiment.py`'s `run_config.json` `models` field's `classification_models` key holds an
ordered list of the *resolved classifier list*'s model ids (FR-1.1 points 8/9) — duplicates
preserved exactly as resolved/expanded, e.g. `["gpt-4o-mini", "gpt-4o-mini", "gpt-4o-mini"]`
for a `--model gpt-4o-mini --n-classifiers 3` run. It is always present, `null` when the
resolved classifier list has length 1 (a single, consistent absent-state representation).
The `model` key is `null` under the **opposite** condition — when the resolved list has
length 2+ (no single model id drove the classification role); when the resolved list has
length 1, `model` holds that one resolved model id, exactly as it already does today
outside multi-classifier mode. `model_failure_counts` is a JSON object mapping each
classifier instance's key — using the same bare-id-unless-repeated / occurrence-suffixed
scheme as FR-1.4/FR-1.7 — to the number of rows in which that instance failed; `null` under the same
condition as `classification_models`. `_construct_classifiers`'s return type and the `run`
subcommand's induction-then-classification merge already carry these two keys unchanged
(both already handle arbitrary values, per the original implementation). The existing
post-run aggregation that builds `model_failure_counts`, however, **does** need a change:
it currently pre-seeds every entry from the raw `--models` list (`{m: 0 for m in
args.models}`) before incrementing keys found in `{category}_model_errors` — for a
resolved list with repeats, this pre-seeding step must instead use the *actual constructed
classifier keys* (occurrence-suffixed where applicable), not the raw `--models`/`--model`
input, or repeated instances' failures are silently never counted (their occurrence-
suffixed keys never match the raw, un-suffixed pre-seed keys). `_SCHEMA_VERSION`
(`experiment.py:85`, currently `2`) is bumped to `3`: `model_failure_counts`' keys may now
be occurrence-suffixed instead of always-bare model ids, which is a shape/contract change
for any consumer that assumed the latter.
**Verify:** a `run_config.json` from a `--models a a b --n-classifiers 3` run has
`models.model == null`, `models.classification_models == ["a", "a", "b"]`, and
`models.model_failure_counts` keyed `"a#1"`/`"a#2"`/`"b"`; a run with a resolved list of 3
distinct models has `model_failure_counts` keyed by bare ids, unchanged from before this
update; a run with a resolved list of length 1 (plain/`--critics` mode) has
`classification_models`/`model_failure_counts` both `null` and `model` holding the single
resolved id; `models.schema_version == 3` in every case (top-level `schema_version` field).

#### FR-1.10: Ship unit tests
Unit tests cover every Verify condition in FR-1.1 through FR-1.9, added to the existing
test files established by the original implementation: `tests/test_multi_model.py`
(orchestration-level: this update requires no new tests here, since `multi_model.py`
itself is unchanged — AR-1.1); `tests/test_debate.py` (`cli.py`-level: all of FR-1.1's new
validation/resolution cases, the occurrence-suffix audit-column cases, the `--n-classifiers`
flag); `tests/test_experiment.py` (`experiment.py`-level: the same, plus the
`run_config.json`/`_SCHEMA_VERSION` cases and the induction-fallback-resolution edge case
from AR-1.8).
**Verify:** `pytest` passes, including the new tests in `tests/test_debate.py`/
`tests/test_experiment.py`.

### Architectural Requirements

#### AR-1.1: `multi_model.py` requires no changes for this update
The module (established by the original implementation) dispatches purely by classifier
dict key and is agnostic to whether those keys are bare or occurrence-suffixed model ids —
it never inspects key *content*, only iterates `dict[str, Classifier]` by position. No
code in `src/query_classification/multi_model.py` changes for this spec-update.

#### AR-1.2: `pipeline.classify_csv`'s contract is unaffected; its "unique ids" framing is corrected
`classify_csv`'s existing `models: dict[str, Classifier] | None` parameter and its own
validation (≥2 entries, non-empty keys) require no code changes. The original spec's note
that this dict is "safe now that FR-1.1 guarantees unique ids" is corrected: `classify_csv`
was never actually guaranteed unique ids by FR-1.1 in the first place — a Python dict
cannot hold a duplicate key by construction, so uniqueness at this layer was always
automatic, not something FR-1.1 uniquely provided. What FR-1.1 previously did was
*disallow the input that would have required disambiguation* (`--models` duplicates); this
update instead handles disambiguation earlier, at construction time (AR-1.3), so
`classify_csv`/`multi_model.py` remain unaware repeated model ids were ever involved.

#### AR-1.3: Classifier-list resolution and occurrence-suffixed identity
`cli.py` and `experiment.py` are responsible for: (a) applying FR-1.1's resolution
algorithm to produce an ordered list of model ids (duplicates preserved when applicable);
(b) constructing one `Classifier` per position in that list, identically configured per
FR-1.2; (c) assigning each instance a key — its bare model id, unless that id occurs more
than once in the list, in which case every occurrence gets an occurrence-suffixed key
(`"<id>#N"`, 1-indexed by position) — before building the `dict[str, Classifier]` handed
to `classify_csv`'s `models` parameter. This is the only place in the system that needs to
know repeated model ids are possible; `pipeline.py` and `multi_model.py` are unaffected
(AR-1.1, AR-1.2) and operate on whatever already-unique keys they're given, exactly as
before this update. The occurrence-suffix scheme assumes model ids don't themselves
already look like `"<other-id>#N"` for some other id in the same resolved list; if
constructing the keyed dict this way would ever produce fewer entries than the resolved
list has positions (i.e. two different positions collide on the same key — only possible
if a supplied model id happens to already contain a `#` in this exact shape), that must be
a validation error, not a silent dropped classifier. This is expected to be vanishingly
rare in practice (model ids are provider-assigned identifiers, not user-chosen strings
that would coincidentally take this form) and is handled as an implementation-level
fail-fast guard, not a new user-facing rule.

#### AR-1.4: Every `critics`-gated branch in `pipeline.py` is unaffected; `experiment.py`'s own completeness check must switch from raw `--models` truthiness to resolved-list length
`pipeline.py`'s five branches (validation/collision check, column creation,
restore-seeding, restore-completeness, non-restore reset) require no changes — none of
that logic depends on key content (bare vs. occurrence-suffixed), only on `models` being
a non-`None` dict, which `cli.py`/`experiment.py` still provide exactly the same way.
`experiment.py`'s own post-run completeness/status check, however, **does** need a change:
it currently gates on raw `args.models` truthiness (`elif args.models:` adds the three
multi-classifier audit columns to its check), which is no longer equivalent to
"multi-classifier mode is active" now that a single-value `--models` is single-classifier
mode (truthy `args.models`, but should *not* add those columns) and a `--model`/default +
`--n-classifiers` replication is multi-classifier mode with `args.models` unset entirely
(should add those columns, but today's check wouldn't). This must switch to checking the
*resolved classifier list's length* (> 1) instead of `args.models`' raw truthiness.

#### AR-1.5: `experiment.py`'s pre-existing source-column collision check must switch from raw `--models` truthiness to resolved-list length
The same problem as AR-1.4 applies here: the check that adds `multi_model.
AUDIT_COLUMN_SUFFIXES` (`_votes`, `_by_model`, `_model_errors`) to the generated-column
set for collision purposes currently gates on `elif args.models:` — this must switch to
the resolved classifier list's length (> 1), for the same reason (a single-value
`--models` shouldn't trigger it; a `--model`+`--n-classifiers` replication should, even
though `args.models` itself is unset in that case).

#### AR-1.6: Concurrency, index-based tallying, and never-raising failure handling
Unchanged — the N instances are still called concurrently via a nested
`ThreadPoolExecutor`, results still collected into a pre-allocated, index-ordered
structure before tallying (so FR-1.3's tie-break-by-position guarantee holds regardless of
completion order), and the orchestration still never raises even on total failure. None of
this depends on whether instances share a model id.

#### AR-1.7: Shared configuration across all N classifier instances
`--allow-new-labels` and the default (`None`) temperature apply identically to every one
of the N instances — no per-instance override, even between instances sharing a model id
(implements FR-1.2). In direct-provider mode, `--cerebus`'s gateway routing and
`--api-base`/env-var-priority resolution likewise apply identically to all N instances.
`CEREBUS_MODE=azure` is rejected only when the resolved classifier list contains 2+
**distinct** model ids (FR-1.1 point 7) — refined from the original spec's blanket
rejection of any multi-classifier usage, now that N same-model instances are a supported,
common case that a single `CEREBUS_CONFIG_ID` serves correctly.

#### AR-1.8: `--model` changes from a single-value flag to `nargs="+"`, `default=None`
This is a mechanical, argparse-level change required to implement FR-1.1 cleanly:
`--model`'s argparse configuration changes from a plain single string
(`type=str` implicitly, with a concrete default) to `nargs="+", default=None`. Two
things this enables, neither achievable with the original configuration: (1) `--model`
given 2+ values can be caught with a clear, custom `ValueError` pointing at `--models`,
instead of argparse's generic "unrecognized arguments" crash — a plain single-value flag
has no way to *receive* multiple tokens to validate against in the first place; (2)
`default=None` lets FR-1.1's resolution algorithm distinguish "the user explicitly passed
`--model`" from "left at its default", which FR-1.1 point 3's override rule requires (a
concrete default value, as before, would make every invocation look "explicit"). Every
existing site that read `args.model` as a plain string — both entry points' classifier
construction, and `experiment.py`'s `_construct_classifiers`'s `--induction-model` fallback
default — must instead use FR-1.1's *resolved single-model value*: the literal default
(`azure/gpt-5-chat`) when neither `--model` nor `--models` was given, `--model`'s sole
value when given, or `--models`' sole value when given with exactly one entry (FR-1.1
point 3). **A fourth case, explicit here rather than left to be inferred:** when
`--models` was given with 2+ values (and `--model` was not separately given — the two are
otherwise mutually exclusive per FR-1.1 point 3), the resolved single-model value used for
`--induction-model`'s fallback is *also* the literal default (`azure/gpt-5-chat`) — never
one of `--models`' 2+ entries. `experiment.py`'s induction-model fallback must always
resolve to one concrete model id — including in this fourth case, in the same `run`
invocation that also induces, since induction has no concept of "multiple models" and this
spec doesn't add one, and there is no principled way to pick a single one of several
explicitly-listed classification models for a role that needs exactly one.

#### AR-1.9: Documentation stays in sync
`README.md`'s Options table and "Multi-model mode"/"Experiment runner" sections document
`--n-classifiers`, the revised `--model`/`--models` interaction rules (`--models a b c`
alone still works exactly as before, unaffected by `--n-classifiers`; `--n-classifiers`
only matters when replicating a single resolved model), and the occurrence-suffixed
audit-key format (`"<model id>#N"`) with a worked example showing both the all-distinct
case (bare keys, unchanged) and the repeated-model case (suffixed keys).

---

## Data Requirements

`run_config.json`'s `models` dict gains no new keys beyond what the original
implementation already added (`classification_models`, `model_failure_counts`) — only
their *content* changes shape when classifier instances repeat a model id:

```json
{
  "model": null,
  "classification_models": ["gpt-4o-mini", "gpt-4o-mini", "gpt-4o-mini"],
  "model_failure_counts": {"gpt-4o-mini#1": 0, "gpt-4o-mini#2": 1, "gpt-4o-mini#3": 0},
  "induction_model": "...",
  "critic_model": null,
  "reconciler_model": null
}
```

versus the all-distinct case (unchanged from the original implementation):

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

`classification_models`/`model_failure_counts` are both `null` when the resolved
classifier list has length 1; `model` is `null` under the opposite condition — length 2+
(otherwise, at length 1, it holds the single resolved model id, unchanged from today).
`schema_version` (top-level) is `3`.

## Integration Points

- `src/query_classification/multi_model.py` — **no changes** (AR-1.1)
- `src/query_classification/pipeline.py` — **no changes** (AR-1.2)
- `src/query_classification/cli.py` — `--model` reconfigured, new `--n-classifiers` flag,
  resolution/validation algorithm, occurrence-suffixed classifier construction (FR-1.1,
  AR-1.3, AR-1.8)
- `src/query_classification/experiment.py` — same as `cli.py`, plus
  `--induction-model` fallback resolution and `_SCHEMA_VERSION` bump (FR-1.1, FR-1.9,
  AR-1.3, AR-1.8)
- `README.md` — documentation sync (AR-1.9)

## Related Specs

| Spec | Relationship | Affected Requirements |
|------|-------------|---------------------|
| Spec 1: Initial classification with critics | **References** — reuses `debate.py`'s vote-bucket/consensus/audit-column/error-sanitization conventions without modifying `debate.py` itself | FR-1.3, FR-1.4, FR-1.7 |
| Spec 2: Experiment runner | **Extends** — wires the resolved classifier list into `experiment.py`'s `classify`/`run` subcommands and `run_config.json` schema | FR-1.1, FR-1.8, FR-1.9 |
| Spec 3: Cerebus gateway | **References/Constrains** — `--cerebus` direct-mode routing applies uniformly to all N instances; Azure mode's single, workspace/model-specific `CEREBUS_CONFIG_ID` is incompatible with 2+ distinct resolved models and is rejected in that case only | AR-1.7 |

## Constraints

- The resolved classifier list always has length ≥1; length 1 is today's existing plain/
  `--critics`-classifier behavior, completely unchanged; length 2+ engages
  multi-classifier voting, regardless of whether the underlying model ids are distinct,
  identical, or a mix.
- `--models` values no longer need to be distinct — duplicates are a first-class,
  supported case (reversed from the original version; see Change Log).
- `--model` accepts only one value; `--n-classifiers` must be ≥1.
- `--models` given with 2+ values always determines the resolved classifier list's length
  by itself (`len(--models)`) — `--n-classifiers` is not applicable and not validated
  against it in that case, so `--models a b c` alone (no `--n-classifiers`) works exactly
  as it did in the original shipped feature. `--n-classifiers` only has an effect when
  `--models` was *not* given with 2+ values (replicating a single resolved model).
- `--n-classifiers` > 1 and `--critics` are mutually exclusive when `--n-classifiers` is
  what's driving the resolved list (i.e. `--models` wasn't separately given with 2+
  values); `--models` given with 2+ values and `--critics` are unconditionally mutually
  exclusive. A resolved single classifier (from `--model`/single-value `--models`/default)
  combined with `--critics` is unaffected, ordinary `--critics` usage.
- `--models` alone or `--model`+`--n-classifiers`, with `CEREBUS_MODE=azure`, is rejected
  only when the resolved list contains 2+ **distinct** model ids — not merely 2+
  instances.
- Occurrence-suffixed audit-trail keys (`"<model id>#N"`) appear only for model ids that
  are actually repeated among the resolved instances; an all-distinct resolved list
  produces bare-id keys exactly as the feature originally shipped.
- All N instances share identical prompt/categories/temperature/`--allow-new-labels` — no
  per-instance overrides, even between instances sharing a model id.
- Mixing native (non-Cerebus) providers within a resolved multi-model list is supported on
  the same "user's responsibility" basis as today's existing `--critic-model`/
  `--reconciler-model` cross-provider support — not separately validated.
- `_SCHEMA_VERSION` is `3` as of this update (was `2` after the original implementation).

## Out of Scope

- Per-instance prompt/temperature/category overrides — every instance of a repeated model
  still shares identical config (FR-1.2); only its position/audit key differs.
- Weighted voting — all N instances count equally; no per-instance weight configuration.
- Combining multi-classifier mode with `--critics`' self-consistency sampling or
  Critic/Reconciler debate in any way.
- Per-model Cerebus Azure-mode configuration (multiple `CEREBUS_CONFIG_ID` values) for the
  2+-distinct-models case — that combination is still rejected outright, not supported.
- Updating `experiments/ag-news/analyze_run.ipynb` (or any other analysis notebook) to
  visualize the new audit columns or the occurrence-suffix key format.
- Any dashboard/UI for visualizing cross-instance disagreement beyond the raw CSV audit
  columns.
- A global request-concurrency cap beyond `--workers` — unchanged from the original
  implementation, still unenforced, matching `--critics`/`--sampling-runs` precedent.
- Validating `--n-classifiers` against `len(--models)` when `--models` has 2+ values —
  considered for this update and dropped in favor of `--n-classifiers` simply not
  applying in that case at all (see Change Log); no mismatch is ever flagged between the
  two flags.

## Spec Completeness Checklist

- [x] **Scope & acceptance criteria** — FR-1.1 through FR-1.10 define scope; Out of Scope
      lists non-scope explicitly; `--models a b c` alone continues to work unchanged from
      the original shipped feature, with `--n-classifiers` purely additive for the
      single-model-replication case.
- [x] **Testing strategy** — FR-1.10 maps every FR's Verify condition to the existing test
      files; explicitly notes `tests/test_multi_model.py` needs no new tests since
      `multi_model.py` is unchanged.
- [x] **Existing patterns** — every FR/AR is compared to the original implementation's own
      conventions and to `debate.py`'s precedent; deviations (the Cerebus-Azure condition
      narrowing, the `_SCHEMA_VERSION` bump) are called out explicitly.
- [N/A] **Dependencies** — no new third-party library; reuses existing `Classifier`/
      `ThreadPoolExecutor`/argparse machinery.
- [x] **Architecture & interfaces** — AR-1.1/AR-1.2 explicitly confirm `multi_model.py`/
      `pipeline.py` need zero changes (a scope-containment finding worth stating, not
      just assuming); AR-1.3/AR-1.8 define the actual interface changes. **No new ADR is
      required** — this update adds no new module and no new import-boundary edge (INV-1/
      INV-7 are unaffected), unlike the original feature.
- [x] **Error handling & failure modes** — FR-1.6/FR-1.7 unchanged in mechanism; FR-1.1's
      7-point validation list is exhaustive and ordered.
- [x] **Security review** — unaffected from the original implementation's assessment; the
      data-egress/CSV-injection considerations don't change based on whether instances
      share a model id.
- [x] **Performance impact** — unaffected; call-volume/concurrency formulas are unchanged
      (still a function of N instances, not of how many distinct models they represent).
- [x] **Rollout & migration** — `_SCHEMA_VERSION` bump to `3` is explicit and justified
      (FR-1.9); `--models a b c` alone (the original shipped feature's primary usage
      pattern) is unaffected by this update, so no migration is needed for existing
      callers of that form.
- [x] **Assumptions & risks** — the considered-and-dropped alternative (validating
      `--n-classifiers` against `len(--models)`) is recorded (Out of Scope, Change Log) so
      a future reviewer can see it was a deliberate choice, not an oversight.

---

## Change Log

### Update from critique-consolidated-v-1.md

[... unchanged from the original implementation's history — see prior revisions of this
file in version control for the full v1/v2 critique Change Log entries, preserved below.]

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

### Update: `--model`/`--models`/`--n-classifiers` redesign (2026-09-10, post-close)

Requested after using the shipped feature: the user wanted "5 classifiers, same model, no
`--critics` debate" to be directly expressible, rather than requiring the
`--critics --sampling-runs 5 --consensus-threshold 1` workaround (functionally correct,
but not what the flag names suggest, and still constructs unused Critic/Reconciler
classifiers). This required reopening a closed, already-implemented, already-merged spec's
CLI contract.

**Applied:**
- Added `--n-classifiers` (new flag, default `1`) and made `--models` accept duplicate
  model ids — FR-1.1. Together these let a single model be replicated into N independent
  classifier instances, the main new capability requested.
- Renumbered/rewrote FR-1.1 into an explicit, ordered 9-point resolution algorithm
  covering `--model`'s new single-value-only constraint, `--models`' override of `--model`
  when given exactly one value, mutual exclusion with `--critics`, and the narrowed
  Cerebus Azure-mode condition. (An intermediate draft of this same pass briefly required
  `--models`' length to exactly equal `--n-classifiers`, which would have made the
  already-shipped `--models a b c` alone into an error; caught and reverted before this
  entry was finalized — see the immediate follow-up entry right after this one.)
- Resolved the audit-trail identity problem duplicates reintroduce (a `dict`/JSON object
  can't hold two entries under one key): classifier instances get occurrence-suffixed keys
  (`"<model id>#1"`, `"<model id>#2"`, ...) **only when their model id actually repeats**
  among the resolved instances — the all-distinct case keeps its original, already-shipped
  bare-id keys unchanged, so this is additive, not a breaking change to that common case —
  FR-1.4, FR-1.7, FR-1.9.
- Traced the change through the whole system and found `multi_model.py` and `pipeline.py`
  require **zero code changes** — both already operate on whatever already-unique
  `dict[str, Classifier]` keys they're handed, agnostic to content. All resolution and
  key-disambiguation logic lives entirely in `cli.py`/`experiment.py`'s classifier-
  construction step (new AR-1.3). This significantly contained the blast radius of what
  first looked like a system-wide redesign.
- Narrowed AR-1.7's (was AR-1.6) Cerebus Azure-mode rejection from "any multi-classifier
  usage" to "2+ *distinct* resolved model ids" — N same-model instances are now a
  supported, common case, and a single `CEREBUS_CONFIG_ID` correctly serves all of them.
- Specified `--model`'s required mechanical change (`nargs="+"`, `default=None`) as its own
  AR (new AR-1.8, was AR-1.6/old-AR-1.7 territory) — this is genuinely dictated by
  argparse's mechanics (a plain single-value flag has no way to receive, and thus validate
  against, multiple tokens), not a stylistic choice, mirroring how the original spec's
  AR-1.2 explained a similar Python-mechanics constraint on `classify_csv`'s signature.
  Traced the ripple effect explicitly: every existing `args.model`-as-plain-string site,
  including `experiment.py`'s `--induction-model` fallback default, must switch to FR-1.1's
  *resolved* single-model value instead.
- Bumped `_SCHEMA_VERSION` from `2` to `3`: `model_failure_counts`' keys may now be
  occurrence-suffixed instead of always-bare, a shape/contract change for any consumer
  that assumed the latter.
- Corrected AR-1.2's framing: `classify_csv` was never actually the thing "guaranteeing"
  unique ids (a Python dict can't hold a duplicate key regardless); the original FR-1.1
  achieved safety by *disallowing the input that would need disambiguation*, not by
  `classify_csv` enforcing anything. This update instead disambiguates at construction
  time (AR-1.3), which is both more capable and doesn't change what `classify_csv` itself
  needs to guarantee.
- No new ADR is required — confirmed no new module or import-boundary edge is introduced;
  INV-1/INV-7 are unaffected by this update, unlike the original feature.

**Rejected:**
- Suffixing every classifier instance's audit key unconditionally (even with no actual
  duplicate) for "consistency" — rejected in favor of only suffixing when a model id
  actually repeats, specifically to avoid changing the already-shipped, already-tested
  all-distinct-models output format. This was the recommended option when the question was
  posed to the user.
- Replacing `--critics`' `--sampling-runs`/`--consensus-threshold` mechanism with
  `--n-classifiers` — rejected in favor of `--critics` staying fully untouched and separate
  (per the user's explicit choice); `--n-classifiers`/`--models` is purely the new
  no-debate voting path.

**Reorganized:**
- Inserted a new AR-1.3 (classifier-list resolution/occurrence-suffixing) immediately
  after AR-1.2, renumbering the original AR-1.3 through AR-1.8 up to AR-1.4 through AR-1.9
  — placed with its most closely related sibling (AR-1.2's `classify_csv` contract) rather
  than appended at the end.
- Retitled Feature 1 from "Multi-model voting classification mode" to "Multi-classifier
  voting mode" throughout FR/AR prose, reflecting that "N classifiers" is now the primary
  concept (which models they run is a secondary, resolved detail) — the spec title itself
  is unchanged since it's still fundamentally about voting across classifiers.

**Stale-plan warning:** `spec/4-multi-model-classification/plan.md` describes the
*original* FR-1.1 (2+ distinct models, no `--n-classifiers`) and is now stale for this
update — regenerate it with `/spec-plan 4-multi-model-classification` before implementing
these changes.

### Immediate follow-up: `--n-classifiers` made inapplicable when `--models` has 2+ values (2026-09-10)

The update above initially required `--models`' length to exactly equal `--n-classifiers`
whenever `--models` had 2+ values — including when `--n-classifiers` was simply left at
its default of `1`, which made the already-shipped, previously-documented `--models a b c`
(alone, no `--n-classifiers`) usage into a validation error. Flagged prominently to the
user rather than applied silently; the user asked for it to keep working as before.

**Applied:**
- FR-1.1 point 8 (was part of the "must equal" rule): when `--models` has 2+ values, the
  resolved classifier list is simply `len(--models)` — `--n-classifiers` plays no role and
  is never validated against it in that case, not even when explicitly given a conflicting
  value (`--models a b c --n-classifiers 99` resolves 3 classifiers, no error).
  `--n-classifiers` only has an effect when replicating a single resolved model (FR-1.1
  point 9 — from `--model`, a single-value `--models`, or the default).
  `--models a b c` alone is unaffected by this whole update, exactly as before.
- Split what was one combined "`--models`(2+)+`--critics`" / "`--n-classifiers`+`--critics`"
  validation point into two explicit points (FR-1.1 points 4 and 5), since they're now
  triggered under different conditions (the former unconditional; the latter only when
  `--n-classifiers` is actually driving the resolved list).
- Updated Constraints, Out of Scope, the Spec Completeness Checklist, and AR-1.9's
  documentation requirement to match — none of them describe a "must match" rule or a
  `--models a b c`-alone breaking change any more.

**Rejected:** none — this was a direct implementation of the user's own correction,
not a new independent design question.

**Reorganized:** none beyond the FR-1.1 point split described above.
