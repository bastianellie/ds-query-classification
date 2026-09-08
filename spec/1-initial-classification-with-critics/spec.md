# Spec 1: Multi-Agent Debate Classification with Critics

## Overview

Adds an opt-in `--critics` mode to the classification CLI. Instead of one LLM call per
row, each row's initial classification is sampled multiple times to measure whether the
model agrees with itself; categories where it doesn't reach consensus are argued out by a
Critic (devil's advocate) and, if a real challenge is raised, settled by an independent
Reconciler. This merges two source documents into one buildable design: an interview-driven
single-shot Critic/Reconciler proposal, and a pre-existing "Initial specification" draft
(`spec/initial-classification-with-critics-spec.md`) describing a larger multi-pass,
self-consistency-sampling Multi-Agent Debate system. Where they conflicted, the resolutions
below (confirmed with the user) take precedence over both source documents.

## Goals

- Turn hidden LLM inconsistency on ambiguous inputs into a measurable, per-category signal
  (vote distribution) instead of a silent single-shot guess.
- Only pay the cost of deeper scrutiny (Critic + Reconciler) on the subset of categories
  that are actually contested, not on every row/category.
- Produce a full audit trail so a user can see exactly why a label was kept or changed.
- Ship entirely opt-in: existing users who never pass `--critics` see no behavior change.

---

## Feature 1: Self-Consistency Sampling & Consensus Voting

**Who & why:** Users classifying subjective or ambiguous text (nuanced reviews, ambiguous
support tickets, contested reasoning traces) get an unreliable signal from a single LLM
call — the same input can get a different top label on a re-run, with no way to tell
confident classifications from lucky guesses. Sampling the classifier multiple times per
row and checking self-agreement exposes that instability directly, and only escalates the
categories where the model is actually inconsistent to Feature 2's deeper (and slower,
costlier) scrutiny.

### Functional Requirements

#### FR-1.1: Opt-in sampling mode
When `--critics` is passed, each row's classification runs `--sampling-runs` (default 5)
independent `.classify()` invocations to the initial Classifier instead of one, each made
at `--sampling-temperature` (default 0.7) — independent of any internal retry/fallback
attempts within a single `.classify()` call. When `--critics` is not passed, behavior is
unchanged from today: exactly one classification call per row, no temperature override. If
the selected model/provider rejects the `temperature` parameter outright, that sampling
call fails and is handled like any other sampling failure (FR-1.6) — no separate fallback
mechanism is required.
**Verify:** with `--critics` and a fake classifier that records call count, classifying one
row triggers exactly `sampling_runs` calls to `.classify()`; without `--critics`, exactly 1
call, matching today's behavior.

#### FR-1.2: Per-category vote tally on the top label
For each category independently, the top-ranked label (position 0 of that category's label
list) from each of the N samples is tallied into a vote count keyed by label value. When
`--allow-new-labels` is not set (default), the Classifier used during sampling is
constrained to the category's predefined labels plus plain `"none"` — no
`"none - <suggestion>"` invention — so votes always converge on a small, fixed set of
options. This constraint is prompt-level only, consistent with today's schema (see
`spec/ARCHITECTURE.md` INV-11: the schema enforces cardinality/type, not label validity) —
it is not a new schema-enforcement capability. If a sample's top label still falls outside
the expected set despite the instruction, it is simply tallied as its own vote bucket like
any other string; it is not rejected or treated as an error.
**Verify:** given 5 fake samples for one category with top labels `[a, a, a, b, b]`, the
tally is `{a: 3, b: 2}`.

#### FR-1.3: Consensus bypass
If the top-voted label's count is `>= --consensus-threshold` (default 4), that category
bypasses debate entirely. Its final 1-3 label list (and its `{cat}_initial` audit value,
see Data Requirements) is taken from the first successful sample (in sample order) whose
top label matches the winning vote.
**Verify:** 5 fake samples with tally `{a: 4, b: 1}` and `consensus_threshold=4` produce
that category's final label list equal to the first sample tagged `a`'s full label list,
with no Critic/Reconciler call made.

#### FR-1.4: Consensus escalation
If no label's vote count reaches `--consensus-threshold`, that category is escalated to
Feature 2. The **leading candidate** is the highest-voted label (its full label list taken
the same way as FR-1.3); the **runner-up candidate** is the second-highest-voted label.
Ties are broken by first-encountered order among the samples. **No-runner-up case:** if
fewer than 2 distinct label values appear among the successful samples (e.g. partial
sampling failures left only unanimous successes that still fall short of
`--consensus-threshold` numerically), there is nothing to escalate — skip debate and use
the sole candidate as final, the same as a consensus bypass (FR-1.3).
**Verify:** 5 fake samples with tally `{a: 3, b: 2}` and `consensus_threshold=4` escalate to
debate with leading=`a`, runner-up=`b`. With `sampling_runs=5` and `consensus_threshold=4`,
if 2 of the 5 attempts fail and the remaining 3 successes all vote `a`, the category's final
label list is `a`'s, with no Critic/Reconciler call made, despite the raw count (3) not
numerically reaching the threshold.

#### FR-1.5: Optional label invention during sampling
`--allow-new-labels` re-enables `"none - <suggestion>"` invention during sampling. When
set, any invented suggestion counts toward one shared "none (new label)" vote bucket
regardless of the exact wording suggested, so votes can still converge; the specific
suggested text is preserved in the final label list of whichever sample is chosen as
representative (FR-1.3/FR-1.4).
**Verify:** two fake samples inventing different new-label text (`"none - urgency"`,
`"none - other"`) both count toward the same bucket in the tally.

#### FR-1.6: Sampling failure handling
If some (but not all) of the `sampling_runs` calls for a row fail after their own retries,
voting proceeds using only the successful samples — this can make consensus harder or
impossible to reach for the affected categories, which is expected and simply escalates
them to debate. If **all** sampling calls for a row fail, the whole row fails, matching
today's per-row failure handling in `pipeline.py`: the row's result cells stay
unmodified/null and the in-memory `failed` counter increments — the row itself is not
removed from the CSV, since `pipeline.py` always writes the full DataFrame
(`pipeline.py:117-118`); it simply carries no new classification data.
**Verify:** 5 samples where 2 raise on a fake classifier and 3 succeed still produce a vote
tally over the 3 successes; a row where the fake classifier raises on all 5 attempts causes
that row to be counted as failed (its category columns remain null in the output CSV, and
the row is not dropped), matching `pipeline.py`'s existing except-and-count pattern
(`pipeline.py:109-110`).

#### FR-1.7: Voting/consensus config validated at startup
`--sampling-runs` must be `>= 1`, `--consensus-threshold` must be between 1 and
`--sampling-runs` inclusive, and `--sampling-temperature` must be a non-negative, finite
number. Invalid combinations raise a clear error before any LLM call is made. Only checked
when `--critics` is passed.
**Verify:** `--critics --sampling-runs 3 --consensus-threshold 5` exits with a clear error
message (`Error: ...`) and no LLM calls attempted. `--critics --sampling-temperature -1`
exits the same way.

### Architectural Requirements

#### AR-1.1: `Classifier` gains an optional sampling temperature
Extend `Classifier.__init__` (`classifier.py:52-70`) with `temperature: float | None = None`,
included in `_completion_kwargs` (`classifier.py:72-78`) only when not `None` — following
the exact pattern already used for `api_base` there. Confirmed the installed `litellm`'s
`completion()` accepts a `temperature` kwarg (verified live this session via
`inspect.signature`).

#### AR-1.2: New orchestration module `debate.py`
`src/query_classification/debate.py` owns the sampling+voting+routing logic for a single
row (and, per Feature 2, the Critic/Reconciler calls), returning a flat `dict[str, Any]`
whose keys match the final category columns plus the new audit columns (Data
Requirements). This is the exact shape `pipeline.py`'s existing per-row result-writing loop
already expects (`for cat, val in classification.items(): df.at[idx, cat] = val`,
`pipeline.py:106-107`) — that loop needs no change; only the callable submitted to the
thread pool changes.

#### AR-1.3: `pipeline.py` wiring
`classify_csv` gains a `critics: bool = False` parameter (plus the sampling/consensus/model
parameters listed under Integration Points). When `True`, the per-row submission at
`pipeline.py:99` (`executor.submit(classifier.classify, str(df.at[idx, column]))`) submits
`debate.run_debate(...)` instead. The "add category columns if not present" step
(`pipeline.py:59-62`) is extended to also pre-create the new audit columns when
`critics=True`.

#### AR-1.4: Sampling and per-category debate calls run concurrently within a row
The N sampling calls for a single row execute concurrently with each other (not in series),
so the added latency stays close to one LLM round-trip rather than `sampling_runs` of them
— implemented as N futures inside `debate.run_debate`, the same concurrency pattern already
used across rows in `pipeline.py:97-101`, applied at the per-row scope. "Sample order" (used
by FR-1.3/FR-1.4's "first successful sample" tie-break) means the logical submission index
(0..N-1) assigned when the N calls are dispatched, tracked explicitly the same way
`pipeline.py:98-101` maps futures back to row indices — not the order in which the futures
happen to complete, which is non-deterministic. The same concurrency principle applies when
a row has more than one escalated category (Feature 2): each category's Critic/Reconciler
debate runs concurrently with the others', not serially — a row with 3 escalated categories
should not take ~3x as long as a row with 1.

#### AR-1.5: Category-name collision protection extended
When `--critics` is enabled, for every category compute its full generated column set (its
own name plus all audit suffixes from AR-2.4, e.g. `sentiment`, `sentiment_votes`,
`sentiment_challenged`, ...). Raise an error (same style as the existing
`pipeline.py:46-51` check) if any name in that set collides with: the `--column` value,
any other category's generated column set, or any column already present in the input CSV
that isn't itself one of these generated names. This is stricter than a same-category-name
check alone — it also protects a user's existing CSV columns from being silently
overwritten by a generated audit column.

#### AR-1.6: Architecture update required
`spec/ARCHITECTURE.md` INV-1 currently fixes `pipeline.py`'s allowed imports to
`categories.py` + `classifier.py` only, and its Module Boundary Map has no `debate.py` row.
Implementing AR-1.2/AR-1.3 (which make `pipeline.py` import `debate.py`, and `debate.py`
import `categories.py`, `classifier.py`, `schema.py`, and `prompts.py`) requires updating
`spec/ARCHITECTURE.md`'s Module Boundary Map and INV-1 to describe this revised dependency
graph, per that document's own stated rule that invariant changes need an ADR
(`spec/{spec_name}/ADR.md`, folded in by `/spec-close`).

#### AR-1.7: `debate.py` is a second accepted direct-import exception for tests
`spec/ARCHITECTURE.md` INV-7 restricts new tests to the public API (`query_classification`)
plus `resources.py` as the one accepted direct-import exception. Tests exercising
`debate.run_debate`, `build_critic_model`, or `build_reconciler_model` directly need
`debate.py` (and, transitively, the new schema/prompt builders) declared as a second
accepted exception — these are not added to `__init__.py`'s public re-exports.

---

## Feature 2: Critic Challenge & Reconciliation for Contested Categories

**Who & why:** When Feature 1's sampling shows the initial Classifier is genuinely
inconsistent about a category, silently picking the plurality winner would paper over real
ambiguity. Users need the tool to argue out the disagreement and reach a reasoned final
answer they can audit later — not an arbitrary tie-break — for exactly the categories where
that scrutiny is warranted.

### Functional Requirements

#### FR-2.1: Critic challenge (devil's advocate)
For each category escalated by Feature 1 (FR-1.4), a Critic LLM call receives the source
text, the category's name/description/full label options, and the leading candidate's top
label. The Critic is prompted to construct the strongest reasonable argument that the
runner-up candidate (or another closed-vocabulary label) fits better, or to explicitly
decline if it cannot find a genuinely compelling alternative. A leading candidate of
`"none"`/`"none - <suggestion>"` is challengeable exactly like any other label — no special
casing. Whether it challenges or declines, the Critic's output (its decline rationale, or
its proposed label + argument) is always recorded in `{cat}_critic_argument` — a decline is
not a silent no-op for audit purposes.
**Verify:** a fake critic that always returns "no compelling alternative" results in the
leading candidate's label list being the category's final output, with `{cat}_challenged`
= `false` and `{cat}_critic_argument` populated with the decline rationale (not empty). A
leading candidate of `"none"` with a fake critic that challenges and proposes a real label
produces a final output containing that real label once reconciled.

#### FR-2.2: Conditional Reconciliation
Reconciliation only runs for a category when the Critic actually raised a challenge in
FR-2.1. The Reconciler LLM call receives the source text, the category's schema, the
leading candidate's full label list, and the Critic's proposed label + argument, then
independently produces a final 1-3 label list for that category — it may agree with the
leading candidate, adopt the Critic's proposal, or land on a different combination
entirely; it is not limited to the two candidates already on the table. "Independent" here
means independence of judgment — the Reconciler is not anchored to defending either the
leading candidate or the Critic's proposal — not necessarily a different underlying model;
model diversity is optional and separately controlled by FR-2.4.
**Verify:** a fake critic that challenges plus a fake reconciler that returns a specific
label list results in that category's final output being exactly the reconciler's label
list, with `{cat}_reconciled` = `true`.

#### FR-2.3: Graceful degradation on Critic/Reconciler failure
If the Critic or Reconciler call fails after its own retries, that category's final label
list falls back to the leading candidate's, and the failure reason is recorded in
`{cat}_debate_error` (sanitized per FR-2.8). The row is not counted as failed solely
because of a Critic/Reconciler error. The two failure points leave different audit states,
since a Reconciler failure necessarily follows a real challenge:
- **Critic call fails:** `{cat}_challenged` = `false`, `{cat}_critic_argument` = empty
  (we never learned whether it would have challenged), `{cat}_reconciled` = `false`.
- **Reconciler call fails:** `{cat}_challenged` = `true` (a real challenge did occur) and
  `{cat}_critic_argument` populated with the Critic's argument, but `{cat}_reconciled` =
  `false` (reconciliation itself did not complete).
**Verify:** a fake critic that raises an exception (after exhausting retries) results in
the leading candidate's label list being used as final output, `{cat}_debate_error`
populated, `{cat}_challenged` = `false`, and the row still counted as classified, not
failed. Separately, a fake critic that challenges paired with a fake reconciler that raises
an exception (after exhausting retries) results in the leading candidate's label list as
final output, `{cat}_challenged` = `true`, `{cat}_reconciled` = `false`, and
`{cat}_debate_error` populated.

#### FR-2.4: Configurable Critic/Reconciler models
`--critic-model` and `--reconciler-model` each independently select the LiteLLM model id
used for FR-2.1/FR-2.2 calls. When not given, each defaults to whatever `--model` is set
to.
**Verify:** running with `--model X --critic-model Y` (and no `--reconciler-model`)
constructs the Critic with `model_id=Y` and the Reconciler with `model_id=X`.

#### FR-2.5: Full audit trail in output CSV
For every category, the output CSV gains the columns in Data Requirements, populated for
every row processed under `--critics` (empty/false where not applicable — e.g. a
consensus-bypassed category has `{cat}_challenged` = `false` and an empty
`{cat}_critic_argument`). Without `--critics`, the output CSV is unchanged from today — no
new columns.
**Verify:** a `--critics` run's output CSV has all 7 new columns per category; the same
input run without `--critics` produces exactly the columns it produces today.

#### FR-2.6: `--restore` correctness under `--critics`
When resuming with both `--restore` and `--critics`, a row only counts as already done if
its category value **and** its `{cat}_votes` audit column are both non-null — not just the
category value alone. This prevents a row classified under the pre-`--critics` schema, or
left mid-way through an interrupted `--critics` run, from being silently treated as
complete. When resuming from a separate prior `--output` file (the existing
`pipeline.py:64-75` behavior that copies `category_names` columns from that file into the
freshly-loaded input before the completeness check), all audit columns must be copied over
the same way, not just the category value — otherwise a fully-completed prior `--critics`
row would look incomplete (missing `_votes`) and be needlessly reprocessed.
**Verify:** a row with a filled `{cat}` column but a null `{cat}_votes` is re-processed on a
`--restore --critics` run, not skipped. A row fully completed (including all audit columns)
in a prior run written to a separate `--output` file is correctly skipped when resumed with
`--restore --critics` pointing `--input` at the original file and the same `--output`.

#### FR-2.8: Sanitized failure text in `{cat}_debate_error`
`{cat}_debate_error` (and any other persisted failure note) stores a sanitized summary —
the exception's type name plus a truncated message — not a raw exception string or
traceback. This avoids leaking secrets, credentials, or internal endpoint URLs that may
appear in an underlying LiteLLM/provider exception's raw text into a CSV file the user may
share or version.
**Verify:** a fake critic/reconciler that raises an exception containing a fake API key or
URL in its message results in `{cat}_debate_error` not containing that raw string verbatim
(e.g. it's truncated and/or the exception type/a fixed-length prefix is stored instead).

#### FR-2.9: Ship unit tests
The implementation ships a `pytest`-runnable test suite (`tests/`, `pythonpath = ["src"]`,
no live network calls, following the existing project convention) covering every FR-1.x and
FR-2.x Verify condition above using fake/duck-typed `Classifier`-shaped objects (any object
exposing `.classify(text) -> dict`), including the failure/edge cases (FR-1.6, FR-1.7,
FR-1.4's no-runner-up case, FR-2.3's Critic-failure and Reconciler-failure states
separately, AR-1.5's collision cases, and FR-2.8's sanitization).
**Verify:** `pytest` passes and includes test functions exercising every FR-1.x/FR-2.x
Verify condition above without making a real LLM call.

### Architectural Requirements

#### AR-2.1: New schema builders
`src/query_classification/schema.py` (alongside `build_classification_model`,
`schema.py:17-49`) gains:
- `build_critic_model(category: Category) -> type[BaseModel]` — fields for whether the
  Critic is challenging, its proposed alternative label, and its argument text (the latter
  two required only when challenging).
- `build_reconciler_model(category: Category) -> type[BaseModel]` — a final 1-3 label list
  for that one category (reusing the `list[str]` + `Field(min_length=1, max_length=3)`
  constraint at `schema.py:44-47`), plus a reasoning text field.

#### AR-2.2: New prompt templates
`resources/prompts/critic_prompt.txt` and `resources/prompts/reconciler_prompt.txt`,
rendered by new builder functions in `src/query_classification/prompts.py` (alongside
`build_system_prompt`, `prompts.py:25-39`), following the same static-system-prompt +
per-call-user-message split already used today (`cli.py` builds the system prompt once;
`classifier.py:99-102` builds a fresh user message per call).

#### AR-2.3: Critic/Reconciler reuse `Classifier` as-is
Both the Critic and the Reconciler are ordinary `Classifier` instances (AR-1.1's
`temperature` param is not needed for these — they use their own role-specific system
prompt (AR-2.2) and schema (AR-2.1)), reusing `classifier.py`'s existing retry/fallback
logic unchanged — no new retry/fallback code is written. `debate.py` (AR-1.2) assembles the
per-call "user message" content (source text + category context + prior-round argument, as
applicable) and passes it as the `text` argument to `.classify()`.

#### AR-2.4: Centralized reserved column names
The reserved audit-column suffixes (`_initial`, `_votes`, `_challenged`,
`_critic_argument`, `_reconciled`, `_reconciler_reasoning`, `_debate_error`) are defined
once as constants in `debate.py`, referenced by both the collision check (AR-1.5) and the
column-creation step (AR-1.3) so they can't drift out of sync.

#### AR-2.5: `{cat}_votes` is written as real JSON text
`{cat}_votes` is serialized via `json.dumps` before being assigned into the DataFrame cell,
producing valid JSON text (e.g. `{"positive": 3, "negative": 2}`). This is a deliberate
improvement over the existing `{cat}` column's behavior — `spec/ARCHITECTURE.md`'s Stable
Contracts section documents that pandas' default `to_csv` cell serialization writes a
Python list/dict *repr* (e.g. `['positive']`), not valid JSON; `{cat}_votes` is a new column
with no existing behavior to preserve, so it should not repeat that footgun.

---

## Data Requirements

New columns added per category `{cat}` when `--critics` is passed (empty/`None`/`false`
when not applicable to that row; entirely absent from the CSV when `--critics` is not
passed):

| Column | Type | Meaning |
|---|---|---|
| `{cat}` | JSON-like list (existing) | Final 1-3 labels: from consensus (FR-1.3), the leading candidate on a declined challenge (FR-2.1), or the Reconciler's verdict (FR-2.2) |
| `{cat}_initial` | JSON-like list | The leading candidate's label list selected by sampling (FR-1.3/FR-1.4), before any Critic/Reconciler revision — identical to `{cat}` unless reconciliation changed it |
| `{cat}_votes` | text, e.g. `{"positive": 3, "negative": 2}` | The per-category vote tally from sampling (FR-1.2) |
| `{cat}_challenged` | bool | Whether the Critic raised a real challenge (FR-2.1); always `false` for consensus-bypassed categories |
| `{cat}_critic_argument` | text | The Critic's output whenever it actually ran: its proposed alternative + reasoning when challenging, or its decline rationale when declining (FR-2.1); empty if the category was never escalated (consensus bypass, FR-1.3/FR-1.4) or if the Critic call itself failed (FR-2.3) |
| `{cat}_reconciled` | bool | Whether the Reconciler ran and returned a verdict; `false` if not challenged or if the Reconciler failed (FR-2.3) |
| `{cat}_reconciler_reasoning` | text | The Reconciler's stated reasoning; empty when reconciliation didn't run |
| `{cat}_debate_error` | text | Sanitized failure note from a Critic/Reconciler error (FR-2.3, FR-2.8); empty otherwise |

## Integration Points

- `src/query_classification/cli.py`: new flags `--critics`, `--sampling-runs` (default 5),
  `--sampling-temperature` (default 0.7), `--consensus-threshold` (default 4),
  `--allow-new-labels`, `--critic-model`, `--reconciler-model` (each defaulting to
  `--model`), wired into the `classify_csv` call (today at `cli.py:145-161`).
- `src/query_classification/pipeline.py`: `classify_csv` gains the `critics`-related
  parameters (AR-1.3) and conditionally submits `debate.run_debate` instead of
  `classifier.classify` (`pipeline.py:99`).
- `src/query_classification/classifier.py`: `Classifier.__init__` gains `temperature`
  (AR-1.1); otherwise unchanged, and reused as-is for Critic/Reconciler (AR-2.3).
- `src/query_classification/schema.py`, `src/query_classification/prompts.py`: extended
  with Critic/Reconciler-specific builders (AR-2.1, AR-2.2).
- New module `src/query_classification/debate.py` (AR-1.2, AR-2.4).
- New prompt templates under `resources/prompts/` (AR-2.2).

## Related Specs

None — this is the first numbered spec in this project (`spec/ARCHITECTURE.md` is
prescriptive project documentation, not a numbered spec). A pre-existing rough draft at
`spec/initial-classification-with-critics-spec.md` ("Initial specification" — a
Multi-Agent Debate proposal) and an interview-driven single-shot Critic/Reconciler design
were merged into the design above, per user resolution of every conflict between them; the
draft file itself is left as-is and is not managed by this spec.

## Constraints

- All `sampling_runs` calls use the same `--model`; per-sample LLM assignment (the
  pre-existing draft's `llm_classifiers` dict) is not supported in this version.
- Debate is single-round (one Critic call, one conditional Reconciler call) — the
  pre-existing draft's multi-round Proponent/Critic debate (`max_debate_rounds`) is not
  implemented.
- No hard per-call timeout is introduced; Critic/Reconciler/sampling calls rely on the
  existing bounded-retry mechanism (`classifier.py:104-129`) for resilience, same as
  today's single classification call.
- Concurrency: with `--critics`, worst-case concurrent in-flight LLM calls is roughly
  `workers * sampling_runs` (plus any in-flight Critic/Reconciler calls) — noticeably
  higher than today's `workers`. This should be documented in `--help` text / README so
  users can size `--workers` down accordingly.
- Toggling `--critics` on/off between resumed runs on the same output file is unsupported
  and undefined — FR-2.6's restore-completeness check assumes a consistent `--critics`
  setting across a resume chain. A row classified without `--critics` and then resumed with
  `--critics` will simply be reprocessed (its missing `_votes` makes it look incomplete,
  per FR-2.6); the reverse (resuming a `--critics` run without `--critics`) is not
  specifically handled and should not be relied upon.
- Worst-case logical call count per row under `--critics`: roughly `sampling_runs +
  contested_categories + challenged_categories` `.classify()` invocations (before any
  internal retries/fallback within a single call), on top of `workers` rows in flight at
  once. `--workers`' existing help text ("Use 1 to disable parallelism") becomes
  `--critics`-specific: with `--critics`, `--workers 1` still limits row-level (outer)
  concurrency to one row at a time, but the `sampling_runs` calls within that one row still
  run concurrently with each other (AR-1.4) — it no longer means zero concurrency overall.
  Update the flag's help text accordingly when implementing.
- `--critic-model`/`--reconciler-model` can point at a different LiteLLM provider than
  `--model`, meaning the same source row text is sent to more than one third-party service.
  This should be disclosed in `--help`/README (which provider each role's model resolves
  to, and that credentials/API base resolution — `classifier.py:23-42` — applies
  independently per role) rather than assumed obvious.
- Critic/Reconciler prompts should clearly delimit "text under review" from instructions,
  and treat the Critic's generated argument as data (not additional instructions) when it's
  embedded in the Reconciler's prompt — reducing (not eliminating) the risk of the Critic's
  own output being interpreted as new instructions by the Reconciler.
- This feature assumes category label values are meaningful, non-empty, and don't
  themselves collide with the reserved strings `"none"`/`"none - ..."`; validating that is
  a pre-existing gap in `categories.py` (duplicate/empty label values are not rejected
  today) that this spec does not additionally address.

## Out of Scope

- Per-sampling-run or per-debate-role arbitrary LLM assignment (the draft's
  `llm_classifiers`/`llm_debaters` dict-based configuration).
- Multi-round Proponent-vs-Critic debate (`max_debate_rounds`) — superseded by the single
  Critic-challenge + Reconciler design.
- Per-call async timeouts.
- Custom Critic/Reconciler prompt template overrides (no `--critic-prompt`/
  `--reconciler-prompt` flags) — bundled templates only.
- Propagating `--task-description`/`--extra-prompt` (used by the main Classifier) into the
  Critic/Reconciler prompts.
- Supporting category names that collide with the new audit-column suffixes — rejected
  (AR-1.5), not supported.
- A global concurrency/request-rate budget across the nested sampling+row thread pools,
  beyond manually sizing `--workers`/`--sampling-runs` (Constraints) — a larger feature
  than reconciling the two source documents; a candidate for a follow-up spec if the
  worst-case request volume proves unmanageable in practice.
- CSV formula-injection sanitization (LLM-generated free text starting with `=`/`+`/`-`/
  `@` being interpreted as a formula by spreadsheet software) for the new free-text audit
  columns — a real risk, but pre-existing and shared by every free-text/label column this
  tool already writes; not unique to this feature and not solved here.

## Spec Completeness Checklist

- [x] **Scope & acceptance criteria** — FR-1.1 through FR-2.9 define exact, testable
  behavior, including the no-runner-up and Critic-vs-Reconciler-failure edge cases; Out of
  Scope explicitly lists what's excluded from the merged pre-existing draft.
- [x] **Testing strategy** — FR-2.9 requires a `pytest` suite covering every FR's Verify
  condition via fake `Classifier`-shaped objects, matching the existing project's
  no-live-network-calls testing convention (`AGENTS.md`).
- [x] **Existing patterns** — AR-1.1/AR-2.3 reuse `Classifier` as-is; AR-1.2/AR-1.3 reuse
  `pipeline.py`'s existing per-row result-writing/thread-pool/restore-copy pattern;
  AR-2.1/AR-2.2 follow `schema.py`/`prompts.py`'s existing builder-function conventions.
- [x] **Dependencies** — no new third-party libraries; reuses the already-installed
  `litellm` (`temperature` confirmed as a supported `litellm.completion` parameter in the
  installed version, verified live this session).
- [x] **Architecture & interfaces** — Integration Points and AR-1.1 through AR-2.5 cover
  every touched module and the new `debate.py` module's interface; AR-1.6 explicitly
  requires the `spec/ARCHITECTURE.md` INV-1/Module Boundary Map update (with ADR) this
  design needs, and AR-1.7 resolves the test-import-strategy question against INV-7.
- [x] **Error handling & failure modes** — FR-1.4 (no-runner-up), FR-1.6 (sampling
  failures), FR-1.7 (invalid config), FR-2.3 (Critic-failure vs. Reconciler-failure,
  handled distinctly) each specify exact fallback behavior.
- [x] **Security review** — FR-2.8 requires sanitized (not raw) failure text in
  `{cat}_debate_error`; Constraints requires disclosing that `--critic-model`/
  `--reconciler-model` may route the same text to a different provider, and requires basic
  prompt data-framing so the Critic's output is treated as data, not instructions, by the
  Reconciler. CSV formula-injection risk is real but pre-existing/tool-wide and explicitly
  logged in Out of Scope rather than silently ignored.
- [x] **Performance impact** — Constraints documents the `workers * sampling_runs`
  concurrency multiplier, the worst-case logical call-count formula, and the `--workers 1`
  help-text caveat; exact latency/cost numbers are not benchmarked here (would require a
  live provider) and are left as a follow-up note for implementation/README. A global
  concurrency budget beyond manual `--workers`/`--sampling-runs` sizing is explicitly Out
  of Scope.
- [x] **Rollout & migration** — entirely opt-in via `--critics`; no schema/behavior change
  for existing users who don't pass it (FR-1.1, FR-2.5). No data migration needed; mixed
  `--critics` on/off resume chains are explicitly called out as unsupported (Constraints).
- [x] **Assumptions & risks** — Constraints and Out of Scope state the assumptions
  inherited from reconciling the two source documents (single model per sampling run,
  single debate round, no timeouts, pre-existing `categories.py` validation gaps not
  additionally addressed); risk: LLM non-determinism means the same input can take a
  different consensus/debate path across runs, which is inherent to self-consistency
  sampling, not a bug.
