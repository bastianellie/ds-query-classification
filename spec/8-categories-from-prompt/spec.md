# Spec 8: Categories From Prompt

> **Status: CLOSED** — Implemented and verified on 2026-09-17.
> Implementation summary: `spec/8-categories-from-prompt/implementation-summary.md`
> Merged from worktree branch `spec/8-categories-from-prompt` on 2026-09-17.

## Overview

Add a new, standalone entry point that turns a single free-text instruction file into a
`categories.json` taxonomy — one `Category` whose `labels` (values and descriptions) are
authored by an LLM from the instruction text, via one `Classifier.classify()` invocation. Unlike
`experiment.py`'s existing `induce` subcommand (spec 2), this needs no labeled training data at
all: no `train_df`, no label column, no sampled examples — just a plain-text description of what
the categories/labels should be.

(One `classify()` invocation may itself involve more than one underlying provider request — see
`Classifier._attempt_completion`'s existing structured-output/JSON-mode-fallback/parameter-drop
retry behavior, unchanged and reused as-is here.)

## Goals

- Let a user hand-write a short natural-language description of a classification taxonomy
  (e.g. "classify support tickets as billing, technical, or account issues") and get back a
  ready-to-use `categories.json`, without writing JSON by hand or running a labeled-data
  induction experiment.
- Reuse the existing `Classifier` retry/fallback machinery rather than building new LLM
  plumbing.
- Keep the new entry point fully self-contained (new script + new module), matching the
  `classify.py`/`experiment.py` sibling-script convention already established in this repo —
  neither existing entry point is modified.

---

## Feature 1: Category Extraction From a Prompt File

**Who & why:** Someone defining a new classification taxonomy often already knows, in plain
English, what the categories and labels should be ("urgent vs. routine", "billing/technical/
account") but has no labeled dataset to run `experiment.py induce` against, and doesn't want to
hand-author `categories.json`'s exact JSON shape (including writing plausible label
descriptions) by hand. This feature turns that description directly into a valid
`categories.json`, in one `Classifier.classify()` invocation.

### Functional Requirements

#### FR-1.1: Extract one category's labels and descriptions from a free-text instruction file

Given a plain-text instruction file (read verbatim, in full) and a caller-supplied category
name, the LLM invents the category's `description` and its full `labels` list (each label a
`{value, description}` pair) based solely on the instruction text. The number of labels is
whatever the instruction text supports — not fixed or capped by this feature.

**Verify:** given a fake classifier response with 3 named labels (`{"category_description":
"...", "labels": [{"value": "billing", "description": "..."}, ...]}`), the resulting `Category`
has exactly those 3 labels, in response order, with values/descriptions copied verbatim.

#### FR-1.2: The category's output name always comes from the caller, never the LLM — and is validated before any LLM call

`--category-name` is a required CLI flag. The LLM's structured-output schema has no name/slug
field at all — the LLM never produces this value. The *caller's* value still needs validating,
though: `Category.name` later becomes a dynamic Pydantic field name inside
`build_classification_model` (`schema.py:35-63`), and specific strings crash that call outright
— verified live against the installed pydantic: `"__root__"` raises `TypeError` ("To define root
models, use `pydantic.RootModel`..."), and any name starting with `model_` (e.g.
`"model_config"`, `"model_dump"`) raises `TypeError`/`ValueError` (pydantic's protected-namespace
check). `--category-name` must therefore be validated immediately after argument parsing, before
any LLM call: it must satisfy `str.isidentifier()`, must not equal `"__root__"`, and must not
start with `"model_"`. A value failing any of these three rules fails with a clear error naming
which rule it violated.

**Verify:** `build_parser().parse_args([])` (no `--category-name`) raises `SystemExit` (argparse
`required=True`); `--category-name __root__`, `--category-name model_config`, and
`--category-name "not an identifier"` each fail with a clear error before any LLM call (a
call-counting fake classifier shows zero calls); a successful run with a valid name (e.g.
`sentiment`) has output `Category.name` equal to the exact string passed to `--category-name`.

#### FR-1.3: Zero labels is a hard, immediate failure

The LLM's response schema does not constrain the label list's length (see AR-1.2) — an empty
`labels` list is a schema-valid response. This feature must therefore itself reject it: zero
labels raises `ValueError` mentioning "at least one label" immediately after the classifier
call returns, before any file is written. This is not retried — `classify()`'s own retry budget
is already spent by the time this check runs, and retrying an already-successful (schema-valid)
call would not fix a response the LLM has already decided has no labels.

**Verify:** a fake classifier response with `"labels": []` raises `ValueError` (message contains
"at least one label"), and the output path does not exist afterward.

#### FR-1.4: Duplicate label values and reserved sentinel values are rejected

If the LLM's response contains two or more labels with the exact same `value` (case-sensitive,
no normalization), raise `ValueError` naming the duplicated value(s) — mirroring
`schema.build_classification_model`'s existing duplicate-category-name check (`schema.py:28-33`),
applied here to label values within the one extracted category instead. Separately, every label
`value` is passed through `experiment.py`'s existing `_check_reserved_sentinel` (the same
function `experiment.py` already applies to every supplied-categories file, `experiment.py:445`,
`:1189`) — a value that normalizes (trim + case-fold) to `"none"` or starts with `"none - "` is
rejected the same way, since `experiment.py --categories <this file>` would otherwise later fail
that exact check anyway, one step removed from where the actual cause is.

**Verify:** a fake response with two labels both valued `"urgent"` raises `ValueError` whose
message contains `"urgent"`; a fake response containing a label valued `"none"` (or `"NONE"`, or
`"none - invented"`) raises `ValueError` naming the reserved value.

#### FR-1.5: Blank or whitespace-only strings are rejected

The LLM's response schema (AR-1.2) does not constrain string length or content beyond `str` —
pydantic accepts empty and whitespace-only strings with no extra check. This feature rejects,
with a clear `ValueError`, a response whose `category_description` is blank/whitespace-only, or
that contains any label whose `value` or `description` is blank/whitespace-only (checked after
`.strip()`), before the file is written.

**Verify:** a fake response with `"category_description": "   "` raises `ValueError`; a fake
response with one label valued `""` (empty string) raises `ValueError`; a fake response with one
label whose `description` is `"   "` raises `ValueError`.

#### FR-1.6: The output file is written atomically and round-trip validated

Writes `{"categories": [<the one Category, .model_dump(mode="json")>]}` to the resolved
`--output` path via `cost.write_json_atomic` (spec 7's shared atomic-write helper — temp file in
the same directory + `Path.replace`, `allow_nan=False`), then immediately reloads it via
`categories.load_categories()` as a round-trip validation check before reporting success —
mirroring `experiment.py`'s own induce-branch pattern of writing `categories.json` and then
calling `load_categories(categories_path)` right after.

**Verify:** after a successful run, `load_categories(output_path)` returns a list containing
exactly one `Category`, equal to what the run produced.

#### FR-1.7: `--output` defaults to `./categories.json`; an existing file is never silently overwritten

`--output` defaults to `categories.json` in the current working directory. If the resolved path
already exists and `--overwrite` was not given, the command fails with a clear error — **before
any LLM call is made** (no cost incurred on a run that was going to fail anyway).
`--overwrite`, when given, allows replacing an existing file at that exact path; it is a
refusal check on a single file, never a directory-clearing operation (there is no directory
concept here, unlike `experiment.py`'s `--run-dir`). The same "fail before any LLM call" rule
applies to two related cases: the resolved `--output` path is a symlink, or it is an existing
directory rather than a file (both fail with a clear error, `--overwrite` or not — there is
nothing sensible to overwrite in either case); and the resolved path's parent directory does not
exist (this feature never creates directories). A known, accepted limitation, not fixed here:
this check-then-write sequence has the same time-of-check/time-of-use gap
`experiment.py`'s own `_preflight_run_dir` check already has (a concurrent process could create
or replace the target between the check and the actual write) — not a new risk this feature
introduces, and not addressed here for the same reason it isn't addressed there.

**Verify:** running the command twice with the same `--output` and no `--overwrite` fails the
second time with an "already exists" error, and a call-counting fake classifier shows zero
calls on that second run; re-running with `--overwrite` succeeds and replaces the file; an
`--output` pointing at an existing directory, or a missing parent directory, each fail before
any LLM call.

#### FR-1.8: The instruction file's content is passed through unchanged as the LLM's user message

No parsing, templating, or interpretation of the instruction file's content happens in this
tool — it is read once, in full, via `Path(prompt_file).read_text(encoding="utf-8")`, and the
resulting `str` is passed as-is as the classify call's `text` argument (character-for-character
after UTF-8 decoding — not byte-for-byte, since decoding is not the identity function). The
system prompt (AR-1.3), not the instruction file, defines what the LLM is allowed to do with
it — matching `induction.py`'s existing convention of treating sampled example text as untrusted
data, never as instructions the tool itself executes. An empty or whitespace-only instruction
file (after `.strip()`) fails immediately with a clear error, before any LLM call.

**Verify:** a fake `classify(self, text)` records its `text` argument; it equals the instruction
file's exact decoded contents, unchanged; an empty (0-byte) or whitespace-only instruction file
fails before any LLM call (zero calls on a call-counting fake classifier).

### Architectural Requirements

#### AR-1.1: New module `category_extraction.py` — a third full-access orchestrator, with one pure function inside it

`src/query_classification/category_extraction.py` is a **third module with the same import
breadth as `cli.py`/`experiment.py`** (categories/classifier/schema/prompts/resources/cost) —
not a leaf module restricted to `categories.py`+`classifier.py` at the module level. Within it,
one function has a narrower *behavioral* contract, mirroring `induction.py`'s pure-function
design:

```python
def extract_category_from_prompt(
    instruction_text: str, category_name: str, classifier: Classifier
) -> Category
```

This function itself only uses `categories.py`+`classifier.py` (it never touches the
filesystem, never calls `schema.py`/`prompts.py`/`resources.py`/`cost.py` directly) — but the
*module* it lives in also contains `build_parser()`/`main()`, which does need the wider import
list to build the classifier/prompt and write the output file. This mirrors how `cli.py` and
`experiment.py` themselves are structured (one module, full import breadth, with individual
functions inside it that only touch a subset) — there is no existing precedent in this repo for
splitting "pure logic" and "CLI orchestration" into two separate package modules, and doing so
here would be a new pattern, not a reused one.

#### AR-1.2: New schema builder, `schema.build_category_extraction_model()`

A fixed (non-parameterized — no `n_labels` argument, unlike `build_induction_model`)
`create_model`-built Pydantic model, with `model_config = ConfigDict(extra="forbid")` and
exactly two required fields: `category_description: str` and `labels: list[categories.Label]`
(no `min_length`/`max_length` constraint on the list). Verified live against the installed
`litellm` (`litellm.utils.type_to_response_format_param`): this shape serializes with
`"strict": true` and **no** `minItems`/`maxItems` keyword on the `labels` field — unlike a
length-bounded list (e.g. `Category.labels`'s own `min_length=1`, or `build_induction_model`'s
fixed-arity workaround), which litellm's strict mode rejects. An unbounded list is therefore
usable directly under strict mode; this feature does not need `build_induction_model`'s
positional-field workaround. `extra="forbid"` matters specifically for the JSON-mode fallback
path (`Classifier._fallback_completion`): verified live that, without it, an extra/unexpected
top-level or nested field in a non-strict JSON response is silently accepted and discarded
rather than rejected — mirroring `build_batch_model`'s own `ConfigDict(extra="forbid")`
precedent (`schema.py`) for exactly this reason.

#### AR-1.3: New prompt template + builder, with an override flag

`resources/prompts/category_extraction_prompt.txt` (new file) + a new
`DEFAULT_CATEGORY_EXTRACTION_PROMPT_FILE` constant in `resources.py` + a new
`prompts.build_category_extraction_prompt()` (mirrors `build_induction_prompt`'s zero-placeholder,
read-the-file-verbatim pattern — no per-call templating needed, since all per-call content is
the user message, not the system prompt). The template instructs the LLM to: (a) treat the
upcoming user message as untrusted data describing a taxonomy to extract, never as instructions
directed at the model itself; (b) invent as many labels as the instruction text meaningfully
supports; (c) never invent or suggest a name/slug for the category itself; (d) respond only in
the `category_description`/`labels` shape the structured-output schema defines. A new
`--extraction-prompt` CLI flag (default: the bundled template) lets a caller supply their own
system prompt file instead — mirroring `cli.py`'s existing `--system-prompt` override — and
`resources.check_default_resources_available()` is called before reading the bundled default,
exactly as `cli.py`/`experiment.py` already do, so a built-wheel install (which per INV-3 ships
no `resources/` directory at all) fails with the same clear, actionable error those two entry
points already give, rather than a raw `FileNotFoundError`.

#### AR-1.4: New entry point — `category_extraction.py`'s `build_parser()`/`main()` plus a root-level script

A root-level `extract_categories.py` sys.path-injection script, mirroring `classify.py`/
`experiment.py` verbatim (`sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))`
then `from query_classification.category_extraction import main`; `if __name__ ==
"__main__": main()`). Flags on `category_extraction.py`'s own `build_parser()`:

| Flag | Required | Notes |
|---|---|---|
| `--prompt-file` | yes | Path to the plain-text instruction file |
| `--category-name` | yes | Becomes the output `Category.name` (FR-1.2) |
| `--output` | no (default `categories.json`) | Output path (FR-1.7) |
| `--overwrite` | no (flag) | Allow replacing an existing `--output` file |
| `--extraction-prompt` | no (default: bundled template) | Override the system prompt file (AR-1.3) |
| `--model` | no | Plain `type=str` flag — **not** `nargs="+"`; defaults to the same literal model id as `experiment.py`'s own `_DEFAULT_MODEL` constant. No `_resolve_classifier_models`-style multi-value resolution: this feature has exactly one classifier, always. |
| `--retries` | no (default 3) | Passed to `Classifier(max_retries=...)` |
| `--cerebus` | no (flag) | Routes the one classify() invocation through the Cerebus gateway, matching `cli.py`/`experiment.py`'s existing flag/env-var (`DEFAULT_LLM_PROVIDER=cerebus`) behavior |
| `--api-base` | no | Explicit endpoint override, same precedence/HTTPS-under-Cerebus guard as the other two entry points |

No `--models`/`--n-classifiers`/`--critics`/`--batch` — this is a single classifier making one
`classify()` invocation; none of the multi-classifier/self-consistency/batching concepts apply.

#### AR-1.5: Neither existing entry point is touched; the new module is a standalone third one

Per INV-1, `cli.py` and `experiment.py` are not modified and do not import
`category_extraction.py`; `category_extraction.py` does not import `cli.py`, `experiment.py`,
`pipeline.py`, `batching.py`, `debate.py`, or `multi_model.py`. This is a standalone third entry
point, not a subcommand bolted onto either existing one, keeping both existing entry points'
own import graphs completely unchanged — INV-1's acyclic guarantee holds because this new module
has zero outbound imports of `cli.py`/`experiment.py` (the two modules that *would* import it,
if either did — neither does).

#### AR-1.6: This spec requires an ADR amending INV-1 and INV-7

Following the exact precedent set by specs 2, 4, 5, and 7 (each of which added a new module and
required an ADR amending `spec/ARCHITECTURE.md`'s Module Boundary Map/INV-1, and, where tests
need direct access, INV-7), the implementation must produce an ADR that: (a) adds
`category_extraction.py` to the Module Boundary Map and INV-1's rule text as a third
full-access orchestrator (mirroring `cli.py`/`experiment.py`'s own entry), and (b) amends INV-7
to add `query_classification.category_extraction` (all functions),
`schema.build_category_extraction_model`, and `prompts.build_category_extraction_prompt` as
accepted test-import exceptions, mirroring the exception every prior spec's own new internals
received.

#### AR-1.7: The CLI's error/exit contract mirrors `cli.py`/`experiment.py`'s existing one

`category_extraction.py`'s `main()` wraps its body in the same
`except (ValueError, FileNotFoundError, RuntimeError) as e: print(f"Error: {e}"); sys.exit(1)`
pattern both existing entry points already use — printed to stdout, exit code 1, no traceback.
No new exception sanitization is introduced: whatever `Classifier`'s own existing
logging/exception behavior already does (e.g. never logging an exception's raw message text for
Cerebus-related failures) is inherited unchanged, not re-implemented here. On success, the
script prints a short confirmation (the resolved output path) and exits 0.

---

## Feature 2: Tests

**Who & why:** This is an LLM-facing feature with several sharp edges (zero labels, duplicate
labels, overwrite refusal) that are easy to get subtly wrong — tests need to ship with the code,
not be left to a future pass, per this repo's established spec convention.

### Functional Requirements

#### FR-2.1: Unit tests ship with the implementation

A `tests/test_category_extraction.py` suite (pytest, this repo's runner) exercises every
distinct Verify *clause* under FR-1.x above (not merely one test per FR number — several FRs'
Verify lines bundle more than one obligation, e.g. FR-1.7's already-exists / symlink /
directory / missing-parent cases) — no live network/provider calls, per `AGENTS.md`'s testing
policy (no real `litellm.completion`). At least one test constructs a real `Classifier` (as
`category_extraction.main()` does) and asserts its `model_id`/system prompt/`max_retries`/
gateway kwargs are wired correctly — mirroring `tests/test_experiment.py`'s `fake_classify`
fixture (monkeypatching `Classifier.classify` at the class level), not only a duck-typed fake
passed directly to the pure `extract_category_from_prompt` function. The full existing `pytest`
suite must still pass — this feature must not regress any existing test.

**Verify:** `.venv/bin/python -m pytest` passes in full (not just the new file), with at least
one dedicated test per distinct Verify clause under FR-1.x, plus at least one CLI-level test
asserting the constructed `Classifier`'s configuration.

---

## Data Requirements

The on-disk `categories.json` shape is entirely unchanged from `categories.py`'s existing
`{"categories": [...]}` format — this feature only ever produces exactly one `Category` in that
array, but the file remains loadable, byte-for-byte compatible, by every existing consumer
(`cli.py --categories`, `experiment.py --categories`) with no changes to either.

## Integration Points

The produced `categories.json` is meant to be handed to `classify.py --categories <output>` or
`experiment.py classify/run --categories <output>` exactly like a hand-written or induced one —
no changes needed to either consumer.

## Related Specs

| Spec | Relationship | Affected Requirements |
|------|-------------|---------------------|
| Spec 2: Experiment Runner | **References** — reuses `induction.py`'s pure-function contract for one function within the module, its write-then-round-trip-validate pattern, and `experiment.py`'s `_check_reserved_sentinel`/error-handling conventions | AR-1.1, FR-1.4, FR-1.6, AR-1.7 |
| Spec 7: Cost and Timing Logging | **References** — reuses `cost.write_json_atomic` for the atomic write | FR-1.6 |
| *(future, not yet written)* | **Depends on** — a later spec may wire this feature's output directly into `experiment.py`'s `induce`/`run` flow (e.g. as an alternative to labeled-data induction, or a preflight step ahead of it) — deliberately deferred, not forgotten; see Out of Scope | — |

## Constraints

- No live network/provider calls in tests (`AGENTS.md`'s hard testing rule).
- `category_extraction.py` is a third full-access orchestrator module (AR-1.1); its own
  `extract_category_from_prompt` function uses only `categories.py`+`classifier.py`, but the
  module itself imports `schema.py`/`prompts.py`/`resources.py`/`cost.py` too, for `main()`'s
  needs. Requires an ADR amending INV-1/INV-7 (AR-1.6).
- `cli.py`, `experiment.py`, `pipeline.py` are not modified.
- This feature invents no input-size or output-label-count cap of its own — the underlying
  provider's own context-window/output-token limits apply exactly as they do to every other
  `Classifier` call in this repo; a very large instruction file or an unusually large invented
  label set is expected to fail (or truncate) the same way any other oversized `classify()` call
  would, not a new, repo-specific failure mode.
- "One LLM call"/"one `classify()` invocation" may itself involve more than one underlying
  provider request, via `Classifier`'s existing, unmodified retry/fallback behavior.

## Out of Scope

- **Any integration with `experiment.py`/`cli.py`** — this spec ships a fully standalone,
  self-contained tool only (AR-1.4/AR-1.5). Wiring its output into `experiment.py`'s `induce`/
  `run` flow (e.g. as an alternative to — or preflight ahead of — labeled-data induction) is a
  deliberate, staged next step, not an oversight: get the standalone extraction tool right and
  independently useful first, then design the integration as its own follow-up spec once this
  one is implemented and in use. Do not add any `experiment.py`/`cli.py` flag, subcommand, or
  import in this pass.
- Extending or merging a new category into an **existing** `categories.json` (multi-category
  taxonomies via this tool) — this always produces a fresh, single-category file at `--output`.
- Iterative refinement (supplying feedback to adjust a previously-generated `categories.json`).
- `--critics`/`--models`/`--batch`-style multi-classifier or self-consistency sampling for this
  one-shot extraction call — a single `Classifier`, a single call, with its own existing
  retry/fallback behavior, is the entire mechanism.
- A `cost_report.json`/timing artifact — this is a lightweight, single-file-output utility, not
  a tracked "run" in the `experiment.py` sense; no run directory, no cost logging.
- Validating that the LLM's invented labels are non-overlapping or mutually exclusive in
  *meaning* — only structural validation is performed (non-empty list, no duplicate/blank/
  reserved-sentinel `value` strings, no blank `description` strings).
- A no-clobber/locking mechanism closing the `--output` preflight-check-then-write race (FR-1.7)
  — accepted, pre-existing-shape risk, not solved here (see FR-1.7).
- A bespoke token/character cap on the instruction file or the invented label count — provider
  limits apply as-is (see Constraints).

## Spec Completeness Checklist

- [x] **Scope & acceptance criteria** — scope is a single new standalone entry point producing
  exactly one category from one instruction file; non-scope and acceptance criteria are stated
  in Out of Scope and each FR's Verify line.
- [x] **Testing strategy** — FR-2.1 requires a dedicated `tests/test_category_extraction.py`
  suite covering every distinct Verify *clause* under FR-1.x (not just one test per FR number),
  at least one CLI-level test against a real `Classifier`, and a full, unregressed `pytest`
  suite — using this repo's established monkeypatched-`Classifier.classify` fake-LLM pattern.
- [x] **Existing patterns** — explicitly modeled on `induction.py` (pure function for one
  function, no filesystem access), `schema.py`'s `create_model` builders (including
  `build_batch_model`'s `extra="forbid"` precedent), `prompts.py`'s read-template pattern,
  `resources.py`'s default-path convention, `experiment.py`'s `_check_reserved_sentinel`/error-
  handling conventions, and the `classify.py`/`experiment.py` sibling-script entry-point shape.
- [x] **Dependencies** — no new external library; reuses `litellm`/`pydantic` already in use.
  Verified live (Step 3 research) that an unbounded `list[Label]` field serializes under strict
  mode without `minItems`/`maxItems`, avoiding `build_induction_model`'s fixed-arity workaround;
  also verified live that specific `--category-name` values crash `build_classification_model`
  downstream (FR-1.2) and that `extra="forbid"` is needed for the JSON-fallback path (AR-1.2).
- [x] **Architecture & interfaces** — AR-1.1 through AR-1.7 pin the new module's shape (a third
  full-access orchestrator, not a restricted leaf), function signature, schema shape,
  prompt/resource additions + override flag, CLI flag surface, the required ADR amending
  INV-1/INV-7, and the CLI's error/exit contract.
- [x] **Error handling & failure modes** — FR-1.2 (invalid `--category-name`), FR-1.3 (zero
  labels), FR-1.4 (duplicate/reserved label values), FR-1.5 (blank strings), FR-1.7 (overwrite
  refusal / symlink / directory / missing-parent, all checked before any LLM call) are explicit,
  testable failure modes; AR-1.7 pins the CLI's exit-code/error-stream contract. Retry/fallback
  behavior on transient LLM failures is inherited unchanged from `Classifier.classify()` — no
  new retry logic is introduced.
- [x] **Security review** — no new secrets or authentication; `--cerebus`/`--api-base` reuse the
  exact same gateway-credential resolution `cli.py`/`experiment.py` already use. This does
  introduce one new, explicit thing worth naming rather than dismissing: the full instruction
  file's content is transmitted to whichever provider is configured (the same trust model as any
  existing `--input` CSV cell, but worth stating since a "prompt file" might otherwise be assumed
  local-only), and the LLM-authored category/label descriptions are later embedded into a
  *future* classification system prompt via `build_classification_model` — an indirect path from
  untrusted instruction-file content into a later, higher-privilege prompt. Mitigation is the
  same "treat as untrusted data" framing `induction.py` already relies on for sampled example
  text — an accepted, pre-existing risk shape in this codebase, not a new one requiring new
  mitigation code.
- [x] **Performance impact** — one `Classifier.classify()` invocation per run (which may itself
  issue more than one underlying provider request via existing retry/fallback); no batching, no
  concurrency. No new input-size/output-count cap is invented — provider-side limits apply as-is
  (see Constraints).
- [x] **Rollout & migration** — purely additive: a new module, a new root script, a new
  resource file, a new `resources.py` constant, and one `spec/ARCHITECTURE.md`/ADR amendment
  (AR-1.6). No existing file's behavior changes. No migration or rollback concerns beyond
  deleting the new files and reverting the ADR-driven `spec/ARCHITECTURE.md` edit.
- [x] **Assumptions & risks** — the main risk is prompt-injection-style content in the
  instruction file attempting to redirect the model, and the same content's indirect promotion
  into a later classification system prompt (see Security review); mitigated the same way
  `induction.py` mitigates it for sampled example text. Assumes the configured model supports
  structured output or falls back to JSON mode automatically, exactly as every other `Classifier`
  caller already assumes. Assumes a source-checkout or editable install for the bundled default
  prompt template (INV-3, unchanged limitation, mitigated the same way `cli.py`/`experiment.py`
  already mitigate it — AR-1.3).
