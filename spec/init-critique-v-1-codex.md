# Critique of `AGENTS.md` and `spec/ARCHITECTURE.md`

## Executive assessment

The reverse-engineered files capture the broad shape of the repository, but
`ARCHITECTURE.md` mixes four different things under the word *invariant*:
prescriptive boundaries, implementation details, observed bugs, and one-off
verification results. That weakens the authority of the genuinely useful
constraints. More importantly, the document overstates validation and resume
semantics while omitting several contracts that protect user data and make the
installed package function.

The highest-priority corrections are:

1. Specify and enforce category-name uniqueness, non-collision with the input
   column, non-empty categories, and valid result-label semantics.
2. Correct the `--restore` contract: the implementation restores from
   `input_path`, not from a separate `output_path`.
3. Make bundled resources work in a built wheel, or explicitly declare that
   only source/editable installs are supported. The current wheel contains no
   resources.
4. Replace non-positive numeric CLI edge cases with explicit validation.
5. Remove current bugs/quirks (`INV-4`, `INV-5`, `INV-9`, and `INV-11`) from the
   invariant section unless they are genuinely intended long-term behavior.

All findings below are labeled as requested. `verified-against-code` means the
claim was checked against repository files or an offline execution. `speculative`
means it is a design risk that cannot be resolved from the current code alone.

## 1. Wrong, unenforceable, or contradicted invariants and contracts

### 1.1 `INV-3` describes an implementation accident and is misstated

**Label: verified-against-code**

`resources.py:13` uses `Path(__file__).resolve().parents[2]`. The file is three
directory edges below the project root (`src/query_classification/resources.py`);
`parents[2]` is a zero-based index, not “exactly 2 levels below.” More
fundamentally, preserving the location of one Python file is not the meaningful
contract. The useful invariant is that default resources remain resolvable in
every supported execution/install mode.

The present formulation actively protects a fragile path traversal. Replace it
with an install-mode/resource-availability contract, preferably implemented with
package data and `importlib.resources`.

### 1.2 `INV-4` freezes an installer deficiency as architecture

**Label: verified-against-code**

`install.sh:111-112` currently installs only `requirements.txt`, so the factual
observation is correct. It should not be a binding invariant that the project
installer “never” installs the project. Doing so makes the documented module and
library entry points fail immediately after the repository's own installer and
prevents a straightforward improvement to `install.sh` without an ADR.

Move this to setup documentation or a known limitation. If retained as a
constraint, it needs a product rationale; none is given.

### 1.3 `INV-5` infers provenance that the repository cannot guarantee

**Label: verified-against-code**

Only `output/` is ignored (`.gitignore:19`). `data/` is not ignored, and a user
can legitimately add another source/input CSV there. The fact that the current
untracked `data/example_queries_classified.csv` matches the install-script example
does not establish that *every future CSV* under `data/` is generated. “Never
hand-edit” is also not mechanically enforceable and is too broad for an arbitrary
data directory.

A narrower rule would identify generated paths by convention or require outputs
under ignored `output/`. The current rule should not be cited as a Module Boundary
Map property because neither `data/` nor `output/` appears in that map.

### 1.4 `INV-7` is false when `max_retries <= 0`

**Label: verified-against-code**

`Classifier.classify` loops over `range(1, self.max_retries + 1)`. With zero or a
negative value, no attempt occurs and `assert last_exc is not None` raises an
`AssertionError`; there is no “last exception” to re-raise. The CLI accepts
`--retries 0` and negative values, and the constructor performs no validation.

Either state a precondition (`max_retries >= 1`) and enforce it at both API and
CLI boundaries, or weaken the invariant to positive retry counts only.

### 1.5 `INV-8` overclaims that no locking is needed

**Label: speculative**

The DataFrame mutation claim is verified: `pipeline.py:84-95` writes only on the
main thread. However, one shared `Classifier` instance and LiteLLM's process-wide
state are called concurrently. The code provides no thread-safety contract for a
custom classifier, LiteLLM, or future mutable classifier state. “No locking is
used” is descriptive; “or needed” is broader than the evidence supports.

Keep the narrow DataFrame ownership rule. If `classify_csv` is intended to accept
only thread-safe classifiers, make that an explicit caller precondition.

### 1.6 `INV-9` turns a known observability/data-quality bug into required behavior

**Label: verified-against-code**

`pipeline.py:89-90` does silently count and discard row failures, so the current
state is accurately described. Calling it an invariant means an implementer must
seek an ADR merely to add error reporting. It is neither necessary for
compatibility nor justified as desired behavior. Put it in `spec/docs/` as a
known limitation, or state the desired failure-reporting contract prescriptively.

### 1.7 `INV-10` contradicts itself and the boundary table

**Label: verified-against-code**

The heading and table say tests never import internal modules directly, but both
the rule and `tests/test_building_blocks.py:12-16` explicitly import
`query_classification.resources`. `resources` is a sibling/internal module and
its constants are not re-exported by `__init__.py`. The stated rationale about
surviving internal ownership refactors therefore does not apply.

Choose one policy: test through the public API and export the required resource
contract, or permit targeted direct imports. The existing test demonstrates the
latter.

### 1.8 `INV-11` records a hazardous quirk rather than a desired invariant

**Label: verified-against-code**

`cli.py:107` does call `load_dotenv(override=True)`, so `.env` values override
exported shell values for the CLI. This behavior is not shared by library usage:
`classifier.py` reads the process environment but never loads `.env` itself.

If override precedence is deliberate, state why it is part of the CLI contract.
Otherwise this belongs in current-state/known-issue documentation, not in the
binding invariant set. The Shared Files wording at `ARCHITECTURE.md:56` is also
imprecise when it says `classifier.py` “implicitly” reads `.env`; it reads
environment variables after the CLI has loaded the file.

### 1.9 The system-prompt placeholder contract is not enforced and is false as a
runtime requirement

**Label: verified-against-code**

`prompts.py:32-36` calls `str.format` with two named arguments, but a template is
not required to reference either argument. A custom template containing neither
placeholder renders successfully. Conversely, an unrelated brace/placeholder
raises `KeyError`. The test checks only that the bundled template's placeholders
were filled; it does not validate custom templates.

“Any custom template must contain both” is therefore a desired rule, not a stable
contract implemented by the code. Either enforce and test it or document that
placeholders are optional inputs to `format`.

### 1.10 The output-label contract claims enforcement that does not exist

**Label: verified-against-code**

`schema.py:36-39` enforces only that each result value is a `list[str]` of length
1-3. Offline validation accepted all of the following:

- an unconfigured label (`"invented"`);
- duplicate labels;
- `"none"` together with a configured label;
- forbidden `"none - positive"`;
- an extra result field, which Pydantic silently discarded.

Ordering by evidence is inherently prompt-level rather than schema-enforceable.
The sentence at `ARCHITECTURE.md:121` saying the complete convention is
“Enforced by `schema.py:36-39`” is wrong. Only cardinality and string type are
enforced.

### 1.11 “JSON list” does not describe the persisted CSV representation

**Label: verified-against-code**

`pipeline.py` assigns Python lists to object cells and delegates serialization to
`pandas.DataFrame.to_csv`. An offline one-row run produced
`hello,['positive']`, using Python's single-quoted list representation rather
than valid JSON. The in-memory value is JSON-serializable, but the persisted cell
is not JSON text. The Stable Contract and README language should distinguish
those two meanings or the pipeline should call `json.dumps` explicitly.

### 1.12 `from __future__ import annotations` “in every module” is contradicted by
the repository

**Label: verified-against-code**

`AGENTS.md:24` requires the future import in every module, but
`src/query_classification/__init__.py`, `src/query_classification/__main__.py`,
and root `classify.py` do not contain it. If the intent is “every module that has
annotations,” say so. If the literal policy is intended, the baseline already
violates it.

### 1.13 The broad-exception lint suffix is not enforceable by the stated toolchain

**Label: verified-against-code**

The explanatory-comment requirement is actionable in review. The mandatory
`# noqa: BLE001` portion is inert because the repository explicitly has no Ruff,
Flake8, or other lint gate. Requiring a tool-specific suppression while declaring
that tool absent is filler unless linting is planned. Preserve the justification
rule and either add the checker or drop the suppression requirement.

## 2. Missing load-bearing constraints

### 2.1 Bundled resources must survive supported installation modes

**Label: verified-against-code**

This is the most consequential missing architectural contract. A wheel built
from the current `pyproject.toml` contains the Python package and metadata only;
it contains no `resources/categories/*.json` or `resources/prompts/*.txt`.
After a normal wheel install, `PROJECT_ROOT = parents[2]` points into the Python
installation hierarchy, where those top-level files do not exist. The CLI's
default categories and prompt consequently fail outside the checkout/editable
layout.

Add a constraint covering wheel contents and resource lookup, plus a smoke test
that installs the wheel into an isolated environment and runs `--help` and a
default-resource load. Alternatively, explicitly scope supported use to a source
checkout and remove claims of a generally installable package.

### 2.2 Category names need identity and collision invariants

**Label: verified-against-code**

No validator requires categories to be non-empty or category names to be unique.
`schema.py` uses a dict keyed by name, so duplicate categories silently collapse
to one model field. The pipeline also uses category names as CSV columns. If a
category has the same name as the source `--column`, `pipeline.py:63` clears that
column before work is submitted, causing the model to classify the string
`"None"` rather than the original text. Existing input columns with category
names are also overwritten when `restore=False`.

At minimum, specify:

- at least one category;
- unique, non-empty category names;
- unique, non-empty label values within each category;
- category names must not equal the input text column;
- an explicit policy for collisions with other existing CSV columns (reject,
  require overwrite confirmation, or intentionally overwrite).

### 2.3 Result keys and label values need a real validation contract

**Label: verified-against-code**

The dynamic result model should define whether extra keys are forbidden and
whether values are limited to configured labels plus a precisely specified
`none` extension. Without this, the structured-output fallback can return
plausible but invalid classifications that are silently accepted. This is a
load-bearing data-quality boundary, not merely prompt wording.

### 2.4 Resume-source semantics are missing and the user-facing wording is
misleading

**Label: verified-against-code**

`classify_csv` always calls `pd.read_csv(input_path)` and only writes to
`output_path`. Therefore rerunning
`--input original.csv --output classified.csv --restore` reloads the original
file and repeats all work; it does not inspect `classified.csv`. Restore works
only when the prior result is now supplied as `--input` or output overwrites
input. The CLI/README promise “resume from a previous run” without this
qualification.

Specify which file is authoritative on resume, how schema/category changes are
handled, and whether partial output must be atomically readable. This matters for
costly LLM workloads.

### 2.5 Numeric parameters need explicit domains

**Label: verified-against-code**

The CLI accepts all integers:

- `--limit 0` means no limit because `pipeline.py:65` tests truthiness;
- a negative limit slices from the end rather than rejecting the input;
- `--workers <= 0` silently becomes one worker via `max(1, workers)`;
- `--retries <= 0` causes the `AssertionError` described above;
- the public `save_every=0` causes modulo-by-zero at `pipeline.py:94`, and a
  negative value has surprising cadence.

Add positive/non-negative contracts and enforce them in both public APIs and
argparse. Decide explicitly whether zero limit means zero rows or is invalid.

### 2.6 Input/output durability and failure semantics are unspecified

**Label: verified-against-code**

When `--output` is omitted, the program overwrites the input directly. Every
incremental save rewrites the entire CSV to the target path without an atomic
temporary-file replacement or backup. There is no invariant covering crash-time
file integrity, output-parent creation, CSV dialect/encoding preservation, or
the exit status when some rows fail. In fact, per-row failures still lead to a
normal return and CLI exit code zero.

These choices can cause data loss or falsely signal successful automation. A
prescriptive architecture should define at least atomicity/backup expectations,
partial-success signaling, and whether overwriting source/category columns is
allowed.

### 2.7 Fallback boundaries need definition

**Label: speculative**

`Classifier._complete` treats every LiteLLM `BadRequestError` from structured
output as evidence that `json_schema` is unsupported and retries immediately in
JSON-object mode. Some bad requests may instead reflect invalid credentials,
model identifiers, schema size, or malformed request configuration. The desired
set of fallback-triggering errors is not documented or tested. Specify it if
this behavior is relied upon; otherwise a future change could hide actionable
provider errors or remove a compatibility behavior.

### 2.8 Public API stability is not actually defined

**Label: verified-against-code**

`__init__.py` re-exports eight names, but `ARCHITECTURE.md` merely says it
“re-exports the public API.” It does not say whether those names/signatures are a
compatibility contract. It also omits resource constants and
`resolve_api_base`, even though tests reach into one internal module. State what
is public and what compatibility changes require coordination.

### 2.9 Dependency declarations must remain synchronized

**Label: verified-against-code**

Runtime dependencies are duplicated in `requirements.txt` and
`pyproject.toml`, while pytest is in both base requirements and the `dev` extra.
The Shared Files section calls out `pyproject.toml` alone and misses this
cross-file drift risk. Add a single-source-of-truth rule or explicitly require
the two declarations to stay aligned.

### 2.10 Tracked source/resource prerequisites are missing

**Label: verified-against-code**

At review time, `git ls-files` showed both `pyproject.toml` and the default
`resources/categories/example_categories.json` as untracked. A clean clone from
the current index therefore lacks the build/test configuration and the CLI's
default categories file. This may simply mean the reverse-engineering changes
have not yet been committed, but completion criteria should require that files
needed by documented commands and defaults are tracked.

## 3. Module Boundary Map and shared-file errors

### 3.1 The dependency diagram omits `__init__.py`

**Label: verified-against-code**

`__init__.py:14-18` imports `categories`, `schema`, `prompts`, `classifier`, and
`pipeline`, but the Mermaid graph has no `__init__` node or edges. These eager
imports are operationally relevant: importing the package pulls in LiteLLM,
pandas, and tqdm even when a caller wants only category parsing.

### 3.2 The `__init__.py` “May import from” cell is inaccurate

**Label: verified-against-code**

“all of the above” includes `cli.py` if read literally and implies
`resources.py`, but the actual module imports neither. If this is permission
rather than an actual-import map, the document should say so and explain why the
public API is allowed to acquire dependencies on the CLI. If it is an actual
map, list the five real imports.

### 3.3 The tests row is internally contradictory

**Label: verified-against-code**

Its “May import” cell permits `resources.py`, while “Must NOT import” forbids
internal modules directly. `resources.py` is internal and is directly imported
by the existing test. This is the map-level form of the `INV-10` problem.

### 3.4 The map omits paths used by its own constraints

**Label: verified-against-code**

`INV-5` sends readers to the Module Boundary Map for `data/`/`output/`, but the
map lists neither. It also omits the packaging/install layer (`pyproject.toml`,
`requirements.txt`, `install.sh`), `.env.example`, and documentation consumers
whose synchronization is discussed under Shared Files. Not all need code-style
import boundaries, but a separate artifact/ownership map is needed if they are
load-bearing.

### 3.5 Shared files are incomplete and one “shared file” is not a file

**Label: verified-against-code**

Missing shared artifacts include:

- `requirements.txt` paired with `pyproject.toml` dependency declarations;
- `resources/prompts/example_task_description.txt`, referenced by README,
  exported as a default/example path, and asserted by tests;
- `.env.example`, referenced by README and copied by `install.sh`;
- `README.md`, which mirrors flags, category semantics, resource paths, and
  entry-point commands;
- root `classify.py`, coupled to the src-layout and CLI entry point.

The bullet beginning “The categories JSON shape...” is a shared *contract*, not
a shared file. It belongs under Stable Contracts. Its “keep in sync by hand”
wording also omits `categories.py`, the actual input model, and incorrectly
bundles schema-enforced cardinality with prompt-only label semantics.

### 3.6 Arrow semantics should be made explicit

**Label: speculative**

The Mermaid arrows run from dependency/provider to importer/consumer
(`categories --> schema` means schema imports categories). That is internally
consistent, but many architecture diagrams use the opposite “imports/depends
on” direction. A one-line legend would prevent readers from reversing the
boundary.

## 4. Command problems

### 4.1 Commands do not reliably select the created virtual environment

**Label: verified-against-code**

After `./install.sh`, `AGENTS.md` recommends bare `pip`, `python`, and `pytest`
without first requiring `source .venv/bin/activate`. Those commands may use a
global or unrelated interpreter. `install.sh` itself correctly uses the venv's
absolute interpreter and only prints activation at the end.

Use reproducible forms such as:

```bash
./install.sh
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
.venv/bin/python classify.py --help
```

or make activation an explicit preceding command.

### 4.2 The editable-install guidance is muddled

**Label: verified-against-code**

“Add `.[dev]` for pytest if not already via `requirements.txt`” forces the reader
to reason about two dependency paths. Today `install.sh` installs pytest from
base `requirements.txt`, while `pyproject.toml` calls it optional development
software. Pick one supported setup flow. Also quote `'.[dev]'` in shell examples
to avoid glob interpretation.

### 4.3 The canonical classifier command omits the model/provider requirement

**Label: verified-against-code**

The abbreviated AGENTS command omits `--model`, causing the Azure-specific
default `azure/gpt-5-chat` to be used. Whether that works depends on provider
credentials and deployment configuration not stated in the command. A command
intended only to show syntax should say so; an executable smoke command should
include a configured model or remain `--help`/offline.

### 4.4 Verification commands mix durable instructions with stale observations

**Label: verified-against-code**

“All commands ... were run live,” Python `3.12.7`, “4 passed in ~10s,” and the
credential availability of one session are historical results, not commands.
The current offline run still passes four tests but took about 2.25 seconds,
illustrating why timing does not belong in prescriptive architecture. Keep only
the commands and expected pass/fail criteria; move the session report to
`spec/docs/` or a review artifact.

### 4.5 Missing verification covers the most fragile supported mode

**Label: verified-against-code**

There is no wheel build/install/resource smoke test. `pip install -e .` masks the
resource-packaging defect because editable imports point back into the checkout.
A critical verification sequence is build wheel, install into a clean temporary
environment, and load the bundled defaults. CLI parser tests should also cover
the documented flags without network calls.

### 4.6 `./install.sh --force` is an unnecessarily destructive default check

**Label: verified-against-code**

The verification table presents `--force`, which deletes `.venv`, as the normal
dependency-install check. Fresh-environment verification is useful, but everyday
agent guidance should use `./install.sh`; reserve `--force` for an explicit clean
rebuild because it discards an existing environment and incurs a full reinstall.

## 5. `AGENTS.md` rule quality and duplication

### 5.1 The testing policy conflates offline testing with module exclusion

**Label: verified-against-code**

`classifier.py`, `pipeline.py`, and `cli.py` can all be tested offline with a fake
classifier, temporary CSVs, monkeypatching, and parser/unit tests. The README
project-layout section does not document an intentional no-coverage boundary;
the test module docstring merely says the *existing* tests are offline building-
block tests. “Unless you're deliberately expanding this boundary” provides no
criterion or approval path.

Retain “tests must not make real provider/network calls.” Remove the prohibition
on offline coverage of orchestration code, or identify a concrete reason and
process for exceptions.

### 5.2 The generated-CSV prohibition duplicates and inherits the flaws of
`INV-5`

**Label: verified-against-code**

`AGENTS.md:23` repeats the same overbroad claim and incorrectly points to the
Module Boundary Map. Prefer one precise rule in AGENTS: do not modify known
generated artifacts unless the task explicitly targets them. Define those paths
accurately elsewhere.

### 5.3 Required patterns mix language compatibility, path architecture, and lint
style in one bullet

**Label: verified-against-code**

`AGENTS.md:24` compresses three unrelated rules, two duplicated from
`ARCHITECTURE.md`. This makes individual requirements easy to miss and leaves
scope ambiguous (“every module,” “new default resource paths,” “one-line
comment”). Split them, keep agent-operational style rules locally, and link to
architecture rather than restating resource and exception invariants.

### 5.4 “4/4 as of this writing” is brittle filler

**Label: verified-against-code**

The enforceable rule is `pytest` exits successfully. A fixed test count becomes
wrong as soon as tests are added and can discourage needed coverage. Remove the
count from both AGENTS and architecture.

### 5.5 “No lint/typecheck/CI gate” and tool inventory are current-state facts

**Label: verified-against-code**

These facts are useful for agent orientation but not rules. Compress them into a
single “required verification today” statement. Do not repeat them in the
overview, command section, build rule, Verification Commands, and Unverified
section.

### 5.6 The spec workflow is not self-contained

**Label: speculative**

`/spec-help`, `/spec-docs`, and `/spec-close` are not scripts or documentation in
this repository. They may be supplied by an external skill, but a fresh
implementer cannot discover or verify them from the checkout. If they are
required workflow, identify the owning tool and prerequisite. “Do not implement
unspecced features without flagging it” also does not say to whom/how to flag or
what qualifies as a feature versus a fix.

## 6. Descriptive content that belongs in `spec/docs/`

### 6.1 Current implementation inventory

**Label: verified-against-code**

The Overview's “single-package batch CLI,” “there is no server/database,” and
examples of supported domains describe the present repository. Keep only the
domain-agnostic product boundary if it is truly prescriptive; move the rest to
generated/current-state docs.

### 6.2 Observed quirks and known gaps

**Label: verified-against-code**

`INV-4` (installer omits package), `INV-5` (current artifact provenance),
`INV-9` (silent failures), and `INV-11` (`.env` override behavior) are current
behavior or known issues. They should not become permanent merely because they
were reverse-engineered.

### 6.3 Evidence line references and session transcript

**Label: verified-against-code**

The repeated `Evidence:` clauses, “verified live in this session,” exact Python
patch version, elapsed test time, current git status, unavailable credentials,
and absence of tools/CI are audit notes. They are valuable in this critique or a
baseline report, but they conflict with the file's declaration that it contains
prescriptive constraints.

### 6.4 “No console script,” README drift, and “no specs yet” status

**Label: verified-against-code**

These are repository-status facts, not architecture. Put them in current-state
docs or actionable backlog items. In particular, “flag drift if you touch either”
is weaker than a stable contract requiring CLI help and user documentation to be
kept synchronized.

### 6.5 Module responsibilities versus module permissions

**Label: verified-against-code**

The Responsibility column is descriptive; the May/Must-not columns are
prescriptive. Keeping both can be useful, but the document should explicitly
separate “current owner” from “allowed dependency.” Otherwise routine refactors
appear to violate architecture even when no load-bearing boundary changes.

## 7. Size, scope, and adherence

### 7.1 Too many low-value invariants obscure the important ones

**Label: verified-against-code**

Of eleven invariants, four mainly preserve current defects/quirks (`INV-4`,
`INV-5`, `INV-9`, `INV-11`), one preserves a fragile file location (`INV-3`), and
one is a lint-comment convention (`INV-6`). Meanwhile resource installability,
category collisions, resume semantics, and output validity are absent. This
allocation makes an implementer more likely to comply with cosmetic constraints
while breaking data integrity.

### 7.2 Repetition creates drift without adding authority

**Label: verified-against-code**

The no-tooling status, test count, editable-install requirement, generated CSV
rule, default resource ownership, and broad-exception rule are repeated across
AGENTS and architecture. Several repetitions are already inconsistent in scope.
AGENTS should be a short execution guide; architecture should own durable
boundaries and contracts; `spec/docs/` should own observed state.

### 7.3 File/line citations will decay quickly

**Label: verified-against-code**

Nearly every invariant embeds exact line numbers. They aid this initial audit but
become stale after harmless edits and create maintenance work unrelated to
behavior. Prefer symbol names and tests as enforcement. Keep exact citations in
the reverse-engineering report, not the durable specification.

### 7.4 The empty Aspirational section is filler

**Label: verified-against-code**

“None—this repository has no specs yet” conveys project status, not a constraint.
Remove the empty section until it has content. More importantly, do not use
“aspirational invariant” for unenforced requirements; use planned work/specs with
acceptance criteria.

## Recommended rewrite shape

Keep `AGENTS.md` compact:

- exact environment/install/test commands using `.venv/bin/python`;
- the offline/no-live-provider test rule;
- generated-artifact and secret-handling cautions;
- language/style conventions that agents must apply;
- a link to architecture for actual boundaries.

Reduce `ARCHITECTURE.md` to durable, testable rules:

- supported entry points and install modes, including packaged resources;
- dependency directions, with complete `__init__` edges;
- category/input/result contracts and collision rules;
- retry/fallback and partial-failure semantics;
- CSV overwrite, atomicity, serialization, and resume-source semantics;
- public API compatibility surface;
- one authoritative verification criterion per invariant where practical.

Move implementation inventory, exact current imports, live command results,
known shortcomings, missing tooling, and drift observations into `spec/docs/` or
a baseline audit document. This would make violations materially meaningful and
substantially improve adherence.
