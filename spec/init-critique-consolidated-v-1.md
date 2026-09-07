# Init Critique — Consolidated v1

Critics: Claude (in-session), Codex (`gpt-5.6-sol`, background). Gemini was unavailable
(the `gemini` CLI is not installed on this machine) — only 2 of 3 planned critiques ran.

## Executive Summary

Claude's in-session pass found 3 real gaps (missing `.env`-override invariant, missing
shared-file, missing stable contract) — all applied directly, see `spec/init-critique-v-1-claude.md`.

Codex's pass was substantially more consequential: it found that v1 of `ARCHITECTURE.md`
conflated **binding invariants** with **current bugs/quirks reverse-engineered from the
code**, and — more importantly — surfaced several **verified, load-bearing correctness
gaps** in the actual codebase that v1 didn't capture at all: unvalidated numeric CLI
params, a category-name collision that corrupts input data, `--restore` not resuming from
a separate `--output` file, and default resources being entirely absent from a built
wheel. All of the findings below were independently re-verified (empirically, not just by
re-reading Codex's report) before being applied. See `spec/init-critique-v-1-codex.md` for
the full text.

## Blocking Items (fixed)

All labeled `verified-against-code` by Codex, and independently re-verified here before
being applied to `AGENTS.md` / `spec/ARCHITECTURE.md`:

1. **Wheel install has no bundled resources at all** — built `pip wheel . --no-deps` and
   inspected the archive: it contains only `query_classification/*.py` files, zero
   `resources/*.json|*.txt`. `PROJECT_ROOT`'s `parents[2]` traversal resolves into
   site-packages in that mode, where `resources/` doesn't exist. → Added as INV-3 (resource
   packaging caveat), replacing the narrower "fragile path" framing.
2. **`max_retries <= 0` crashes with `AssertionError`, not a normal exception** —
   reproduced live: `Classifier(..., max_retries=0).classify(...)` raises
   `AssertionError`, not whatever the actual failure was. CLI's `--retries` has no
   validation. → Added as INV-5 (precondition `max_retries >= 1`), replacing the
   overclaim that "the last exception is always re-raised."
3. **A category named the same as `--column` corrupts the source text** —
   `pipeline.py:53-63` nulls all category columns (when not `--restore`) before the
   classify loop reads `df.at[idx, column]` at line 79. If a category's `name` equals
   `--column`, the text gets nulled to `None` before it's ever classified. Confirmed by
   reading the exact line order (mutation at line 63, read at line 79, same `df`, no copy).
   → Added as INV-8.
4. **Duplicate category names silently collapse** — reproduced live: two `Category`
   entries both named `x` produce a single Pydantic field `x`, keeping only the second
   category's description; the first is silently dropped. → Folded into INV-8.
5. **The result schema enforces far less than v1 claimed** — reproduced live: the dynamic
   model accepts an unconfigured label (`"invented"`), duplicate labels, `"none"` combined
   with a real label, and the explicitly-forbidden `"none - <existing label>"` form. Only
   cardinality (1-3) and string type are enforced (`schema.py:36-39`). v1's Stable
   Contracts entry said this convention was "Enforced by `schema.py:36-39`" — that was
   wrong for everything except cardinality/type. → Corrected the Stable Contracts entry;
   added INV-11 to state precisely what is/isn't schema-enforced.
6. **The persisted CSV cell is not JSON text** — reproduced live: writing a list into a
   DataFrame cell and calling `to_csv` produces `['positive']` (Python repr), not
   `["positive"]` (JSON). The in-memory value is JSON-*serializable*; what's on disk is
   not JSON. → Corrected the Output CSV Stable Contract wording.
7. **`--restore` resumes from `--input`, never from a separate `--output`** —
   `classify_csv` always calls `pd.read_csv(input_path)` (`pipeline.py:46`) and only
   writes to `output_path`. Re-running with `--input original.csv --output out.csv
   --restore` re-reads the untouched original and repeats all work. → Added as INV-10.
8. **Numeric CLI params are unvalidated in several surprising ways** — confirmed by
   reading the exact lines: `--limit 0` means "no limit" (`if limit:` is falsy at 0,
   `pipeline.py:65`); `--workers <= 0` silently becomes 1 (`max(1, workers)`,
   `pipeline.py:77`); the library-only `save_every` raises `ZeroDivisionError` at 0
   (`pipeline.py:94`, not reachable via the shipped CLI, which has no `--save-every`
   flag). → Added as INV-9.
9. **`INV-4`/`INV-5`/`INV-9`/`INV-11` (v1 numbering) were bugs/quirks framed as binding
   invariants** — freezing "the installer never installs the package," "`data/` CSVs are
   always generated," and "row failures are silently dropped" as *invariants* would force
   a future ADR just to fix them, and none of the three has a stated rationale for being
   permanent. → Moved to a new **Known Gaps** section (not "Invariants"), reworded to
   describe current behavior without implying it's protected/desired. `INV-5`'s claim was
   also narrowed: only `output/` is actually gitignored (`.gitignore:19`); `data/` is not,
   so a user can legitimately add other input CSVs there — v1's blanket "never hand-edit
   `data/`" was too broad.
10. **Module Boundary Map errors** — `__init__.py` was missing from the dependency
    diagram despite importing 5 modules eagerly (pulling in litellm/pandas/tqdm on any
    import); its "May import from" cell said "all of the above" which isn't literally true
    (it doesn't import `cli.py` or `resources.py`). The `tests/` row was self-contradictory
    (permits `resources.py` in one cell, forbids "internal modules directly" in the next) —
    matches the real code (`tests/test_building_blocks.py:12-16` imports `resources.py`
    directly). → Fixed the diagram and both cells; reworded the tests invariant (was
    INV-10) to state the direct-`resources.py`-import exception explicitly instead of
    contradicting it.
11. **Missing Shared Files** — `requirements.txt` (duplicates `pyproject.toml`'s
    dependency list — confirmed both list the same 5 runtime deps + pytest), `.env.example`,
    `resources/prompts/example_task_description.txt`, `README.md`, and root `classify.py`
    were all touched-by-multiple-areas but not listed. → Added.
12. **AGENTS.md commands don't reliably hit the created venv** — bare `pip`/`python`/
    `pytest` depend on activation or `$PATH` ordering; `install.sh` itself uses the venv's
    absolute interpreter rather than relying on activation. → Reworded to lead with
    `source .venv/bin/activate` (matching `install.sh`'s own printed instructions) and note
    `.venv/bin/python`/`.venv/bin/pytest` as the unambiguous alternative.
13. **`from __future__ import annotations` "in every module" is false** — `__init__.py`,
    `__main__.py`, and root `classify.py` don't have it (confirmed by re-reading each — none
    use type hints, so there was nothing to future-import). → Scoped the rule to modules
    under `src/query_classification/` that use type hints.
14. **Testing-policy rule overstated an "intentional" coverage boundary** — the module
    docstring and README describe the *existing* tests as offline, not a rule against ever
    testing `classifier.py`/`pipeline.py`/`cli.py`. → Reworded: the hard rule is "no live
    network/provider calls in tests," not "don't add coverage for those modules."
- Brittleness fixes: dropped exact "4/4"/timing figures from `AGENTS.md`'s Build/verify
  rule (kept in `ARCHITECTURE.md`'s Verification Commands table only, where a baseline is
  explicitly asked for by the template, and reworded to avoid implying a fixed count is a
  gate).

## Non-Blocking / Noted, Not Changed

- **Sample size on INV-4 (formerly INV-6, noqa/BLE001) and the tests-import-resources
  rule** — both generalize from a small number of sites (2 and 1, respectively). Left
  as-is: the evidence is honestly cited, not overstated, and both are still the only
  observed pattern.
- **`save_every` resume/atomicity semantics beyond what's stated** — Codex raised
  (2.6/2.4) that there's no atomicity/backup guarantee on the incremental CSV write and no
  distinct exit code for partial failure. Real and worth a future decision, but fixing the
  *code* is out of scope for this skill (`spec-init` documents, it doesn't patch bugs) —
  recorded in Known Gaps rather than invented as an invariant, since no current behavior
  guarantees atomicity to document as a contract.
- **Fallback-trigger scope** (`litellm.BadRequestError` triggers JSON-mode fallback
  regardless of cause) — real ambiguity, recorded in Known Gaps, not elevated to an
  invariant since the desired behavior isn't specified anywhere.
- **Line-number citations will decay** — Codex is right that they'll go stale after
  unrelated edits. Kept them anyway (all other `/spec-*` skills and ADRs cite line/file
  evidence the same way) but scoped expectations: treat a citation mismatch as a signal to
  re-verify the invariant, not proof the invariant is false.
- **Public API compatibility statement** — added a short Stable Contract entry listing the
  8 re-exported names from `__init__.py`, without over-promising semantic-versioning-style
  guarantees this 0.1.0, single-commit project hasn't made anywhere else.

## Scope Note

Per this skill's boundary, no product code was changed to fix any of the bugs above
(category collision, unvalidated numeric params, wheel packaging, schema permissiveness,
`--restore` semantics, `AssertionError` on `max_retries<=0`). All are documented as either
Invariants (things to respect/work around today) or Known Gaps (current behavior that is
not desired/protected and can be fixed without an ADR). Fixing them is a normal follow-up
task, not part of this backfill.
