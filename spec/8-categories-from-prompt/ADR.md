# Architectural Decision Records: 8-categories-from-prompt

## ADR-001: New `category_extraction.py` node (INV-1), a third but narrower orchestrator, plus this spec's own INV-7 test-import names

- **Date:** 2026-09-17
- **Status:** Accepted
- **Context:** This spec adds a standalone entry point that extracts a single `Category`
  taxonomy from a free-text instruction file, via one `Classifier.classify()` call, with no
  labeled training data involved. A new module, `src/query_classification/category_extraction.py`,
  owns this: `extract_category_from_prompt()` (a pure function, mirroring `induction.py`'s own
  no-filesystem-access contract) plus `build_parser()`/`main()` (the CLI orchestration, reading
  the instruction file and writing `categories.json`). `cli.py` and `experiment.py` were
  previously the only two modules INV-1 authorized to import the full set of shared building
  blocks; this spec adds a **third**, `category_extraction.py`, but a **narrower** one than
  either existing orchestrator — it may import exactly `categories.py`, `classifier.py`,
  `schema.py`, `prompts.py`, `resources.py`, and `cost.py` (not `pipeline.py`, `batching.py`,
  `debate.py`, or `multi_model.py`, since none of those are relevant to a single-classifier,
  single-call tool with no CSV loop, no batching, and no multi-classifier concept). INV-1's rule
  text and the Module Boundary Map are closed per-module lists, so this new orchestrator and its
  exact import list must be recorded, or the invariant would contradict the shipped import
  graph. Separately, `tests/test_category_extraction.py` needs direct imports of
  `category_extraction` (all functions), `schema.build_category_extraction_model`, and
  `prompts.build_category_extraction_prompt` — none of which INV-7 previously named.
- **Decision:** INV-1's rule text and the Module Boundary Map gain `category_extraction.py` as
  a new, third orchestrator node — importing exactly the six modules named above, and
  explicitly **not** the full breadth `cli.py`/`experiment.py` each have — plus the mermaid
  graph's corresponding inbound edges and one new outbound edge to a new
  `extract_categories.py (repo root)` node, mirroring `classify.py`/`experiment.py (repo root)`'s
  own entries. `cost.py`'s own row/docstring gain a note that a third module (not just
  `pipeline.py`/`cli.py`/`experiment.py`) now imports it, for its shared `write_json_atomic`
  helper specifically. INV-7's accepted-exception list and the `tests/` Module Boundary Map row
  gain, from this spec: `category_extraction` (all functions), `schema.
  build_category_extraction_model`, and `prompts.build_category_extraction_prompt`.
- **Rationale:** Mirrors the precedent set by spec 5's `batching.py` and spec 7's `cost.py` (new
  sibling modules, each with a deliberately narrower import list than the two full orchestrators,
  recorded as new Module Boundary Map rows plus inbound mermaid edges) for INV-1, and by every
  prior spec's own INV-7 amendment (spec 1's `debate.py`, spec 2's `dataset_io.py`/
  `induction.py`/`experiment.py`, spec 3's Cerebus functions, spec 4's `multi_model.py`, spec 5's
  `batching.py`, spec 6's failure-classification primitives, spec 7's `cost`/
  `_capture_usage_from_response`) for INV-7 — each gives tests direct access to internals with
  no public-API equivalent, rather than forcing indirect exercise through a live provider. Unlike
  `batching.py`/`cost.py`, though, this module is itself an **orchestrator** (it has its own
  `build_parser()`/`main()`, constructs a real `Classifier`, and writes a file directly) rather
  than a building block consumed by `cli.py`/`experiment.py` — hence "a third orchestrator," not
  "a new building block," even though its import list is narrower than the other two
  orchestrators' own.
- **Consequences:** `category_extraction.py` becomes a new, independently-runnable standalone
  tool, not a subcommand of either existing entry point — `cli.py` and `experiment.py` are
  unmodified and do not import it, and it does not import either of them (INV-1's acyclic
  guarantee holds: it has zero outbound imports of `pipeline.py`, `batching.py`, `debate.py`,
  `multi_model.py`, `cli.py`, or `experiment.py`). Its effectively-public surface for test
  purposes widens INV-7's list by three names; nothing else in `classifier.py`/`categories.py`/
  `schema.py`/`prompts.py`/`resources.py`/`cost.py` becomes importable beyond what's named.
- **Invariants affected:** INV-1 (amended, new module + six inbound edges, one outbound edge to
  a new repo-root script) and INV-7 (amended, three new test-import names — distinct from, and
  not duplicating, any prior spec's own recorded names).
