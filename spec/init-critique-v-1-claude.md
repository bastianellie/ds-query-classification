# Init Critique v1 — Claude

Self-critique of `AGENTS.md` and `spec/ARCHITECTURE.md`, written in-session by re-reading
the generated files against the source they cite.

## Findings

### 1. Missing invariant: `.env` silently overrides real shell env vars (verified-against-code)
`cli.py:107` calls `load_dotenv(override=True)` before parsing args. This means a value
present in `.env` takes precedence over an already-exported shell environment variable of
the same name — the opposite of `python-dotenv`'s usual non-destructive default
(`override=False`). This is exactly the kind of non-obvious, surprising behavior that
belongs in Invariants, and it wasn't captured in v1.
**Action: add INV-11.**

### 2. Missing shared file: `.env`/`.env.example` (verified-against-code)
The "Shared files" list omits `.env`/`.env.example`, even though `cli.py` loads it
(`load_dotenv`) and `classifier.py` implicitly reads the env vars it sets
(`_API_BASE_ENV_VARS`, `classifier.py:23-29`) — two modules depend on its contents.
**Action: add a bullet.**

### 3. Missing stable contract: provider endpoint env-var priority (verified-against-code)
The priority order `LITELLM_API_BASE > AZURE_API_BASE > AZURE_OPENAI_ENDPOINT >
OPENAI_BASE_URL > OPENAI_API_BASE` (`classifier.py:23-29`) is restated verbatim in
`cli.py`'s `--api-base` help text (`cli.py:49-51`), making it user-facing documented
behavior, not an internal implementation detail. Silently reordering it would change
CLI behavior without changing the CLI's own stated contract.
**Action: add to Stable Contracts.**

### 4. Thin evidence base on INV-6 and INV-10 (verified-against-code, non-blocking)
- INV-6 (broad-except-requires-noqa-comment) generalizes from exactly 2 existing sites
  (`classifier.py:108`, `pipeline.py:89`).
- INV-10 (tests only use the public API) generalizes from the single existing test file.
Both are accurately cited and not contradicted by anything in the code, but the sample
size is small. No change recommended — the evidence is honestly presented, not
overstated — flagging only so a future critique doesn't need to re-derive this.

### 5. Non-blocking: `save_every` resume behavior isn't a Stable Contract
`--restore`'s correctness depends on the incremental-write cadence (`save_every=20`,
`pipeline.py:29`) writing before a crash. This might be worth a Stable Contract entry,
but confirming its edge cases (e.g. exact interaction with `--limit`) would need runtime
testing I haven't done. Left as a suggestion for the user, not fixed.

## Verdict
Findings 1-3 are verified against code and get applied directly. Finding 4 requires no
change. Finding 5 is a non-blocking suggestion, not applied.
