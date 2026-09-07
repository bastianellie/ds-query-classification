# ds-query-classification

Classify arbitrary text queries in a CSV against user-defined categories using an LLM
(via LiteLLM). Fully domain-agnostic: the output schema and system prompt are built at
runtime from a categories JSON file. Python >=3.10 (developed/tested on 3.12), src-layout
package, pandas + pydantic + litellm, no web framework — this is a batch CLI tool.

## Commands

Package manager: **pip**, no lockfile (plain `requirements.txt` + `pyproject.toml`).

- Install deps + venv: `./install.sh` (creates `.venv` on Python 3.12; `--force` recreates it; `--dev` is currently a no-op), then `source .venv/bin/activate`
- Install the package itself editable — **required** for `python -m query_classification` or `import query_classification`, and **not** done by `install.sh`: `pip install -e '.[dev]'` (quoted, to avoid shell glob expansion; `.[dev]` also pulls in pytest, already installed separately via `requirements.txt`)
- Run the classifier (works without the editable install — `classify.py` sys.path-injects `src/`): `python classify.py --input <csv> --column <col> --categories <json> [--output <csv>] [--model <litellm-model-id>]`
- Same, as a module (requires the editable install): `python -m query_classification ...`
- Tests: `pytest` (pythonpath/testpaths are set in `pyproject.toml`; no editable install needed)

If you're not sure the venv is active, prefix commands with `.venv/bin/` (e.g. `.venv/bin/pytest`) rather than relying on bare `pip`/`python`/`pytest`.

No lint, typecheck, or CI is configured in this repo (no ruff/mypy/pre-commit config, no `.github/`).

## Rules

- **Testing policy:** the hard rule is **no live network/provider calls in tests** — `tests/test_building_blocks.py` uses pytest + `tmp_path` fixtures and only covers `categories.py`/`schema.py`/`prompts.py`/`resources.py` today. `classifier.py`/`pipeline.py`/`cli.py` have zero coverage, but that's a gap, not a boundary — new tests for those are welcome as long as they fake the LLM call (no real `litellm.completion`).
- **Forbidden:** never hand-edit `*.egg-info/`, `build/`, `dist/`, or anything under `output/` — all generated. `data/` is not fully generated (see `spec/ARCHITECTURE.md` Known Gaps) — check before assuming a file there is disposable.
- **Required patterns:** every module under `src/query_classification/` that uses type hints starts with `from __future__ import annotations` and uses PEP 604 unions (`str | None`, not `Optional[...]`) — `__init__.py`/`__main__.py`/root `classify.py` don't need it since they don't use hints. New default resource paths belong in `resources.py`, never recomputed inline elsewhere. A broad `except Exception` must carry a `# noqa: BLE001` plus a one-line comment explaining why the broad catch is intentional (see `classifier.py:108`, `pipeline.py:89` — this is a review convention, not lint-enforced; no linter is configured).
- **Build/verify:** `pytest` must pass before a change is considered done. There is no lint/typecheck/CI gate to satisfy.

## Architecture

Binding constraints, module boundaries, and invariants live in `spec/ARCHITECTURE.md` — read it before any structural change (new modules, changed import direction, changed CLI/schema contracts).

## Spec Workflow

Features are spec-driven: `spec/{N}-{name}/spec.md`. See `/spec-help` for the full command set. This project has no specs yet — `spec/ARCHITECTURE.md` was reverse-engineered from the code as of 2026-09-07. Do not implement unspecced features without flagging it.
