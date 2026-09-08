"""Locating the bundled default resources (prompt template, examples).

Resources live in the top-level ``resources/`` directory of the project so they
are easy to find and edit. These paths are used only as defaults; every entry
point lets the caller point at their own files instead.
"""

from __future__ import annotations

from pathlib import Path

# .../src/query_classification/resources.py -> project root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCES_DIR = PROJECT_ROOT / "resources"

DEFAULT_SYSTEM_PROMPT_FILE = RESOURCES_DIR / "prompts" / "system_prompt.txt"
DEFAULT_CATEGORIES_FILE = RESOURCES_DIR / "categories" / "example_categories.json"
EXAMPLE_TASK_DESCRIPTION_FILE = RESOURCES_DIR / "prompts" / "example_task_description.txt"
DEFAULT_CRITIC_PROMPT_FILE = RESOURCES_DIR / "prompts" / "critic_prompt.txt"
DEFAULT_RECONCILER_PROMPT_FILE = RESOURCES_DIR / "prompts" / "reconciler_prompt.txt"


def check_default_resources_available() -> None:
    """Raise a clear, actionable error if the bundled ``resources/`` dir is missing.

    Only a source checkout or an editable install (``pip install -e .``) has
    ``resources/`` next to the installed package; a plain wheel install does not
    currently package it (see ``spec/ARCHITECTURE.md`` INV-3). Call this before
    relying on any of the ``DEFAULT_*``/``EXAMPLE_*`` paths above so the failure
    is an actionable message instead of a bare ``FileNotFoundError`` the first
    time a default file is opened.
    """
    if not RESOURCES_DIR.is_dir():
        raise FileNotFoundError(
            f"Bundled resources directory not found: {RESOURCES_DIR}. Default "
            "categories/prompts are only available from a source checkout or an "
            "editable install (`pip install -e .`) - a plain wheel install does "
            "not include them. Pass explicit --categories/--system-prompt/"
            "--task-description paths instead."
        )
