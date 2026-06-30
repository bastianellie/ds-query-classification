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
