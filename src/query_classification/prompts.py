"""Building the system prompt from the template + schema + task context.

The system prompt template carries two placeholders:

* ``{task_description}`` - free-text framing of *what* is being classified and
  *how* to interpret the labels for a given domain. This is what makes the tool
  general: swap the task description (and categories) and the same machinery
  classifies an entirely different kind of query.
* ``{schema_description}`` - an auto-generated summary of the category fields.

``extra_prompt`` appends ad-hoc instructions for a single run without editing
the template.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from query_classification.resources import (
    DEFAULT_CRITIC_PROMPT_FILE,
    DEFAULT_RECONCILER_PROMPT_FILE,
    DEFAULT_SYSTEM_PROMPT_FILE,
)
from query_classification.schema import schema_description

_INVENTION_ALLOWED_POLICY = (
    'If none of the specific labels apply, use "none" alone, or "none - <new '
    'label>" to suggest a concept not already present in the predefined set. '
    'Never use "none - <existing label>": if an existing label fits, assign '
    "it directly."
)
_INVENTION_FORBIDDEN_POLICY = (
    'If none of the specific labels apply, use "none" alone. Do not invent a '
    "new label."
)


def build_system_prompt(
    model: type[BaseModel],
    system_prompt_file: str | Path = DEFAULT_SYSTEM_PROMPT_FILE,
    task_description: str | None = None,
    extra_prompt: str | None = None,
    allow_new_labels: bool = True,
) -> str:
    """Render the system prompt for the given classification model.

    ``allow_new_labels`` controls the ``{invention_policy}`` slot: whether the
    model is told it may suggest a new, not-yet-listed label (the default,
    matching today's behavior) or must stick to the predefined labels plus
    plain "none".
    """
    template = Path(system_prompt_file).read_text()
    prompt = template.format(
        task_description=(task_description or "").strip(),
        schema_description=schema_description(model),
        invention_policy=(
            _INVENTION_ALLOWED_POLICY if allow_new_labels else _INVENTION_FORBIDDEN_POLICY
        ),
    )
    if extra_prompt:
        prompt += f"\n\nAdditional instructions:\n{extra_prompt.strip()}"
    return prompt


def build_critic_prompt(
    category_name: str,
    category_description: str,
    label_options: str,
    system_prompt_file: str | Path = DEFAULT_CRITIC_PROMPT_FILE,
) -> str:
    """Render the Critic's (devil's advocate) system prompt for one category.

    Plain-string parameters (not a ``Category``) so this module doesn't need
    to import ``categories.py`` — callers extract these fields themselves.
    """
    template = Path(system_prompt_file).read_text()
    return template.format(
        category_name=category_name,
        category_description=category_description,
        label_options=label_options,
    )


def build_reconciler_prompt(
    category_name: str,
    category_description: str,
    label_options: str,
    system_prompt_file: str | Path = DEFAULT_RECONCILER_PROMPT_FILE,
) -> str:
    """Render the Reconciler's (independent verdict) system prompt for one category."""
    template = Path(system_prompt_file).read_text()
    return template.format(
        category_name=category_name,
        category_description=category_description,
        label_options=label_options,
    )
