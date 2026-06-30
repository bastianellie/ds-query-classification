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

from query_classification.resources import DEFAULT_SYSTEM_PROMPT_FILE
from query_classification.schema import schema_description


def build_system_prompt(
    model: type[BaseModel],
    system_prompt_file: str | Path = DEFAULT_SYSTEM_PROMPT_FILE,
    task_description: str | None = None,
    extra_prompt: str | None = None,
) -> str:
    """Render the system prompt for the given classification model."""
    template = Path(system_prompt_file).read_text()
    prompt = template.format(
        task_description=(task_description or "").strip(),
        schema_description=schema_description(model),
    )
    if extra_prompt:
        prompt += f"\n\nAdditional instructions:\n{extra_prompt.strip()}"
    return prompt
