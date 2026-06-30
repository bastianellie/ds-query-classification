"""Dynamic construction of the classification output schema.

The categories are turned into a Pydantic model at runtime so the LLM can be
asked for structured output that exactly matches the user-defined categories.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, create_model

from query_classification.categories import Category


def build_classification_model(categories: list[Category]) -> type[BaseModel]:
    """Dynamically build a Pydantic model from the categories definition.

    Each category becomes a field holding a list of 1-3 label strings, ordered
    by strength of evidence.
    """
    fields: dict[str, Any] = {}
    for cat in categories:
        label_docs = "; ".join(
            f'"{label.value}": {label.description}' for label in cat.labels
        )
        field_description = (
            f"{cat.description}. "
            f"Provide 1-3 labels ordered by strength of evidence. "
            f'If none of the specific labels apply, use "none" alone, or '
            f'"none - <new label>" to suggest a label not already in the '
            f'predefined set. Do not use "none - <existing label>": if an '
            f"existing label fits, assign it directly. "
            f"Options - {label_docs}."
        )
        fields[cat.name] = (
            list[str],
            Field(min_length=1, max_length=3, description=field_description),
        )

    return create_model("ClassificationResult", **fields)


def schema_description(model: type[BaseModel]) -> str:
    """Produce a human-readable summary of the model fields for the prompt."""
    lines = [
        "Classification schema (list of 1-3 labels per field, ordered by evidence):"
    ]
    for name, info in model.model_fields.items():
        lines.append(f"\n  [{name}]")
        lines.append(f"  Description: {info.description}")
    return "\n".join(lines)
