"""Loading and validation of the category definitions.

A categories file is a JSON document of the form::

    {
      "categories": [
        {
          "name": "sentiment",
          "description": "The overall sentiment expressed in the text.",
          "labels": [
            {"value": "positive", "description": "..."},
            {"value": "negative", "description": "..."}
          ]
        }
      ]
    }

These definitions are domain-agnostic: any set of categories and labels can be
supplied, and the rest of the pipeline adapts to them dynamically.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class Label(BaseModel):
    """A single label option within a category."""

    value: str = Field(description="The label string assigned to a text.")
    description: str = Field(description="What this label means / when to assign it.")


class Category(BaseModel):
    """A category the text is classified against, with its candidate labels."""

    name: str = Field(description="Field name used in the output (e.g. a CSV column).")
    description: str = Field(description="What this category captures.")
    labels: list[Label] = Field(min_length=1, description="Candidate labels.")


def load_categories(path: str | Path) -> list[Category]:
    """Load and validate the category definitions from a JSON file."""
    with open(path) as f:
        raw = json.load(f)
    return [Category.model_validate(cat) for cat in raw["categories"]]
