"""Classify arbitrary text queries against arbitrary, user-defined categories.

The public API mirrors the building blocks of the classification pipeline:

    from query_classification import (
        load_categories,
        build_classification_model,
        build_system_prompt,
        Classifier,
        classify_csv,
    )
"""

from query_classification.categories import Category, Label, load_categories
from query_classification.schema import build_classification_model, schema_description
from query_classification.prompts import build_system_prompt
from query_classification.classifier import Classifier
from query_classification.pipeline import classify_csv

__all__ = [
    "Category",
    "Label",
    "load_categories",
    "build_classification_model",
    "schema_description",
    "build_system_prompt",
    "Classifier",
    "classify_csv",
]
