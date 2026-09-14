"""Dynamic construction of the classification output schema.

The categories are turned into a Pydantic model at runtime so the LLM can be
asked for structured output that exactly matches the user-defined categories.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, Field, create_model, model_validator

from query_classification.categories import Category


def build_classification_model(
    categories: list[Category], allow_new_labels: bool = True
) -> type[BaseModel]:
    """Dynamically build a Pydantic model from the categories definition.

    Each category becomes a field holding a list of 1-3 label strings, ordered
    by strength of evidence. ``allow_new_labels`` only affects the generated
    field *description* text (whether it mentions the "none - <new label>"
    invention option) — it does not change the schema/``Field`` constraints,
    which remain cardinality/type only either way.
    """
    duplicates = [name for name, count in Counter(cat.name for cat in categories).items() if count > 1]
    if duplicates:
        raise ValueError(
            f"Duplicate category name(s): {duplicates}. Each category's `name` "
            f"becomes a distinct output field/column and must be unique."
        )

    fields: dict[str, Any] = {}
    for cat in categories:
        label_docs = "; ".join(
            f'"{label.value}": {label.description}' for label in cat.labels
        )
        if allow_new_labels:
            none_clause = (
                'If none of the specific labels apply, use "none" alone, or '
                '"none - <new label>" to suggest a label not already in the '
                'predefined set. Do not use "none - <existing label>": if an '
                "existing label fits, assign it directly. "
            )
        else:
            none_clause = (
                'If none of the specific labels apply, use "none" alone. '
                "Do not invent a new label. "
            )
        field_description = (
            f"{cat.description}. "
            f"Provide 1-3 labels ordered by strength of evidence. "
            f"{none_clause}"
            f"Options - {label_docs}."
        )
        fields[cat.name] = (
            list[str],
            Field(min_length=1, max_length=3, description=field_description),
        )

    return create_model("ClassificationResult", **fields)


def build_critic_model(category: Category) -> type[BaseModel]:
    """Build the Critic's output schema for a single category.

    The Critic either challenges the leading label (providing a proposed
    alternative and its argument) or declines (providing only a rationale in
    ``argument``) — ``proposed_label`` is required exactly when
    ``challenges=True``.
    """

    class CriticVerdict(BaseModel):
        challenges: bool = Field(
            description="Whether a genuinely compelling alternative label exists."
        )
        proposed_label: str | None = Field(
            default=None,
            description="The alternative label being argued for. Required when "
            "challenges is true; must be omitted/null when challenges is false.",
        )
        argument: str = Field(
            description="The argument for the proposed alternative when "
            "challenging, or the rationale for declining when not."
        )

        @model_validator(mode="after")
        def _proposed_label_matches_challenges(self) -> "CriticVerdict":
            if self.challenges and not self.proposed_label:
                raise ValueError("proposed_label is required when challenges is true")
            if not self.challenges and self.proposed_label:
                raise ValueError("proposed_label must be omitted when challenges is false")
            return self

    CriticVerdict.__name__ = f"CriticVerdict_{category.name}"
    return CriticVerdict


def build_reconciler_model(category: Category) -> type[BaseModel]:
    """Build the Reconciler's output schema for a single category: a fresh
    1-3 label list for that category, plus the Reconciler's reasoning."""

    class ReconcilerVerdict(BaseModel):
        labels: list[str] = Field(
            min_length=1,
            max_length=3,
            description=f"Final 1-3 labels for '{category.name}', ordered by "
            "strength of evidence.",
        )
        reasoning: str = Field(description="Why this final verdict was reached.")

    ReconcilerVerdict.__name__ = f"ReconcilerVerdict_{category.name}"
    return ReconcilerVerdict


def build_induction_model(n_labels: int) -> type[BaseModel]:
    """Build the output schema for the category-induction call: one
    **required** ``description_i`` field per label position (1-indexed), plus
    ``category_description``. Carries no label names at all -- label identity
    comes from the runner's own sorted label list (``induction.py``), never
    from the model's response, so a missing/invented label is structurally
    impossible rather than something to reconcile after the fact.

    Field names are positional indices, not label values, and arity is pinned
    by making every ``description_i`` field required -- not by a length-bounded
    ``list`` field. Both choices are forced, not stylistic:

    - Label values are dataset-chosen strings that routinely aren't valid
      Python/Pydantic identifiers (``"very negative"``, ``"class 1"``), so a
      field-*per-label-value* model (unlike ``build_classification_model``'s
      dynamic-field approach) would break for exactly the datasets this
      targets. Indices are always valid identifiers.
    - Verified against the installed stack: ``litellm`` serializes a Pydantic
      ``response_format`` with ``"strict": true``, and strict structured
      output does not support ``minItems``/``maxItems`` -- a length-bounded
      ``list`` would be rejected by the provider, silently degrading every
      induction call to JSON-object-mode fallback
      (``Classifier._attempt_completion``'s ``except litellm.BadRequestError``).
      Required fields plus strict mode's ``additionalProperties: false``
      express exactly-``n_labels`` in a form the strict path accepts.

    Uses the same ``create_model`` mechanism as ``build_classification_model``
    above, differing only in that the dynamic field names are positional
    indices rather than caller-chosen names.
    """
    if n_labels < 1:
        raise ValueError(f"n_labels must be >= 1, got {n_labels}")

    fields: dict[str, Any] = {"category_description": (str, ...)}
    for i in range(1, n_labels + 1):
        fields[f"description_{i}"] = (str, ...)

    return create_model("InductionResult", **fields)


def schema_description(model: type[BaseModel]) -> str:
    """Produce a human-readable summary of the model fields for the prompt."""
    lines = [
        "Classification schema (list of 1-3 labels per field, ordered by evidence):"
    ]
    for name, info in model.model_fields.items():
        lines.append(f"\n  [{name}]")
        lines.append(f"  Description: {info.description}")
    return "\n".join(lines)
