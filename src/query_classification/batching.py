"""Batched classification: token-budget resolution, batch sizing, payload
assembly, and the batched-call-with-bisection routine.

Owns everything spec 5 (batched-classification) adds beyond `classifier.py`'s
own primitives. Per INV-1, this module may import `categories.py`,
`classifier.py`, `schema.py`, and `prompts.py` -- it must not import
`pipeline.py`, `debate.py`, `multi_model.py`, `cli.py`, or `experiment.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import litellm
from litellm.utils import type_to_response_format_param
from pydantic import BaseModel

from query_classification.categories import Category

# Duplicated deliberately from induction.py (`induction.py:31-32`) rather than
# imported, since AR-1.1 forbids batching.py from importing induction.py. A
# test asserts the two modules' values stay identical (AR-2.3).
_DATA_START = "<<<DATA>>>"
_DATA_END = "<<<END DATA>>>"


@dataclass(frozen=True)
class TokenBudgets:
    """The resolved (or overridden) input/output token budgets for a batched
    run's model, plus which candidate model id each side actually resolved
    to (``None`` when that side was overridden rather than resolved)."""

    max_input: int | None
    max_output: int | None
    resolved_input_model: str | None
    resolved_output_model: str | None


def _candidate_ids(model_id: str) -> list[str]:
    """The ordered candidate walk FR-1.2 specifies: the id as given, with a
    leading ``openai/`` marker stripped, with a leading ``@workspace/``
    segment stripped, then progressively dropping trailing `-`-delimited
    segments, longest candidate first. Each step only applies when its
    precondition holds (e.g. step 2 only if the id actually starts with
    ``openai/``), so the list stays as short as possible."""
    candidates = [model_id]
    current = model_id
    if current.startswith("openai/"):
        current = current[len("openai/") :]
        candidates.append(current)
    if current.startswith("@"):
        slash_idx = current.find("/")
        if slash_idx != -1:
            current = current[slash_idx + 1 :]
            candidates.append(current)
    parts = current.split("-")
    for i in range(len(parts) - 1, 0, -1):
        candidates.append("-".join(parts[:i]))

    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def resolve_token_budgets(
    model_id: str, input_override: int | None, output_override: int | None
) -> TokenBudgets:
    """Resolve the input/output token budgets for ``model_id``, independently
    per side (FR-1.2). A side already supplied via override is never walked
    -- the override value is used as-is and that side's ``resolved_*_model``
    stays ``None``, since no candidate was actually resolved for it."""
    litellm.suppress_debug_info = True  # AR-1.2: suppress get_model_info's stderr noise

    max_input = input_override
    max_output = output_override
    resolved_input_model: str | None = None
    resolved_output_model: str | None = None

    if max_input is None or max_output is None:
        for candidate in _candidate_ids(model_id):
            try:
                info = litellm.get_model_info(candidate)
            except Exception:  # noqa: BLE001 -- get_model_info raises a bare
                # Exception (not a litellm-specific type) for an unmapped
                # model id (AR-1.3); the walk must keep trying candidates.
                continue
            if max_input is None and info.get("max_input_tokens") is not None:
                max_input = info["max_input_tokens"]
                resolved_input_model = candidate
            if max_output is None and info.get("max_output_tokens") is not None:
                max_output = info["max_output_tokens"]
                resolved_output_model = candidate
            if max_input is not None and max_output is not None:
                break

    return TokenBudgets(
        max_input=max_input,
        max_output=max_output,
        resolved_input_model=resolved_input_model,
        resolved_output_model=resolved_output_model,
    )


@dataclass(frozen=True)
class BatchPlan:
    """The result of sizing one batch iteration."""

    arity: int
    was_trimmed: bool
    oversized_row_index: int | None  # position (0-based, relative to `start`) of the
    # single row that alone exceeds a budget -- set only when `arity == 1` was forced
    # by that condition, never by ordinary sizing/trimming.


def _render_payload(texts: list[str]) -> str:
    """The ordered, 1-based, delimited JSON payload FR-2.2 specifies. A
    query's own text can only ever appear as a JSON string value, so it
    cannot structurally forge a delimiter, a position field, or a
    neighboring array entry."""
    entries = [{"position": i, "text": text} for i, text in enumerate(texts, start=1)]
    return f"{_DATA_START}\n{json.dumps(entries, sort_keys=False)}\n{_DATA_END}"


def _measure_rendered_payload(
    texts: list[str],
    system_prompt: str,
    batch_model_for: Callable[[int], type[BaseModel]],
    model_for_counting: str,
) -> int:
    """FR-1.6's input measurement: the complete rendered payload (system
    prompt plus the fully serialized user message) plus the serialized
    ``response_format`` JSON Schema, which is charged as input and grows
    with arity."""
    user_message = _render_payload(texts)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    message_tokens = litellm.token_counter(model=model_for_counting, messages=messages)
    schema = type_to_response_format_param(batch_model_for(len(texts)))
    schema_tokens = litellm.token_counter(model=model_for_counting, text=json.dumps(schema))
    return message_tokens + schema_tokens


def _estimate_output_tokens_per_row(categories: list[Category], model_for_counting: str) -> int:
    """FR-1.6's output estimate for a single row: for every category, the
    maximum label count the schema permits (3) times that category's
    longest configured label value, plus the nested ``result_i`` object's
    JSON overhead -- computed together by counting the serialized skeleton
    object, which naturally includes both the label content and the
    field-name/bracket structure around it."""
    skeleton: dict[str, list[str]] = {}
    for cat in categories:
        if cat.labels:
            longest = max(cat.labels, key=lambda label: len(label.value))
            skeleton[cat.name] = [longest.value] * 3
        else:
            skeleton[cat.name] = []
    return litellm.token_counter(model=model_for_counting, text=json.dumps(skeleton))


def _output_ceiling_arity(
    categories: list[Category], budgets: TokenBudgets, model_for_counting: str
) -> int | None:
    """The arity ceiling FR-1.6's output estimate implies, with the
    safety factor of 2. `None` when the output budget is unknown -- this
    figure is an estimate used for sizing only, never an upper bound, and
    is skipped entirely rather than guessed when there's nothing to size
    against."""
    if budgets.max_output is None:
        return None
    per_row = _estimate_output_tokens_per_row(categories, model_for_counting)
    if per_row <= 0:
        return None
    return max(1, budgets.max_output // (per_row * 2))


def plan_batch(
    rows: list[str],
    *,
    start: int,
    mode: str,
    fixed_size: int | None,
    max_size: int,
    budgets: TokenBudgets,
    system_prompt: str,
    categories: list[Category],
    batch_model_for: Callable[[int], type[BaseModel]],
) -> BatchPlan:
    """Decide one batch's arity, starting at position `start` in `rows`.

    Dynamic mode grows greedily from an additive per-row estimate (cheap,
    imprecise) up to `max_size`/the output ceiling, then the chosen arity is
    validated -- and reduced if needed -- against the exact rendered-payload
    measurement (FR-1.6). Fixed mode starts from `fixed_size` and only ever
    reduces via that same measurement (FR-1.7); reduction below what
    availability alone would produce is the only thing counted as a "trim".
    Either mode floors at arity 1, at which point a still-oversized single
    row is reported via `oversized_row_index` rather than reduced further.
    """
    remaining = rows[start:]
    available = len(remaining)
    if available == 0:
        raise ValueError("plan_batch called with no remaining rows")

    model_for_counting = budgets.resolved_input_model or ""

    requested = min(fixed_size, max_size) if mode == "fixed" else max_size
    target_before_budget = min(requested, available)
    target = target_before_budget

    if mode == "dynamic" and budgets.max_input is not None:
        cumulative = 0
        shortlisted = 0
        for i in range(target):
            cumulative += litellm.token_counter(model=model_for_counting, text=remaining[i])
            if cumulative > budgets.max_input:
                break
            shortlisted = i + 1
        target = max(1, shortlisted)

    output_ceiling = _output_ceiling_arity(categories, budgets, model_for_counting)
    if output_ceiling is not None:
        target = max(1, min(target, output_ceiling))

    was_trimmed = False
    oversized_row_index: int | None = None
    if budgets.max_input is not None:
        measured = _measure_rendered_payload(
            remaining[:target], system_prompt, batch_model_for, model_for_counting
        )
        while measured > budgets.max_input and target > 1:
            target -= 1
            measured = _measure_rendered_payload(
                remaining[:target], system_prompt, batch_model_for, model_for_counting
            )
        if measured > budgets.max_input:
            oversized_row_index = 0
            target = 1
        elif mode == "fixed":
            was_trimmed = target < target_before_budget

    return BatchPlan(arity=target, was_trimmed=was_trimmed, oversized_row_index=oversized_row_index)
