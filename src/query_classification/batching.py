"""Batched classification: token-budget resolution, batch sizing, payload
assembly, and the batched-call-with-bisection routine.

Owns everything spec 5 (batched-classification) adds beyond `classifier.py`'s
own primitives. Per INV-1, this module may import `categories.py`,
`classifier.py`, `schema.py`, and `prompts.py` -- it must not import
`pipeline.py`, `debate.py`, `multi_model.py`, `cli.py`, or `experiment.py`.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import litellm
from litellm.utils import type_to_response_format_param
from pydantic import BaseModel

from query_classification.categories import Category
from query_classification.classifier import Classifier, classify_failure

logger = logging.getLogger(__name__)

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


class BatchStats:
    """Thread-safe counters for FR-4.1's `run_config.json` `batch` block. Not
    a module global -- scoped to a single run, constructed by the caller and
    passed into `BatchRunner`."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls = 0
        self._arities: list[int] = []
        self._trims = 0
        self._bisections = 0
        self._batch_costs: list[tuple[int, float | None]] = []

    def record_call(self, arity: int) -> None:
        with self._lock:
            self._calls += 1
            self._arities.append(arity)

    def record_trim(self) -> None:
        with self._lock:
            self._trims += 1

    def record_bisection(self) -> None:
        with self._lock:
            self._bisections += 1

    def record_cost(self, arity: int, cost: float | None) -> None:
        """The cost of one top-level formed batch's first attempt only --
        never a retry-at-original-size or bisection call. See
        `BatchRunner.run`'s own `_top_level` gating."""
        with self._lock:
            self._batch_costs.append((arity, cost))

    def snapshot(self) -> dict[str, int | float | None]:
        """A freshly-built dict each call -- never a stored reference the
        caller could mutate back into this object's own counters. Arity
        fields are `None` (not `0`) until the first batched call is
        recorded, matching FR-4.1's null-vs-zero rule."""
        with self._lock:
            arities = list(self._arities)
            return {
                "batched_calls": self._calls,
                "min_arity": min(arities) if arities else None,
                "mean_arity": (sum(arities) / len(arities)) if arities else None,
                "max_arity": max(arities) if arities else None,
                "trims": self._trims,
                "bisections": self._bisections,
                "batch_costs": list(self._batch_costs),
            }


class BatchRunner:
    """Owns the arity-keyed classifier cache, batch formation (the sole
    caller of `plan_batch`), and the batched call with bisection (FR-3.2)."""

    def __init__(
        self,
        classifier_factory: Callable[[int], Classifier],
        budgets: TokenBudgets,
        stats: BatchStats,
        max_size: int,
        mode: str,
        fixed_size: int | None,
        system_prompt: str,
        categories: list[Category],
        batch_model_for: Callable[[int], type[BaseModel]],
        cost_fn: Callable[[int, int], float | None] | None = None,
    ) -> None:
        self._classifier_factory = classifier_factory
        self.budgets = budgets
        self.stats = stats
        self.max_size = max_size
        self.mode = mode
        self.fixed_size = fixed_size
        self.system_prompt = system_prompt
        self.categories = categories
        self.batch_model_for = batch_model_for
        # A plain callable, not a `cost.CostPerToken`/`cost.py` import -- keeps
        # this module's import list exactly as it is today (`categories.py`/
        # `classifier.py` only). The caller (experiment.py/cli.py, which does
        # import cost.py) builds it as a closure over one resolved price.
        self._cost_fn = cost_fn
        self._cache: dict[int, Classifier] = {}
        self._cache_lock = threading.Lock()

    def total_usage(self) -> tuple[int, int]:
        """Sum of `usage_totals()` across every classifier currently cached
        by arity -- so callers never need to reach into `_cache` directly."""
        with self._cache_lock:
            classifiers = list(self._cache.values())
        total_prompt = 0
        total_completion = 0
        for classifier in classifiers:
            p, c = classifier.usage_totals()
            total_prompt += p
            total_completion += c
        return (total_prompt, total_completion)

    def _get_classifier(self, arity: int) -> Classifier:
        # Lookup-and-construction under one lock, not a check-then-insert --
        # a plain check-then-insert is a compound operation two threads can
        # execute simultaneously, producing two models for the same arity.
        with self._cache_lock:
            classifier = self._cache.get(arity)
            if classifier is None:
                classifier = self._classifier_factory(arity)
                self._cache[arity] = classifier
            return classifier

    def iter_batches(self, rows: list[tuple[int, str]]) -> Iterator[list[int]]:
        """`rows` is `(original_dataframe_index, text)` pairs, in remaining
        order -- carrying the true row identity through `--restore`/`limit`
        filtering. Yields successive lists of the covered rows' original
        DataFrame indices, never positions into `rows`, never row text."""
        texts = [text for _, text in rows]
        start = 0
        total = len(texts)
        while start < total:
            plan = plan_batch(
                texts,
                start=start,
                mode=self.mode,
                fixed_size=self.fixed_size,
                max_size=self.max_size,
                budgets=self.budgets,
                system_prompt=self.system_prompt,
                categories=self.categories,
                batch_model_for=self.batch_model_for,
            )
            if plan.was_trimmed:
                self.stats.record_trim()
            if plan.oversized_row_index is not None:
                original_index = rows[start + plan.oversized_row_index][0]
                logger.warning(
                    "Row %s alone exceeds the configured token budget; sending it as a "
                    "batch of one.",
                    original_index,
                )
            batch_indices = [rows[start + i][0] for i in range(plan.arity)]
            yield batch_indices
            start += plan.arity

    def _attempt(self, classifier: Classifier, payload: str, arity: int) -> dict[str, Any]:
        self.stats.record_call(arity)
        return classifier.classify(payload)

    @staticmethod
    def _unpack(raw: dict[str, Any], arity: int) -> list[dict[str, Any]]:
        return [raw[f"result_{i}"] for i in range(1, arity + 1)]

    def run(
        self, texts: list[str], *, _top_level: bool = True
    ) -> list[dict[str, Any] | Exception]:
        """Classify one batch, applying FR-3.2's bisection policy on
        failure. `classify_failure` -- spec 6's, not re-derived here --
        supplies both axes; the branch is exhaustive by construction over
        the two booleans, so there is no "unrecognized" case to default.

        `_top_level` (keyword-only, default `True`) marks the very first,
        outermost call for a formed batch -- the recursive bisection calls
        below pass `_top_level=False`. Only a top-level call's first
        `_attempt()` (before any retry-at-original-size or bisection) has its
        cost recorded, via `self._cost_fn`, when one is configured."""
        arity = len(texts)
        classifier = self._get_classifier(arity)
        payload = _render_payload(texts)
        record_cost = _top_level and self._cost_fn is not None

        exc: Exception | None = None
        try:
            if record_cost:
                with classifier.track_usage() as sink:
                    try:
                        raw = self._attempt(classifier, payload, arity)
                    finally:
                        cost = self._cost_fn(*sink.usage) if sink.usage is not None else None
                        self.stats.record_cost(arity, cost)
            else:
                raw = self._attempt(classifier, payload, arity)
        except Exception as e:  # noqa: BLE001 -- classified immediately below via
            # classify_failure (spec 6); every branch either re-raises nothing
            # (isolable split / retry / fail) so nothing is swallowed silently.
            exc = e
        else:
            return self._unpack(raw, arity)

        kind = classify_failure(exc)

        if kind.isolable:
            self.stats.record_bisection()
            if arity == 1:
                return [exc]
            mid = (arity + 1) // 2  # first half takes the extra row when odd
            return self.run(texts[:mid], _top_level=False) + self.run(texts[mid:], _top_level=False)

        if kind.retryable:
            delay = classifier.retry_delay
            for _ in range(2):  # at most 2 further attempts, beyond Classifier.classify's own
                time.sleep(delay)
                try:
                    raw = self._attempt(classifier, payload, arity)
                except Exception as e:  # noqa: BLE001 -- same classify_failure contract as above
                    exc = e
                    delay *= 2
                    continue
                return self._unpack(raw, arity)
            return [exc] * arity

        return [exc] * arity
