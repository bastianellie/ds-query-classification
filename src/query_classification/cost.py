"""Cost and timing primitives (spec 7): per-model pricing resolution, cost
arithmetic, sample statistics, a live-readable timer, and an atomic JSON
writer. Owns every pure/reusable primitive the two entry points
(`experiment.py`/`cli.py`) need to assemble a `cost_report.json`.

Per INV-1, this module may import `categories.py`/`classifier.py` only -- it
must not import `pipeline.py`, `batching.py`, `debate.py`, `multi_model.py`,
`cli.py`, or `experiment.py`.
"""

from __future__ import annotations

import json
import math
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import litellm


# Duplicated deliberately from `batching._candidate_ids` rather than imported
# (batching.py is not one of this module's declared imports; see the module
# docstring) -- a test asserts the two sequences stay identical for the same
# input (FR-2.3).
def _candidate_ids(model_id: str) -> list[str]:
    """The ordered candidate walk FR-2.3 specifies: the id as given, with a
    leading ``openai/`` marker stripped, with a leading ``@workspace/``
    segment stripped, then progressively dropping trailing `-`-delimited
    segments, longest candidate first. Identical to
    `batching._candidate_ids` -- duplicated, not imported, per this module's
    own import boundary."""
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


@dataclass(frozen=True)
class CostPerToken:
    """Resolved per-token pricing for one model id, plus which candidate
    resolved each side (`None` when that side never resolved), mirroring
    spec 5's own `resolved_input_model`/`resolved_output_model` diagnostic
    precedent in `run_config.json`."""

    input: float | None
    output: float | None
    resolved_input_model: str | None
    resolved_output_model: str | None


def _finite_or_none(value: float | None) -> float | None:
    """Convert a non-finite (`NaN`/`±Infinity`) value to `None` -- realistically
    only reachable via a corrupted/zero pricing table, not normal arithmetic."""
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return value


class PricingCache:
    """Run-scoped, lock-protected cache of `CostPerToken` by model id.
    `resolve(model_id)` walks the candidate sequence only on a cache miss
    (FR-2.1); resolution itself happens under the lock (simple correctness
    over a single-flight optimization -- a redundant walk on a rare
    concurrent cache miss costs a few local dict lookups, not a network
    call)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[str, CostPerToken] = {}

    def resolve(self, model_id: str) -> CostPerToken:
        with self._lock:
            cached = self._cache.get(model_id)
            if cached is not None:
                return cached

            input_cost: float | None = None
            output_cost: float | None = None
            resolved_input_model: str | None = None
            resolved_output_model: str | None = None

            for candidate in _candidate_ids(model_id):
                try:
                    info = litellm.get_model_info(candidate)
                except Exception:  # noqa: BLE001 -- get_model_info raises a bare
                    # Exception (not a litellm-specific type) for an unmapped
                    # model id; the walk must keep trying candidates.
                    continue
                if input_cost is None and info.get("input_cost_per_token") is not None:
                    input_cost = _finite_or_none(info["input_cost_per_token"])
                    if input_cost is not None:
                        resolved_input_model = candidate
                if output_cost is None and info.get("output_cost_per_token") is not None:
                    output_cost = _finite_or_none(info["output_cost_per_token"])
                    if output_cost is not None:
                        resolved_output_model = candidate
                if input_cost is not None and output_cost is not None:
                    break

            result = CostPerToken(
                input=input_cost,
                output=output_cost,
                resolved_input_model=resolved_input_model,
                resolved_output_model=resolved_output_model,
            )
            self._cache[model_id] = result
            return result


def compute_cost(
    prompt_tokens: int, completion_tokens: int, cost_per_token: CostPerToken
) -> float | None:
    """FR-2.2's arithmetic: `prompt_tokens * input + completion_tokens *
    output`. `None` if either side of `cost_per_token` is unresolved."""
    if cost_per_token.input is None or cost_per_token.output is None:
        return None
    total = prompt_tokens * cost_per_token.input + completion_tokens * cost_per_token.output
    return _finite_or_none(total)


def stats_from_samples(samples: list[float | None]) -> dict[str, float | int | None]:
    """{"mean_usd": ..., "stddev_usd": ..., "count": int}. For REAL,
    independently-observed samples. count=0 -> both None. count=1 ->
    mean=that value, stddev=None (sample stddev is mathematically undefined
    for a single real observation). Any None in `samples` poisons
    mean/stddev to None but does not change count. Sample (not population)
    standard deviation for count>=2.

    Do NOT use this for --critics/--models' uniform estimate -- see
    `uniform_estimate` below.
    """
    count = len(samples)
    if count == 0:
        return {"mean_usd": None, "stddev_usd": None, "count": 0}
    if any(sample is None for sample in samples):
        return {"mean_usd": None, "stddev_usd": None, "count": count}

    values = [float(s) for s in samples]  # type: ignore[arg-type]
    mean = sum(values) / count
    if count == 1:
        return {"mean_usd": _finite_or_none(mean), "stddev_usd": None, "count": count}

    variance = sum((v - mean) ** 2 for v in values) / (count - 1)
    stddev = math.sqrt(variance)
    return {
        "mean_usd": _finite_or_none(mean),
        "stddev_usd": _finite_or_none(stddev),
        "count": count,
    }


def uniform_estimate(total_cost: float | None, count: int) -> dict[str, float | int | None]:
    """FR-3.3's --critics/--models case only: every row is assigned the
    identical estimate `total_cost / count` by construction, so `stddev_usd`
    is always exactly `0.0` (never `None`), including at `count == 1` --
    there is no variance to be undefined about, unlike `stats_from_samples`'
    real-sample case."""
    if total_cost is None or count <= 0:
        return {"mean_usd": None, "stddev_usd": None, "count": count}
    mean = _finite_or_none(total_cost / count)
    if mean is None:
        return {"mean_usd": None, "stddev_usd": None, "count": count}
    return {"mean_usd": mean, "stddev_usd": 0.0, "count": count}


def expand_batch_costs_to_query_samples(
    batch_costs: list[tuple[int, float | None]]
) -> list[float | None]:
    """Each (arity, cost) pair becomes `arity` copies of (cost/arity if cost
    is not None else None) -- the per-row estimate FR-3.2 describes."""
    samples: list[float | None] = []
    for arity, cost in batch_costs:
        share = _finite_or_none(cost / arity) if cost is not None else None
        samples.extend([share] * arity)
    return samples


class Timer:
    """A plain, live-readable object -- NOT a contextmanager. Records
    `time.monotonic()` at construction; `elapsed_seconds()` is callable at
    any later point, including from inside a broad `except` handler before
    `sys.exit`. A contextmanager would only expose the final elapsed time
    after its `with` block closes, which is too late for a failure handler
    that needs "elapsed so far" *before* deciding to exit."""

    def __init__(self) -> None:
        self._start = time.monotonic()

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._start


def write_json_atomic(path: Path, data: Any) -> None:
    """Temp file (same directory) + `Path.replace`, identical pattern to
    `experiment.py`'s existing private `_write_json_atomic`. Serializes with
    `allow_nan=False` as a final defense-in-depth boundary: a non-finite
    (`NaN`/`Infinity`) value can never reach the on-disk JSON, regardless of
    which upstream function let one slip through."""
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(data, f, indent=2, allow_nan=False)
        Path(tmp_name).replace(path)
    except BaseException:  # noqa: BLE001 -- any failure mid-write must not leave
        # a partial/temp file behind; the exception itself is always re-raised.
        Path(tmp_name).unlink(missing_ok=True)
        raise


class QueryCostCollector:
    """Thread-safe accumulator of per-row real cost samples (plain mode
    only, FR-3.1) plus the run-wide rows-attempted count every mode
    populates (FR-3.3's denominator). Mirrors `BatchStats`' own
    lock-protected pattern."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: list[float | None] = []
        self._rows_attempted: int | None = None

    def record(self, cost: float | None) -> None:
        with self._lock:
            self._samples.append(cost)

    def samples(self) -> list[float | None]:
        with self._lock:
            return list(self._samples)

    def set_rows_attempted(self, n: int) -> None:
        with self._lock:
            self._rows_attempted = n

    def rows_attempted(self) -> int | None:
        with self._lock:
            return self._rows_attempted
