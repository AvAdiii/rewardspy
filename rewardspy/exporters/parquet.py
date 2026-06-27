"""Parquet batch export.

Same column layout as the CSV export: the base fields plus one
``component.<key>`` column per reward component, unioned across all records.
Unlike CSV, Parquet keeps the numeric columns typed and compresses well, so it
is the format to reach for once a run has produced a lot of history. ``pyarrow``
is an optional extra (``pip install rewardspy[parquet]``) to keep it out of the
core install.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..records import RolloutRecord

_BASE_FIELDS = [
    "step",
    "timestamp",
    "call_id",
    "scalar_reward",
    "call_duration_ms",
    "input_length",
    "output_length",
]


def write_parquet(records: Sequence[RolloutRecord], path: str | Path) -> Path:
    """Write records to ``path`` as Parquet and return the path written."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "parquet export needs pyarrow; install it with 'pip install rewardspy[parquet]'"
        ) from exc

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    component_keys = sorted({key for r in records for key in r.components})
    columns: dict[str, list] = {name: [] for name in _BASE_FIELDS}
    for key in component_keys:
        columns[f"component.{key}"] = []

    for r in records:
        columns["step"].append(r.step)
        columns["timestamp"].append(r.timestamp)
        columns["call_id"].append(r.call_id)
        columns["scalar_reward"].append(r.scalar_reward)
        columns["call_duration_ms"].append(r.call_duration_ms)
        columns["input_length"].append(r.input_length)
        columns["output_length"].append(r.output_length)
        for key in component_keys:
            columns[f"component.{key}"].append(r.components.get(key))

    pq.write_table(pa.table(columns), out)
    return out
