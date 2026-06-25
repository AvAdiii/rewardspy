"""The records that flow through rewardspy.

``RolloutRecord`` is one observation of a reward call. ``Alert`` is one finding
raised by a detector. Both are plain dataclasses so they serialize cleanly to
JSONL and CSV.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class RolloutRecord:
    """A single intercepted reward call.

    For large tensor inputs (such as logprobs) the wrapper stores derived
    lengths rather than the raw payload, so a record stays small and cheap to
    keep in memory by the thousands.
    """

    call_id: str
    timestamp: float
    step: int
    scalar_reward: float
    components: dict[str, float] = field(default_factory=dict)
    call_duration_ms: float = 0.0
    input_length: int = 0
    output_length: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Alert:
    """A finding raised by a detector when a record or window looks suspicious.

    ``status`` is ``WARNING`` or ``ALERT``. ``severity`` is an optional finer
    grade (``LOW`` / ``MEDIUM`` / ``HIGH``) used to rank alerts in the UI.
    """

    step: int
    timestamp: float
    detector: str
    status: str
    message: str
    detail: str = ""
    severity: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
