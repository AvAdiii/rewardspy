"""rewardspy: a plug-in debugger and visualizer for RL reward functions.

One import, zero boilerplate. Wrap a reward function and rewardspy intercepts
every call to track the statistical signatures of reward hacking in real time.

    import rewardspy

    reward_fn = rewardspy.watch(my_reward_fn)
    r = reward_fn(response, ground_truth)   # identical to the original
"""

from typing import Any

from .detectors import DetectionEngine, DetectionResult, Status
from .exporters import JSONLExporter, read_jsonl, write_csv
from .records import Alert, RolloutRecord
from .store import MetricStore
from .wrapper import Session, get_store, log_eval, registered_stores, watch

__version__ = "0.1.0"


def show(target: Any = None, *, interval: float = 0.5) -> None:
    """Launch the live terminal dashboard.

    Imported lazily so that simply importing rewardspy does not pull in Textual.
    """
    from .tui import show as _show

    _show(target, interval=interval)

__all__ = [
    "watch",
    "Session",
    "log_eval",
    "show",
    "MetricStore",
    "RolloutRecord",
    "Alert",
    "registered_stores",
    "get_store",
    "JSONLExporter",
    "read_jsonl",
    "write_csv",
    "DetectionEngine",
    "DetectionResult",
    "Status",
    "__version__",
]
