"""Optional integrations for popular RL stacks.

Each integration imports its heavy dependency lazily, so importing this package
never pulls in TRL or W&B. Install the extra you need:

    pip install rewardspy[trl]
    pip install rewardspy[wandb]
"""

from .grpo import GroupRecord, GRPOSpy
from .trl import watch_trl

__all__ = ["GRPOSpy", "GroupRecord", "watch_trl"]
