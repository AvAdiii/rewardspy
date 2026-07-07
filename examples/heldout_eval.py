"""Catch a hack the reward stats alone cannot see, using a held-out eval.

This is the case the distribution detectors miss. A proxy reward that keeps its
own distribution looking healthy, no variance collapse, no ceiling saturation,
while the policy quietly games something the reward under-specifies. The only
thing that gives it away is a held-out eval on the true objective.

Here a fake policy is scored by a proxy reward that rewards a surface pattern
correlated with the real answer. The policy learns to produce the pattern
without the reasoning, so the proxy reward keeps climbing smoothly while true
held-out accuracy stalls and then drops. We log the held-out accuracy with
``rewardspy.log_eval`` every few steps, and the divergence detector fires.

Run it:

    python examples/heldout_eval.py

Press q to quit the dashboard.
"""

from __future__ import annotations

import random
import threading
import time

import rewardspy

random.seed(11)

STEPS = 1600
STEP_DELAY = 0.03  # seconds between training steps
EVAL_EVERY = 25    # log a held-out eval score this often


def true_accuracy(step: int) -> float:
    """The real objective. Rises early, then the policy trades it for the hack."""
    if step < 350:
        return 0.30 + 0.42 * (step / 350)   # genuine learning to ~0.72
    # After it discovers the surface pattern, real accuracy slips the rest of
    # the run, so the divergence is still visible in the latest window.
    return max(0.25, 0.72 - (step - 350) / 1250 * 0.47)


def proxy_reward_level(step: int) -> float:
    """The proxy the model optimizes. Climbs smoothly for nearly the whole run."""
    return 0.30 + 0.62 * min(1.0, step / 1500)


@rewardspy.watch(name="heldout_demo", max_reward=1.0)
def reward(response: str) -> float:
    """Proxy reward. Its own distribution stays healthy the entire run."""
    # 'response' encodes the proxy level plus noise; the reward just reads it.
    base = float(response)
    return max(0.0, min(1.0, base + random.gauss(0, 0.05)))


def train() -> None:
    for step in range(1, STEPS + 1):
        response = f"{proxy_reward_level(step):.4f}"
        reward(response)
        if step % EVAL_EVERY == 0:
            # Evaluate on held-out data the reward never touches, and log it.
            acc = true_accuracy(step) + random.gauss(0, 0.01)
            rewardspy.log_eval(reward, acc)
        time.sleep(STEP_DELAY)


def main() -> None:
    trainer = threading.Thread(target=train, daemon=True)
    trainer.start()
    rewardspy.show(reward, interval=0.3)


if __name__ == "__main__":
    main()
