<div align="center">

# rewardspy

**A plug-in debugger and visualizer for RL reward functions.**
Detects reward hacking before it derails your training run.

</div>

---

Reward hacking is the number one practical failure mode in RL today. An agent
learns to maximize the *proxy* reward by exploiting loopholes, without actually
solving the task. The reward curve goes up and to the right, so it looks fine,
right up until you read the rollouts and discover the model overwrote its own
unit tests or learned to spam a format token.

`rewardspy` wraps your reward function and watches every call. It tracks the
statistical signatures of reward hacking in real time and surfaces them in a
live terminal dashboard. One import. Zero boilerplate. Your reward function is
never modified; rewardspy is a pure observer.

```python
import rewardspy

# Before
reward = my_reward_fn(response, ground_truth)

# After: full observability, zero other changes
reward = rewardspy.watch(my_reward_fn)(response, ground_truth)
```

## Status

Early development. The public API is taking shape and may change before `0.1.0`
is released. Follow along, open issues, and check the roadmap.

## Install

```bash
pip install rewardspy
```

## License

MIT. See [LICENSE](LICENSE).
