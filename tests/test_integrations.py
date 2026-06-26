from rewardspy.integrations import GRPOSpy, watch_trl
from rewardspy.integrations import wandb as rspy_wandb

# -- GRPO ----------------------------------------------------------------------

def test_grpospy_records_and_is_transparent():
    def reward(response, gt):
        return 1.0 if gt in response else 0.0

    spy = GRPOSpy(reward, group_size=4, name="grpo_t1")
    with spy.step(step=1):
        assert spy.reward("has 42", "42") == 1.0
        assert spy.reward("nope", "42") == 0.0
    assert spy.store.count == 2
    assert len(spy.groups) == 1
    group = spy.groups[0]
    assert group.size == 2
    assert group.mean == 0.5
    assert not group.all_same


def test_grpospy_flags_group_collapse():
    def reward(response, gt):
        return 1.0  # every response scores the same: collapsed groups

    spy = GRPOSpy(reward, group_size=4, name="grpo_collapse", min_groups=10)
    for step in range(30):
        with spy.step(step=step):
            for _ in range(4):
                spy.reward("x", "y")

    assert spy.collapse_rate == 1.0
    assert any(a.detector == "grpo_group" for a in spy.store.alerts)


def test_grpospy_healthy_groups_no_collapse_alert():
    values = iter([0.2, 0.8] * 1000)

    def reward(response, gt):
        return next(values)

    spy = GRPOSpy(reward, group_size=2, name="grpo_healthy", min_groups=10)
    for step in range(30):
        with spy.step(step=step):
            spy.reward("a", "b")
            spy.reward("c", "d")

    assert spy.collapse_rate == 0.0
    assert not any(a.detector == "grpo_group" for a in spy.store.alerts)


# -- TRL -----------------------------------------------------------------------

def test_watch_trl_records_each_completion_and_returns_list():
    def batch_reward(prompts=None, completions=None, **kwargs):
        return [float(len(c)) for c in completions]

    watched = watch_trl(batch_reward, name="trl_t1")
    completions = ["aa", "bbbb", "cccccc"]
    result = watched(prompts=["p", "p", "p"], completions=completions)

    assert result == [2.0, 4.0, 6.0]
    assert watched.store.count == 3
    record = watched.store.recent(1)[0]
    assert record.output_length == 6  # last completion length


def test_watch_trl_handles_chat_completions():
    def batch_reward(prompts, completions):
        return [1.0 for _ in completions]

    watched = watch_trl(batch_reward, name="trl_chat")
    completions = [[{"role": "assistant", "content": "hello there"}]]
    watched(["prompt"], completions)
    assert watched.store.count == 1
    assert watched.store.recent(1)[0].output_length == len("hello there")


def test_watch_trl_runs_detection():
    def batch_reward(prompts, completions):
        return [1.0 for _ in completions]  # always max: should saturate

    watched = watch_trl(batch_reward, name="trl_detect", window_size=20)
    for _ in range(60):
        watched(["p"] * 5, ["resp"] * 5)
    assert watched.engine is not None
    assert len(watched.store.alerts) > 0


# -- W&B (with a stub run) -----------------------------------------------------

class _StubRun:
    def __init__(self):
        self.logged = []

    def log(self, data, step=None):
        self.logged.append((data, step))


def test_wandb_log_metrics_with_stub_run():
    import rewardspy

    @rewardspy.watch(name="wandb_t1")
    def reward(response, gt):
        return 1.0 if gt in response else 0.0

    for _ in range(10):
        reward("has 7", "7")

    run = _StubRun()
    data = rspy_wandb.log_metrics(reward.store, run=run, step=10)
    assert run.logged[0][1] == 10
    assert data["rewardspy/reward_mean"] == 1.0
    assert "rewardspy/health/overall" in data


def test_wandb_alert_callback_with_stub_run():
    from rewardspy.records import Alert

    run = _StubRun()
    cb = rspy_wandb.alert_callback(run=run)
    cb(Alert(step=5, timestamp=0.0, detector="variance", status="ALERT", message="boom"))
    assert run.logged
    assert run.logged[0][0]["rewardspy/alert_detector"] == "variance"
