import asyncio

from rich.console import Console

import rewardspy
from rewardspy.store import MetricStore
from rewardspy.tui import render
from rewardspy.tui.app import RewardSpyApp


def render_to_text(renderable, width: int = 80) -> str:
    console = Console(width=width)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


def populated_store(n: int = 130):
    @rewardspy.watch(name="tui_demo")
    def reward(response, gt):
        return {"correctness": 0.0, "format": 1.0, "total": 1.0}

    for _ in range(n):
        reward("a short response", "gt")
    return reward.store


# -- pure render functions -----------------------------------------------------

def test_sparkline_length_and_charset():
    line = render.sparkline([0, 1, 2, 3, 4, 5, 6, 7, 8], width=9)
    assert len(line) == 9
    assert all(ch in render.theme.BLOCKS for ch in line)


def test_sparkline_ascii_mode_uses_ascii_ramp():
    line = render.sparkline([0, 1, 2, 3, 4, 5, 6, 7, 8], width=9, ascii=True)
    assert len(line) == 9
    assert all(ord(ch) < 128 for ch in line)
    assert all(ch in render.theme.ASCII_BLOCKS for ch in line)


def test_curve_empty_store_shows_placeholder():
    out = render_to_text(render.reward_curve(MetricStore("e"), width=40, height=6))
    assert "Waiting" in out


def test_curve_renders_multiple_rows():
    store = MetricStore("c")
    for i in range(60):
        store.append(_rec(i, (i % 10) / 10.0))
    out = render_to_text(render.reward_curve(store, width=50, height=8))
    assert out.count("\n") >= 8


def test_curve_ascii_mode_uses_plain_chart_glyphs():
    store = MetricStore("ascii")
    for i in range(60):
        store.append(_rec(i, (i % 10) / 10.0))
    out = render_to_text(render.reward_curve(store, width=50, height=8, ascii=True))
    assert out.count("\n") >= 8
    assert all(ord(ch) < 128 for ch in out)
    assert "+" in out and "-" in out and "|" in out


def test_overview_shows_stats():
    store = MetricStore("o")
    for i in range(50):
        store.append(_rec(i, 0.5))
    out = render_to_text(render.reward_overview(store))
    assert "mean" in out and "trend" in out and "ceiling" in out
    assert "0.500" in out


def test_diagnosis_healthy_vs_hacking():
    healthy = MetricStore("hh")
    for i in range(150):
        healthy.append(_rec(i, (i % 11) / 10.0))
    text = render_to_text(render.diagnosis(healthy))
    assert "Collecting" in text or "Healthy" in text

    store = populated_store()
    out = render_to_text(render.diagnosis(store))
    assert "hacking" in out.lower()
    assert "step" in out.lower()


def test_hack_status_lists_detectors_and_overall():
    store = populated_store()
    out = render_to_text(render.hack_status(store))
    for label in ("Variance", "Slope", "Component", "Ceiling", "Length"):
        assert label in out
    assert "Overall" in out


def test_component_bars_render_names():
    store = populated_store()
    out = render_to_text(render.component_bars(store, width=40))
    assert "correctness" in out
    assert "format" in out


def test_alerts_log_healthy_then_populated():
    healthy = render_to_text(render.alerts_log(MetricStore("h")))
    assert "healthy" in healthy.lower()

    store = populated_store()
    assert store.alerts  # detectors fired on the hacky reward
    out = render_to_text(render.alerts_log(store))
    assert "step" in out.lower()


def test_rollouts_table_has_header():
    store = populated_store()
    out = render_to_text(render.rollouts_table(store))
    assert "step" in out and "reward" in out


# -- app mounts headless -------------------------------------------------------

def test_app_mounts_and_refreshes():
    store = populated_store()

    async def run():
        app = RewardSpyApp(store, interval=10.0)
        async with app.run_test() as pilot:
            await pilot.pause()
            # All panels mounted and the header is present.
            from rewardspy.tui.app import Panel

            assert len(app.query(Panel)) == 7

    asyncio.run(run())


def _rec(step, reward):
    from rewardspy.records import RolloutRecord

    return RolloutRecord(
        call_id=f"c{step}",
        timestamp=float(step),
        step=step,
        scalar_reward=reward,
        output_length=10,
    )
