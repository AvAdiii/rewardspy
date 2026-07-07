import csv as csvlib

from click.testing import CliRunner

from rewardspy.cli import _load_store, main
from rewardspy.exporters import JSONLExporter, read_jsonl
from rewardspy.records import RolloutRecord


def _rec(step, reward, components=None, length=10):
    return RolloutRecord(
        call_id=f"c{step}",
        timestamp=float(step),
        step=step,
        scalar_reward=reward,
        components=components or {},
        output_length=length,
    )


def write_log(path, records):
    with JSONLExporter(path, mode="w") as exporter:
        exporter.write_many(records)


def hacky_log(path, n=120):
    records = [
        _rec(i, 1.1, {"correctness": 1.0, "format": 0.1, "total": 1.1}) for i in range(n)
    ]
    write_log(path, records)


def healthy_log(path, n=200):
    records = [
        _rec(i, (i % 10) / 10.0, {"a": 0.25, "b": 0.25, "total": (i % 10) / 10.0})
        for i in range(n)
    ]
    write_log(path, records)


def test_load_store_replays_and_detects(tmp_path):
    path = tmp_path / "h.jsonl"
    hacky_log(path)
    store, engine = _load_store(path, window=50, sensitivity="medium")
    assert store.count == 120
    assert engine.overall.value == "ALERT"


def test_summary_runs(tmp_path):
    path = tmp_path / "h.jsonl"
    hacky_log(path)
    result = CliRunner().invoke(main, ["summary", str(path), "--window", "50"])
    assert result.exit_code == 0
    assert "mean" in result.output
    assert "hacking" in result.output.lower()


def test_summary_empty_log(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    result = CliRunner().invoke(main, ["summary", str(path)])
    assert result.exit_code == 0
    assert "No records" in result.output


def test_audit_exit_code_on_hacking(tmp_path):
    path = tmp_path / "h.jsonl"
    hacky_log(path)
    result = CliRunner().invoke(main, ["audit", str(path), "--window", "50"])
    assert result.exit_code == 1
    assert "hacking" in result.output.lower()


def test_audit_exit_code_on_healthy(tmp_path):
    path = tmp_path / "ok.jsonl"
    healthy_log(path)
    result = CliRunner().invoke(main, ["audit", str(path), "--window", "50"])
    assert result.exit_code == 0


def test_export_to_csv(tmp_path):
    path = tmp_path / "h.jsonl"
    hacky_log(path, n=10)
    out = tmp_path / "out.csv"
    result = CliRunner().invoke(main, ["export", str(path), "-o", str(out)])
    assert result.exit_code == 0
    with open(out, newline="") as fh:
        rows = list(csvlib.DictReader(fh))
    assert len(rows) == 10
    assert "component.correctness" in rows[0]


def test_export_to_jsonl_with_last(tmp_path):
    path = tmp_path / "h.jsonl"
    hacky_log(path, n=50)
    out = tmp_path / "trim.jsonl"
    result = CliRunner().invoke(
        main, ["export", str(path), "-o", str(out), "--format", "jsonl", "--last", "5"]
    )
    assert result.exit_code == 0
    assert len(read_jsonl(out)) == 5


def test_probe_runs_reward_function(tmp_path, monkeypatch):
    module = tmp_path / "my_reward.py"
    module.write_text(
        "def reward(response, answer):\n"
        "    return 1.0 if answer in response else 0.0\n"
    )
    cases = tmp_path / "cases.json"
    cases.write_text(
        '[{"response": "the answer is 42", "answer": "42"},'
        ' {"response": "nope", "answer": "42"}]'
    )
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        main, ["probe", "my_reward:reward", "-p", str(cases)]
    )
    assert result.exit_code == 0
    assert "records" in result.output
    assert "0.5" in result.output  # mean of [1.0, 0.0]


def test_probe_bad_target():
    result = CliRunner().invoke(main, ["probe", "no_colon", "-p", "x"])
    assert result.exit_code != 0


def test_help_lists_commands():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    for command in ("show", "summary", "audit", "export", "probe"):
        assert command in result.output


def test_show_help_lists_ascii_charts_option():
    result = CliRunner().invoke(main, ["show", "--help"])
    assert result.exit_code == 0
    assert "--ascii-charts" in result.output
