import csv as csvlib

import pytest

import rewardspy
from rewardspy.exporters import JSONLExporter, read_jsonl, write_csv, write_parquet
from rewardspy.records import RolloutRecord


def make_record(step, reward, components=None):
    return RolloutRecord(
        call_id=f"c{step}",
        timestamp=float(step),
        step=step,
        scalar_reward=reward,
        components=components or {},
        call_duration_ms=1.5,
        input_length=20,
        output_length=10,
    )


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "rewards.jsonl"
    records = [make_record(i, i * 0.1, {"correctness": 0.5}) for i in range(5)]
    with JSONLExporter(path) as exporter:
        exporter.write_many(records)

    loaded = read_jsonl(path)
    assert loaded == records


def test_jsonl_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "deep" / "rewards.jsonl"
    with JSONLExporter(path) as exporter:
        exporter.write(make_record(0, 1.0))
    assert path.exists()
    assert len(read_jsonl(path)) == 1


def test_jsonl_appends_across_sessions(tmp_path):
    path = tmp_path / "rewards.jsonl"
    with JSONLExporter(path) as e:
        e.write(make_record(0, 0.1))
    with JSONLExporter(path) as e:
        e.write(make_record(1, 0.2))
    assert len(read_jsonl(path)) == 2


def test_watch_streams_to_jsonl(tmp_path):
    path = tmp_path / "logs" / "run.jsonl"

    @rewardspy.watch(name="streamed", export_path=path)
    def reward(response, gt):
        return {"correctness": 1.0, "total": 1.0}

    for _ in range(3):
        reward("a response", "gt")

    loaded = read_jsonl(path)
    assert len(loaded) == 3
    assert loaded[0].components["correctness"] == 1.0
    assert loaded[0].output_length == len("a response")


def test_csv_export_has_component_columns(tmp_path):
    records = [
        make_record(0, 0.9, {"correctness": 0.8, "format": 0.1}),
        make_record(1, 0.2, {"correctness": 0.1, "format": 0.1}),
    ]
    path = tmp_path / "out.csv"
    write_csv(records, path)

    with open(path, newline="") as fh:
        rows = list(csvlib.DictReader(fh))

    assert "component.correctness" in rows[0]
    assert "component.format" in rows[0]
    assert rows[0]["scalar_reward"] == "0.9"
    assert rows[0]["component.correctness"] == "0.8"
    assert len(rows) == 2


def test_csv_handles_varying_component_keys(tmp_path):
    records = [
        make_record(0, 1.0, {"a": 0.5}),
        make_record(1, 1.0, {"b": 0.5}),
    ]
    path = tmp_path / "out.csv"
    write_csv(records, path)
    with open(path, newline="") as fh:
        rows = list(csvlib.DictReader(fh))
    # Union of keys becomes columns; missing values are blank.
    assert rows[0]["component.a"] == "0.5"
    assert rows[0]["component.b"] == ""
    assert rows[1]["component.b"] == "0.5"


def test_session_separate_files_per_function(tmp_path):
    path = tmp_path / "rewards.jsonl"
    spy = rewardspy.Session(name="multi", export_path=path)

    @spy.watch(name="reward_a")
    def reward_a(response, gt):
        return 0.1

    @spy.watch(name="reward_b")
    def reward_b(response, gt):
        return 0.2

    reward_a("r", "g")
    reward_b("r", "g")

    # First function uses the base path; the second is suffixed with its name.
    assert path.exists()
    second = tmp_path / "rewards.reward_b.jsonl"
    assert second.exists()
    assert len(read_jsonl(path)) == 1
    assert len(read_jsonl(second)) == 1


def test_parquet_round_trip(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    records = [
        make_record(0, 0.9, {"correctness": 0.8, "format": 0.1}),
        make_record(1, 0.2, {"correctness": 0.1, "format": 0.1}),
    ]
    path = tmp_path / "out.parquet"
    write_parquet(records, path)

    table = pq.read_table(path)
    cols = table.column_names
    assert "scalar_reward" in cols
    assert "component.correctness" in cols
    assert table.num_rows == 2
    assert table.column("scalar_reward").to_pylist() == [0.9, 0.2]
    assert table.column("component.correctness").to_pylist() == [0.8, 0.1]


def test_parquet_missing_component_is_null(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    records = [
        make_record(0, 1.0, {"a": 0.5}),
        make_record(1, 1.0, {"b": 0.5}),
    ]
    path = tmp_path / "out.parquet"
    write_parquet(records, path)

    table = pq.read_table(path)
    assert table.column("component.a").to_pylist() == [0.5, None]
    assert table.column("component.b").to_pylist() == [None, 0.5]


def test_parquet_creates_parent_dirs(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "nested" / "deep" / "out.parquet"
    write_parquet([make_record(0, 1.0)], path)
    assert path.exists()
