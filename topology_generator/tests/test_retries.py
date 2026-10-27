from argparse import Namespace
import json

import generate_topology as topology


def test_failed_batch_isolated_and_retried_after_successes_are_cached(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(topology.time, "sleep", lambda _: None)
    scenarios = [
        {"task_id": "task_a", "env_id": "env_a"},
        {"task_id": "task_b", "env_id": "env_b"},
    ]
    args = Namespace(
        batch_size=1,
        workers=2,
        sweep_retries=2,
        model="test-model",
        api_key_file=str(tmp_path / "key"),
        output_dir=str(tmp_path / "output"),
    )
    (tmp_path / "key").write_text("test-key")
    calls = {"task_a": 0, "task_b": 0}
    cached = []

    monkeypatch.setattr(topology, "get_env_record", lambda records, item: {})
    monkeypatch.setattr(topology, "read_cache", lambda *unused: None)

    def fake_call(batch, env_record, call_args, api_key):
        task_id = batch[0]["task_id"]
        calls[task_id] += 1
        if task_id == "task_a" and calls[task_id] == 1:
            raise RuntimeError("transient failure")
        return {task_id: {"task_id": task_id}}

    def fake_write(call_args, scenario, env_record, result):
        cached.append(scenario["task_id"])

    monkeypatch.setattr(topology, "call_with_fallback", fake_call)
    monkeypatch.setattr(topology, "write_cache", fake_write)

    topology.run_tagging(args, scenarios, {})

    report = json.loads((tmp_path / "output" / "tag_failures.json").read_text())
    assert calls == {"task_a": 2, "task_b": 1}
    assert sorted(cached) == ["task_a", "task_b"]
    assert report["failed_task_count"] == 0
