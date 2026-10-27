import copy
import json
import sys
from types import SimpleNamespace

import pytest

import generate_topology as topology


def inputs(tmp_path):
    task = {
        "task_id": "task-1", "env_id": "env-1", "env_class_name": "TinyEnv",
        "task": "Set the value to one.", "init_config": {"value": 0},
        "checklist_with_func": [{"check_item": "Value is one", "check_func": "def check_func(state):\n    return state['value'] == 1\n"}],
    }
    envs = {"env-1": {"env_class_name": "TinyEnv", "tools": [{"type": "function", "function": {"name": "set_value", "description": "Set the value"}}]}}
    tasks_path = tmp_path / "tasks.json"
    envs_path = tmp_path / "envs.json"
    tasks_path.write_text(json.dumps([task]))
    envs_path.write_text(json.dumps(envs))
    args = topology.parse_args([
        "--scenarios", str(tasks_path), "--env-metadata", str(envs_path),
        "--model", "test-model", "--output-dir", str(tmp_path / "output"),
    ])
    return args, task, envs


def raw_topology(task_id):
    return {
        "task_id": task_id,
        "operations": [{"id": "update", "family": "update", "tool_candidates": ["set_value"], "affected_checks": ["c0"]}],
        "reward_groups": [{"check_ids": ["c0"], "producer_operations": ["update"], "relation": "shared_operation"}],
    }


def test_add_tasks_reuses_cache_and_exports_full_pool(tmp_path, monkeypatch):
    args, task, envs = inputs(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls = []

    def fake_batch(batch, env_record, call_args, api_key):
        calls.extend(item["task_id"] for item in batch)
        return {item["task_id"]: topology.validate_topology(raw_topology(item["task_id"]), item, env_record) for item in batch}

    monkeypatch.setattr(topology, "call_with_fallback", fake_batch)
    scenarios, _ = topology.load_selected(args)
    topology.run_tagging(args, scenarios, envs)
    second = copy.deepcopy(task)
    second["task_id"] = "task-2"
    scenarios.append(second)
    topology.run_tagging(args, scenarios, envs)
    topology.export_topologies(args, scenarios, envs)
    assert calls == ["task-1", "task-2"]
    rows = [json.loads(line) for line in (tmp_path / "output/task_topologies.jsonl").read_text().splitlines()]
    assert [row["task_id"] for row in rows] == ["task-1", "task-2"]
    assert set(rows[0]) == {"task_id", "env_id", "env_class_name", "topology"}
    # A complete cache needs neither a key nor another model call.
    monkeypatch.delenv("OPENAI_API_KEY")
    topology.run_tagging(args, scenarios, envs)
    assert calls == ["task-1", "task-2"]
    # Changing static input invalidates only the corresponding task.
    scenarios[0]["task"] = "Set the value to two."
    assert topology.read_cache(args, scenarios[0], envs["env-1"]) is None
    assert topology.read_cache(args, scenarios[1], envs["env-1"]) is not None


def test_missing_cache_cannot_export_partial_pool(tmp_path):
    args, task, envs = inputs(tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing valid topology cache"):
        topology.export_topologies(args, [task], envs)
    assert not (tmp_path / "output/task_topologies.jsonl").exists()


def test_duplicate_ids_rejected_before_generation(tmp_path):
    args, task, _ = inputs(tmp_path)
    from pathlib import Path
    Path(args.scenarios).write_text(json.dumps([task, task]))
    with pytest.raises(ValueError, match="Duplicate task_id"):
        topology.load_selected(args)


@pytest.mark.parametrize("task_id", ["../escape", "/absolute", "bad\\path"])
def test_cache_cannot_escape_directory(tmp_path, task_id):
    args, _, _ = inputs(tmp_path)
    with pytest.raises(ValueError, match="filename-safe"):
        topology.cache_path(args, task_id)


def test_streamed_model_response_is_validated(tmp_path, monkeypatch):
    args, task, envs = inputs(tmp_path)
    args.max_retries = 1
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        answer = json.dumps({"tasks": [raw_topology(task["task_id"])]})
        return [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=answer[:15]))]), SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=answer[15:]))])]

    fake = SimpleNamespace(OpenAI=lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setitem(sys.modules, "openai", fake)
    result = topology.call_batch([task], envs["env-1"], args, "test-key")
    assert result["task-1"]["operations"][0]["id"] == "o0"
    assert result["task-1"]["reward_groups"][0]["check_ids"] == ["c0"]
    payload = json.loads(captured["messages"][1]["content"])
    assert payload["tasks"][0]["verifier_checks"][0]["initially_true"] is False
    assert captured["stream"] is True
