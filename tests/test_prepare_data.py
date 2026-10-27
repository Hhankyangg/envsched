import hashlib
import json

import pytest

from scripts import prepare_data

SCENARIOS = prepare_data.SCENARIOS
ENVIRONMENTS = "vendor/data/environment_metadata.json"
TOPOLOGIES = "data/task_topologies.jsonl"


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def fixture(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(prepare_data, "ROOT", root)
    source = tmp_path / "EnvScaler"
    scenarios = json.dumps(
        [{"task_id": "t1", "env_id": "env_b"}, {"task_id": "t2", "env_id": "env_a"}]
    ).encode()
    environments = json.dumps(
        {"env_a": {"name": "A"}, "env_unused": {"name": "-"}, "env_b": {"name": "B"}}
    ).encode()
    # The subset keeps upstream order and drops environments no scenario uses.
    subset = b'{"env_a":{"name":"A"},"env_b":{"name":"B"}}\n'
    for relative, content in (
        ("rl/scenarios.json", scenarios),
        ("rl/191_env_metadata.json", environments),
    ):
        (source / relative).parent.mkdir(parents=True, exist_ok=True)
        (source / relative).write_bytes(content)
    manifest = {
        "source": {"raw_url": "unused", "commit": "unused"},
        "files": {
            SCENARIOS: {
                "source": "rl/scenarios.json",
                "source_sha256": digest(scenarios),
                "sha256": digest(scenarios),
                "size": len(scenarios),
            },
            ENVIRONMENTS: {
                "source": "rl/191_env_metadata.json",
                "source_sha256": digest(environments),
                "derive": "scenario_environments",
                "sha256": digest(subset),
                "size": len(subset),
            },
        },
        "generated": {TOPOLOGIES: "topology_generator/generate_topology.py"},
    }
    return root, source, manifest


def test_installs_verified_data_and_is_idempotent(tmp_path, monkeypatch):
    root, source, manifest = fixture(tmp_path, monkeypatch)
    prepare_data.install(prepare_data.build_files(manifest, source))
    prepare_data.install(prepare_data.build_files(manifest, source))
    assert (root / ENVIRONMENTS).read_bytes() == b'{"env_a":{"name":"A"},"env_b":{"name":"B"}}\n'
    assert prepare_data.data_status(manifest, include_generated=False) == []


def test_flat_source_directory_is_accepted(tmp_path, monkeypatch):
    root, source, manifest = fixture(tmp_path, monkeypatch)
    flat = tmp_path / "downloads"
    flat.mkdir()
    for path in source.rglob("*.json"):
        (flat / path.name).write_bytes(path.read_bytes())
    prepare_data.install(prepare_data.build_files(manifest, flat))
    assert prepare_data.data_status(manifest, include_generated=False) == []


def test_rejects_changed_upstream_before_installing(tmp_path, monkeypatch):
    root, source, manifest = fixture(tmp_path, monkeypatch)
    (source / "rl/scenarios.json").write_bytes(b"[]")
    with pytest.raises(ValueError, match="Upstream SHA-256"):
        prepare_data.build_files(manifest, source)
    assert not list(root.iterdir())


def test_rejects_unexpected_derived_content(tmp_path, monkeypatch):
    root, source, manifest = fixture(tmp_path, monkeypatch)
    manifest["files"][ENVIRONMENTS]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Data SHA-256"):
        prepare_data.build_files(manifest, source)
    assert not list(root.iterdir())


def test_preserves_different_existing_data(tmp_path, monkeypatch):
    root, source, manifest = fixture(tmp_path, monkeypatch)
    target = root / ENVIRONMENTS
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing data")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        prepare_data.install(prepare_data.build_files(manifest, source))
    assert target.read_bytes() == b"existing data"
    assert not (root / SCENARIOS).exists()


def test_rejects_link_destination(tmp_path, monkeypatch):
    root, source, manifest = fixture(tmp_path, monkeypatch)
    target = root / ENVIRONMENTS
    target.parent.mkdir(parents=True)
    target.symlink_to(tmp_path / "elsewhere.json")
    with pytest.raises(ValueError, match="Invalid data destination"):
        prepare_data.install(prepare_data.build_files(manifest, source))


def test_check_requires_topologies_for_the_scenario_pool(tmp_path, monkeypatch):
    root, source, manifest = fixture(tmp_path, monkeypatch)
    prepare_data.install(prepare_data.build_files(manifest, source))
    assert any("generate it" in problem for problem in prepare_data.data_status(manifest))
    topologies = root / TOPOLOGIES
    topologies.parent.mkdir(parents=True)
    topologies.write_text(json.dumps({"task_id": "t1"}) + "\n")
    assert any("do not match" in problem for problem in prepare_data.data_status(manifest))
    topologies.write_text(
        "".join(json.dumps({"task_id": task_id}) + "\n" for task_id in ("t1", "t2"))
    )
    assert prepare_data.data_status(manifest) == []
