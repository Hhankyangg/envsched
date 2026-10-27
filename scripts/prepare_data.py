#!/usr/bin/env python3
"""Fetch, verify, and install the EnvScaler training data used by EnvSched."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = "vendor/ROLL/roll/pipeline/agentic/env/envscaler_env/data/envscaler_rl_scenario_metadata.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict:
    return json.loads((ROOT / "data/manifest.json").read_text())


def read_source(manifest: dict, entry: dict, source_dir: Path | None) -> bytes:
    """Read one pinned upstream file from a local copy or the pinned commit."""
    relative = entry["source"]
    if source_dir is not None:
        for candidate in (source_dir / relative, source_dir / Path(relative).name):
            if candidate.is_file():
                return candidate.read_bytes()
        raise ValueError(f"{Path(relative).name} not found under {source_dir}")
    source = manifest["source"]
    url = source["raw_url"].format(commit=source["commit"], path=relative)
    with urllib.request.urlopen(url, timeout=300) as response:
        return response.read()


def scenario_environments(environments: bytes, scenarios: bytes) -> bytes:
    """Keep the environments that the RL scenarios use, in upstream order."""
    used = {str(item["env_id"]) for item in json.loads(scenarios)}
    records = json.loads(environments)
    missing = used - set(records)
    if missing:
        raise ValueError(f"Environment metadata lacks scenario environments: {sorted(missing)[:5]}")
    subset = {env_id: record for env_id, record in records.items() if env_id in used}
    return (json.dumps(subset, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def build_files(manifest: dict, source_dir: Path | None) -> dict[str, bytes]:
    """Return every installed file's verified content without writing anything."""
    built: dict[str, bytes] = {}
    sources: dict[str, bytes] = {}
    for name, entry in manifest["files"].items():
        content = read_source(manifest, entry, source_dir)
        if hashlib.sha256(content).hexdigest() != entry["source_sha256"]:
            raise ValueError(f"Upstream SHA-256 mismatch: {entry['source']}")
        sources[name] = content
    for name, entry in manifest["files"].items():
        if entry.get("derive") == "scenario_environments":
            content = scenario_environments(sources[name], sources[SCENARIOS])
        elif "derive" in entry:
            raise ValueError(f"Unknown derivation for {name}: {entry['derive']}")
        else:
            content = sources[name]
        if len(content) != entry["size"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"Data SHA-256 mismatch: {name}")
        built[name] = content
    return built


def install(files: dict[str, bytes]) -> None:
    targets = {}
    for name, content in files.items():
        target = ROOT / name
        if target.is_symlink() or not target.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError(f"Invalid data destination: {name}")
        if target.exists() and (
            not target.is_file()
            or sha256(target) != hashlib.sha256(content).hexdigest()
        ):
            raise ValueError(f"Refusing to overwrite different existing data: {name}")
        targets[name] = target
    with tempfile.TemporaryDirectory(prefix=".data-", dir=ROOT) as staging:
        staged = []
        for index, (name, content) in enumerate(files.items()):
            path = Path(staging) / str(index)
            path.write_bytes(content)
            staged.append((path, targets[name]))
        for path, target in staged:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.replace(path, target)


def data_status(manifest: dict, *, include_generated: bool = True) -> list[str]:
    problems = []
    for name, expected in manifest["files"].items():
        path = ROOT / name
        if not path.is_file():
            problems.append(f"Missing: {name}")
        elif path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
            problems.append(f"Checksum mismatch: {name}")
    if not include_generated:
        return problems
    for name, generator in manifest["generated"].items():
        path = ROOT / name
        if not path.is_file():
            problems.append(f"Missing: {name} (generate it with {generator})")
        elif (ROOT / SCENARIOS).is_file():
            task_ids = {
                str(json.loads(line)["task_id"])
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            scenario_ids = {
                str(item["task_id"])
                for item in json.loads((ROOT / SCENARIOS).read_text(encoding="utf-8"))
            }
            if task_ids != scenario_ids:
                problems.append(f"Task IDs in {name} do not match the scenario pool")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Local EnvScaler checkout or directory with the upstream files; default downloads them",
    )
    parser.add_argument("--check", action="store_true", help="Verify already installed data")
    args = parser.parse_args()
    manifest = load_manifest()
    try:
        if not args.check:
            source_dir = args.source_dir.expanduser().resolve() if args.source_dir else None
            install(build_files(manifest, source_dir))
        problems = data_status(manifest, include_generated=args.check)
        if problems:
            parser.exit(1, "\n".join(problems) + "\n")
    except (OSError, ValueError, KeyError, urllib.error.URLError) as exc:
        parser.exit(1, f"Data preparation failed: {exc}\n")
    if args.check:
        print("Verified EnvScaler data and task topologies.")
    else:
        print(
            f"Installed {len(manifest['files'])} verified EnvScaler data files. "
            "Next, generate data/task_topologies.jsonl with topology_generator/."
        )


if __name__ == "__main__":
    main()
