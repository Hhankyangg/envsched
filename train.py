#!/usr/bin/env python3
"""Portable, synchronous Qwen3 selector Direct RL entry point."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "vendor/ROLL/roll/pipeline/agentic/env/envscaler_env/data"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", required=True, choices=["1.7b", "4b", "8b"])
    parser.add_argument("--model-path", type=Path, help="Local original Qwen3 thinking model directory")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="Physical GPU IDs assigned to this job")
    parser.add_argument("--run-dir", type=Path, required=True, help="Output directory for this training run")
    parser.add_argument("--max-steps", type=int, help="Pipeline iterations: default 201 for all model sizes")
    parser.add_argument("--selector-python", default=sys.executable, help="Python with selector CPU dependencies")
    parser.add_argument("--ray-cpus", type=int, default=48)
    parser.add_argument("--prepare-only", action="store_true", help="Resolve config and build step 0 without CUDA, Ray or model loading")
    return parser.parse_args(argv)


def resolve_config(args):
    from omegaconf import OmegaConf

    profile = OmegaConf.to_container(OmegaConf.load(ROOT / f"configs/qwen3_{args.model_size}.yaml"))
    gpu_ids = args.gpus.split(",")
    if not gpu_ids or any(not gpu.isdigit() for gpu in gpu_ids) or len({int(gpu) for gpu in gpu_ids}) != len(gpu_ids):
        raise ValueError("--gpus requires distinct comma-separated integer GPU IDs")
    n = len(gpu_ids)
    pp = profile["pipeline_parallel_size"]
    if n % pp:
        raise ValueError(f"GPU count must be divisible by pipeline size {pp}")
    max_steps = args.max_steps if args.max_steps is not None else profile["max_steps"]
    if max_steps <= 0 or args.ray_cpus <= 0:
        raise ValueError("--max-steps and --ray-cpus must be positive")
    run = args.run_dir.expanduser().resolve()
    model = (args.model_path or ROOT / "models" / profile["model_id"].split("/")[-1]).expanduser().resolve()
    selector_python = shutil.which(args.selector_python)
    if selector_python is None:
        raise ValueError(f"Selector interpreter not found: {args.selector_python}")
    paths = [ROOT / "python_patches", ROOT / "selector", ROOT / "vendor/ROLL/mcore_adapter/src", ROOT / "vendor/ROLL", ROOT]
    # Use only this release's sources; inherited workspace paths can shadow them.
    env = {}
    env.update({
        "WORKSPACE": str(ROOT), "RUN_NAME": run.name, "RUN_DIR": str(run),
        "MODEL_PATH": str(model), "MAX_STEPS": str(max_steps), "NUM_GPUS": str(n),
        "PIPELINE_PARALLEL_SIZE": str(pp), "GRADIENT_ACCUMULATION_STEPS": str((256 + n // pp - 1) // (n // pp)),
        "DEVICE_MAPPING": str(list(range(n))), "CUDA_VISIBLE_DEVICES": ",".join(gpu_ids),
        "ROLLOUT_GPU_MEMORY_UTILIZATION": str(profile["rollout_gpu_memory_utilization"]),
        "PYTHONPATH": os.pathsep.join(map(str, paths)), "SELECTOR_PYTHON": selector_python,
        "CURRICULUM_OUTPUT_DIR": str(run / "selector"),
        "CURRICULUM_MANIFEST_PATH": str(run / "selector/task_manifest.json"),
        "CURRICULUM_SCENARIOS": str(DATA / "envscaler_rl_scenario_metadata.json"),
        "CURRICULUM_ENVIRONMENT_METADATA": str(DATA / "environment_metadata.json"),
        "CURRICULUM_TOPOLOGIES": str(ROOT / "data/task_topologies.jsonl"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "RAY_USAGE_STATS_ENABLED": "0", "HF_HUB_DISABLE_TELEMETRY": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "ROLL_DEFER_ROLLOUT_SCHEDULER_CREATION": "1", "ROLL_STREAMING_DIST_OPT_SAVE": "1",
        "ROLL_STREAMING_DIST_OPT_SAVE_COPY_AHEAD_BYTES": "1048576",
        "ROLL_LOG_DIR": str(run / "output/logs"),
    })
    os.environ.update(env)
    cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/selector.yaml"), resolve=True)
    cfg["zero_loss_batch_padding_enabled"] = bool(256 % (n // pp))
    return cfg, profile, env


def prepare(args, cfg, profile, env):
    from omegaconf import OmegaConf

    run = Path(env["RUN_DIR"])
    run.mkdir(parents=True, exist_ok=True)
    if not (run / "selector/task_manifest.step_0.json").exists():
        subprocess.run([
            env["SELECTOR_PYTHON"], str(ROOT / "selector/select_static_cold_start.py"),
            "--scenarios", env["CURRICULUM_SCENARIOS"], "--topologies", env["CURRICULUM_TOPOLOGIES"],
            "--output-dir", env["CURRICULUM_OUTPUT_DIR"],
            "--session-id", run.name, "--batch-size", "64", "--max-per-environment", "2",
        ], cwd=ROOT, env=os.environ.copy(), check=True)
    OmegaConf.save(OmegaConf.create(cfg), run / "resolved_config.yaml")
    print(f"Prepared {profile['model_id']}: {env['NUM_GPUS']} GPUs, TP1/PP{profile['pipeline_parallel_size']}, "
          f"{cfg['max_steps']} iterations, 64 tasks x 8 rollouts; {run}", flush=True)


def run_training(args, cfg, env):
    # The interpreter has already started, so load the same compatibility hooks
    # that PYTHONPATH makes available to child processes.
    for path in reversed(env["PYTHONPATH"].split(os.pathsep)):
        sys.path.insert(0, path)
    import sitecustomize  # noqa: F401
    from dacite import from_dict
    import ray
    from roll.pipeline.agentic.agentic_config import AgenticConfig
    from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline
    from roll.utils.constants import RAY_NAMESPACE

    config = from_dict(data_class=AgenticConfig, data=cfg)
    runtime_vars = dict(cfg["system_envs"])
    runtime_vars.update({k: v for k, v in env.items() if k != "CUDA_VISIBLE_DEVICES"})
    temp = tempfile.mkdtemp(prefix="envsched-", dir="/tmp")
    ray.init(address="local", namespace=RAY_NAMESPACE, num_gpus=int(env["NUM_GPUS"]),
             num_cpus=args.ray_cpus, include_dashboard=False, _temp_dir=temp,
             runtime_env={"env_vars": runtime_vars})
    try:
        AgenticPipeline(pipeline_config=config).run()
    finally:
        ray.shutdown()


def main(argv=None):
    args = parse_args(argv)
    required = [
        DATA / "envscaler_rl_scenario_metadata.json",
        DATA / "environment_metadata.json",
        ROOT / "data/task_topologies.jsonl",
        ROOT / "vendor/ROLL/roll/pipeline/agentic/env/bfcl_env/data/data_multi_turn_base.json",
    ]
    if any(not path.is_file() for path in required):
        raise SystemExit(
            "Training data is missing. Run: python scripts/prepare_data.py, then generate "
            "data/task_topologies.jsonl with topology_generator/ (see README)."
        )
    cfg, profile, env = resolve_config(args)
    if not args.prepare_only and not Path(env["MODEL_PATH"]).is_dir():
        raise SystemExit("--model-path must point to a local original Qwen3 model directory")
    prepare(args, cfg, profile, env)
    if not args.prepare_only:
        run_training(args, cfg, env)


if __name__ == "__main__":
    main()
