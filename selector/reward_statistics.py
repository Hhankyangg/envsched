"""Read completed rollout groups and compute reward statistics."""

from __future__ import annotations

import glob
import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

DEFAULT_SCENARIOS = "vendor/ROLL/roll/pipeline/agentic/env/envscaler_env/data/envscaler_rl_scenario_metadata.json"


@dataclass
class RolloutObservation:
    task: str
    step: int
    group_id: str
    rewards: list[float]
    variance_rewards: list[float]

    @property
    def mean_reward(self) -> float:
        return statistics.mean(self.rewards)

    @property
    def reward_variance(self) -> float:
        return statistics.pvariance(self.variance_rewards)


def read_rollout_observations(
    run_dir: Path,
    variance_reward_builder: Callable[[list[dict[str, Any]], Path, str], list[float]],
    max_step: int | None = None,
    min_step: int = 0,
) -> list[RolloutObservation]:
    paths = glob.glob(str(run_dir / "output/cases/rollout_cases.step_*.jsonl"))
    paths.sort(key=lambda value: int(re.search(r"step_(\d+)", value).group(1)))
    observations: list[RolloutObservation] = []
    for path_str in paths:
        path = Path(path_str)
        step = int(re.search(r"step_(\d+)", path.name).group(1))
        if step < min_step or (max_step is not None and step > max_step):
            continue
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with path.open() as handle:
            for line in handle:
                item = json.loads(line)
                groups[item["traj_group_id"]].append(item)
        for group_id, items in groups.items():
            prompt = items[0]["prompt"]
            user_parts = re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt, re.S)
            if not user_parts:
                raise ValueError(f"Could not extract task from {path}:{group_id}")
            observations.append(
                RolloutObservation(
                    task=user_parts[-1].strip(),
                    step=step,
                    group_id=group_id,
                    rewards=[float(item["episode_score"]) for item in items],
                    variance_rewards=variance_reward_builder(items, path, group_id),
                )
            )
    if not observations:
        raise FileNotFoundError(f"No rollout case logs found under {run_dir}")
    return observations
