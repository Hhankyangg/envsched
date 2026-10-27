#!/usr/bin/env python3
"""Create a diverse step-0 curriculum without reading rollout outcomes."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from incremental_similarity_graph import (
    build_static_similarity_provider,
    similarity_column,
    similarity_submatrix,
)
from select_online_high_variance import (
    DEFAULT_SCENARIOS,
    DEFAULT_TOPOLOGIES,
    GRAPH_EXECUTION_MODES,
    atomic_write_json,
    build_static_similarity,
    load_pool,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    parser.add_argument("--topologies", default=DEFAULT_TOPOLOGIES)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-per-environment", type=int, default=2)
    parser.add_argument(
        "--graph-execution-mode",
        choices=GRAPH_EXECUTION_MODES,
        default="dense_legacy",
    )
    parser.add_argument(
        "--incremental-graph-state-dir",
        help=(
            "Derived graph cache shared with the online selector. Defaults to "
            "OUTPUT_DIR/incremental_graph_state."
        ),
    )
    return parser.parse_args()


def novelty(candidate: int, selected: list[int], similarity: Any) -> float:
    if not selected:
        return 1.0
    values = similarity_submatrix(
        similarity,
        np.asarray([candidate], dtype=int),
        np.asarray(selected, dtype=int),
    )
    return float(1.0 - np.max(values))


def mean_similarity_to_candidates(
    rows: np.ndarray,
    candidates: np.ndarray,
    similarity: Any,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    """Compute row means without materializing a candidate-square matrix."""
    means = np.empty(len(rows), dtype=float)
    for start in range(0, len(rows), chunk_size):
        stop = min(start + chunk_size, len(rows))
        means[start:stop] = similarity_submatrix(
            similarity,
            rows[start:stop],
            candidates,
        ).mean(axis=1)
    return means


def choose_environment_medoids(
    pool: pd.DataFrame,
    similarity: Any,
) -> list[int]:
    medoids = []
    for _, group in pool.groupby("env_id", sort=True):
        indices = group["task_index"].to_numpy(dtype=int)
        local_mean = mean_similarity_to_candidates(indices, indices, similarity)
        medoids.append(int(indices[int(np.argmax(local_mean))]))
    return medoids


def diverse_subset(
    candidates: list[int],
    count: int,
    similarity: Any,
) -> list[int]:
    if len(candidates) <= count:
        return list(candidates)
    candidate_array = np.asarray(candidates, dtype=int)
    average_similarity = mean_similarity_to_candidates(
        candidate_array,
        candidate_array,
        similarity,
    )
    selected = [int(candidate_array[int(np.argmin(average_similarity))])]
    remaining = set(candidates) - set(selected)
    redundancy = similarity_column(similarity, selected[0])[candidate_array]
    candidate_position = {
        int(task_index): position
        for position, task_index in enumerate(candidate_array)
    }
    while len(selected) < count:
        best = min(
            remaining,
            key=lambda index: (redundancy[candidate_position[index]], index),
        )
        selected.append(int(best))
        remaining.remove(best)
        redundancy = np.maximum(
            redundancy,
            similarity_column(similarity, best)[candidate_array],
        )
    return selected


def select_cold_start_tasks(
    pool: pd.DataFrame,
    similarity: Any,
    batch_size: int,
    max_per_environment: int,
) -> tuple[list[int], dict[int, str]]:
    if batch_size <= 0 or batch_size > len(pool):
        raise ValueError("batch_size must be between 1 and the task-pool size")
    if max_per_environment <= 0:
        raise ValueError("max_per_environment must be positive")

    medoids = choose_environment_medoids(pool, similarity)
    selected = diverse_subset(medoids, batch_size, similarity)
    roles = {index: "environment_medoid" for index in selected}
    environment_count = Counter(str(pool.iloc[index]["env_id"]) for index in selected)
    task_indices = pool["task_index"].to_numpy(dtype=int)
    env_ids = pool["env_id"].astype(str).to_numpy()
    global_redundancy = np.zeros(len(pool), dtype=float)
    local_redundancy = np.zeros(len(pool), dtype=float)
    for selected_index in selected:
        column = similarity_column(similarity, selected_index)[task_indices]
        global_redundancy = np.maximum(global_redundancy, column)
        same_environment = env_ids == env_ids[selected_index]
        local_redundancy[same_environment] = np.maximum(
            local_redundancy[same_environment],
            column[same_environment],
        )

    while len(selected) < batch_size:
        best: int | None = None
        best_score = -np.inf
        for index in pool["task_index"].to_numpy(dtype=int):
            index = int(index)
            if index in roles:
                continue
            env_id = str(pool.iloc[index]["env_id"])
            if environment_count[env_id] >= max_per_environment:
                continue
            global_novelty = 1.0 - global_redundancy[index]
            local_novelty = 1.0 - local_redundancy[index]
            score = 0.65 * global_novelty + 0.35 * local_novelty
            if score > best_score or (score == best_score and (best is None or index < best)):
                best = index
                best_score = score
        if best is None:
            raise RuntimeError("Could not fill the cold-start batch under the environment quota")
        selected.append(best)
        roles[best] = "diversity_fill"
        environment_count[str(pool.iloc[best]["env_id"])] += 1
        column = similarity_column(similarity, best)[task_indices]
        global_redundancy = np.maximum(global_redundancy, column)
        same_environment = env_ids == env_ids[best]
        local_redundancy[same_environment] = np.maximum(
            local_redundancy[same_environment],
            column[same_environment],
        )
    return selected, roles


def write_outputs(
    args: argparse.Namespace,
    output_dir: Path,
    pool: pd.DataFrame,
    scenarios: list[dict[str, Any]],
    similarity: Any,
    static_weights: dict[str, float],
    selected_indices: list[int],
    roles: dict[int, str],
) -> None:
    rows = []
    earlier: list[int] = []
    for rank, index in enumerate(selected_indices, start=1):
        row = pool.iloc[index]
        rows.append(
            {
                "selection_rank": rank,
                "task_id": str(row["task_id"]),
                "env_id": str(row["env_id"]),
                "env_class_name": str(row["env_class_name"]),
                "cold_start_role": roles[index],
                "static_novelty_at_selection": novelty(index, earlier, similarity),
                "task": str(row["task"]),
            }
        )
        earlier.append(index)
    selected = pd.DataFrame(rows)
    selected.to_csv(output_dir / "selected_tasks.csv", index=False)
    selected.to_csv(output_dir / "selected_tasks.step_0.csv", index=False)

    task_ids = selected["task_id"].tolist()
    scenario_by_id = {str(item["task_id"]): item for item in scenarios}
    manifest = {
        "schema_version": 1,
        "session_id": args.session_id,
        "history_max_step": None,
        "target_policy_step": 0,
        "selection_method": "static_diverse_cold_start",
        "selection_uses_current_target_rollout": False,
        "selection_uses_historical_rollouts": False,
        "selection_uses_previous_run_rollouts": False,
        "rollout_files_read": [],
        "task_ids": task_ids,
        "tasks": selected.drop(columns="task").to_dict(orient="records"),
    }
    audit = {
        "session_id": args.session_id,
        "target_policy_step": 0,
        "pool_task_count": len(pool),
        "selected_task_count": len(selected),
        "selected_environment_count": int(selected["env_id"].nunique()),
        "environment_task_count_histogram": {
            str(key): int(value)
            for key, value in sorted(
                Counter(selected["env_id"].value_counts().tolist()).items()
            )
        },
        "selection_method": "static_diverse_cold_start",
        "selection_uses_historical_rollouts": False,
        "selection_uses_previous_run_rollouts": False,
        "rollout_files_read": [],
        "source_run_dirs": [],
        "static_view_weights": static_weights,
        "max_per_environment": args.max_per_environment,
    }
    if args.graph_execution_mode == "incremental_exact":
        manifest["graph_execution_mode"] = args.graph_execution_mode
        audit["graph_execution_mode"] = args.graph_execution_mode
    for name in ("task_manifest.json", "task_manifest.step_0.json"):
        atomic_write_json(output_dir / name, manifest)
    for name in ("input_audit.json", "input_audit.step_0.json"):
        atomic_write_json(output_dir / name, audit)
    atomic_write_json(
        output_dir / "selected_task_items.json",
        [scenario_by_id[task_id] for task_id in task_ids],
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    # A first attempt can be interrupted after the deterministic static cache is
    # atomically committed but before any selector state is written.  Reusing
    # that input-only cache is still a fresh cold start and makes the launcher
    # safely retryable without reading any rollout history.
    existing_names = {path.name for path in output_dir.iterdir()} if output_dir.exists() else set()
    incremental_root = Path(
        args.incremental_graph_state_dir
        or output_dir / "incremental_graph_state"
    )
    if args.graph_execution_mode == "incremental_exact":
        allowed_cache_names = (
            {incremental_root.name}
            if incremental_root.parent == output_dir
            else set()
        )
    else:
        allowed_cache_names = {"static_similarity.npz"}
    unexpected_names = existing_names - allowed_cache_names
    if unexpected_names:
        raise FileExistsError(
            f"Cold-start output directory contains selector outputs: {output_dir} "
            f"({sorted(unexpected_names)})"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    pool, scenarios = load_pool(args.scenarios)
    if args.graph_execution_mode == "incremental_exact":
        similarity, static_weights = build_static_similarity_provider(
            pool,
            scenarios,
            args.topologies,
            incremental_root / "static_provider",
        )
    else:
        similarity, static_weights = build_static_similarity(
            pool,
            scenarios,
            args.topologies,
            output_dir / "static_similarity.npz",
        )
    selected, roles = select_cold_start_tasks(
        pool,
        similarity,
        args.batch_size,
        args.max_per_environment,
    )
    write_outputs(
        args,
        output_dir,
        pool,
        scenarios,
        similarity,
        static_weights,
        selected,
        roles,
    )
    print(
        f"selection_method=static_diverse_cold_start selected={len(selected)} "
        f"manifest={output_dir / 'task_manifest.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
