"""The incremental_exact graph backend must reproduce the dense selector."""

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from incremental_similarity_graph import (
    IncrementalGraphState,
    build_abstract_topology_similarity_provider,
    build_static_similarity_provider,
    similarity_submatrix,
)
from select_online_high_variance import (
    aggregate_history,
    build_abstract_topology_similarity,
    build_static_similarity,
    graph_neighbor_weights,
    predict_tasks,
    select_diverse_tasks,
)
from select_static_cold_start import select_cold_start_tasks
from task_progress_retirement import estimate_task_progress

ENTITIES = ("user", "order", "room", "patient", "policy", "ticket")
FAMILIES = ("lookup", "update", "create", "delete", "validate", "authorize")
GATE_TYPES = ("identifier_resolution", "state_precondition", "authorization")
PROVENANCE = ("task_literal", "unique_lookup", "broad_lookup", "prior_tool_output")
RELATIONS = ("shared_operation", "shared_gate", "initially_satisfied")
VALUES = ("active", "closed", "pending", "high")


def synthetic_pool(tmp_path: Path) -> tuple[pd.DataFrame, list[dict], str]:
    rng = np.random.default_rng(0)
    scenarios = []
    topologies = []
    for env, entity in enumerate(ENTITIES):
        fields = (f"{entity}_status", f"{entity}_owner", "priority", "notes")
        for task in range(10):
            field = fields[rng.integers(len(fields))]
            value = VALUES[rng.integers(len(VALUES))]
            task_id = f"env_{env}-task_{task}"
            records = {
                f"{entity}_{index}": {
                    "status": VALUES[rng.integers(len(VALUES))],
                    field: VALUES[rng.integers(len(VALUES))],
                }
                for index in range(int(rng.integers(2, 7)))
            }
            scenarios.append(
                {
                    "task_id": task_id,
                    "env_id": f"env_{env}",
                    "env_class_name": f"{entity.title()}Env",
                    "task": (
                        f"Find {entity} number {task} in system {env} "
                        f"and set its {field} to {value}."
                    ),
                    "init_config": {f"{entity}s": records},
                    "checklist_with_func": [
                        {
                            "check_item": f"The {entity} {field} is {value}.",
                            "check_func": (
                                "def check(state):\n"
                                f"    return state['{entity}s']['{entity}_0']"
                                f"['{field}'] == '{value}'\n"
                            ),
                        }
                    ],
                }
            )
            # The abstract vocabulary is small, so many cross-environment
            # scores tie and exercise the boundary-tie fallback.
            operation_count = int(rng.integers(1, 4))
            topologies.append(
                {
                    "task_id": task_id,
                    "topology": {
                        "operations": [
                            {
                                "family": FAMILIES[rng.integers(len(FAMILIES))],
                                "depends_on_operations": ["o0"] * position,
                                "gate_ids": ["g0"] * int(rng.integers(0, 2)),
                            }
                            for position in range(operation_count)
                        ],
                        "gates": [
                            {
                                "type": GATE_TYPES[rng.integers(len(GATE_TYPES))],
                                "provenance": PROVENANCE[rng.integers(len(PROVENANCE))],
                            }
                            for _ in range(int(rng.integers(0, 3)))
                        ],
                        "reward_groups": [
                            {"relation": RELATIONS[rng.integers(len(RELATIONS))]}
                        ],
                        "conflicts": [],
                        "independent_branches": [],
                    },
                }
            )
    topology_path = tmp_path / "task_topologies.jsonl"
    topology_path.write_text(
        "".join(json.dumps(item) + "\n" for item in topologies), encoding="utf-8"
    )
    pool = pd.DataFrame(
        {
            "task_index": np.arange(len(scenarios)),
            "task_id": [item["task_id"] for item in scenarios],
            "env_id": [item["env_id"] for item in scenarios],
            "env_class_name": [item["env_class_name"] for item in scenarios],
            "task": [item["task"] for item in scenarios],
        }
    )
    return pool, scenarios, str(topology_path)


def synthetic_history(pool: pd.DataFrame, steps: int) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for step in range(steps):
        for task_index in rng.choice(len(pool), size=20, replace=False):
            task = pool.iloc[int(task_index)]
            rows.append(
                {
                    "task_index": int(task_index),
                    "task_id": task["task_id"],
                    "env_id": task["env_id"],
                    "step": step,
                    "reward_variance": float(rng.uniform(0.0, 0.25)),
                    "mean_reward": float(rng.uniform(0.0, 1.0)),
                    "dispersion": float(rng.uniform(0.0, 1.0)),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def backends(tmp_path: Path):
    pool, scenarios, topology_path = synthetic_pool(tmp_path)
    dense_static, dense_weights = build_static_similarity(
        pool, scenarios, topology_path, tmp_path / "static_similarity.npz"
    )
    dense_abstract = build_abstract_topology_similarity(
        pool, topology_path, tmp_path / "abstract_topology_similarity.npz"
    )
    sparse_static, sparse_weights = build_static_similarity_provider(
        pool, scenarios, topology_path, tmp_path / "state/static_provider"
    )
    sparse_abstract = build_abstract_topology_similarity_provider(
        pool, topology_path, tmp_path / "state/abstract_provider"
    )
    assert sparse_weights == pytest.approx(dense_weights)
    assert "llm_topology" in sparse_weights
    return Namespace(
        pool=pool,
        dense_static=dense_static,
        dense_abstract=dense_abstract,
        sparse_static=sparse_static,
        sparse_abstract=sparse_abstract,
        state_dir=tmp_path / "state/neighbors",
    )


def predictor_args() -> Namespace:
    return Namespace(
        half_life_steps=8.0,
        neighbors=12,
        within_env_neighbors=6,
        graph_shrinkage=2.0,
        within_env_graph_weight=0.5,
        own_history_shrinkage=2.0,
    )


def test_sparse_providers_match_dense_similarity(backends) -> None:
    indices = np.arange(len(backends.pool))
    off_diagonal = ~np.eye(len(indices), dtype=bool)
    for dense, sparse in (
        (backends.dense_static, backends.sparse_static),
        (backends.dense_abstract, backends.sparse_abstract),
    ):
        np.testing.assert_allclose(
            similarity_submatrix(sparse, indices, indices)[off_diagonal],
            dense[off_diagonal],
            rtol=0.0,
            atol=1e-6,
        )


def test_incremental_state_matches_dense_across_steps(backends) -> None:
    pool = backends.pool
    history = synthetic_history(pool, steps=4)
    args = predictor_args()
    task_ids = pool["task_id"].tolist()
    actions = []
    for target_step in range(1, 5):
        observed = history[history["step"] < target_step]
        # Reload from disk each step, as the online selector does.
        state = IncrementalGraphState(backends.state_dir, task_ids)
        dense = predict_tasks(
            observed,
            pool["task_index"].to_numpy(),
            target_step,
            pool,
            backends.dense_static,
            backends.dense_abstract,
            args,
        )
        incremental = predict_tasks(
            observed,
            pool["task_index"].to_numpy(),
            target_step,
            pool,
            backends.sparse_static,
            backends.sparse_abstract,
            args,
            state,
            "propagation",
        )
        pd.testing.assert_frame_equal(
            incremental, dense, check_exact=False, rtol=0.0, atol=1e-6
        )

        aggregates = aggregate_history(observed, pool, target_step, 8.0)
        for target in range(len(pool)):
            for scope, k, dense_similarity, plan in (
                ("cross", 12, backends.dense_abstract, "propagation.cross"),
                ("within", 6, backends.dense_static, "propagation.within"),
            ):
                dense_neighbors, _ = graph_neighbor_weights(
                    target, dense_similarity, aggregates, pool, k, scope
                )
                sparse_neighbors, _ = state.prepared_neighbors(target, plan)[:2]
                assert sorted(sparse_neighbors.tolist()) == sorted(
                    dense_neighbors.tolist()
                )

        dense_progress, _ = estimate_task_progress(
            observed, pool, backends.dense_static, target_step,
            minimum_training_transitions=4,
        )
        incremental_progress, _ = estimate_task_progress(
            observed, pool, backends.sparse_static, target_step,
            minimum_training_transitions=4,
            graph_state=state,
            graph_plan_name="progress.all",
            history_max_step=target_step - 1,
        )
        pd.testing.assert_frame_equal(
            incremental_progress, dense_progress,
            check_exact=False, rtol=0.0, atol=1e-9,
        )
        state.save()
        actions.append(state.audit()["plans"]["propagation.cross"]["action"])

    assert actions[0] == "rebuild"
    assert set(actions[1:]) == {"incremental_update"}


def test_batches_match_dense(backends) -> None:
    pool = backends.pool
    history = synthetic_history(pool, steps=3)
    predictions = predict_tasks(
        history,
        pool["task_index"].to_numpy(),
        3,
        pool,
        backends.dense_static,
        backends.dense_abstract,
        predictor_args(),
    ).merge(pool[["task_index", "env_id"]], on="task_index")
    rng = np.random.default_rng(2)
    predictions["selection_count_before"] = rng.integers(0, 4, len(pool))
    predictions["recent_selection_count_before"] = 0
    predictions["consecutive_selection_count"] = rng.integers(0, 3, len(pool))
    predictions["retire_candidate"] = rng.random(len(pool)) < 0.1
    predictions["online_id_hack_blocked"] = False
    args = Namespace(
        batch_size=16,
        least_observed_per_batch=8,
        max_per_environment=4,
        diversity_penalty=0.08,
        exact_task_exposure_scale=16.0,
        max_consecutive_selections=2,
        environment_window_steps=0,
        max_per_environment_window=0,
    )

    assert select_diverse_tasks(
        predictions, backends.sparse_static, "rank_ensemble", args
    ) == select_diverse_tasks(
        predictions, backends.dense_static, "rank_ensemble", args
    )
    assert select_cold_start_tasks(
        pool, backends.sparse_static, 12, 2
    ) == select_cold_start_tasks(pool, backends.dense_static, 12, 2)
