from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from select_online_high_variance import (
    abstract_topology_text,
    graph_neighbor_weights,
    predict_tasks,
    select_diverse_tasks,
)


def with_selection_defaults(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add the history and guard columns that main() always provides."""
    defaults = {
        "selection_count_before": 0,
        "recent_selection_count_before": 0,
        "consecutive_selection_count": 0,
        "retire_candidate": False,
        "online_id_hack_blocked": False,
    }
    for column, value in defaults.items():
        if column not in predictions:
            predictions[column] = value
    return predictions


def test_abstract_topology_drops_domain_identity() -> None:
    topology = {
        "task_id": "env_141_rl-task_1",
        "operations": [
            {
                "family": "update",
                "description": "Update an insurance policy deductible.",
                "tool_candidates": ["update_policy_deductible"],
                "depends_on_operations": ["o0"],
                "gate_ids": ["g0", "g1"],
            }
        ],
        "gates": [
            {
                "type": "argument_exactness",
                "description": "The deductible field must be exact.",
                "provenance": "task_literal",
                "reachability": "direct",
                "observability": "explicit_result",
                "recoverability": "retryable",
            }
        ],
        "reward_groups": [{"relation": "shared_operation"}],
        "conflicts": [{"kind": "order"}],
        "independent_branches": [{"id": "b0"}],
    }

    text = abstract_topology_text(topology)

    assert "operation_update" in text
    assert "gate_argument_exactness" in text
    assert "reward_shared_operation" in text
    assert "update_policy_deductible" not in text
    assert "insurance" not in text
    assert "deductible" not in text
    assert "env_141_rl" not in text


def test_graph_neighbors_enforce_environment_scope() -> None:
    pool = pd.DataFrame(
        {
            "env_id": ["env_a", "env_a", "env_b", "env_c"],
        }
    )
    similarity = np.asarray(
        [
            [1.0, 0.7, 0.9, 0.8],
            [0.7, 1.0, 0.6, 0.5],
            [0.9, 0.6, 1.0, 0.4],
            [0.8, 0.5, 0.4, 1.0],
        ],
        dtype=np.float32,
    )
    aggregates = {"task_weight": np.ones(4, dtype=float)}

    local, _ = graph_neighbor_weights(
        0, similarity, aggregates, pool, 3, "within"
    )
    cross, _ = graph_neighbor_weights(
        0, similarity, aggregates, pool, 3, "cross"
    )

    assert local.tolist() == [1]
    assert set(cross.tolist()) == {2, 3}


def test_candidate_pool_expands_without_relaxing_environment_quota() -> None:
    task_count = 30
    environment = (
        ["env_a"] * 6
        + ["env_b"] * 6
        + ["env_c"] * 6
        + ["env_d"] * 4
        + ["env_e"] * 4
        + ["env_f"] * 4
    )
    predictions = with_selection_defaults(pd.DataFrame(
        {
            "task_index": np.arange(task_count),
            "env_id": environment,
            "rank_ensemble": np.linspace(1.0, 0.0, task_count),
            "history_group_count": np.zeros(task_count, dtype=int),
            "consecutive_selection_count": np.zeros(task_count, dtype=int),
        }
    ))
    similarity = np.eye(task_count, dtype=np.float32)
    args = Namespace(
        batch_size=8,
        exact_task_exposure_scale=16.0,
        max_per_environment=2,
        diversity_penalty=0.08,
        least_observed_per_batch=0,
        max_consecutive_selections=0,
        environment_window_steps=0,
        max_per_environment_window=0,
    )

    selected = select_diverse_tasks(
        predictions, similarity, "rank_ensemble", args
    )
    selected_environments = predictions.iloc[selected]["env_id"].value_counts()

    assert len(selected) == 8
    assert selected_environments.max() <= 2
    assert len(selected_environments) >= 4


def test_coverage_phase_visits_environments_before_reuse() -> None:
    predictions = with_selection_defaults(pd.DataFrame(
        {
            "task_index": np.arange(6),
            "env_id": ["env_a", "env_a", "env_b", "env_b", "env_c", "env_c"],
            "rank_ensemble": np.linspace(1.0, 0.0, 6),
            "history_group_count": [0, 1, 1, 2, 3, 4],
            "consecutive_selection_count": np.zeros(6, dtype=int),
            "selection_count_before": [0, 1, 1, 2, 3, 4],
        }
    ))
    args = Namespace(
        batch_size=4,
        exact_task_exposure_scale=16.0,
        max_per_environment=2,
        diversity_penalty=0.08,
        least_observed_per_batch=3,
        max_consecutive_selections=0,
        environment_window_steps=0,
        max_per_environment_window=0,
    )

    selected = select_diverse_tasks(
        predictions, np.eye(6, dtype=np.float32), "rank_ensemble", args
    )

    assert len(selected) == 4
    assert predictions.iloc[selected[:3]]["env_id"].nunique() == 3
    assert predictions.iloc[selected[:3]]["history_group_count"].tolist() == [0, 1, 3]


def test_coverage_phase_prioritizes_least_exposed_environments() -> None:
    predictions = with_selection_defaults(pd.DataFrame(
        {
            "task_index": np.arange(8),
            "env_id": np.repeat(["env_a", "env_b", "env_c", "env_d"], 2),
            "rank_ensemble": [1.0, 0.99, 0.8, 0.79, 0.7, 0.69, 0.6, 0.59],
            "history_group_count": np.zeros(8, dtype=int),
            "selection_count_before": [5, 5, 0, 0, 1, 1, 2, 2],
            "consecutive_selection_count": np.zeros(8, dtype=int),
        }
    ))
    args = Namespace(
        batch_size=3,
        exact_task_exposure_scale=16.0,
        max_per_environment=2,
        diversity_penalty=0.08,
        least_observed_per_batch=3,
        max_consecutive_selections=0,
        environment_window_steps=0,
        max_per_environment_window=0,
    )

    selected = select_diverse_tasks(
        predictions, np.eye(8, dtype=np.float32), "rank_ensemble", args
    )

    assert predictions.iloc[selected]["env_id"].tolist() == ["env_b", "env_c", "env_d"]


def test_rolling_environment_quota_limits_current_step_capacity() -> None:
    predictions = with_selection_defaults(pd.DataFrame(
        {
            "task_index": np.arange(12),
            "env_id": np.repeat(["env_a", "env_b", "env_c"], 4),
            "rank_ensemble": np.linspace(1.0, 0.0, 12),
            "history_group_count": np.zeros(12, dtype=int),
            "recent_selection_count_before": [1, 1, 1, 1] + [0] * 8,
            "consecutive_selection_count": np.zeros(12, dtype=int),
        }
    ))
    args = Namespace(
        batch_size=4,
        exact_task_exposure_scale=16.0,
        max_per_environment=4,
        diversity_penalty=0.08,
        least_observed_per_batch=0,
        max_consecutive_selections=0,
        environment_window_steps=3,
        max_per_environment_window=5,
    )

    selected = select_diverse_tasks(
        predictions, np.eye(12, dtype=np.float32), "rank_ensemble", args
    )
    selected_environments = predictions.iloc[selected]["env_id"].value_counts()

    assert len(selected) == 4
    assert selected_environments["env_a"] == 1


def test_full_pool_shortfall_still_raises() -> None:
    predictions = with_selection_defaults(pd.DataFrame(
        {
            "task_index": np.arange(4),
            "env_id": ["env_a", "env_a", "env_b", "env_b"],
            "rank_ensemble": np.linspace(1.0, 0.0, 4),
            "history_group_count": np.ones(4, dtype=int),
            "consecutive_selection_count": np.ones(4, dtype=int),
        }
    ))
    args = Namespace(
        batch_size=2,
        exact_task_exposure_scale=16.0,
        max_per_environment=2,
        diversity_penalty=0.08,
        least_observed_per_batch=2,
        max_consecutive_selections=1,
        environment_window_steps=0,
        max_per_environment_window=0,
    )

    with pytest.raises(RuntimeError, match="Could not fill batch under quotas/cooldown"):
        select_diverse_tasks(
            predictions, np.eye(4, dtype=np.float32), "rank_ensemble", args
        )


def test_progress_retired_tasks_are_excluded_from_exploitation() -> None:
    predictions = with_selection_defaults(pd.DataFrame(
        {
            "task_index": np.arange(6),
            "env_id": [f"env_{index}" for index in range(6)],
            "rank_ensemble": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
            "history_group_count": np.ones(6, dtype=int),
            "consecutive_selection_count": np.zeros(6, dtype=int),
            "retire_candidate": [True, False, True, False, False, False],
        }
    ))
    args = Namespace(
        batch_size=3,
        exact_task_exposure_scale=16.0,
        max_per_environment=1,
        diversity_penalty=0.08,
        least_observed_per_batch=0,
        max_consecutive_selections=0,
        environment_window_steps=0,
        max_per_environment_window=0,
    )

    selected = select_diverse_tasks(
        predictions, np.eye(6, dtype=np.float32), "rank_ensemble", args
    )

    assert selected == [1, 3, 4]
    assert not predictions.iloc[selected]["retire_candidate"].any()


def predictor_args(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "half_life_steps": 8.0,
        "neighbors": 3,
        "within_env_neighbors": 3,
        "graph_shrinkage": 0.1,
        "within_env_graph_weight": 0.5,
        "own_history_shrinkage": 2.0,
    }
    values.update(overrides)
    return Namespace(**values)


def predictor_fixture() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    pool = pd.DataFrame(
        {
            "task_index": np.arange(4),
            "env_id": ["env_a", "env_a", "env_a", "env_b"],
        }
    )
    # Task 1 is cold. Task 2 is its high-variance same-environment sibling;
    # task 3 is the lower-variance cross-environment observation.
    history = pd.DataFrame(
        {
            "task_index": [0, 2, 3],
            "step": [0, 0, 0],
            "reward_variance": [0.02, 0.18, 0.05],
            "mean_reward": [0.70, 0.45, 0.65],
            "dispersion": [0.10, 0.40, 0.15],
        }
    )
    within = np.eye(4, dtype=np.float32)
    within[0, 2] = within[2, 0] = 0.8
    within[1, 2] = within[2, 1] = 0.9
    cross = np.eye(4, dtype=np.float32)
    cross[:3, 3] = 0.9
    cross[3, :3] = 0.9
    return history, pool, within, cross


def test_cold_task_borrows_variance_from_graph_neighbors() -> None:
    history, pool, within, cross = predictor_fixture()

    def cold_prediction(within_env_graph_weight: float) -> float:
        prediction = predict_tasks(
            history,
            np.asarray([1]),
            1,
            pool,
            within,
            cross,
            predictor_args(within_env_graph_weight=within_env_graph_weight),
        ).set_index("task_index")
        assert prediction.loc[1, "history_group_count"] == 0
        return float(prediction.loc[1, "history_direct"])

    # Task 1's within-environment sibling is high variance, while its only
    # cross-environment neighbor is low variance.
    assert cold_prediction(1.0) > cold_prediction(0.0)
