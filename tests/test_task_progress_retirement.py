import numpy as np
import pandas as pd
import pytest

import task_progress_retirement
from task_progress_retirement import estimate_task_progress


def singleton_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    observations = pd.DataFrame(
        [
            {
                "task_index": 0,
                "task_id": "task-0",
                "env_id": "singleton-0",
                "step": 0,
                "mean_reward": 0.1,
                "reward_variance": 0.01,
            },
            {
                "task_index": 0,
                "task_id": "task-0",
                "env_id": "singleton-0",
                "step": 1,
                "mean_reward": 0.2,
                "reward_variance": 0.02,
            },
        ]
    )
    pool = observations.iloc[[0]][["task_index", "task_id", "env_id"]]
    return observations, pool


def use_residuals(monkeypatch: pytest.MonkeyPatch, values: list[float]) -> None:
    residuals = pd.DataFrame(
        [
            {
                "task_index": 0,
                "env_id": "singleton-0",
                "after_step": step,
                "residual": value,
            }
            for step, value in enumerate(values, start=1)
        ]
    )
    monkeypatch.setattr(
        task_progress_retirement,
        "causal_transition_residuals",
        lambda transitions, minimum_training_transitions: residuals,
    )


def test_singleton_environment_prior_equals_weighted_residual_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations, pool = singleton_fixture()
    use_residuals(monkeypatch, [0.2, 0.4])

    progress, audit = estimate_task_progress(
        observations,
        pool,
        np.zeros((1, 1)),
        3,
        environment_prior_weight=1.0,
        task_shrinkage=3.0,
        minimum_own_transitions=1,
    )

    weights = np.exp2(-np.array([2.0, 1.0]) / 10.0)
    expected = float(np.dot(weights, [0.2, 0.4]) / weights.sum())
    assert progress.loc[0, "progress_posterior"] == pytest.approx(expected)
    assert not progress.loc[0, "retire_candidate"]
    assert audit["retired_task_ids"] == []


def test_task_retires_once_posterior_progress_is_non_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations, pool = singleton_fixture()
    use_residuals(monkeypatch, [-0.2, -0.1])

    def estimate(minimum_own_transitions: int) -> tuple[pd.DataFrame, dict]:
        return estimate_task_progress(
            observations,
            pool,
            np.zeros((1, 1)),
            3,
            environment_prior_weight=1.0,
            minimum_own_transitions=minimum_own_transitions,
        )

    progress, audit = estimate(minimum_own_transitions=2)
    assert progress.loc[0, "progress_posterior"] < 0.0
    assert progress.loc[0, "retire_candidate"]
    assert audit["retired_task_ids"] == ["task-0"]

    # A task with fewer own transitions than required is never retired.
    progress, audit = estimate(minimum_own_transitions=3)
    assert not progress.loc[0, "retire_candidate"]
    assert audit["retired_task_ids"] == []
