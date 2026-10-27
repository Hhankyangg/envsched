from argparse import Namespace
import json

import numpy as np
import pandas as pd
import pytest

from select_online_high_variance import (
    exact_task_exposure_weights,
    load_selection_history,
    select_diverse_tasks,
)


def selection_args(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "batch_size": 2,
        "exact_task_exposure_scale": 16.0,
        "max_per_environment": 4,
        "diversity_penalty": 0.0,
        "least_observed_per_batch": 1,
        "max_consecutive_selections": 0,
        "environment_window_steps": 0,
        "max_per_environment_window": 0,
    }
    values.update(overrides)
    return Namespace(**values)


def exposure_fixture() -> tuple[pd.DataFrame, np.ndarray]:
    predictions = pd.DataFrame(
        {
            "task_index": np.arange(4),
            "env_id": ["env_a"] * 4,
            "rank_ensemble": [1.0, 0.9, 0.8, 0.7],
            "history_group_count": [0, 1, 1, 1],
            # Task 1 outranks task 2, but after 32 selections its exposure
            # weight 1/sqrt(3) drops it below the never-selected task 2.
            "selection_count_before": [64, 32, 0, 0],
            "recent_selection_count_before": np.zeros(4, dtype=int),
            "consecutive_selection_count": np.zeros(4, dtype=int),
            "retire_candidate": np.zeros(4, dtype=bool),
            "online_id_hack_blocked": np.zeros(4, dtype=bool),
        }
    )
    return predictions, np.eye(4, dtype=np.float32)


def test_exact_task_exposure_inverse_sqrt_formula() -> None:
    weights = exact_task_exposure_weights(
        np.asarray([0, 16, 48], dtype=float),
        scale=16.0,
    )

    np.testing.assert_allclose(weights, [1.0, 1.0 / np.sqrt(2.0), 0.5])


@pytest.mark.parametrize(
    ("counts", "scale", "message"),
    [
        (np.asarray([-1.0]), 8.0, "selection counts"),
        (np.asarray([0.0]), 0.0, "scale"),
    ],
)
def test_exact_task_exposure_rejects_invalid_inputs(
    counts: np.ndarray,
    scale: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        exact_task_exposure_weights(counts, scale)


def test_exact_task_exposure_changes_exploitation_not_coverage() -> None:
    predictions, similarity = exposure_fixture()

    selected = select_diverse_tasks(
        predictions,
        similarity,
        "rank_ensemble",
        selection_args(),
    )

    assert selected == [0, 2]


@pytest.mark.parametrize(
    ("coverage_slots", "expected_position"),
    [(0, 2), (1, 1)],
)
def test_inverse_sqrt_online_guard_blocks_both_selection_phases(
    coverage_slots: int,
    expected_position: int,
) -> None:
    predictions, similarity = exposure_fixture()
    predictions["online_id_hack_blocked"] = [True, False, False, False]

    selected = select_diverse_tasks(
        predictions,
        similarity,
        "rank_ensemble",
        selection_args(
            batch_size=1,
            least_observed_per_batch=coverage_slots,
        ),
    )

    assert selected == [expected_position]


def test_inverse_sqrt_fails_closed_without_online_guard_column() -> None:
    predictions, similarity = exposure_fixture()
    predictions = predictions.drop(columns="online_id_hack_blocked")

    with pytest.raises(ValueError, match="online rewarded-ID-hack guard"):
        select_diverse_tasks(
            predictions,
            similarity,
            "rank_ensemble",
            selection_args(),
        )


def test_online_guard_capacity_check_counts_only_eligible_tasks() -> None:
    predictions, similarity = exposure_fixture()
    predictions["online_id_hack_blocked"] = [True, True, True, False]

    with pytest.raises(RuntimeError, match="capacity=1, requested=2"):
        select_diverse_tasks(
            predictions,
            similarity,
            "rank_ensemble",
            selection_args(),
        )


def test_selection_history_updates_lifetime_counts_recursively(tmp_path) -> None:
    for step, task_ids in enumerate((["a", "b"], ["b", "c"])):
        (tmp_path / f"task_manifest.step_{step}.json").write_text(
            json.dumps({"target_policy_step": step, "task_ids": task_ids}),
            encoding="utf-8",
        )

    history = load_selection_history(
        tmp_path,
        target_step=2,
        task_ids=["a", "b", "c"],
        recent_step_count=2,
    ).set_index("task_id")

    assert history["selection_count_before"].to_dict() == {
        "a": 1,
        "b": 2,
        "c": 1,
    }
    assert history["consecutive_selection_count"].to_dict() == {
        "a": 0,
        "b": 2,
        "c": 1,
    }


def test_selection_history_counts_duplicate_groups_within_step(tmp_path) -> None:
    for step, task_ids in enumerate((["a", "a", "b"], ["a", "c", "c"])):
        (tmp_path / f"task_manifest.step_{step}.json").write_text(
            json.dumps({"target_policy_step": step, "task_ids": task_ids}),
            encoding="utf-8",
        )

    history = load_selection_history(
        tmp_path,
        target_step=2,
        task_ids=["a", "b", "c"],
        recent_step_count=1,
    ).set_index("task_id")

    assert history["selection_count_before"].to_dict() == {
        "a": 3,
        "b": 1,
        "c": 2,
    }
    assert history["recent_selection_count_before"].to_dict() == {
        "a": 1,
        "b": 0,
        "c": 2,
    }
    assert history["consecutive_selection_count"].to_dict() == {
        "a": 2,
        "b": 0,
        "c": 1,
    }
