from __future__ import annotations

import json
import statistics
from pathlib import Path

import pandas as pd
import pytest

from select_online_high_variance import (
    load_observation_frame,
    verifier_conditional_scalar_rewards,
)


TASK_ID = "task-0"
TASK_TEXT = "Update the linked records."


def pool_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "task_index": [0],
            "task_id": [TASK_ID],
            "env_id": ["env-0"],
            "task": [TASK_TEXT],
        }
    )


def write_group(
    case_dir: Path,
    step: int,
    scores: list[float],
    reached: list[bool] | None,
) -> None:
    rows = []
    for index, score in enumerate(scores):
        row = {
            "traj_id": f"traj-{step}-{index}",
            "traj_group_id": f"group-{step}",
            "task_id": TASK_ID,
            "prompt": f"<|im_start|>user\n{TASK_TEXT}<|im_end|>",
            "episode_score": score,
        }
        if reached is not None:
            row["verifier_reached"] = reached[index]
        rows.append(row)
    (case_dir / f"rollout_cases.step_{step}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_verifier_conditional_target_keeps_scalar_mean_and_shrinks_variance(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    case_dir = run_dir / "output/cases"
    case_dir.mkdir(parents=True)
    scores = [0.2, 0.8, 0.0, 0.0]
    write_group(case_dir, 0, scores, [True, True, False, False])

    frame, max_step = load_observation_frame(
        str(run_dir),
        pool_fixture(),
        0,
    )

    assert max_step == 0
    assert frame.loc[0, "mean_reward"] == pytest.approx(0.25)
    assert frame.loc[0, "reward_variance"] == pytest.approx(0.045)
    assert frame.loc[0, "dispersion"] == pytest.approx(0.24)
    assert frame.loc[0, "rollout_count"] == 4
    audit = frame.attrs["variance_label_audit"]
    assert audit["variance_target"] == "verifier_conditional_scalar_reward"
    assert audit["label_trajectory_count"] == 4
    assert audit["verifier_trajectory_count"] == 2
    assert audit["non_verifier_trajectory_count"] == 2
    assert audit["group_count"] == 1
    assert audit["group_size_histogram"] == {"4": 1}
    assert audit["verifier_count_histogram"] == {"2": 1}
    assert audit["insufficient_verifier_group_count"] == 0
    assert audit["raw_scalar_variance_sum"] == pytest.approx(0.1075)
    assert audit["target_variance_sum"] == pytest.approx(0.045)
    assert audit["check_error_policy"] == "not_consumed"


@pytest.mark.parametrize(
    "reached, expected",
    [
        ([True, True, True, True], 0.1075),
        ([True, False, False, False], 0.0),
        ([False, False, False, False], 0.0),
    ],
)
def test_verifier_conditional_variance_boundaries(
    reached: list[bool],
    expected: float,
) -> None:
    scores = [0.2, 0.8, 0.0, 0.0]
    rows = [
        {"episode_score": score, "verifier_reached": flag}
        for score, flag in zip(scores, reached)
    ]

    variance_rewards = verifier_conditional_scalar_rewards(
        rows,
        context="unit-test",
    )

    assert len(variance_rewards) == len(scores)
    assert statistics.pvariance(variance_rewards) == pytest.approx(expected)


@pytest.mark.parametrize("bad_value", [None, 1, "true"])
def test_verifier_conditional_target_requires_strict_boolean_reachability(
    bad_value: object,
) -> None:
    with pytest.raises(ValueError, match="verifier_reached must be a bool"):
        verifier_conditional_scalar_rewards(
            [{"episode_score": 0.5, "verifier_reached": bad_value}],
            context="strict-test",
        )


def test_verifier_conditional_target_can_skip_pre_schema_history(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    case_dir = run_dir / "output/cases"
    case_dir.mkdir(parents=True)
    write_group(case_dir, 0, [0.5], None)
    write_group(case_dir, 1, [0.2, 0.8], [True, True])

    frame, max_step = load_observation_frame(
        str(run_dir),
        pool_fixture(),
        1,
        history_min_step=1,
    )

    assert max_step == 1
    assert frame["step"].tolist() == [1]
    assert frame.loc[0, "reward_variance"] == pytest.approx(0.09)
