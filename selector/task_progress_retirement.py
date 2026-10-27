"""Causal task-progress estimates for online curriculum retirement.

The estimator separates ordinary reward movement from task-specific progress.
Each completed repeat transition is scored by a baseline trained only on
transitions that had completed before that transition's decision step. Recent
residual progress is then pooled by task with weak environment/graph shrinkage.
A task with enough own transitions retires once its posterior mean progress is
non-positive.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from incremental_similarity_graph import (
    IncrementalGraphState,
    similarity_submatrix,
)

NUMERIC_FEATURES = (
    "current_mean",
    "current_mean_sq",
    "current_variance",
    "origin_step_scaled",
    "history_count_log",
)


def _baseline_model() -> Any:
    transformer = ColumnTransformer(
        [
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "environment",
                OneHotEncoder(handle_unknown="ignore"),
                ["env_id"],
            ),
        ]
    )
    return make_pipeline(transformer, Ridge(alpha=10.0, solver="lsqr"))


def build_reward_transitions(observations: pd.DataFrame) -> pd.DataFrame:
    """Build adjacent observations without introducing future task outcomes."""
    required = {
        "task_index",
        "task_id",
        "env_id",
        "step",
        "mean_reward",
        "reward_variance",
    }
    missing = required - set(observations.columns)
    if missing:
        raise KeyError(f"Missing observation columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    ordered = observations.sort_values(["task_index", "step"])
    for _, group in ordered.groupby("task_index", sort=False):
        records = list(group.itertuples(index=False))
        for position, (before, after) in enumerate(zip(records, records[1:])):
            rows.append(
                {
                    "task_index": int(before.task_index),
                    "task_id": str(before.task_id),
                    "env_id": str(before.env_id),
                    "origin_step": int(before.step),
                    "after_step": int(after.step),
                    "step_gap": int(after.step - before.step),
                    "history_count": position + 1,
                    "current_mean": float(before.mean_reward),
                    "current_variance": float(before.reward_variance),
                    "after_mean": float(after.mean_reward),
                    "gain": float(after.mean_reward - before.mean_reward),
                }
            )
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows).sort_values(
        ["origin_step", "task_index", "after_step"]
    )
    frame["current_mean_sq"] = frame["current_mean"] ** 2
    frame["origin_step_scaled"] = frame["origin_step"] / max(
        float(frame["origin_step"].max()), 1.0
    )
    frame["history_count_log"] = np.log1p(frame["history_count"])
    return frame.reset_index(drop=True)


def causal_transition_residuals(
    transitions: pd.DataFrame,
    minimum_training_transitions: int,
) -> pd.DataFrame:
    """Score transitions with models using only previously completed outcomes."""
    if transitions.empty:
        return pd.DataFrame()
    predictions: list[pd.DataFrame] = []
    for decision_step in sorted(transitions["origin_step"].unique()):
        train = transitions[transitions["after_step"] < decision_step]
        target = transitions[transitions["origin_step"] == decision_step]
        if len(train) < minimum_training_transitions or target.empty:
            continue
        model = _baseline_model()
        model.fit(train, train["after_mean"].to_numpy(dtype=float))
        scored = target.copy()
        scored["baseline_prediction"] = np.clip(model.predict(target), 0.0, 1.0)
        scored["residual"] = scored["after_mean"] - scored["baseline_prediction"]
        predictions.append(scored)
    if not predictions:
        return pd.DataFrame()
    return pd.concat(predictions, ignore_index=True).sort_values(
        ["origin_step", "task_index", "after_step"]
    )


def _weighted_group_stats(
    history: pd.DataFrame,
    key: str | list[str],
    target_step: int,
    half_life_steps: float,
) -> pd.DataFrame:
    frame = history.copy()
    frame["weight"] = np.exp2(
        -(target_step - frame["after_step"].to_numpy(dtype=float))
        / half_life_steps
    )
    frame["weighted_residual"] = frame["weight"] * frame["residual"]
    grouped = frame.groupby(key, as_index=True).agg(
        residual_sum=("weighted_residual", "sum"),
        residual_weight=("weight", "sum"),
        residual_count=("residual", "size"),
        residual_last_step=("after_step", "max"),
    )
    grouped["residual_mean"] = (
        grouped["residual_sum"] / grouped["residual_weight"]
    )
    return grouped


def _environment_prior_mean(env_id: str, env_stats: pd.DataFrame) -> float:
    if env_id not in env_stats.index:
        return 0.0
    return float(env_stats.loc[env_id, "residual_mean"])


def _graph_prior(
    task_index: int,
    task_stats: pd.DataFrame,
    similarity: Any,
    neighbors: int,
    target_step: int,
    half_life_steps: float,
    graph_state: IncrementalGraphState | None = None,
    graph_plan_name: str = "progress.all",
) -> tuple[float, float, int]:
    if graph_state is not None:
        selected, selected_scores, _ = graph_state.prepared_neighbors(
            task_index, graph_plan_name
        )
        count = len(selected)
    else:
        candidates = task_stats.index.to_numpy(dtype=int)
        candidates = candidates[candidates != task_index]
        if not len(candidates):
            return 0.0, 0.0, 0
        scores = similarity_submatrix(
            similarity,
            np.asarray([task_index], dtype=int),
            candidates,
        )[0]
        positive = scores > 0
        candidates = candidates[positive]
        scores = scores[positive]
        if not len(candidates):
            return 0.0, 0.0, 0
        count = min(neighbors, len(candidates))
        positions = np.argpartition(-scores, count - 1)[:count]
        selected = candidates[positions]
        selected_scores = scores[positions]
    stats = task_stats.loc[selected]
    recency = np.exp2(
        -(target_step - stats["residual_last_step"].to_numpy(dtype=float))
        / half_life_steps
    )
    weights = (
        np.square(selected_scores)
        * np.sqrt(np.clip(stats["residual_weight"].to_numpy(dtype=float), 0.1, 4.0))
        * recency
    )
    total = float(weights.sum())
    if total <= 0:
        return 0.0, 0.0, 0
    value = float(np.dot(weights, stats["residual_mean"].to_numpy()) / total)
    return value, total, count


def empty_progress_frame(pool: pd.DataFrame) -> pd.DataFrame:
    frame = pool[["task_index", "task_id"]].copy()
    frame["progress_transition_count"] = 0
    frame["progress_effective_count"] = 0.0
    frame["progress_posterior"] = 0.0
    frame["progress_posterior_se"] = 0.0
    frame["progress_probability_positive"] = 0.5
    frame["retire_candidate"] = False
    return frame


def estimate_task_progress(
    observations: pd.DataFrame,
    pool: pd.DataFrame,
    similarity: Any,
    target_step: int,
    *,
    minimum_training_transitions: int = 96,
    half_life_steps: float = 10.0,
    neighbors: int = 12,
    task_shrinkage: float = 3.0,
    environment_prior_weight: float = 0.5,
    minimum_own_transitions: int = 3,
    graph_state: IncrementalGraphState | None = None,
    graph_plan_name: str = "progress.all",
    history_max_step: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Estimate progress for every task and conservatively flag plateaus."""
    if half_life_steps <= 0:
        raise ValueError("half_life_steps must be positive")
    if task_shrinkage <= 0:
        raise ValueError("task_shrinkage must be positive")
    if not 0.0 <= environment_prior_weight <= 1.0:
        raise ValueError("environment_prior_weight must be between 0 and 1")
    if minimum_own_transitions < 1:
        raise ValueError("minimum_own_transitions must be positive")

    transitions = build_reward_transitions(observations)
    residuals = causal_transition_residuals(
        transitions, minimum_training_transitions
    )
    if residuals.empty:
        return empty_progress_frame(pool), {
            "transition_count": int(len(transitions)),
            "causal_residual_count": 0,
            "eligible_task_count": 0,
            "retired_task_count": 0,
            "retired_task_ids": [],
            "environment_prior_weight": environment_prior_weight,
            "minimum_own_transitions": minimum_own_transitions,
            "uses_only_completed_transitions_before_target": True,
        }
    history = residuals[residuals["after_step"] < target_step].copy()
    if history.empty:
        return empty_progress_frame(pool), {
            "transition_count": int(len(transitions)),
            "causal_residual_count": 0,
            "eligible_task_count": 0,
            "retired_task_count": 0,
            "retired_task_ids": [],
            "environment_prior_weight": environment_prior_weight,
            "minimum_own_transitions": minimum_own_transitions,
            "uses_only_completed_transitions_before_target": True,
        }

    task_stats = _weighted_group_stats(
        history, "task_index", target_step, half_life_steps
    )
    if graph_state is not None:
        active_mask = np.zeros(len(pool), dtype=bool)
        active_mask[task_stats.index.to_numpy(dtype=int)] = True
        graph_state.prepare_plan(
            similarity,
            active_mask,
            neighbors,
            "all",
            pool["env_id"].astype(str).to_numpy(),
            graph_plan_name,
            history_max_step=history_max_step,
            positive_only=True,
        )
    env_stats = _weighted_group_stats(history, "env_id", target_step, half_life_steps)
    residual_scale = max(float(history["residual"].std(ddof=1)), 1e-3)
    rows: list[dict[str, Any]] = []
    for row in pool.itertuples(index=False):
        task_index = int(row.task_index)
        env_id = str(row.env_id)
        env_mean = _environment_prior_mean(env_id, env_stats)
        graph_mean, _, _ = _graph_prior(
            task_index,
            task_stats,
            similarity,
            neighbors,
            target_step,
            half_life_steps,
            graph_state,
            graph_plan_name,
        )
        prior = (
            environment_prior_weight * env_mean
            + (1.0 - environment_prior_weight) * graph_mean
        )
        if task_index in task_stats.index:
            own_sum = float(task_stats.loc[task_index, "residual_sum"])
            own_weight = float(task_stats.loc[task_index, "residual_weight"])
            own_count = int(task_stats.loc[task_index, "residual_count"])
        else:
            own_sum = 0.0
            own_weight = 0.0
            own_count = 0
        posterior_weight = own_weight + task_shrinkage
        posterior = (own_sum + task_shrinkage * prior) / posterior_weight
        posterior_se = residual_scale / math.sqrt(posterior_weight)
        z_value = posterior / posterior_se
        probability_positive = float(norm.cdf(z_value))
        retired = own_count >= minimum_own_transitions and posterior <= 0.0
        rows.append(
            {
                "task_index": task_index,
                "task_id": str(row.task_id),
                "progress_transition_count": own_count,
                "progress_effective_count": own_weight,
                "progress_posterior": posterior,
                "progress_posterior_se": posterior_se,
                "progress_probability_positive": probability_positive,
                "retire_candidate": bool(retired),
            }
        )
    result = pd.DataFrame(rows)
    retired_ids = result.loc[result["retire_candidate"], "task_id"].tolist()
    audit = {
        "transition_count": int(len(transitions)),
        "causal_residual_count": int(len(history)),
        "residual_scale": residual_scale,
        "eligible_task_count": int(
            (result["progress_transition_count"] >= minimum_own_transitions).sum()
        ),
        "retired_task_count": len(retired_ids),
        "retired_task_ids": retired_ids,
        "minimum_training_transitions": minimum_training_transitions,
        "half_life_steps": half_life_steps,
        "neighbors": neighbors,
        "task_shrinkage": task_shrinkage,
        "environment_prior_weight": environment_prior_weight,
        "minimum_own_transitions": minimum_own_transitions,
        "uses_only_completed_transitions_before_target": True,
    }
    return result, audit
