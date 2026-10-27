#!/usr/bin/env python3
"""Predict next-step reward variance and emit a high-variance task manifest.

The predictor is deliberately causal in time: a prediction for rollout step t
uses case logs from steps strictly smaller than t. Full static task context
defines a within-environment graph, while abstract execution/reward topology
defines a cross-environment graph. Historical rollouts provide the signals
propagated over those graphs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from incremental_similarity_graph import (
    ABSTRACT_TOPOLOGY_SCHEMA_VERSION,
    IncrementalGraphState,
    abstract_topology_text,
    build_abstract_topology_similarity_provider,
    build_static_similarity_provider,
    normalize_similarity_text,
    similarity_column,
    similarity_submatrix,
)
from online_rewarded_id_hack_guard import build_online_id_hack_guard
from reward_statistics import (
    DEFAULT_SCENARIOS,
    read_rollout_observations,
)
from task_features import state_signature, verifier_text
from task_progress_retirement import estimate_task_progress


DEFAULT_TOPOLOGIES = "data/task_topologies.jsonl"
DEFAULT_ENVIRONMENT_METADATA = (
    "vendor/ROLL/roll/pipeline/agentic/env/envscaler_env/data/"
    "environment_metadata.json"
)
DEFAULT_EXACT_TASK_EXPOSURE_SCALE = 16.0
GRAPH_EXECUTION_MODES = ("dense_legacy", "incremental_exact")
ONLINE_ID_HACK_BLOCK_COLUMN = "online_id_hack_blocked"
VERIFIER_CONDITIONAL_VARIANCE_TARGET = "verifier_conditional_scalar_reward"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    parser.add_argument("--topologies", default=DEFAULT_TOPOLOGIES)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--graph-execution-mode",
        choices=GRAPH_EXECUTION_MODES,
        default="dense_legacy",
        help=(
            "Similarity-graph execution backend. dense_legacy keeps dense N-by-N "
            "similarity caches. incremental_exact stores sparse TF-IDF features "
            "plus persistent K+1 neighbor plans and computes only newly eligible "
            "similarity columns; it selects the same tasks."
        ),
    )
    parser.add_argument(
        "--incremental-graph-state-dir",
        help=(
            "Derived incremental graph cache. Defaults to "
            "OUTPUT_DIR/incremental_graph_state."
        ),
    )
    parser.add_argument(
        "--incremental-target-chunk-size",
        type=int,
        default=2048,
        help="Target rows per exact incremental top-k update block.",
    )
    parser.add_argument(
        "--incremental-candidate-chunk-size",
        type=int,
        default=64,
        help="New candidate columns per exact incremental top-k update block.",
    )
    parser.add_argument(
        "--history-min-step",
        type=int,
        default=0,
        help=(
            "First rollout step eligible for predictor labels. Selection "
            "history is still loaded independently through target_policy_step."
        ),
    )
    parser.add_argument("--history-max-step", type=int)
    parser.add_argument(
        "--target-policy-step",
        type=int,
        help="Policy step being selected for; defaults to history_max_step + 1.",
    )
    parser.add_argument("--half-life-steps", type=float, default=8.0)
    parser.add_argument("--neighbors", type=int, default=12)
    parser.add_argument("--within-env-neighbors", type=int, default=6)
    parser.add_argument("--graph-shrinkage", type=float, default=2.0)
    parser.add_argument(
        "--within-env-graph-weight",
        type=float,
        default=0.5,
        help=(
            "Weight assigned to the within-environment graph; the remaining "
            "weight is assigned to the abstract cross-environment graph."
        ),
    )
    parser.add_argument("--own-history-shrinkage", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-per-environment", type=int, default=4)
    parser.add_argument("--diversity-penalty", type=float, default=0.08)
    parser.add_argument(
        "--exact-task-exposure-scale",
        type=float,
        default=DEFAULT_EXACT_TASK_EXPOSURE_SCALE,
        help=(
            "Selection-count scale c of the exploitation weight "
            "1/sqrt(1 + selection_count / c). The weight reaches 1/sqrt(2) "
            "after c prior selections."
        ),
    )
    parser.add_argument(
        "--environment-metadata",
        default=DEFAULT_ENVIRONMENT_METADATA,
        help="Executable environment metadata replayed by the rewarded-ID-hack guard.",
    )
    parser.add_argument("--selection-history-dir")
    parser.add_argument(
        "--least-observed-per-batch",
        type=int,
        default=0,
        help=(
            "Number of coverage slots selected environment-first, then by the "
            "smallest environment/task history before the variance/MMR tie-breaker."
        ),
    )
    parser.add_argument("--max-consecutive-selections", type=int, default=0)
    parser.add_argument(
        "--environment-window-steps",
        type=int,
        default=0,
        help=(
            "Rolling environment-quota window including the target step; 0 disables "
            "the rolling quota."
        ),
    )
    parser.add_argument(
        "--max-per-environment-window",
        type=int,
        default=0,
        help=(
            "Maximum selections from one environment across the rolling window; "
            "0 disables the rolling quota."
        ),
    )
    parser.add_argument("--recency-guard-half-life", type=float, default=0.0)
    parser.add_argument("--recency-guard-strength", type=float, default=0.0)
    parser.add_argument("--progress-min-training-transitions", type=int, default=96)
    parser.add_argument("--progress-half-life-steps", type=float, default=10.0)
    parser.add_argument("--progress-neighbors", type=int, default=12)
    parser.add_argument("--progress-task-shrinkage", type=float, default=3.0)
    parser.add_argument("--progress-environment-prior-weight", type=float, default=0.5)
    parser.add_argument("--progress-min-own-transitions", type=int, default=3)
    return parser.parse_args()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def verifier_conditional_scalar_rewards(
    items: list[dict[str, Any]],
    *,
    context: str,
) -> list[float]:
    """Return variance-only rewards with non-verifier outcomes mean-imputed.

    Population variance over the returned group equals the population variance
    among verifier-reached scores multiplied by verifier_count / group_count.
    Scalar means, progress labels, and training rewards remain unchanged.
    """
    if not items:
        raise ValueError(f"{context}: trajectory group is empty")

    scores: list[float] = []
    verifier_reached: list[bool] = []
    reached_scores: list[float] = []
    for row_index, item in enumerate(items):
        row_context = f"{context}:row={row_index}"
        reached = item.get("verifier_reached")
        if type(reached) is not bool:
            raise ValueError(
                f"{row_context}: verifier_reached must be a bool, got "
                f"{reached!r}"
            )
        try:
            score = float(item["episode_score"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{row_context}: episode_score must be numeric"
            ) from error
        if not math.isfinite(score):
            raise ValueError(
                f"{row_context}: episode_score must be finite, got {score!r}"
            )
        scores.append(score)
        verifier_reached.append(reached)
        if reached:
            reached_scores.append(score)

    conditional_mean = (
        math.fsum(reached_scores) / len(reached_scores)
        if reached_scores
        else 0.0
    )
    return [
        score if reached else conditional_mean
        for score, reached in zip(scores, verifier_reached)
    ]


def load_pool(path: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    scenarios = json.loads(Path(path).read_text(encoding="utf-8"))
    task_ids = [str(item["task_id"]) for item in scenarios]
    task_texts = [str(item["task"]).strip() for item in scenarios]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Scenario metadata contains duplicate task IDs")
    if len(task_texts) != len(set(task_texts)):
        raise ValueError("Scenario metadata contains duplicate task texts")
    pool = pd.DataFrame(
        {
            "task_index": np.arange(len(scenarios), dtype=int),
            "task_id": task_ids,
            "env_id": [str(item["env_id"]) for item in scenarios],
            "env_class_name": [str(item["env_class_name"]) for item in scenarios],
            "task": task_texts,
        }
    )
    return pool, scenarios


def load_observation_frame(
    run_dir: str,
    pool: pd.DataFrame,
    history_max_step: int | None,
    *,
    history_min_step: int = 0,
) -> tuple[pd.DataFrame, int]:
    if history_min_step < 0:
        raise ValueError("history_min_step must be non-negative")
    if history_max_step is not None and history_min_step > history_max_step:
        raise ValueError("history_min_step cannot exceed history_max_step")
    variance_label_audit: dict[str, Any] = {
        "variance_target": VERIFIER_CONDITIONAL_VARIANCE_TARGET,
        "label_trajectory_count": 0,
        "verifier_trajectory_count": 0,
        "non_verifier_trajectory_count": 0,
        "check_error_policy": "not_consumed",
        "group_count": 0,
        "group_size_histogram": {},
        "verifier_count_histogram": {},
        "insufficient_verifier_group_count": 0,
        "raw_scalar_variance_sum": 0.0,
        "target_variance_sum": 0.0,
        "reachability_weight_formula": "verifier_count_over_group_count",
    }

    def variance_reward_builder(
        items: list[dict[str, Any]],
        path: Path,
        group_id: str,
    ) -> list[float]:
        context = f"{path}:traj_group_id={group_id}"
        rewards = verifier_conditional_scalar_rewards(
            items,
            context=context,
        )
        verifier_count = sum(
            item["verifier_reached"] is True for item in items
        )
        group_size = len(items)
        group_size_key = str(group_size)
        verifier_count_key = str(verifier_count)
        variance_label_audit["label_trajectory_count"] += group_size
        variance_label_audit["verifier_trajectory_count"] += verifier_count
        variance_label_audit["non_verifier_trajectory_count"] += (
            group_size - verifier_count
        )
        variance_label_audit["group_count"] += 1
        variance_label_audit["group_size_histogram"][group_size_key] = (
            variance_label_audit["group_size_histogram"].get(
                group_size_key, 0
            )
            + 1
        )
        variance_label_audit["verifier_count_histogram"][
            verifier_count_key
        ] = (
            variance_label_audit["verifier_count_histogram"].get(
                verifier_count_key, 0
            )
            + 1
        )
        variance_label_audit["insufficient_verifier_group_count"] += int(
            verifier_count < 2
        )
        variance_label_audit["raw_scalar_variance_sum"] += float(
            np.var([float(item["episode_score"]) for item in items])
        )
        variance_label_audit["target_variance_sum"] += float(
            np.var(rewards)
        )
        return rewards

    observations = read_rollout_observations(
        Path(run_dir),
        variance_reward_builder=variance_reward_builder,
        max_step=history_max_step,
        min_step=history_min_step,
    )
    available_max = max(item.step for item in observations)
    max_step = available_max if history_max_step is None else history_max_step
    observations = [item for item in observations if item.step <= max_step]
    task_index = dict(zip(pool["task"], pool["task_index"]))
    missing = sorted({item.task for item in observations} - set(task_index))
    if missing:
        raise KeyError(f"Could not map {len(missing)} rollout tasks to metadata")
    env_ids = pool.set_index("task_index")["env_id"].to_dict()
    rows = []
    for item in observations:
        index = int(task_index[item.task])
        mean = float(item.mean_reward)
        variance = float(item.reward_variance)
        ceiling = max(mean * (1.0 - mean), 0.01)
        rows.append(
            {
                "task_index": index,
                "task_id": str(pool.iloc[index]["task_id"]),
                "env_id": str(env_ids[index]),
                "step": int(item.step),
                "mean_reward": mean,
                "reward_variance": variance,
                "dispersion": float(np.clip(variance / ceiling, 0.0, 1.0)),
                "rollout_count": len(item.rewards),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["step", "task_index"]).reset_index(drop=True)
    frame.attrs["variance_label_audit"] = variance_label_audit
    return frame, max_step


def optional_static_texts(
    path: str,
    task_ids: list[str],
    field: str | None,
) -> list[str] | None:
    artifact = Path(path)
    if not artifact.exists():
        return None
    records = read_jsonl(artifact)
    by_id = {str(item.get("task_id")): item for item in records}
    if set(by_id) != set(task_ids):
        return None
    texts = []
    for task_id in task_ids:
        record = by_id[task_id]
        value = record.get(field, record) if field else record
        texts.append(json.dumps(value, ensure_ascii=True, sort_keys=True))
    return texts


def build_abstract_topology_similarity(
    pool: pd.DataFrame,
    topology_path: str,
    cache_path: Path,
) -> np.ndarray:
    path = Path(topology_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Abstract cross-environment graph requires topology labels: {path}"
        )
    stat = path.stat()
    cache_key = hashlib.sha256(
        (
            f"abstract-topology-v{ABSTRACT_TOPOLOGY_SCHEMA_VERSION}:"
            f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:"
            + "|".join(pool["task_id"].astype(str))
        ).encode("utf-8")
    ).hexdigest()
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            if (
                str(cached["cache_key"].item()) == cache_key
                and cached["similarity"].shape == (len(pool), len(pool))
            ):
                return cached["similarity"].astype(np.float32, copy=False)

    records = read_jsonl(path)
    by_id = {str(item.get("task_id")): item for item in records}
    task_ids = pool["task_id"].astype(str).tolist()
    if set(by_id) != set(task_ids):
        raise ValueError(
            "Abstract topology graph requires exactly one topology for every task"
        )
    texts = [
        abstract_topology_text(by_id[task_id].get("topology", by_id[task_id]))
        for task_id in task_ids
    ]
    similarity = tfidf_similarity(texts)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            similarity=similarity,
            cache_key=np.asarray(cache_key),
        )
    os.replace(temporary, cache_path)
    return similarity


def tfidf_similarity(
    texts: list[str],
    *,
    analyzer: str = "word",
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int = 35_000,
) -> np.ndarray:
    normalized = [normalize_similarity_text(text) for text in texts]
    vectors = TfidfVectorizer(
        lowercase=True,
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=1,
        max_features=max_features,
        sublinear_tf=True,
    ).fit_transform(normalized)
    return np.clip(cosine_similarity(vectors), 0.0, 1.0).astype(np.float32)


def build_static_similarity(
    pool: pd.DataFrame,
    scenarios: list[dict[str, Any]],
    topology_path: str,
    cache_path: Path,
) -> tuple[np.ndarray, dict[str, float]]:
    cache_hasher = hashlib.sha256(
        json.dumps(scenarios, ensure_ascii=True, sort_keys=True).encode("utf-8")
    )
    path = Path(topology_path)
    if path.exists():
        stat = path.stat()
        cache_hasher.update(
            f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
        )
    else:
        cache_hasher.update(f"missing:{path}".encode("utf-8"))
    cache_key = cache_hasher.hexdigest()
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_key = str(cached["cache_key"].item())
            cached_similarity = cached["similarity"]
            cached_weights = json.loads(str(cached["weights"].item()))
        if cached_key == cache_key and cached_similarity.shape == (len(pool), len(pool)):
            return cached_similarity.astype(np.float32, copy=False), cached_weights

    task_texts: list[str] = []
    verifier_natural: list[str] = []
    verifier_structural: list[str] = []
    static_context: list[str] = []
    for scenario in scenarios:
        task = str(scenario["task"])
        natural, structural, _, _ = verifier_text(scenario)
        state, _, _ = state_signature(scenario.get("init_config", {}), task)
        task_texts.append(task)
        verifier_natural.append(natural)
        verifier_structural.append(structural)
        static_context.append(f"TASK {task} CHECKS {natural} STATE {state}")

    views: list[tuple[str, float, list[str], dict[str, Any]]] = [
        ("task_word", 0.35, task_texts, {}),
        ("task_char", 0.10, task_texts, {"analyzer": "char_wb", "ngram_range": (3, 5)}),
        ("verifier_natural", 0.20, verifier_natural, {}),
        ("verifier_structural", 0.15, verifier_structural, {}),
        ("static_context", 0.20, static_context, {}),
    ]
    task_ids = pool["task_id"].astype(str).tolist()
    topology = optional_static_texts(topology_path, task_ids, "topology")
    if topology is not None:
        views.append(("llm_topology", 0.25, topology, {}))

    total_weight = sum(weight for _, weight, _, _ in views)
    similarity = np.zeros((len(pool), len(pool)), dtype=np.float32)
    weights: dict[str, float] = {}
    for name, raw_weight, texts, kwargs in views:
        weight = raw_weight / total_weight
        similarity += weight * tfidf_similarity(texts, **kwargs)
        weights[name] = weight
    similarity = np.clip(0.5 * (similarity + similarity.T), 0.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            similarity=similarity,
            cache_key=np.asarray(cache_key),
            weights=np.asarray(json.dumps(weights, sort_keys=True)),
        )
    os.replace(temporary, cache_path)
    return similarity, weights


def percentile_rank(values: np.ndarray) -> np.ndarray:
    if len(values) == 1:
        return np.ones(1, dtype=float)
    return (rankdata(values, method="average") - 0.5) / len(values)


def exact_task_exposure_weights(
    selection_counts: np.ndarray,
    scale: float,
) -> np.ndarray:
    counts = np.asarray(selection_counts, dtype=float)
    if not np.isfinite(counts).all() or np.any(counts < 0.0):
        raise ValueError(
            "exact-task selection counts must be finite and non-negative"
        )
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("exact_task_exposure_scale must be finite and positive")
    return 1.0 / np.sqrt(1.0 + counts / scale)


def load_selection_history(
    history_dir: Path,
    target_step: int,
    task_ids: list[str],
    recent_step_count: int = 0,
) -> pd.DataFrame:
    if recent_step_count < 0:
        raise ValueError("recent_step_count cannot be negative")
    selections_by_step: dict[int, Counter[str]] = {}
    for path in history_dir.glob("task_manifest.step_*.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        step = int(manifest["target_policy_step"])
        if step >= target_step:
            continue
        selected = Counter(str(task_id) for task_id in manifest["task_ids"])
        if step in selections_by_step and selections_by_step[step] != selected:
            raise ValueError(f"Conflicting selection manifests for step {step}")
        selections_by_step[step] = selected

    expected_steps = set(range(target_step))
    if set(selections_by_step) != expected_steps:
        raise ValueError(
            "Selection history is incomplete: "
            f"observed={sorted(selections_by_step)}, expected={sorted(expected_steps)}"
        )

    rows = []
    recent_start_step = max(0, target_step - recent_step_count)
    for task_id in task_ids:
        selected_steps = [
            step
            for step, selected in selections_by_step.items()
            if selected[task_id] > 0
        ]
        selection_count = sum(
            selected[task_id] for selected in selections_by_step.values()
        )
        consecutive = 0
        for step in range(target_step - 1, -1, -1):
            if selections_by_step[step][task_id] == 0:
                break
            consecutive += 1
        rows.append(
            {
                "task_id": task_id,
                "selection_count_before": selection_count,
                "recent_selection_count_before": sum(
                    selected[task_id]
                    for step, selected in selections_by_step.items()
                    if step >= recent_start_step
                ),
                "last_selected_step": max(selected_steps, default=-1),
                "consecutive_selection_count": consecutive,
            }
        )
    return pd.DataFrame(rows)


def weighted_recent_step_baseline(
    history: pd.DataFrame,
    target_step: int,
    column: str,
    half_life: float,
) -> tuple[float, dict[int, float]]:
    by_step = history.groupby("step")[column].mean().sort_index()
    ages = target_step - by_step.index.to_numpy(dtype=float)
    weights = np.exp2(-ages / half_life)
    current = float(np.average(by_step.to_numpy(dtype=float), weights=weights))
    return current, by_step.to_dict()


def aggregate_history(
    history: pd.DataFrame,
    pool: pd.DataFrame,
    target_step: int,
    half_life: float,
) -> dict[str, Any]:
    if history.empty:
        raise ValueError("At least one historical rollout step is required")
    current_variance, step_variance = weighted_recent_step_baseline(
        history, target_step, "reward_variance", half_life
    )
    current_mean, step_mean = weighted_recent_step_baseline(
        history, target_step, "mean_reward", half_life
    )
    current_dispersion, step_dispersion = weighted_recent_step_baseline(
        history, target_step, "dispersion", half_life
    )
    frame = history.copy()
    frame["weight"] = np.exp2(
        -(target_step - frame["step"].to_numpy(dtype=float)) / half_life
    )
    frame["adjusted_variance"] = np.clip(
        current_variance
        + frame["reward_variance"]
        - frame["step"].map(step_variance),
        0.0,
        0.25,
    )
    frame["adjusted_mean"] = np.clip(
        current_mean + frame["mean_reward"] - frame["step"].map(step_mean),
        0.0,
        1.0,
    )
    frame["adjusted_dispersion"] = np.clip(
        current_dispersion + frame["dispersion"] - frame["step"].map(step_dispersion),
        0.0,
        1.0,
    )
    count = len(pool)
    task_indices = frame["task_index"].to_numpy(dtype=int)
    weight = frame["weight"].to_numpy(dtype=float)
    task_weight = np.bincount(task_indices, weights=weight, minlength=count)
    task_count = np.bincount(task_indices, minlength=count)
    values: dict[str, np.ndarray] = {}
    for source, name in (
        ("adjusted_variance", "variance"),
        ("adjusted_mean", "mean"),
        ("adjusted_dispersion", "dispersion"),
    ):
        sums = np.bincount(
            task_indices,
            weights=weight * frame[source].to_numpy(dtype=float),
            minlength=count,
        )
        default = {
            "variance": current_variance,
            "mean": current_mean,
            "dispersion": current_dispersion,
        }[name]
        values[name] = np.divide(
            sums,
            task_weight,
            out=np.full(count, default, dtype=float),
            where=task_weight > 0,
        )

    last_step = (
        frame.groupby("task_index")["step"]
        .max()
        .reindex(np.arange(count), fill_value=-1)
        .to_numpy(dtype=int)
    )
    return {
        "current": {
            "variance": current_variance,
            "mean": current_mean,
            "dispersion": current_dispersion,
        },
        "task_weight": task_weight,
        "task_count": task_count,
        "task_values": values,
        "last_step": last_step,
    }


def graph_neighbor_weights(
    target: int,
    similarity: Any,
    aggregates: dict[str, Any],
    pool: pd.DataFrame,
    neighbors: int,
    environment_scope: str,
    graph_state: IncrementalGraphState | None = None,
    plan_name: str | None = None,
    history_max_step: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if environment_scope not in {"all", "within", "cross"}:
        raise ValueError(f"Unknown environment scope: {environment_scope}")
    if graph_state is not None:
        if plan_name is None:
            raise ValueError("incremental graph lookup requires plan_name")
        try:
            selected, selected_scores, _ = graph_state.prepared_neighbors(
                target, plan_name
            )
        except RuntimeError:
            active_mask = np.asarray(aggregates["task_weight"] > 0, dtype=bool)
            env_ids = pool["env_id"].astype(str).to_numpy()
            selected, selected_scores, _ = graph_state.neighbors(
                target,
                similarity,
                active_mask,
                neighbors,
                environment_scope,
                env_ids,
                plan_name,
                history_max_step=history_max_step,
            )
    else:
        active_mask = np.asarray(aggregates["task_weight"] > 0, dtype=bool)
        env_ids = pool["env_id"].astype(str).to_numpy()
        candidates = np.flatnonzero(active_mask)
        candidates = candidates[candidates != target]
        target_env = str(env_ids[target])
        candidate_envs = env_ids[candidates]
        if environment_scope == "within":
            candidates = candidates[candidate_envs == target_env]
        elif environment_scope == "cross":
            candidates = candidates[candidate_envs != target_env]
        if not len(candidates):
            return np.asarray([], dtype=int), np.asarray([], dtype=float)
        count = min(neighbors, len(candidates))
        scores = similarity_submatrix(
            similarity,
            np.asarray([target], dtype=int),
            candidates,
        )[0]
        selected_positions = np.argpartition(-scores, count - 1)[:count]
        selected = candidates[selected_positions]
        selected_scores = scores[selected_positions]
    if not len(selected):
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
    distance = np.clip(1.0 - selected_scores, 0.0, 1.0)
    bandwidth = max(float(np.median(distance)), 0.05)
    kernel = np.exp(-0.5 * np.square(distance / bandwidth))
    reliability = np.sqrt(np.clip(aggregates["task_weight"][selected], 0.1, 3.0))
    return selected, kernel * reliability


def shrunken_graph_value(
    selected: np.ndarray,
    weights: np.ndarray,
    task_values: np.ndarray,
    prior: float,
    shrinkage: float,
) -> float:
    if not len(selected):
        return prior
    return float(
        (np.dot(weights, task_values[selected]) + shrinkage * prior)
        / (weights.sum() + shrinkage)
    )


def predict_tasks(
    history: pd.DataFrame,
    target_indices: np.ndarray,
    target_step: int,
    pool: pd.DataFrame,
    local_similarity: Any,
    global_similarity: Any,
    args: argparse.Namespace,
    graph_state: IncrementalGraphState | None = None,
    graph_plan_namespace: str = "propagation",
) -> pd.DataFrame:
    aggregates = aggregate_history(
        history, pool, target_step, args.half_life_steps
    )
    observed_history_max_step = int(history["step"].max())
    if graph_state is not None:
        active_mask = np.asarray(aggregates["task_weight"] > 0, dtype=bool)
        env_ids = pool["env_id"].astype(str).to_numpy()
        graph_state.prepare_plan(
            global_similarity,
            active_mask,
            args.neighbors,
            "cross",
            env_ids,
            f"{graph_plan_namespace}.cross",
            history_max_step=observed_history_max_step,
        )
        graph_state.prepare_plan(
            local_similarity,
            active_mask,
            args.within_env_neighbors,
            "within",
            env_ids,
            f"{graph_plan_namespace}.within",
            history_max_step=observed_history_max_step,
        )
    current = aggregates["current"]
    rows = []
    for target in target_indices:
        target = int(target)
        own_weight = float(aggregates["task_weight"][target])
        global_selected, global_weights = graph_neighbor_weights(
            target,
            global_similarity,
            aggregates,
            pool,
            args.neighbors,
            "cross",
            graph_state,
            f"{graph_plan_namespace}.cross",
            observed_history_max_step,
        )
        within_selected, within_weights = graph_neighbor_weights(
            target,
            local_similarity,
            aggregates,
            pool,
            args.within_env_neighbors,
            "within",
            graph_state,
            f"{graph_plan_namespace}.within",
            observed_history_max_step,
        )
        history_values: dict[str, float] = {}
        for name in ("variance", "mean", "dispersion"):
            prior = float(current[name])
            global_graph = shrunken_graph_value(
                global_selected,
                global_weights,
                aggregates["task_values"][name],
                prior,
                args.graph_shrinkage,
            )
            within_graph = shrunken_graph_value(
                within_selected,
                within_weights,
                aggregates["task_values"][name],
                prior,
                args.graph_shrinkage,
            )
            graph_blend = (
                (1.0 - args.within_env_graph_weight) * global_graph
                + args.within_env_graph_weight * within_graph
            )
            own = float(aggregates["task_values"][name][target])
            own_confidence = own_weight / (
                own_weight + args.own_history_shrinkage
            )
            history_values[name] = (
                own_confidence * own
                + (1.0 - own_confidence) * graph_blend
            )

        mean_prediction = float(np.clip(history_values["mean"], 0.0, 1.0))
        dispersion_prediction = float(
            np.clip(history_values["dispersion"], 0.0, 1.0)
        )
        mean_frontier = dispersion_prediction * mean_prediction * (1.0 - mean_prediction)
        rows.append(
            {
                "task_index": target,
                "target_step": target_step,
                "history_direct": history_values["variance"],
                "mean_frontier": mean_frontier,
                "predicted_mean_reward": mean_prediction,
                "predicted_dispersion": dispersion_prediction,
                "history_effective_count": own_weight,
                "history_group_count": int(aggregates["task_count"][target]),
                "last_observed_step": int(aggregates["last_step"][target]),
            }
        )
    result = pd.DataFrame(rows)
    rank_score = (
        0.65 * percentile_rank(result["history_direct"].to_numpy(dtype=float))
        + 0.35 * percentile_rank(result["mean_frontier"].to_numpy(dtype=float))
    )
    reference = np.sort(result["history_direct"].to_numpy(dtype=float))
    result["rank_ensemble"] = np.quantile(reference, np.clip(rank_score, 0.0, 1.0))
    for column in ("history_direct", "mean_frontier", "rank_ensemble"):
        result[column] = np.clip(result[column], 0.0, 0.25)
    return result


def select_diverse_tasks(
    pool_predictions: pd.DataFrame,
    similarity: Any,
    method: str,
    args: argparse.Namespace,
) -> list[int]:
    if not 0 <= args.least_observed_per_batch <= args.batch_size:
        raise ValueError(
            "least_observed_per_batch must be between 0 and batch_size"
        )
    if args.max_consecutive_selections < 0:
        raise ValueError("max_consecutive_selections cannot be negative")
    environment_window_steps = args.environment_window_steps
    max_per_environment_window = args.max_per_environment_window
    if environment_window_steps < 0:
        raise ValueError("environment_window_steps cannot be negative")
    if max_per_environment_window < 0:
        raise ValueError("max_per_environment_window cannot be negative")
    if bool(environment_window_steps) != bool(max_per_environment_window):
        raise ValueError(
            "environment_window_steps and max_per_environment_window must both "
            "be zero or both be positive"
        )
    if ONLINE_ID_HACK_BLOCK_COLUMN not in pool_predictions:
        raise ValueError(
            "Selection requires the online rewarded-ID-hack guard column"
        )
    guard_values = pool_predictions[ONLINE_ID_HACK_BLOCK_COLUMN]
    if guard_values.dtype != bool or guard_values.isna().any():
        raise ValueError(
            f"{ONLINE_ID_HACK_BLOCK_COLUMN} must contain booleans"
        )
    online_id_hack_blocked = guard_values.to_numpy(dtype=bool)

    score = pool_predictions[method].to_numpy(dtype=float)
    percentile = percentile_rank(score)
    exact_task_exposure_weight = exact_task_exposure_weights(
        pool_predictions["selection_count_before"].to_numpy(dtype=float),
        args.exact_task_exposure_scale,
    )
    coverage_score = percentile
    exploit_score = percentile * exact_task_exposure_weight
    task_indices = pool_predictions["task_index"].to_numpy(dtype=int)
    env_codes, _ = pd.factorize(pool_predictions["env_id"], sort=True)
    environment_count_total = int(env_codes.max()) + 1
    history_count = pool_predictions["history_group_count"].to_numpy(dtype=int)
    consecutive = pool_predictions["consecutive_selection_count"].to_numpy(dtype=int)
    retired = pool_predictions["retire_candidate"].fillna(False).to_numpy(dtype=bool)
    environment_history_count = np.bincount(
        env_codes,
        weights=pool_predictions["selection_count_before"].to_numpy(dtype=int),
        minlength=environment_count_total,
    ).astype(int)
    recent_environment_count = np.bincount(
        env_codes,
        weights=pool_predictions["recent_selection_count_before"].to_numpy(
            dtype=int
        ),
        minlength=environment_count_total,
    ).astype(int)
    cooldown_blocked = (
        consecutive >= args.max_consecutive_selections
        if args.max_consecutive_selections > 0
        else np.zeros(len(pool_predictions), dtype=bool)
    )
    blocked = cooldown_blocked | online_id_hack_blocked
    selected_count = min(args.batch_size, len(score))
    eligible_per_environment = np.bincount(
        env_codes[~blocked], minlength=environment_count_total
    )
    current_environment_capacity = np.full(
        environment_count_total, args.max_per_environment, dtype=int
    )
    if max_per_environment_window > 0:
        current_environment_capacity = np.minimum(
            current_environment_capacity,
            np.maximum(
                max_per_environment_window - recent_environment_count,
                0,
            ),
        )
    eligible_capacity = int(
        np.minimum(eligible_per_environment, current_environment_capacity).sum()
    )
    if eligible_capacity < selected_count:
        raise RuntimeError(
            "Could not fill batch under quotas/cooldown/online guard: "
            f"capacity={eligible_capacity}, requested={selected_count}"
        )

    selected: list[int] = []
    chosen = np.zeros(len(pool_predictions), dtype=bool)
    redundancy = np.zeros(len(pool_predictions), dtype=float)
    environment_count = np.zeros(environment_count_total, dtype=int)
    coverage_environment_count = np.zeros(environment_count_total, dtype=int)

    def select_phase(
        count: int,
        prefer_least_observed: bool,
    ) -> None:
        for _ in range(count):
            allowed = (
                ~chosen
                & ~blocked
                & (
                    environment_count[env_codes]
                    < current_environment_capacity[env_codes]
                )
            )
            if prefer_least_observed and allowed.any():
                least_coverage_count = coverage_environment_count[env_codes][
                    allowed
                ].min()
                allowed &= (
                    coverage_environment_count[env_codes]
                    == least_coverage_count
                )
                least_environment_history = environment_history_count[env_codes][
                    allowed
                ].min()
                allowed &= (
                    environment_history_count[env_codes]
                    == least_environment_history
                )
                least_history_count = history_count[allowed].min()
                allowed &= history_count == least_history_count
            if not prefer_least_observed:
                allowed &= ~retired
            phase_score = coverage_score if prefer_least_observed else exploit_score
            mmr_score = phase_score - args.diversity_penalty * redundancy
            mmr_score[~allowed] = -np.inf
            best = int(np.argmax(mmr_score))
            if not np.isfinite(mmr_score[best]):
                scope = "least-observed" if prefer_least_observed else "all"
                raise RuntimeError(
                    f"Could not fill the {scope} selection phase under "
                    "quotas/cooldown/online guard"
                )
            selected.append(best)
            chosen[best] = True
            environment_count[env_codes[best]] += 1
            if prefer_least_observed:
                coverage_environment_count[env_codes[best]] += 1
            redundancy[:] = np.maximum(
                redundancy,
                similarity_column(similarity, task_indices[best])[task_indices],
            )

    select_phase(
        min(args.least_observed_per_batch, selected_count),
        True,
    )
    select_phase(selected_count - len(selected), False)
    return selected


def main() -> None:
    args = parse_args()
    if args.history_min_step < 0:
        raise ValueError("history_min_step must be non-negative")
    if (
        args.history_max_step is not None
        and args.history_min_step > args.history_max_step
    ):
        raise ValueError("history_min_step cannot exceed history_max_step")
    if not 0.0 <= args.within_env_graph_weight <= 1.0:
        raise ValueError("within_env_graph_weight must be between 0 and 1")
    if args.max_per_environment < 1:
        raise ValueError("max_per_environment must be positive")
    if not 0.0 <= args.progress_environment_prior_weight <= 1.0:
        raise ValueError(
            "progress_environment_prior_weight must be between 0 and 1"
        )
    if args.progress_min_training_transitions < 1:
        raise ValueError("progress_min_training_transitions must be positive")
    if args.progress_neighbors < 1:
        raise ValueError("progress_neighbors must be positive")
    if (
        not math.isfinite(args.exact_task_exposure_scale)
        or args.exact_task_exposure_scale <= 0.0
    ):
        raise ValueError(
            "exact_task_exposure_scale must be finite and positive"
        )
    if args.incremental_target_chunk_size < 1:
        raise ValueError("incremental_target_chunk_size must be positive")
    if args.incremental_candidate_chunk_size < 1:
        raise ValueError("incremental_candidate_chunk_size must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir).resolve()
    pool, scenarios = load_pool(args.scenarios)
    observations, max_step = load_observation_frame(
        str(run_dir),
        pool,
        args.history_max_step,
        history_min_step=args.history_min_step,
    )
    variance_runtime_label_audit = dict(
        observations.attrs.get("variance_label_audit", {})
    )
    observed_steps = sorted(observations["step"].astype(int).unique().tolist())
    required_steps = list(range(args.history_min_step, max_step + 1))
    if observed_steps != required_steps:
        raise ValueError(
            f"Current-run rollout history is incomplete: observed={observed_steps}, "
            f"required={required_steps}"
        )
    case_paths = sorted(
        path.resolve()
        for path in (run_dir / "output" / "cases").glob("rollout_cases.step_*.jsonl")
        if args.history_min_step
        <= int(re.search(r"step_(\d+)", path.name).group(1))
        <= max_step
    )
    for path in case_paths:
        if run_dir not in path.parents:
            raise ValueError(f"Rollout case escaped the bound run directory: {path}")
    graph_state: IncrementalGraphState | None = None
    if args.graph_execution_mode == "incremental_exact":
        incremental_root = Path(
            args.incremental_graph_state_dir
            or output_dir / "incremental_graph_state"
        )
        similarity, static_weights = build_static_similarity_provider(
            pool,
            scenarios,
            args.topologies,
            incremental_root / "static_provider",
        )
        global_similarity = build_abstract_topology_similarity_provider(
            pool,
            args.topologies,
            incremental_root / "abstract_provider",
        )
        graph_state = IncrementalGraphState(
            incremental_root / "neighbors",
            pool["task_id"].astype(str).tolist(),
            rebuild_on_error=True,
            target_chunk_size=args.incremental_target_chunk_size,
            candidate_chunk_size=args.incremental_candidate_chunk_size,
        )
    else:
        incremental_root = None
        similarity, static_weights = build_static_similarity(
            pool,
            scenarios,
            args.topologies,
            output_dir / "static_similarity.npz",
        )
        global_similarity = build_abstract_topology_similarity(
            pool,
            args.topologies,
            output_dir / "abstract_topology_similarity.npz",
        )
    selected_method = "rank_ensemble"

    target_step = (
        max_step + 1
        if args.target_policy_step is None
        else args.target_policy_step
    )
    if target_step < max_step:
        raise ValueError("target_policy_step cannot precede history_max_step")
    deployment = predict_tasks(
        observations,
        pool["task_index"].to_numpy(dtype=int),
        target_step,
        pool,
        similarity,
        global_similarity,
        args,
        graph_state,
        "propagation",
    ).merge(
        pool[
            [
                "task_index",
                "task_id",
                "env_id",
                "env_class_name",
                "task",
            ]
        ],
        on="task_index",
        validate="one_to_one",
    )
    if args.recency_guard_half_life < 0:
        raise ValueError("recency_guard_half_life cannot be negative")
    if not 0.0 <= args.recency_guard_strength <= 1.0:
        raise ValueError("recency_guard_strength must be between 0 and 1")
    deployment[f"{selected_method}_long_term"] = deployment[selected_method]
    deployment[f"{selected_method}_recent"] = deployment[selected_method]
    if args.recency_guard_half_life > 0 and args.recency_guard_strength > 0:
        recent_args = copy.copy(args)
        recent_args.half_life_steps = args.recency_guard_half_life
        recent = predict_tasks(
            observations,
            pool["task_index"].to_numpy(dtype=int),
            target_step,
            pool,
            similarity,
            global_similarity,
            recent_args,
            graph_state,
            "propagation",
        ).set_index("task_index")[selected_method]
        deployment[f"{selected_method}_recent"] = deployment["task_index"].map(recent)
        downward_gap = np.maximum(
            deployment[f"{selected_method}_long_term"]
            - deployment[f"{selected_method}_recent"],
            0.0,
        )
        deployment[selected_method] = np.clip(
            deployment[f"{selected_method}_long_term"]
            - args.recency_guard_strength * downward_gap,
            0.0,
            0.25,
        )
    deployment["recency_downward_adjustment"] = (
        deployment[f"{selected_method}_long_term"] - deployment[selected_method]
    )
    deployment["selected_method"] = selected_method
    deployment["predicted_variance"] = deployment[selected_method]
    deployment["predicted_variance_percentile"] = percentile_rank(
        deployment["predicted_variance"].to_numpy(dtype=float)
    )
    deployment = deployment.sort_values("task_index").reset_index(drop=True)
    progress, progress_audit = estimate_task_progress(
        observations,
        pool,
        similarity,
        target_step,
        minimum_training_transitions=args.progress_min_training_transitions,
        half_life_steps=args.progress_half_life_steps,
        neighbors=args.progress_neighbors,
        task_shrinkage=args.progress_task_shrinkage,
        environment_prior_weight=args.progress_environment_prior_weight,
        minimum_own_transitions=args.progress_min_own_transitions,
        graph_state=graph_state,
        graph_plan_name="progress.all",
        history_max_step=max_step,
    )
    deployment = deployment.merge(
        progress,
        on=["task_index", "task_id"],
        how="left",
        validate="one_to_one",
    )
    selection_history_dir = Path(args.selection_history_dir or output_dir)
    selection_history = load_selection_history(
        selection_history_dir,
        target_step,
        deployment["task_id"].astype(str).tolist(),
        recent_step_count=max(args.environment_window_steps - 1, 0),
    )
    deployment = deployment.merge(
        selection_history,
        on="task_id",
        how="left",
        validate="one_to_one",
    )
    online_guard_case_paths = sorted(
        path.resolve()
        for path in (run_dir / "output" / "cases").glob(
            "rollout_cases.step_*.jsonl"
        )
        if int(re.search(r"step_(\d+)", path.name).group(1))
        < target_step
    )
    for path in online_guard_case_paths:
        if run_dir not in path.parents:
            raise ValueError(
                f"Online ID-hack case escaped the bound run directory: {path}"
            )
    online_id_hack_guard_frame, online_id_hack_guard_audit = (
        build_online_id_hack_guard(
            case_paths=online_guard_case_paths,
            task_ids=deployment["task_id"].astype(str).tolist(),
            target_step=target_step,
            task_path=args.scenarios,
            env_path=args.environment_metadata,
            cache_dir=output_dir / "online_id_hack_guard_cache",
            expected_group_size=8,
        )
    )
    required_guard_columns = {
        "task_id",
        "blocked",
        "evidence_count",
        "first_confirmation_step",
    }
    missing_guard_columns = required_guard_columns - set(
        online_id_hack_guard_frame.columns
    )
    if missing_guard_columns:
        raise ValueError(
            "Online ID-hack guard is missing columns: "
            f"{sorted(missing_guard_columns)}"
        )
    if online_id_hack_guard_frame["task_id"].duplicated().any():
        raise ValueError("Online ID-hack guard returned duplicate task IDs")
    if set(online_id_hack_guard_frame["task_id"].astype(str)) != set(
        deployment["task_id"].astype(str)
    ):
        raise ValueError(
            "Online ID-hack guard task universe differs from deployment"
        )
    if (
        online_id_hack_guard_frame["blocked"].dtype != bool
        or online_id_hack_guard_frame["blocked"].isna().any()
    ):
        raise ValueError("Online ID-hack guard blocked column must be boolean")
    guard_selection_columns = online_id_hack_guard_frame[
        [
            "task_id",
            "blocked",
            "evidence_count",
            "first_confirmation_step",
        ]
    ].rename(
        columns={
            "blocked": ONLINE_ID_HACK_BLOCK_COLUMN,
            "evidence_count": "online_id_hack_evidence_count",
            "first_confirmation_step": (
                "online_id_hack_first_confirmation_step"
            ),
        }
    )
    deployment = deployment.merge(
        guard_selection_columns,
        on="task_id",
        how="left",
        validate="one_to_one",
    )
    exposure_weights = exact_task_exposure_weights(
        deployment["selection_count_before"].to_numpy(dtype=float),
        args.exact_task_exposure_scale,
    )
    deployment["exact_task_exposure_weight"] = exposure_weights
    deployment["exposure_adjusted_exploit_score"] = (
        deployment["predicted_variance_percentile"] * exposure_weights
    )
    exact_task_exposure_audit: dict[str, Any] = {
        "scale": args.exact_task_exposure_scale,
        "count_source": "selection_count_before",
        "history_scope": "all_prior_policy_steps",
        "applies_to": "exploitation_only",
        "formula": "1 / sqrt(1 + selection_count_before / scale)",
        "pool_mean_weight": float(np.mean(exposure_weights)),
        "pool_min_weight": float(np.min(exposure_weights)),
        "pool_max_selection_count_before": int(
            deployment["selection_count_before"].max()
        ),
    }
    deployment["environment_selection_count_before"] = deployment.groupby(
        "env_id"
    )["selection_count_before"].transform("sum")
    deployment["recent_environment_selection_count_before"] = deployment.groupby(
        "env_id"
    )["recent_selection_count_before"].transform("sum")
    selected_positions = select_diverse_tasks(
        deployment,
        similarity,
        selected_method,
        args,
    )
    selection_policy = "coverage_quota_then_inverse_sqrt_online_id_hack_guard_mmr"
    selected = deployment.iloc[selected_positions].copy()
    selected_blocked_count = int(selected[ONLINE_ID_HACK_BLOCK_COLUMN].sum())
    if selected_blocked_count:
        raise AssertionError(
            "Online rewarded-ID-hack guard allowed blocked tasks into selection"
        )
    online_id_hack_guard_audit["selected_blocked_task_count"] = (
        selected_blocked_count
    )
    selected["selection_rank"] = np.arange(1, len(selected) + 1)
    selected = selected[
        [
            "selection_rank",
            "task_id",
            "env_id",
            "env_class_name",
            "predicted_variance",
            "predicted_variance_percentile",
            f"{selected_method}_long_term",
            f"{selected_method}_recent",
            "recency_downward_adjustment",
            "predicted_mean_reward",
            "predicted_dispersion",
            "history_group_count",
            "history_effective_count",
            "last_observed_step",
            "selection_count_before",
            "recent_selection_count_before",
            "environment_selection_count_before",
            "recent_environment_selection_count_before",
            "last_selected_step",
            "consecutive_selection_count",
            "progress_transition_count",
            "progress_effective_count",
            "progress_posterior",
            "progress_posterior_se",
            "progress_probability_positive",
            "exact_task_exposure_weight",
            "exposure_adjusted_exploit_score",
            "retire_candidate",
            "task",
        ]
    ]
    exploitation_start = min(
        args.least_observed_per_batch,
        len(selected),
    )
    selected_exploitation = selected.iloc[exploitation_start:]
    selected_exploitation_mean_weight = (
        float(selected_exploitation["exact_task_exposure_weight"].mean())
        if not selected_exploitation.empty
        else None
    )
    selected_exploitation_min_weight = (
        float(selected_exploitation["exact_task_exposure_weight"].min())
        if not selected_exploitation.empty
        else None
    )
    selected_exploitation_mean_count_before = (
        float(selected_exploitation["selection_count_before"].mean())
        if not selected_exploitation.empty
        else None
    )
    exact_task_exposure_audit.update(
        {
            "selected_exploitation_count": int(len(selected_exploitation)),
            "selected_exploitation_mean_weight": (
                selected_exploitation_mean_weight
            ),
            "selected_exploitation_min_weight": (
                selected_exploitation_min_weight
            ),
            "selected_exploitation_mean_count_before": (
                selected_exploitation_mean_count_before
            ),
        }
    )
    if graph_state is not None:
        graph_state.save()
        graph_execution_audit = {
            "mode": args.graph_execution_mode,
            "state_root": str(incremental_root.resolve()),
            "static_provider": similarity.audit(),
            "abstract_provider": global_similarity.audit(),
            "state": graph_state.audit(),
        }
    deployment.sort_values("predicted_variance", ascending=False).to_csv(
        output_dir / "pool_predictions.csv", index=False
    )
    selected.to_csv(output_dir / "selected_tasks.csv", index=False)
    online_guard_payload = {
        "audit": online_id_hack_guard_audit,
        "blocked_tasks": online_id_hack_guard_frame.loc[
            online_id_hack_guard_frame["blocked"]
        ].to_dict(orient="records"),
    }
    atomic_write_json(
        output_dir / "online_rewarded_id_hack_guard.json",
        online_guard_payload,
    )
    atomic_write_json(
        output_dir
        / f"online_rewarded_id_hack_guard.step_{target_step}.json",
        online_guard_payload,
    )

    scenario_by_id = {str(item["task_id"]): item for item in scenarios}
    task_ids = selected.sort_values("selection_rank")["task_id"].astype(str).tolist()
    variance_label_contract = {
        "variance_target": VERIFIER_CONDITIONAL_VARIANCE_TARGET,
        "mean_and_progress_target": "scalar_episode_score",
        "variance_formula": (
            "pvariance(verifier_reached_scalar_scores) * "
            "verifier_count / group_count"
        ),
        "non_verifier_trajectory": (
            "conditional_mean_imputation_for_variance_only"
        ),
        "fewer_than_two_verifier_trajectories": "zero_variance",
        "check_error_policy": "not_consumed",
        "label_validation": "strict_verifier_reached_bool",
    }
    manifest = {
        "schema_version": 1,
        "variance_target": VERIFIER_CONDITIONAL_VARIANCE_TARGET,
        "variance_label_contract": variance_label_contract,
        "variance_runtime_label_audit": variance_runtime_label_audit,
        "history_min_step": args.history_min_step,
        "history_max_step": max_step,
        "target_policy_step": target_step,
        "selection_method": selected_method,
        "selection_policy": selection_policy,
        "selection_uses_current_target_rollout": False,
        "selection_uses_historical_rollouts": True,
        "selection_uses_previous_run_rollouts": False,
        "source_current_run_dir": str(run_dir),
        "predictor_weights": {
            "global_graph": 1.0 - args.within_env_graph_weight,
            "within_environment_graph": args.within_env_graph_weight,
        },
        "predictor_graphs": {
            "global": {
                "similarity": "abstract_execution_reward_topology",
                "environment_scope": "cross_environment_only",
                "neighbor_count": args.neighbors,
            },
            "local": {
                "similarity": "full_static_multiview",
                "environment_scope": "within_environment_only",
                "neighbor_count": args.within_env_neighbors,
            },
            "abstract_topology_schema_version": ABSTRACT_TOPOLOGY_SCHEMA_VERSION,
        },
        "exact_task_exposure": exact_task_exposure_audit,
        "online_rewarded_id_hack_guard": online_id_hack_guard_audit,
        "progress_retirement": progress_audit,
        "rollout_files_read": [str(path) for path in case_paths],
        "task_ids": task_ids,
        "tasks": selected.drop(columns="task").to_dict(orient="records"),
    }
    if graph_state is not None:
        manifest["graph_execution"] = graph_execution_audit
    atomic_write_json(output_dir / "task_manifest.json", manifest)
    atomic_write_json(output_dir / f"task_manifest.step_{target_step}.json", manifest)
    atomic_write_json(
        output_dir / "selected_task_items.json",
        [scenario_by_id[task_id] for task_id in task_ids],
    )
    least_observed_task_count = min(
        args.least_observed_per_batch,
        args.batch_size,
        len(deployment),
    )
    least_observed_history_counts = (
        selected.iloc[:least_observed_task_count]["history_group_count"]
        .astype(int)
        .value_counts()
        .sort_index()
    )
    selected_unobserved_task_count = int(
        (selected["history_group_count"] == 0).sum()
    )
    coverage_selection = selected.iloc[:least_observed_task_count]
    coverage_environment_history_counts = (
        coverage_selection.drop_duplicates("env_id")[
            "environment_selection_count_before"
        ]
        .astype(int)
        .value_counts()
        .sort_index()
    )
    selected_environment_counts = selected["env_id"].value_counts()
    selected_environment_frame = selected.drop_duplicates("env_id").set_index(
        "env_id"
    )
    rolling_environment_totals = (
        selected_environment_counts
        + selected_environment_frame[
            "recent_environment_selection_count_before"
        ].astype(int)
    )
    audit = {
        "variance_target": VERIFIER_CONDITIONAL_VARIANCE_TARGET,
        "variance_label_contract": variance_label_contract,
        "variance_runtime_label_audit": variance_runtime_label_audit,
        "pool_task_count": len(pool),
        "historically_observed_task_count": int(observations["task_id"].nunique()),
        "historical_group_count": len(observations),
        "history_min_step": args.history_min_step,
        "history_max_step": max_step,
        "target_policy_step": target_step,
        "selected_method": selected_method,
        "selection_policy": selection_policy,
        "predictor_weights": {
            "global_graph": 1.0 - args.within_env_graph_weight,
            "within_environment_graph": args.within_env_graph_weight,
        },
        "predictor_graphs": {
            "global": {
                "similarity": "abstract_execution_reward_topology",
                "environment_scope": "cross_environment_only",
                "neighbor_count": args.neighbors,
            },
            "local": {
                "similarity": "full_static_multiview",
                "environment_scope": "within_environment_only",
                "neighbor_count": args.within_env_neighbors,
            },
            "abstract_topology_schema_version": ABSTRACT_TOPOLOGY_SCHEMA_VERSION,
        },
        "exact_task_exposure": exact_task_exposure_audit,
        "online_rewarded_id_hack_guard": online_id_hack_guard_audit,
        "selection_history_dir": str(selection_history_dir.resolve()),
        "least_observed_per_batch": args.least_observed_per_batch,
        "least_observed_task_count": least_observed_task_count,
        "least_observed_history_count_distribution": {
            str(history_count): int(count)
            for history_count, count in least_observed_history_counts.items()
        },
        "coverage_environment_count": int(
            coverage_selection["env_id"].nunique()
        ),
        "max_per_environment": args.max_per_environment,
        "diversity_penalty": args.diversity_penalty,
        "coverage_environment_history_count_distribution": {
            str(history_count): int(count)
            for history_count, count in coverage_environment_history_counts.items()
        },
        "selected_environment_count": int(selected["env_id"].nunique()),
        "max_selected_per_environment": int(selected_environment_counts.max()),
        "environment_window_steps": args.environment_window_steps,
        "max_per_environment_window": args.max_per_environment_window,
        "max_selected_per_environment_in_window": int(
            rolling_environment_totals.max()
        ),
        "max_consecutive_selections": args.max_consecutive_selections,
        "progress_retirement": progress_audit,
        "selected_retire_candidate_count": int(selected["retire_candidate"].sum()),
        "mean_selected_progress_posterior": float(
            selected["progress_posterior"].mean()
        ),
        "mean_selected_progress_probability_positive": float(
            selected["progress_probability_positive"].mean()
        ),
        "recency_guard_half_life": args.recency_guard_half_life,
        "recency_guard_strength": args.recency_guard_strength,
        "mean_selected_recency_downward_adjustment": float(
            selected["recency_downward_adjustment"].mean()
        ),
        "selected_unobserved_task_count": selected_unobserved_task_count,
        "selected_previously_observed_task_count": int(
            (selected["history_group_count"] > 0).sum()
        ),
        "selected_from_previous_step_count": int(
            (selected["last_selected_step"] == target_step - 1).sum()
        ),
        "static_view_weights": static_weights,
        "llm_topology_complete": "llm_topology" in static_weights,
        "selection_uses_current_target_rollout": False,
        "selection_uses_historical_rollouts": True,
        "selection_uses_previous_run_rollouts": False,
        "source_current_run_dir": str(run_dir),
        "rollout_files_read": [str(path) for path in case_paths],
    }
    if graph_state is not None:
        audit["graph_execution"] = graph_execution_audit
    atomic_write_json(output_dir / "input_audit.json", audit)
    atomic_write_json(output_dir / f"input_audit.step_{target_step}.json", audit)
    print(
        f"selected_method={selected_method} history_steps="
        f"{args.history_min_step}..{max_step} "
        f"manifest={output_dir / 'task_manifest.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
