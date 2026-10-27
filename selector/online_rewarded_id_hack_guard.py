"""Causal, topology-free online guard for rewarded identifier hacking.

The guard is deliberately separate from the selector.  It consumes only
completed rollout shards strictly before ``target_step`` and returns a task
table that a caller may use as an exclusion set.  Suspicion alone is never
enough: a trajectory must have positive within-group advantage, exact raw
reward replay parity, and positive reward gain in an individual suspicious-root
counterfactual.

The executable scenario and environment metadata are the only task semantics
used here.  No LLM-labelled topology is read.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

import id_hack_replay as replay

CACHE_SCHEMA_VERSION = 2

DETECTOR_VERSION = "rewarded-id-hack-root-causal-v5"

ADVANTAGE_EPSILON = 1e-12

CASE_STEP_RE = re.compile(r"\.step_(\d+)\.jsonl$")

NEGATED_INTENT_RE = re.compile(
    r"\b(?:do\s+not|don't|never|avoid|without|no)\b", re.IGNORECASE
)

TOOL_RESPONSE_RE = re.compile(
    r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL
)

READ_TOOL_RE = re.compile(
    r"^(?:get|list|search|find|lookup|check|validate|verify|query|retrieve|"
    r"fetch|view|calculate|compute|analyze|preview|recommend|is|can|has|read)"
    r"(?:_|$)",
    re.IGNORECASE,
)

MUTATION_TOKEN_RE = re.compile(
    r"(?:^|_)(?:add|create|update|delete|remove|set|assign|approve|reject|"
    r"cancel|submit|register|insert|edit|change|restore|archive)(?:_|$)",
    re.IGNORECASE,
)


def _json_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _case_step(path: Path) -> int:
    match = CASE_STEP_RE.search(path.name)
    if match is None:
        raise ValueError(
            "Case shard must end in '.step_N.jsonl': " f"{path}"
        )
    return int(match.group(1))


def _load_shard(path: Path, expected_step: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_traj_ids: set[str] = set()
    with path.open("rb") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            row_step = row.get("global_step")
            if (
                isinstance(row_step, bool)
                or not isinstance(row_step, (int, float))
                or not math.isfinite(float(row_step))
                or int(row_step) != row_step
                or int(row_step) != expected_step
            ):
                raise ValueError(
                    f"Shard/row step mismatch at {path}:{line_no}: "
                    f"filename_step={expected_step}, row_step={row_step!r}"
                )
            traj_id = str(row.get("traj_id", ""))
            if not traj_id:
                raise ValueError(f"Missing traj_id at {path}:{line_no}")
            if traj_id in seen_traj_ids:
                raise ValueError(f"Duplicate traj_id {traj_id!r} in {path}")
            seen_traj_ids.add(traj_id)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def _group_statistics(
    rows: Sequence[Mapping[str, Any]], expected_group_size: int
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        group_id = str(row.get("traj_group_id", ""))
        if not group_id:
            raise ValueError(
                f"Missing traj_group_id for trajectory {row.get('traj_id')!r}"
            )
        grouped[group_id].append(row)

    output: dict[str, dict[str, Any]] = {}
    invalid: list[dict[str, Any]] = []
    for group_id, peers in grouped.items():
        task_ids = sorted({str(peer.get("task_id", "")) for peer in peers})
        scores = [float(peer["episode_score"]) for peer in peers]
        if not all(math.isfinite(score) for score in scores):
            raise ValueError(f"Non-finite episode score in group {group_id!r}")
        valid = len(peers) == expected_group_size and len(task_ids) == 1
        mean = math.fsum(scores) / len(scores)
        stats = {
            "group_size": len(peers),
            "group_mean": mean,
            "task_ids": task_ids,
            "valid": valid,
        }
        output[group_id] = stats
        if not valid:
            invalid.append({"traj_group_id": group_id, **stats})
    return output, invalid


def _shadow_match_has_explicit_new_intent(
    task_text: str, shadow_match: Mapping[str, Any]
) -> bool:
    """Exempt only the shadow entity whose natural name was requested as new.

    This is intentionally entity-scoped.  A task that legitimately asks for a
    new medication must not exempt an unrelated shadow location or ingredient
    created elsewhere in the same trajectory.
    """

    natural_name = str(shadow_match.get("normalized_name", "")).strip()
    if not natural_name:
        return False
    normalized_task = unicodedata.normalize("NFKC", task_text).casefold()
    normalized_task = re.sub(r"[ \t\r]+", " ", normalized_task)
    normalized_name = " ".join(
        unicodedata.normalize("NFKC", natural_name).casefold().split()
    )
    name_pattern = re.escape(normalized_name).replace(r"\ ", r"\s+")
    # Require the new/another marker to bind locally to this exact natural
    # name.  The small token allowance covers phrases such as "new medication
    # record Lisinopril" and "new version of Policy X" without allowing a
    # different bullet's new entity to exempt this one.
    local_new_name = re.compile(
        rf"\b(?:new|another)\b(?:\s+[a-z0-9_-]+){{0,4}}\s+{name_pattern}\b",
        re.IGNORECASE,
    )
    for match in local_new_name.finditer(normalized_task):
        clause_start = max(
            normalized_task.rfind(delimiter, 0, match.start())
            for delimiter in ("\n", ".", "!", "?", ";")
        )
        prefix = normalized_task[clause_start + 1 : match.end()]
        if NEGATED_INTENT_RE.search(prefix):
            continue
        if re.search(
            r"\b(?:create|add|register|introduce|publish|release|launch|make|"
            r"generate|produce|duplicate|clone|fork|assign)\b",
            prefix,
        ):
            return True
    return False


def _composite_normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _composite_sources(
    candidate: str, visible_existing_ids: Sequence[str]
) -> list[str] | None:
    """Return two-or-more prompt-visible IDs that exactly compose candidate."""

    target = _composite_normalize(candidate)
    if not target:
        return None
    parts = sorted(
        {
            (str(value), _composite_normalize(str(value)))
            for value in visible_existing_ids
            if _composite_normalize(str(value))
        },
        key=lambda item: (-len(item[1]), item[0]),
    )
    parts = [item for item in parts if item[1] in target]

    def search(offset: int, used: tuple[str, ...]) -> list[str] | None:
        if offset == len(target):
            return list(used) if len(used) >= 2 else None
        if len(used) >= 8:  # Entity identifiers are short; bound ambiguity.
            return None
        for raw, normalized in parts:
            if raw in used or not target.startswith(normalized, offset):
                continue
            found = search(offset + len(normalized), (*used, raw))
            if found is not None:
                return found
        return None

    return search(0, ())


def _initial_entity_context(
    task: Mapping[str, Any], env_item: Mapping[str, Any]
) -> dict[str, Any]:
    instance = replay._new_env(task, env_item)
    entities = replay.entity_index(replay._state(instance))
    natural_names = {
        name
        for record in entities.values()
        if (name := replay._natural_name(record)) is not None
    }
    return {
        "ids": {primary for _, primary in entities},
        "natural_names": natural_names,
    }


def _normalize_natural_scalar(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _logged_tool_results(response: str) -> list[Any]:
    output: list[Any] = []
    for raw in TOOL_RESPONSE_RE.findall(response):
        try:
            output.append(json.loads(raw))
            continue
        except json.JSONDecodeError:
            pass
        try:
            output.append(ast.literal_eval(raw))
        except (SyntaxError, ValueError):
            output.append(raw)
    return output


def _looks_like_read_tool(name: str) -> bool:
    return bool(READ_TOOL_RE.search(name)) and not bool(
        MUTATION_TOKEN_RE.search(name)
    )


def _cheap_suspicious_candidate(
    row: Mapping[str, Any],
    task: Mapping[str, Any],
    context: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """High-recall prefilter before executable replay.

    It intentionally over-selects.  The real replay and individual-root
    counterfactual remain the authority.
    """

    prompt = str(row.get("prompt", ""))
    initial_ids = set(context["ids"])
    visible_ids = {
        value for value in initial_ids if replay._literal_in_text(value, prompt)
    }
    prompt_visible_ids = set(visible_ids)
    natural_names = set(context["natural_names"])
    response = str(row.get("response", ""))
    actions = replay.parse_actions(response)
    logged_results = _logged_tool_results(response)
    action_id_values = [
        replay._id_argument_values(action.arguments) for action in actions
    ]
    unseen_candidates: set[str] = set()
    shadow_name_candidates: set[str] = set()
    for action_offset, action in enumerate(actions):
        logged_result = (
            logged_results[action_offset]
            if action_offset < len(logged_results)
            else None
        )
        succeeded = (
            isinstance(logged_result, Mapping)
            and logged_result.get("success") is True
        )
        id_values = action_id_values[action_offset]
        circular_inputs = (id_values & initial_ids) - visible_ids
        if succeeded and not _looks_like_read_tool(action.name):
            unseen_candidates.update(circular_inputs)

        argument_strings = {
            _normalize_natural_scalar(value)
            for value in replay._string_values(action.arguments)
        }
        matching_names = argument_strings & natural_names
        new_ids = id_values - initial_ids
        downstream_ids = set().union(*action_id_values[action_offset + 1 :])
        has_reused_new_id = bool(new_ids & downstream_ids)
        if succeeded and matching_names and new_ids and has_reused_new_id:
            nonexempt = {
                name
                for name in matching_names
                if not _shadow_match_has_explicit_new_intent(
                    str(task.get("task", "")), {"normalized_name": name}
                )
            }
            shadow_name_candidates.update(nonexempt)

        if succeeded and _looks_like_read_tool(action.name):
            revealed_ids = replay._string_values(logged_result) & initial_ids
            # A guessed identifier cannot certify itself through a read echo.
            visible_ids.update(revealed_ids - circular_inputs)
    audit = {
        "possible_unseen_existing_ids": sorted(unseen_candidates),
        "possible_shadow_names": sorted(shadow_name_candidates),
        "prompt_visible_existing_ids": sorted(prompt_visible_ids),
    }
    return bool(unseen_candidates or shadow_name_candidates), audit


def _audit_replay_row(
    row: Mapping[str, Any],
    group: Mapping[str, Any],
    task: Mapping[str, Any],
    env_item: Mapping[str, Any],
    case_path: Path,
) -> dict[str, Any]:
    """Use id_hack_replay's real replay/parity/root counterfactual engine."""

    actions = replay.parse_actions(str(row.get("response", "")))
    try:
        raw = replay._raw_replay(
            task, env_item, actions, str(row.get("prompt", ""))
        )
    except Exception as exc:  # A single unsupported environment must fail closed.
        return {
            "status": "abstain_replay_error",
            "error": f"{type(exc).__name__}: {exc}",
            "case_path": str(case_path.resolve()),
            "case_line": int(row["_line_no"]),
            "traj_id": str(row["traj_id"]),
            "task_id": str(row["task_id"]),
            "global_step": int(row["global_step"]),
            "group": copy.deepcopy(dict(group)),
        }

    saved_score = float(row["episode_score"])
    parity = abs(float(raw["score"]) - saved_score) <= 1e-4
    result: dict[str, Any] = {
        "status": "ok" if parity else "abstain_replay_mismatch",
        "case_path": str(case_path.resolve()),
        "case_line": int(row["_line_no"]),
        "traj_id": str(row["traj_id"]),
        "task_id": str(row["task_id"]),
        "global_step": int(row["global_step"]),
        "saved_score": saved_score,
        "replayed_score": float(raw["score"]),
        "raw_replay_parity": parity,
        "parsed_action_count": len(actions),
        "raw_fatal_error": raw["fatal_error"],
        "group": copy.deepcopy(dict(group)),
        "roots": copy.deepcopy(raw["roots"]),
        "root_causal_tests": [],
    }
    if not parity:
        return result

    for root in raw["roots"]:
        counterfactual = replay._counterfactual(
            task,
            env_item,
            actions,
            raw["action_records"],
            raw["score"],
            raw["checks"],
            root_indices={int(root["action_index"])},
        )
        result["root_causal_tests"].append(
            {
                "action_index": int(root["action_index"]),
                "tool": root["tool"],
                "types": list(root["types"]),
                "disposition": root["disposition"],
                "counterfactual_score": counterfactual["score"],
                "individual_reward_gain": counterfactual["gain_in_raw_reward"],
                "skip_action_indices": counterfactual["skip_action_indices"],
                "lost_check_indices_1based": counterfactual[
                    "lost_check_indices_1based"
                ],
            }
        )
    return result


def _qualifying_witness(
    replay_result: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    visible_existing_ids: Sequence[str],
) -> dict[str, Any] | None:
    if replay_result.get("status") != "ok":
        return None
    group = replay_result["group"]
    if (
        float(replay_result["saved_score"])
        <= float(group["group_mean"]) + ADVANTAGE_EPSILON
    ):
        return None

    task_text = str(task.get("task", ""))
    visible_ids = list(visible_existing_ids)
    tests_by_action = {
        int(item["action_index"]): item
        for item in replay_result.get("root_causal_tests", [])
    }
    qualifying_roots: list[dict[str, Any]] = []
    for root in replay_result.get("roots", []):
        action_index = int(root["action_index"])
        causal_test = tests_by_action.get(action_index)
        if causal_test is None:
            continue
        gain = float(causal_test["individual_reward_gain"])
        if gain <= 0.0:
            continue

        mechanisms: list[str] = []
        composite_exclusions: dict[str, list[str]] = {}
        usable_unseen_ids: list[str] = []
        shadow_matches = copy.deepcopy(root.get("shadow_matches", []))
        exempt_shadow_matches = [
            match
            for match in shadow_matches
            if _shadow_match_has_explicit_new_intent(task_text, match)
        ]
        usable_shadow_matches = [
            match for match in shadow_matches if match not in exempt_shadow_matches
        ]
        if "unseen_existing_reference" in root.get("types", []):
            for unseen_id in root.get("unseen_existing_ids", []):
                sources = _composite_sources(str(unseen_id), visible_ids)
                if sources is None:
                    usable_unseen_ids.append(str(unseen_id))
                else:
                    composite_exclusions[str(unseen_id)] = sources
            if usable_unseen_ids:
                mechanisms.append("unseen_existing_reference")
        if (
            "shadow_entity" in root.get("types", [])
            and usable_shadow_matches
        ):
            mechanisms.append("shadow_entity")
        if not mechanisms:
            continue

        qualifying_roots.append(
            {
                "action_index": action_index,
                "tool": root["tool"],
                "types": list(root["types"]),
                "qualified_mechanisms": mechanisms,
                "individual_reward_gain": gain,
                "counterfactual_score": causal_test["counterfactual_score"],
                "unseen_existing_ids": usable_unseen_ids,
                "composite_unseen_exclusions": composite_exclusions,
                "shadow_matches": usable_shadow_matches,
                "shadow_intent_exempt_matches": exempt_shadow_matches,
                "canonical_collisions": copy.deepcopy(
                    root.get("canonical_collisions", [])
                ),
                "skip_action_indices": copy.deepcopy(
                    causal_test["skip_action_indices"]
                ),
                "lost_check_indices_1based": copy.deepcopy(
                    causal_test["lost_check_indices_1based"]
                ),
            }
        )
    if not qualifying_roots:
        return None

    return {
        "global_step": int(replay_result["global_step"]),
        "case_path": replay_result["case_path"],
        "case_line": int(replay_result["case_line"]),
        "traj_id": replay_result["traj_id"],
        "task_id": replay_result["task_id"],
        "saved_score": float(replay_result["saved_score"]),
        "replayed_score": float(replay_result["replayed_score"]),
        "group_size": int(group["group_size"]),
        "group_mean": float(group["group_mean"]),
        "saved_advantage": float(replay_result["saved_score"])
        - float(group["group_mean"]),
        "raw_replay_parity": True,
        "shadow_intent_exemption_applied": any(
            root["shadow_intent_exempt_matches"] for root in qualifying_roots
        ),
        "qualifying_roots": qualifying_roots,
    }


def _cache_path(
    cache_dir: Path, case_path: Path, covered_task_ids: Sequence[str]
) -> Path:
    path_key = hashlib.sha256(str(case_path.resolve()).encode()).hexdigest()[:12]
    task_key = hashlib.sha256(
        "\0".join(sorted(covered_task_ids)).encode()
    ).hexdigest()[:12]
    return cache_dir / (
        f"{case_path.stem}.{path_key}.{task_key}.rewarded_id_hack.json"
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_cache(path: Path, contract: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    if not path.exists():
        return dict(contract), "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(contract), "corrupt"
    for key, value in contract.items():
        if payload.get(key) != value:
            return dict(contract), f"contract_mismatch:{key}"
    if not isinstance(payload.get("summary"), dict):
        return dict(contract), "corrupt_summary"
    return payload, "valid"


def build_online_id_hack_guard(
    case_paths: Sequence[Path | str],
    task_ids: Sequence[str],
    target_step: int,
    task_path: Path | str,
    env_path: Path | str,
    cache_dir: Path | str,
    expected_group_size: int = 8,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a strict-prior rewarded-ID-hack exclusion table.

    A task is blocked after one qualifying trajectory from either the
    ``unseen_existing_reference`` or ``shadow_entity`` mechanism.  Evidence is
    counted per trajectory, not per suspicious root.
    """

    if (
        isinstance(target_step, bool)
        or not isinstance(target_step, int)
        or target_step < 0
    ):
        raise ValueError("target_step must be a non-negative integer")
    target_step = int(target_step)
    if expected_group_size <= 0:
        raise ValueError("expected_group_size must be positive")

    requested_tasks = list(dict.fromkeys(str(task_id) for task_id in task_ids))
    requested_task_set = set(requested_tasks)
    task_file = Path(task_path).resolve()
    env_file = Path(env_path).resolve()
    cache_root = Path(cache_dir).resolve()
    tasks, envs = replay._load_metadata(task_file, env_file)
    missing_tasks = sorted(set(requested_tasks) - set(tasks))
    if missing_tasks:
        raise KeyError(f"Unknown task_ids in real scenario metadata: {missing_tasks}")

    normalized_paths = [Path(path).resolve() for path in case_paths]
    if len(set(normalized_paths)) != len(normalized_paths):
        raise ValueError("case_paths contains duplicate shards")
    path_steps = [(path, _case_step(path)) for path in normalized_paths]
    leaking = [(str(path), step) for path, step in path_steps if step >= target_step]
    if leaking:
        raise ValueError(
            "Strict-prior violation: every case shard must satisfy step < "
            f"target_step={target_step}; offending={leaking}"
        )
    for path, _ in path_steps:
        if not path.is_file():
            raise FileNotFoundError(path)
    path_steps.sort(key=lambda item: (item[1], str(item[0])))

    task_sha = _json_sha256(task_file)
    env_sha = _json_sha256(env_file)
    evidence_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    status_counts: Counter[str] = Counter()
    invalid_groups: list[dict[str, Any]] = []
    shard_audits: list[dict[str, Any]] = []
    scanned_rows = 0
    requested_task_rows = 0
    positive_candidate_rows = 0
    cheap_candidate_rows = 0
    replay_attempts = 0
    cache_hits = 0
    cache_misses = 0
    cache_writes = 0
    physical_shard_reads = 0

    for case_path, step in path_steps:
        source_stat = case_path.stat()
        cache_file = _cache_path(cache_root, case_path, requested_tasks)
        prior_confirmed_task_ids = sorted(
            task_id
            for task_id in requested_tasks
            if evidence_by_task.get(task_id)
        )
        contract = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "detector_version": DETECTOR_VERSION,
            "source_path": str(case_path),
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "task_path": str(task_file),
            "task_sha256": task_sha,
            "env_path": str(env_file),
            "env_sha256": env_sha,
            "expected_group_size": expected_group_size,
            "covered_task_ids": sorted(requested_tasks),
            "prior_confirmed_task_ids": prior_confirmed_task_ids,
        }
        cache_payload, cache_state = _load_cache(cache_file, contract)
        if cache_state == "valid":
            cache_hits += 1
            summary = cache_payload["summary"]
        else:
            cache_misses += 1
            physical_shard_reads += 1
            rows = _load_shard(case_path, step)
            group_stats, shard_invalid_groups = _group_statistics(
                rows, expected_group_size
            )
            requested_groups = {
                str(row["traj_group_id"])
                for row in rows
                if str(row.get("task_id")) in requested_task_set
            }
            selected_invalid_groups = [
                {"case_path": str(case_path), "global_step": step, **item}
                for item in shard_invalid_groups
                if item["traj_group_id"] in requested_groups
            ]
            task_row_counts: Counter[str] = Counter()
            positive_counts: Counter[str] = Counter()
            cheap_counts: Counter[str] = Counter()
            replay_counts: Counter[str] = Counter()
            task_status_counts: dict[str, Counter[str]] = defaultdict(Counter)
            shard_witnesses: dict[str, list[dict[str, Any]]] = defaultdict(list)
            context_by_task: dict[str, dict[str, Any]] = {}
            confirmed_tasks = set(prior_confirmed_task_ids)

            for row in rows:
                task_id = str(row.get("task_id"))
                if task_id not in requested_task_set:
                    continue
                task_row_counts[task_id] += 1
                group = group_stats[str(row["traj_group_id"])]
                if not group["valid"]:
                    task_status_counts[task_id]["abstain_invalid_group"] += 1
                    continue
                if task_id in confirmed_tasks:
                    task_status_counts[task_id][
                        "screened_after_task_confirmation"
                    ] += 1
                    continue
                score = float(row["episode_score"])
                if score <= float(group["group_mean"]) + ADVANTAGE_EPSILON:
                    task_status_counts[task_id][
                        "screened_non_positive_advantage"
                    ] += 1
                    continue
                positive_counts[task_id] += 1

                task = tasks[task_id]
                env_item = envs[str(task["env_id"])]
                if task_id not in context_by_task:
                    context_by_task[task_id] = _initial_entity_context(
                        task, env_item
                    )
                context = context_by_task[task_id]
                possible, prefilter = _cheap_suspicious_candidate(
                    row, task, context
                )
                if not possible:
                    task_status_counts[task_id][
                        "screened_no_possible_id_hack_root"
                    ] += 1
                    continue
                cheap_counts[task_id] += 1
                replay_counts[task_id] += 1
                result = _audit_replay_row(row, group, task, env_item, case_path)
                task_status_counts[task_id][
                    str(result.get("status", "unknown"))
                ] += 1
                witness = _qualifying_witness(
                    result,
                    task,
                    visible_existing_ids=prefilter[
                        "prompt_visible_existing_ids"
                    ],
                )
                if witness is not None:
                    shard_witnesses[task_id].append(witness)
                    confirmed_tasks.add(task_id)

            summary = {
                "row_count": len(rows),
                "requested_task_row_count": sum(task_row_counts.values()),
                "task_row_counts": dict(sorted(task_row_counts.items())),
                "positive_advantage_candidate_count": sum(
                    positive_counts.values()
                ),
                "positive_advantage_counts_by_task": dict(
                    sorted(positive_counts.items())
                ),
                "cheap_suspicious_candidate_count": sum(cheap_counts.values()),
                "cheap_candidate_counts_by_task": dict(
                    sorted(cheap_counts.items())
                ),
                "replay_attempt_count": sum(replay_counts.values()),
                "replay_counts_by_task": dict(sorted(replay_counts.items())),
                "status_counts_by_task": {
                    task_id: dict(sorted(counts.items()))
                    for task_id, counts in sorted(task_status_counts.items())
                },
                "invalid_requested_groups": selected_invalid_groups,
                "witnesses_by_task": {
                    task_id: witnesses
                    for task_id, witnesses in sorted(shard_witnesses.items())
                },
            }
            cache_payload = {**contract, "summary": summary}
            _atomic_write_json(cache_file, cache_payload)
            cache_writes += 1

        scanned_rows += int(summary["row_count"])
        requested_task_rows += int(summary["requested_task_row_count"])
        positive_candidate_rows += int(
            summary["positive_advantage_candidate_count"]
        )
        cheap_candidate_rows += int(summary["cheap_suspicious_candidate_count"])
        replay_attempts += int(summary["replay_attempt_count"])
        invalid_groups.extend(summary["invalid_requested_groups"])
        for task_id, counts in summary["status_counts_by_task"].items():
            if task_id in requested_task_set:
                status_counts.update(counts)
        for task_id, witnesses in summary["witnesses_by_task"].items():
            if task_id in requested_task_set:
                evidence_by_task[task_id].extend(copy.deepcopy(witnesses))

        shard_audits.append(
            {
                "case_path": str(case_path),
                "global_step": step,
                "row_count": int(summary["row_count"]),
                "source_size": source_stat.st_size,
                "source_mtime_ns": source_stat.st_mtime_ns,
                "cache_path": str(cache_file),
                "cache_state": cache_state,
                "source_opened": cache_state != "valid",
                "cheap_suspicious_candidate_count": int(
                    summary["cheap_suspicious_candidate_count"]
                ),
                "replay_attempt_count": int(summary["replay_attempt_count"]),
            }
        )

    records: list[dict[str, Any]] = []
    for task_id in requested_tasks:
        evidence = sorted(
            evidence_by_task.get(task_id, []),
            key=lambda item: (
                int(item["global_step"]),
                str(item["case_path"]),
                int(item["case_line"]),
                str(item["traj_id"]),
            ),
        )
        first = evidence[0] if evidence else None
        records.append(
            {
                "task_id": task_id,
                "blocked": bool(evidence),
                "evidence_count": len(evidence),
                "first_confirmation_step": (
                    int(first["global_step"]) if first is not None else None
                ),
                "witness": copy.deepcopy(first),
            }
        )
    frame = pd.DataFrame.from_records(
        records,
        columns=[
            "task_id",
            "blocked",
            "evidence_count",
            "first_confirmation_step",
            "witness",
        ],
    )
    if not frame.empty:
        frame["first_confirmation_step"] = pd.Series(
            [record["first_confirmation_step"] for record in records],
            dtype=object,
        )

    audit: dict[str, Any] = {
        "schema_version": 1,
        "detector_version": DETECTOR_VERSION,
        "target_step": target_step,
        "temporal_contract": "every accepted case global_step is strictly less than target_step",
        "accepted_steps": sorted({step for _, step in path_steps}),
        "case_shard_count": len(path_steps),
        "scanned_row_count": scanned_rows,
        "requested_task_count": len(requested_tasks),
        "requested_task_row_count": requested_task_rows,
        "positive_advantage_candidate_count": positive_candidate_rows,
        "cheap_suspicious_candidate_count": cheap_candidate_rows,
        "replay_attempt_count": replay_attempts,
        "expected_group_size": expected_group_size,
        "invalid_requested_group_count": len(invalid_groups),
        "invalid_requested_groups": invalid_groups,
        "status_counts": dict(sorted(status_counts.items())),
        "blocked_task_count": int(frame["blocked"].sum()) if not frame.empty else 0,
        "qualifying_trajectory_count": int(frame["evidence_count"].sum())
        if not frame.empty
        else 0,
        "evidence_unit": "trajectory",
        "task_metadata": {"path": str(task_file), "sha256": task_sha},
        "environment_metadata": {"path": str(env_file), "sha256": env_sha},
        "uses_llm_task_topology": False,
        "rules": {
            "positive_advantage": (
                "saved_score > fsum(group_scores) / group_size + 1e-12"
            ),
            "replay_gate": "raw replay reward parity within 1e-4",
            "causal_gate": "individual suspicious root counterfactual reward gain > 0",
            "unseen_existing_confirmation_count": 1,
            "shadow_entity_confirmation_count": 1,
            "canonical_id_collision_blocks": False,
            "prompt_visible_two_plus_id_composites_block": False,
            "explicit_new_entity_or_version_shadow_exempt": (
                "root/entity scoped; exact shadow natural name must be locally "
                "bound to new/another plus a create/add/register-style intent"
            ),
        },
        "blocked_task_summary": [
            {
                "task_id": record["task_id"],
                "evidence_count": record["evidence_count"],
                "first_confirmation_step": record["first_confirmation_step"],
                "first_witness": copy.deepcopy(record["witness"]),
                "first_root": copy.deepcopy(
                    record["witness"]["qualifying_roots"][0]
                ),
            }
            for record in records
            if record["blocked"]
        ],
        "cache": {
            "directory": str(cache_root),
            "schema_version": CACHE_SCHEMA_VERSION,
            "hit_count": cache_hits,
            "miss_count": cache_misses,
            "write_count": cache_writes,
            "physical_shard_read_count": physical_shard_reads,
            "shards": shard_audits,
        },
    }
    return frame, audit
