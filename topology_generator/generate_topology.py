#!/usr/bin/env python3
"""Extract static execution and verifier-reward topology for EnvScaler tasks.

The LLM receives only information available before rollout: task text, the
initial state, verifier definitions, tool schemas, and selected real tool
implementations. It must return factual graph structure and is explicitly not
asked to predict model success, reward, or reward variance.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import os
import re
import textwrap
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from topology_utils import (
    get_env_record,
    parse_json_content,
    safe_model_name,
    summarize_initial_state,
)
from topology_utils import initial_check_results


PROMPT_VERSION = "task-topology-v2-agent-visibility"
DEFAULT_OUTPUT = "outputs/topology"
DEFAULT_API_BASE = "https://api.openai.com/v1"

OPERATION_FAMILIES = {
    "lookup",
    "create",
    "update",
    "delete",
    "transition",
    "link",
    "aggregate",
    "schedule",
    "authorize",
    "validate",
    "report",
    "other",
}
GATE_TYPES = {
    "identifier_resolution",
    "generated_identifier",
    "actor_authorization",
    "state_precondition",
    "ordering",
    "argument_exactness",
    "temporal_conversion",
    "tool_affordance",
    "capacity_or_eligibility",
    "cross_entity_join",
    "termination_horizon",
    "other",
}
PROVENANCE_TYPES = {
    "task_literal",
    "unique_lookup",
    "broad_lookup",
    "prior_tool_output",
    "agent_generated",
    "wall_clock",
    "inferred",
    "unavailable",
    "none",
    "unknown",
}
REACHABILITY_TYPES = {"direct", "indirect", "ambiguous", "unsupported", "unknown"}
OBSERVABILITY_TYPES = {"explicit_result", "state_query", "silent", "unknown"}
RECOVERABILITY_TYPES = {"retryable", "irreversible", "unknown"}
REWARD_RELATIONS = {
    "shared_gate",
    "shared_operation",
    "independent",
    "initially_satisfied",
    "fixed_unreachable",
    "mixed",
}
CONFLICT_TYPES = {"order", "destructive", "mutual_exclusion", "preservation"}

TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9_]{1,}|[A-Za-z]{2,}")
GENERIC_TOKENS = {
    "and",
    "are",
    "data",
    "for",
    "from",
    "get",
    "has",
    "have",
    "into",
    "must",
    "only",
    "record",
    "return",
    "should",
    "state",
    "system",
    "that",
    "the",
    "then",
    "this",
    "tool",
    "true",
    "update",
    "user",
    "using",
    "with",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("tag", "export", "all"), default="all")
    parser.add_argument("--scenarios", required=True, help="Scenario JSON with tasks, initial states, and verifier functions")
    parser.add_argument("--env-metadata", required=True, help="Environment metadata JSON keyed by environment ID")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", help="Defaults to OUTPUT_DIR/cache; reuse it to label only new or changed tasks")
    parser.add_argument("--api-key-file", help="Optional key file; otherwise read OPENAI_API_KEY")
    parser.add_argument("--api-base", default=os.getenv("OPENAI_BASE_URL", DEFAULT_API_BASE))
    parser.add_argument("--model", required=True, help="Model ID exposed by your compatible chat-completions endpoint")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument(
        "--sweep-retries",
        type=int,
        default=3,
        help="Retry only batches that remain failed after a complete concurrent sweep.",
    )
    parser.add_argument("--max-tool-implementations", type=int, default=10)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and report cache coverage without calling the API")
    args = parser.parse_args(argv)
    args.cache_dir = args.cache_dir or str(Path(args.output_dir) / "cache")
    for name in ("workers", "batch_size", "max_retries", "sweep_retries", "max_tool_implementations"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def truncate(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) <= limit:
        return text
    head = max(int(limit * 0.68), 1)
    tail = max(limit - head - 5, 1)
    return f"{text[:head]} ... {text[-tail:]}"


def compact_source(source: Any, limit: int = 2600) -> str:
    raw = textwrap.dedent(str(source)).strip()
    if not raw:
        return ""
    try:
        tree = ast.parse(raw)
        if tree.body and isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
            function = tree.body[0]
            if (
                function.body
                and isinstance(function.body[0], ast.Expr)
                and isinstance(function.body[0].value, ast.Constant)
                and isinstance(function.body[0].value.value, str)
            ):
                function.body = function.body[1:] or [ast.Pass()]
            raw = ast.unparse(function)
    except (SyntaxError, ValueError):
        pass
    return truncate(raw, limit)


def lexical_tokens(value: Any) -> set[str]:
    expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(value)).replace("_", " ").lower()
    return {token for token in TOKEN_PATTERN.findall(expanded) if token not in GENERIC_TOKENS}


def signature_text(detail: dict[str, Any]) -> str:
    signature = detail.get("signature", {})
    parameters = signature.get("parameters", []) if isinstance(signature, dict) else []
    parts = []
    for parameter in parameters:
        if isinstance(parameter, dict):
            parts.append(f"{parameter.get('name')}:{parameter.get('type', 'unknown')}")
    return f"({', '.join(parts)}) -> {signature.get('return', 'unknown') if isinstance(signature, dict) else 'unknown'}"


def all_tool_names(env_record: dict[str, Any]) -> set[str]:
    names = set()
    for tool in env_record.get("tools", []):
        name = tool.get("function", {}).get("name")
        if name:
            names.add(str(name))
    details = env_record.get("env_func_details", {})
    if isinstance(details, dict):
        names.update(map(str, details))
    return names


def compact_tool_catalog(env_record: dict[str, Any]) -> list[dict[str, Any]]:
    schema_by_name = {}
    for tool in env_record.get("tools", []):
        function = tool.get("function", {})
        name = str(function.get("name", ""))
        if not name:
            continue
        parameters = function.get("parameters", {})
        schema_by_name[name] = {
            "name": name,
            "description": truncate(function.get("description", ""), 180),
            "required": list(parameters.get("required", [])),
            "arguments": sorted(parameters.get("properties", {})),
        }
    details = env_record.get("env_func_details", {})
    if isinstance(details, dict):
        for name, detail in details.items():
            if not isinstance(detail, dict):
                continue
            row = schema_by_name.setdefault(str(name), {"name": str(name)})
            row["signature"] = signature_text(detail)
            if not row.get("description"):
                row["description"] = truncate(detail.get("doc", ""), 180)
    return [schema_by_name[name] for name in sorted(schema_by_name)]


def relevant_tool_implementations(
    scenario: dict[str, Any], env_record: dict[str, Any], maximum: int
) -> list[dict[str, Any]]:
    details = env_record.get("env_func_details", {})
    if not isinstance(details, dict):
        return []
    query = " ".join(
        [str(scenario.get("task", ""))]
        + [str(item.get("check_item", "")) for item in scenario.get("checklist_with_func", [])]
        + [str(item.get("check_func", "")) for item in scenario.get("checklist_with_func", [])]
    )
    query_tokens = lexical_tokens(query)
    ranked = []
    for name, detail in details.items():
        if not isinstance(detail, dict):
            continue
        name_tokens = lexical_tokens(name)
        doc_tokens = lexical_tokens(detail.get("doc", ""))
        source_tokens = lexical_tokens(detail.get("source_code", ""))
        score = 8 * len(name_tokens & query_tokens)
        score += 2 * len(doc_tokens & query_tokens)
        score += min(len(source_tokens & query_tokens), 12)
        if str(name).lower() in query.lower():
            score += 20
        ranked.append((score, str(name), detail))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = ranked[: max(maximum, 1)]
    return [
        {
            "name": name,
            "signature": signature_text(detail),
            "documentation": truncate(detail.get("doc", ""), 360),
            "implementation": compact_source(detail.get("source_code", ""), 1500),
        }
        for _, name, detail in selected
    ]


def shared_environment_context(env_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "environment": env_record.get("env_class_name"),
        "summary": truncate(env_record.get("environment_summary", ""), 650),
        "state_space": truncate(env_record.get("state_space_definition", ""), 750),
        "constraints": truncate(env_record.get("constraints_rules", ""), 1200),
        "tool_catalog": compact_tool_catalog(env_record),
    }


def task_payload(
    scenario: dict[str, Any], env_record: dict[str, Any], max_tool_implementations: int
) -> dict[str, Any]:
    checks = []
    initial = initial_check_results(scenario)
    for index, check in enumerate(scenario.get("checklist_with_func", [])):
        checks.append(
            {
                "check_id": f"c{index}",
                "requirement": str(check.get("check_item", "")),
                "initially_true": bool(initial[index]) if index < len(initial) else False,
                "verifier_code": compact_source(check.get("check_func", ""), 1100),
            }
        )
    return {
        "task_id": str(scenario["task_id"]),
        "task": str(scenario.get("task", "")).strip(),
        "initial_state_summary": summarize_initial_state(
            scenario.get("init_config", {}), str(scenario.get("task", ""))
        ),
        "verifier_checks": checks,
        "relevant_tool_implementations": relevant_tool_implementations(
            scenario, env_record, max_tool_implementations
        ),
    }


def system_prompt() -> str:
    return f"""You extract factual static topology from tool-agent tasks. You receive no rollout,
trajectory, action trace, model output, reward observation, success rate, or variance. Do not predict
any of them and do not assign difficulty, probability, capability, or high/low labels.

Build two connected structures from the supplied task, initial state, exact verifier checks, tool
catalog, and selected real tool implementations:
1. execution topology: the minimum correct operations, their dependencies, and factual prerequisite
   gates such as identifier provenance, state preconditions, ordering, authorization, exact arguments,
   and missing tool affordances;
2. reward topology: a partition of verifier check IDs into groups that are controlled by the same
   operation or prerequisite gate. Check weights are not yours to estimate; downstream code computes
   them from the exact check IDs.

A gate is a prerequisite whose resolution can jointly enable or block one or more operations/checks.
Do not create a gate for every ordinary literal. Distinguish a direct task literal, unique lookup,
broad lookup, prior tool output, agent-generated value, and genuinely unavailable value. Mark
reachability unsupported only when the supplied tools/implementation establish that no path exists;
otherwise use unknown. Describe topology, not likely model behavior.

The initial_state_summary is evidence available to this static analyzer, not information shown to the
agent. Never mark an identifier direct merely because it appears in that summary. From the agent's
execution perspective, an identifier is task_literal only when the task supplies the execution-ready
value; otherwise classify the actual tool path as unique_lookup, broad_lookup, prior_tool_output,
inferred, or unavailable. Use the initial state only to verify that records, matches, and preconditions
exist and to determine whether a supplied lookup can resolve them.

Use these enums exactly:
- operation.family: {sorted(OPERATION_FAMILIES)}
- gate.type: {sorted(GATE_TYPES)}
- gate.provenance: {sorted(PROVENANCE_TYPES)}
- gate.reachability: {sorted(REACHABILITY_TYPES)}
- gate.observability: {sorted(OBSERVABILITY_TYPES)}
- gate.recoverability: {sorted(RECOVERABILITY_TYPES)}
- reward_group.relation: {sorted(REWARD_RELATIONS)}
- conflict.kind: {sorted(CONFLICT_TYPES)}

Return JSON with key "tasks". Every task item must have exactly this structure:
{{
  "task_id": "...",
  "operations": [
    {{"id":"o0","family":"lookup","description":"...","tool_candidates":["..."],
      "depends_on_operations":[],"gate_ids":["g0"],"affected_checks":["c0"]}}
  ],
  "gates": [
    {{"id":"g0","type":"identifier_resolution","description":"...",
      "provenance":"unique_lookup","reachability":"indirect",
      "observability":"explicit_result","recoverability":"retryable",
      "depends_on_gates":[],"affected_operations":["o0"],"affected_checks":["c0"]}}
  ],
  "reward_groups": [
    {{"id":"r0","check_ids":["c0"],"shared_gate_ids":["g0"],
      "producer_operations":["o0"],"relation":"shared_gate"}}
  ],
  "conflicts": [{{"left_operation":"o0","right_operation":"o1","kind":"order"}}],
  "independent_branches": [
    {{"id":"b0","operation_ids":["o0"],"check_ids":["c0"]}}
  ],
  "uncertainties": ["short factual ambiguity in the supplied static evidence"]
}}

Rules:
- Use only supplied check IDs and tool names.
- reward_groups must partition every verifier check exactly once. Group checks only when they share a
  concrete gate/producer; otherwise use singleton or independent groups.
- A check already true in the untouched initial state belongs in an initially_satisfied group unless
  later required operations can invalidate it.
- Operations describe the shortest correct state-changing/query plan, not speculative retries.
- IDs must be local sequential identifiers o0..., g0..., r0..., b0....
- Keep descriptions concise and case-specific. At most 30 operations, 30 gates, 30 reward groups,
  20 conflicts, 20 branches, and 5 uncertainties per task.
- Return JSON only."""


def clean_enum(value: Any, allowed: set[str], default: str) -> str:
    text = str(value)
    return text if text in allowed else default


def clean_refs(raw: Any, mapping: dict[str, str]) -> list[str]:
    values = raw if isinstance(raw, list) else []
    return sorted({mapping[str(value)] for value in values if str(value) in mapping})


def validate_topology(
    raw: dict[str, Any], scenario: dict[str, Any], env_record: dict[str, Any]
) -> dict[str, Any]:
    task_id = str(scenario["task_id"])
    if str(raw.get("task_id")) != task_id:
        raise ValueError(f"Unexpected task_id {raw.get('task_id')!r}, wanted {task_id!r}")
    valid_checks = {
        f"c{index}" for index, _ in enumerate(scenario.get("checklist_with_func", []))
    }
    valid_tools = all_tool_names(env_record)

    raw_operations = [item for item in raw.get("operations", []) if isinstance(item, dict)][:30]
    operation_map = {
        str(item.get("id", f"o{index}")): f"o{index}"
        for index, item in enumerate(raw_operations)
    }
    raw_gates = [item for item in raw.get("gates", []) if isinstance(item, dict)][:30]
    gate_map = {
        str(item.get("id", f"g{index}")): f"g{index}" for index, item in enumerate(raw_gates)
    }

    operations = []
    for index, item in enumerate(raw_operations):
        tools = item.get("tool_candidates", [])
        operations.append(
            {
                "id": f"o{index}",
                "family": clean_enum(item.get("family"), OPERATION_FAMILIES, "other"),
                "description": truncate(item.get("description", ""), 220),
                "tool_candidates": sorted(
                    {str(value) for value in tools if str(value) in valid_tools}
                )[:8],
                "depends_on_operations": clean_refs(
                    item.get("depends_on_operations"), operation_map
                ),
                "gate_ids": clean_refs(item.get("gate_ids"), gate_map),
                "affected_checks": sorted(
                    {str(value) for value in item.get("affected_checks", [])}
                    & valid_checks
                ),
            }
        )

    gates = []
    for index, item in enumerate(raw_gates):
        gates.append(
            {
                "id": f"g{index}",
                "type": clean_enum(item.get("type"), GATE_TYPES, "other"),
                "description": truncate(item.get("description", ""), 220),
                "provenance": clean_enum(
                    item.get("provenance"), PROVENANCE_TYPES, "unknown"
                ),
                "reachability": clean_enum(
                    item.get("reachability"), REACHABILITY_TYPES, "unknown"
                ),
                "observability": clean_enum(
                    item.get("observability"), OBSERVABILITY_TYPES, "unknown"
                ),
                "recoverability": clean_enum(
                    item.get("recoverability"), RECOVERABILITY_TYPES, "unknown"
                ),
                "depends_on_gates": clean_refs(item.get("depends_on_gates"), gate_map),
                "affected_operations": clean_refs(
                    item.get("affected_operations"), operation_map
                ),
                "affected_checks": sorted(
                    {str(value) for value in item.get("affected_checks", [])}
                    & valid_checks
                ),
            }
        )

    # The model may state a gate-operation edge on only one endpoint. Make the
    # validated graph bidirectionally consistent without inventing a new edge.
    operations_by_id = {item["id"]: item for item in operations}
    gates_by_id = {item["id"]: item for item in gates}
    for operation in operations:
        for gate_id in operation["gate_ids"]:
            gate = gates_by_id[gate_id]
            if operation["id"] not in gate["affected_operations"]:
                gate["affected_operations"].append(operation["id"])
    for gate in gates:
        gate["affected_operations"] = sorted(set(gate["affected_operations"]))
        for operation_id in gate["affected_operations"]:
            operation = operations_by_id[operation_id]
            if gate["id"] not in operation["gate_ids"]:
                operation["gate_ids"].append(gate["id"])
    for operation in operations:
        operation["gate_ids"] = sorted(set(operation["gate_ids"]))

    raw_groups = [item for item in raw.get("reward_groups", []) if isinstance(item, dict)][:30]
    assigned: set[str] = set()
    reward_groups = []
    for item in raw_groups:
        checks = sorted(
            ({str(value) for value in item.get("check_ids", [])} & valid_checks) - assigned
        )
        if not checks:
            continue
        assigned.update(checks)
        reward_groups.append(
            {
                "id": f"r{len(reward_groups)}",
                "check_ids": checks,
                "shared_gate_ids": clean_refs(item.get("shared_gate_ids"), gate_map),
                "producer_operations": clean_refs(
                    item.get("producer_operations"), operation_map
                ),
                "relation": clean_enum(
                    item.get("relation"), REWARD_RELATIONS, "mixed"
                ),
            }
        )
    initially_true = initial_check_results(scenario)
    for check_id in sorted(valid_checks - assigned, key=lambda value: int(value[1:])):
        check_index = int(check_id[1:])
        relation = (
            "initially_satisfied"
            if check_index < len(initially_true) and initially_true[check_index]
            else "independent"
        )
        reward_groups.append(
            {
                "id": f"r{len(reward_groups)}",
                "check_ids": [check_id],
                "shared_gate_ids": [],
                "producer_operations": [],
                "relation": relation,
            }
        )

    conflicts = []
    seen_conflicts = set()
    for item in raw.get("conflicts", []):
        if not isinstance(item, dict) or len(conflicts) >= 20:
            continue
        left = operation_map.get(str(item.get("left_operation")))
        right = operation_map.get(str(item.get("right_operation")))
        if not left or not right or left == right:
            continue
        kind = clean_enum(item.get("kind"), CONFLICT_TYPES, "order")
        key = (min(left, right), max(left, right), kind)
        if key in seen_conflicts:
            continue
        seen_conflicts.add(key)
        conflicts.append({"left_operation": left, "right_operation": right, "kind": kind})

    branches = []
    for item in raw.get("independent_branches", []):
        if not isinstance(item, dict) or len(branches) >= 20:
            continue
        operation_ids = clean_refs(item.get("operation_ids"), operation_map)
        check_ids = sorted(
            {str(value) for value in item.get("check_ids", [])} & valid_checks
        )
        if not operation_ids and not check_ids:
            continue
        branches.append(
            {
                "id": f"b{len(branches)}",
                "operation_ids": operation_ids,
                "check_ids": check_ids,
            }
        )

    uncertainties = [
        truncate(value, 220)
        for value in raw.get("uncertainties", [])
        if str(value).strip()
    ][:5]
    return {
        "task_id": task_id,
        "operations": operations,
        "gates": gates,
        "reward_groups": reward_groups,
        "conflicts": conflicts,
        "independent_branches": branches,
        "uncertainties": uncertainties,
    }


def input_hash(
    scenario: dict[str, Any], env_record: dict[str, Any], args: argparse.Namespace
) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "model": args.model,
        "environment": shared_environment_context(env_record),
        "task": task_payload(scenario, env_record, args.max_tool_implementations),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def cache_path(args: argparse.Namespace, task_id: str) -> Path:
    if task_id in {"", ".", ".."} or any(char in task_id for char in ("/", "\\", "\0")):
        raise ValueError("task_id must be a non-empty filename-safe identifier")
    return Path(args.cache_dir) / safe_model_name(args.model) / f"{task_id}.json"


def read_cache(
    args: argparse.Namespace, scenario: dict[str, Any], env_record: dict[str, Any]
) -> dict[str, Any] | None:
    path = cache_path(args, str(scenario["task_id"]))
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
        if record.get("prompt_version") != PROMPT_VERSION or record.get("model") != args.model:
            return None
        if record.get("input_hash") != input_hash(scenario, env_record, args):
            return None
        return validate_topology(record["topology"], scenario, env_record)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def write_cache(
    args: argparse.Namespace,
    scenario: dict[str, Any],
    env_record: dict[str, Any],
    topology: dict[str, Any],
) -> None:
    path = cache_path(args, str(scenario["task_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "prompt_version": PROMPT_VERSION,
        "model": args.model,
        "input_hash": input_hash(scenario, env_record, args),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "topology": topology,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def call_batch(
    scenarios: list[dict[str, Any]],
    env_record: dict[str, Any],
    args: argparse.Namespace,
    api_key: str,
) -> dict[str, dict[str, Any]]:
    from openai import OpenAI

    expected = {str(scenario["task_id"]): scenario for scenario in scenarios}
    payload = {
        "shared_environment": shared_environment_context(env_record),
        "tasks": [
            task_payload(scenario, env_record, args.max_tool_implementations)
            for scenario in scenarios
        ],
    }
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
    ]
    last_error: Exception | None = None
    for attempt in range(args.max_retries):
        started = time.monotonic()
        try:
            client = OpenAI(api_key=api_key, base_url=args.api_base, timeout=600)
            stream = client.chat.completions.create(
                model=args.model,
                messages=messages,
                response_format={"type": "json_object"},
                stream=True,
            )
            pieces = []
            for chunk in stream:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if content:
                    pieces.append(content)
            parsed = parse_json_content("".join(pieces))
            items = parsed.get("tasks")
            if not isinstance(items, list):
                raise ValueError("Response does not contain a tasks list")
            by_id = {str(item.get("task_id")): item for item in items if isinstance(item, dict)}
            if set(by_id) != set(expected):
                raise ValueError(
                    f"Response task IDs differ: got {sorted(by_id)}, wanted {sorted(expected)}"
                )
            return {
                task_id: validate_topology(by_id[task_id], scenario, env_record)
                for task_id, scenario in expected.items()
            }
        except Exception as exc:
            last_error = exc
            elapsed = time.monotonic() - started
            print(
                f"Topology API attempt {attempt + 1}/{args.max_retries} failed after {elapsed:.1f}s "
                f"model={args.model} tasks={','.join(sorted(expected))}: "
                f"{type(exc).__name__}: {truncate(exc, 320)}",
                flush=True,
            )
            if attempt + 1 < args.max_retries:
                time.sleep(min(20, 2**attempt))
    raise RuntimeError(f"Topology extraction failed after {args.max_retries} attempts: {last_error}")


def call_with_fallback(
    scenarios: list[dict[str, Any]],
    env_record: dict[str, Any],
    args: argparse.Namespace,
    api_key: str,
) -> dict[str, dict[str, Any]]:
    try:
        return call_batch(scenarios, env_record, args, api_key)
    except Exception:
        if len(scenarios) <= 1:
            raise
        midpoint = len(scenarios) // 2
        left = call_with_fallback(scenarios[:midpoint], env_record, args, api_key)
        right = call_with_fallback(scenarios[midpoint:], env_record, args, api_key)
        return {**left, **right}


def load_selected(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    scenarios = json.loads(Path(args.scenarios).read_text())
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("--scenarios must contain a non-empty JSON list")
    required = {"task_id", "env_id", "env_class_name", "task", "init_config", "checklist_with_func"}
    seen = set()
    for item in scenarios:
        if not isinstance(item, dict) or required - item.keys():
            raise ValueError(f"Each scenario requires: {', '.join(sorted(required))}")
        task_id = str(item["task_id"])
        cache_path(args, task_id)
        if task_id in seen:
            raise ValueError(f"Duplicate task_id: {task_id}")
        seen.add(task_id)
        if not isinstance(item["checklist_with_func"], list) or not item["checklist_with_func"]:
            raise ValueError(f"Task {task_id} needs at least one verifier check")
    selected = scenarios
    selected.sort(key=lambda item: str(item["task_id"]))
    if args.task_id:
        requested = set(args.task_id)
        selected = [item for item in selected if str(item["task_id"]) in requested]
        missing = requested - {str(item["task_id"]) for item in selected}
        if missing:
            raise KeyError(f"Requested task IDs are unavailable: {sorted(missing)}")
    if args.limit is not None:
        selected = selected[: max(args.limit, 0)]
    env_records = json.loads(Path(args.env_metadata).read_text())
    if not isinstance(env_records, dict) or not env_records:
        raise ValueError("--env-metadata must contain a non-empty JSON object")
    for scenario in selected:
        get_env_record(env_records, scenario)
    return selected, env_records


def run_tagging(
    args: argparse.Namespace,
    scenarios: list[dict[str, Any]],
    env_records: dict[str, dict[str, Any]],
) -> None:
    pending_by_env: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cached = 0
    for scenario in scenarios:
        env_record = get_env_record(env_records, scenario)
        if read_cache(args, scenario, env_record) is None:
            pending_by_env[str(scenario["env_id"])].append(scenario)
        else:
            cached += 1
    batches: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
    for env_scenarios in pending_by_env.values():
        env_record = get_env_record(env_records, env_scenarios[0])
        batches.extend(
            (batch, env_record)
            for batch in chunks(env_scenarios, max(args.batch_size, 1))
        )
    print(
        f"Topology extraction: total={len(scenarios)} cached={cached} "
        f"pending={len(scenarios) - cached} batches={len(batches)} model={args.model}",
        flush=True,
    )
    if not batches:
        return
    api_key = (
        Path(args.api_key_file).expanduser().read_text().strip()
        if args.api_key_file
        else os.getenv("OPENAI_API_KEY", "").strip()
    )
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY or provide --api-key-file before tagging uncached tasks")
    completed = cached
    remaining = batches
    failure_rows: list[dict[str, Any]] = []
    sweep_count = max(args.sweep_retries, 1)
    for sweep in range(1, sweep_count + 1):
        failures: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
        failure_rows = []
        print(
            f"Topology sweep {sweep}/{sweep_count}: batches={len(remaining)} ",
            f"workers={max(args.workers, 1)}",
            flush=True,
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(args.workers, 1)
        ) as executor:
            futures = {
                executor.submit(
                    call_with_fallback, batch, env_record, args, api_key
                ): (batch, env_record)
                for batch, env_record in remaining
            }
            for future in concurrent.futures.as_completed(futures):
                batch, env_record = futures[future]
                try:
                    topologies = future.result()
                    for scenario in batch:
                        task_id = str(scenario["task_id"])
                        write_cache(
                            args, scenario, env_record, topologies[task_id]
                        )
                    completed += len(batch)
                    print(f"Extracted {completed}/{len(scenarios)}", flush=True)
                except Exception as exc:
                    failures.append((batch, env_record))
                    failure_rows.append(
                        {
                            "task_ids": [str(item["task_id"]) for item in batch],
                            "environment": str(batch[0].get("env_id", "unknown")),
                            "error_type": type(exc).__name__,
                            "error": truncate(exc, 1000),
                            "sweep": sweep,
                        }
                    )
                    print(
                        f"Topology batch isolated after sweep {sweep}: "
                        f"tasks={','.join(failure_rows[-1]['task_ids'])} "
                        f"error={type(exc).__name__}: {truncate(exc, 320)}",
                        flush=True,
                    )
        remaining = failures
        if not remaining:
            break
        if sweep < sweep_count:
            print(
                f"Retrying {sum(len(batch) for batch, _ in remaining)} failed "
                f"tasks in sweep {sweep + 1}/{sweep_count}",
                flush=True,
            )
            time.sleep(min(20, 2**sweep))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_report = {
        "failed_batch_count": len(remaining),
        "failed_task_count": sum(len(batch) for batch, _ in remaining),
        "completed_task_count": completed,
        "total_task_count": len(scenarios),
        "failures": failure_rows,
    }
    report_path = output_dir / "tag_failures.json"
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(failure_report, ensure_ascii=False, indent=2) + "\n"
    )
    temporary.replace(report_path)
    if remaining:
        raise RuntimeError(
            f"Topology extraction left {failure_report['failed_task_count']} failed "
            f"tasks after {sweep_count} sweeps; see {report_path}"
        )


def export_topologies(
    args: argparse.Namespace,
    scenarios: list[dict[str, Any]],
    env_records: dict[str, dict[str, Any]],
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_rows = []
    summary_rows = []
    missing = []
    partition_violations = []
    asymmetric_gate_operation_edges = []
    reachability_counts: dict[str, int] = defaultdict(int)
    provenance_counts: dict[str, int] = defaultdict(int)
    reward_relation_counts: dict[str, int] = defaultdict(int)
    for scenario in scenarios:
        env_record = get_env_record(env_records, scenario)
        topology = read_cache(args, scenario, env_record)
        if topology is None:
            missing.append(str(scenario["task_id"]))
            continue
        record = {
            "task_id": str(scenario["task_id"]),
            "env_id": str(scenario["env_id"]),
            "env_class_name": str(scenario["env_class_name"]),
            "topology": topology,
        }
        json_rows.append(record)
        expected_checks = {
            f"c{index}" for index, _ in enumerate(scenario.get("checklist_with_func", []))
        }
        grouped_checks = [
            str(check_id)
            for group in topology["reward_groups"]
            for check_id in group["check_ids"]
        ]
        if set(grouped_checks) != expected_checks or len(grouped_checks) != len(
            set(grouped_checks)
        ):
            partition_violations.append(record["task_id"])
        operations_by_id = {item["id"]: item for item in topology["operations"]}
        gates_by_id = {item["id"]: item for item in topology["gates"]}
        for operation in topology["operations"]:
            for gate_id in operation["gate_ids"]:
                if operation["id"] not in gates_by_id[gate_id]["affected_operations"]:
                    asymmetric_gate_operation_edges.append(record["task_id"])
        for gate in topology["gates"]:
            for operation_id in gate["affected_operations"]:
                if gate["id"] not in operations_by_id[operation_id]["gate_ids"]:
                    asymmetric_gate_operation_edges.append(record["task_id"])
            reachability_counts[gate["reachability"]] += 1
            provenance_counts[gate["provenance"]] += 1
        for group in topology["reward_groups"]:
            reward_relation_counts[group["relation"]] += 1
        summary_rows.append(
            {
                "task_id": record["task_id"],
                "env_id": record["env_id"],
                "env_class_name": record["env_class_name"],
                "operation_count": len(topology["operations"]),
                "gate_count": len(topology["gates"]),
                "reward_group_count": len(topology["reward_groups"]),
                "conflict_count": len(topology["conflicts"]),
                "independent_branch_count": len(topology["independent_branches"]),
                "uncertainty_count": len(topology["uncertainties"]),
                "operation_families": "|".join(
                    sorted({item["family"] for item in topology["operations"]})
                ),
                "gate_types": "|".join(sorted({item["type"] for item in topology["gates"]})),
                "gate_provenance": "|".join(
                    sorted({item["provenance"] for item in topology["gates"]})
                ),
                "gate_reachability": "|".join(
                    sorted({item["reachability"] for item in topology["gates"]})
                ),
            }
        )
    if missing:
        raise FileNotFoundError(f"Missing valid topology cache for {len(missing)} tasks: {missing[:5]}")
    jsonl_path = output_dir / "task_topologies.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in json_rows)
    )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "task_topology_summary.csv", index=False
    )
    audit = {
        "prompt_version": PROMPT_VERSION,
        "model": args.model,
        "task_count": len(json_rows),
        "included_static_inputs": [
            "task text",
            "initial-state summary",
            "verifier requirements and code",
            "initial verifier truth flags",
            "environment description and rules",
            "tool schema",
            "selected real tool implementations",
        ],
        "excluded_inputs": [
            "rollout action or trajectory",
            "rollout reward",
            "success rate",
            "reward variance",
            "model-generated outcome or pilot",
            "direct LLM variance or success score",
        ],
        "topology_fields": [
            "operations",
            "gates",
            "reward_groups",
            "conflicts",
            "independent_branches",
            "uncertainties",
        ],
    }
    (output_dir / "extraction_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    )
    summary_frame = pd.DataFrame(summary_rows)
    quality = {
        "task_count": len(json_rows),
        "partition_violation_count": len(set(partition_violations)),
        "partition_violation_tasks": sorted(set(partition_violations)),
        "asymmetric_gate_operation_edge_task_count": len(
            set(asymmetric_gate_operation_edges)
        ),
        "asymmetric_gate_operation_edge_tasks": sorted(
            set(asymmetric_gate_operation_edges)
        ),
        "empty_operation_task_count": int((summary_frame["operation_count"] == 0).sum()),
        "empty_gate_task_count": int((summary_frame["gate_count"] == 0).sum()),
        "empty_reward_group_task_count": int(
            (summary_frame["reward_group_count"] == 0).sum()
        ),
        "count_distributions": {
            column: {
                "min": int(summary_frame[column].min()),
                "median": float(summary_frame[column].median()),
                "mean": float(summary_frame[column].mean()),
                "max": int(summary_frame[column].max()),
            }
            for column in (
                "operation_count",
                "gate_count",
                "reward_group_count",
                "conflict_count",
                "independent_branch_count",
                "uncertainty_count",
            )
        },
        "gate_reachability_counts": dict(sorted(reachability_counts.items())),
        "gate_provenance_counts": dict(sorted(provenance_counts.items())),
        "reward_relation_counts": dict(sorted(reward_relation_counts.items())),
    }
    (output_dir / "topology_quality_audit.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Exported {len(json_rows)} topologies to {jsonl_path}", flush=True)


def main() -> None:
    args = parse_args()
    scenarios, env_records = load_selected(args)
    if args.dry_run:
        cached = sum(
            read_cache(args, scenario, get_env_record(env_records, scenario)) is not None
            for scenario in scenarios
        )
        print(json.dumps({
            "tasks": len(scenarios), "cached": cached, "pending": len(scenarios) - cached,
            "output": str(Path(args.output_dir) / "task_topologies.jsonl"),
            "api_called": False,
        }, indent=2))
        return
    if args.stage in {"tag", "all"}:
        run_tagging(args, scenarios, env_records)
    if args.stage in {"export", "all"}:
        export_topologies(args, scenarios, env_records)


if __name__ == "__main__":
    main()
