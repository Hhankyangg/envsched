"""Static input helpers extracted from the EnvSched research implementation."""
from __future__ import annotations

import copy
import json
import re
from typing import Any

def get_env_record(env_records: dict[str, dict[str, Any]], scenario: dict[str, Any]) -> dict[str, Any]:
    env_id = scenario["env_id"]
    if env_id in env_records:
        return env_records[env_id]
    matches = [record for record in env_records.values() if record.get("env_class_name") == scenario["env_class_name"]]
    if not matches:
        raise KeyError(f"No environment metadata for {env_id} / {scenario['env_class_name']}")
    return matches[0]


def truncate(text: Any, limit: int) -> str:
    value = re.sub(r"\s+", " ", str(text)).strip()
    return value if len(value) <= limit else value[: limit - 3] + "..."


def task_tokens(task: str) -> set[str]:
    stop = {
        "about", "after", "before", "been", "being", "could", "each", "from", "have",
        "into", "must", "only", "should", "that", "their", "then", "there", "these",
        "this", "those", "through", "under", "using", "with", "would", "your",
    }
    return {token for token in re.findall(r"[a-z0-9_-]{3,}", task.lower()) if token not in stop}


def record_relevance(record: Any, tokens: set[str]) -> int:
    serialized = json.dumps(record, ensure_ascii=False, sort_keys=True).lower()
    return sum(token in serialized for token in tokens)


def summarize_initial_state(init_config: dict[str, Any], task: str) -> dict[str, Any]:
    tokens = task_tokens(task)
    summary: dict[str, Any] = {}
    for name, value in init_config.items():
        if name == "init_config":
            continue
        if isinstance(value, dict):
            records = list(value.values())
            keys = list(value.keys())
        elif isinstance(value, list):
            records = value
            keys = []
        else:
            summary[name] = {"type": type(value).__name__, "value": truncate(value, 160)}
            continue
        fields: set[str] = set()
        for record in records[:10]:
            if isinstance(record, dict):
                fields.update(str(key) for key in record)
        ranked = sorted(
            ((record_relevance(record, tokens), index, record) for index, record in enumerate(records)),
            reverse=True,
            key=lambda item: (item[0], -item[1]),
        )
        relevant = []
        for score, _, record in ranked[:2]:
            if score <= 0:
                break
            relevant.append(json.loads(json.dumps(record, ensure_ascii=False, default=str)))
        summary[name] = {
            "count": len(records),
            "key_examples": keys[:4],
            "fields": sorted(fields)[:24],
            "task_matching_examples": relevant,
        }
    serialized = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    if len(serialized) <= 6500:
        return summary
    for item in summary.values():
        if isinstance(item, dict):
            item["task_matching_examples"] = []
    return summary


def parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def safe_model_name(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model)


def initial_check_results(scenario: dict[str, Any]) -> list[bool]:
    results = []
    for check in scenario.get("checklist_with_func", []):
        namespace: dict[str, Any] = {}
        try:
            exec(str(check.get("check_func", "")), namespace)
            result = bool(namespace["check_func"](copy.deepcopy(scenario.get("init_config", {}))))
        except Exception:
            result = False
        results.append(result)
    return results
