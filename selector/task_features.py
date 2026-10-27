"""Static task and verifier features used to construct the similarity graph."""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from typing import Any


def normalize_literal(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[a-z]{1,8}[-_]?\d{2,}", " id_token ", value)
    value = re.sub(r"\d{4}-\d{2}-\d{2}", " date_token ", value)
    value = re.sub(r"\b\d+(?:\.\d+)?\b", " number_token ", value)
    return re.sub(r"[^a-z_]+", " ", value).strip()


class VerifierVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.tokens: list[str] = []

    def generic_visit(self, node: ast.AST) -> None:
        self.tokens.append(f"node_{type(node).__name__.lower()}")
        super().generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = ""
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        elif isinstance(node.func, ast.Name):
            name = node.func.id
        if name:
            self.tokens.append(f"call_{name.lower()}")
        if name == "get" and node.args and isinstance(node.args[0], ast.Constant):
            field = normalize_literal(str(node.args[0].value)).replace(" ", "_")
            if field:
                self.tokens.append(f"field_{field}")
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        for operation in node.ops:
            self.tokens.append(f"compare_{type(operation).__name__.lower()}")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and len(node.value) <= 100:
            literal = normalize_literal(node.value)
            for token in literal.split()[:8]:
                if len(token) >= 2:
                    self.tokens.append(f"literal_{token}")
        self.generic_visit(node)


def verifier_text(scenario: dict[str, Any]) -> tuple[str, str, int, int]:
    natural: list[str] = []
    structural: list[str] = []
    node_count = 0
    code_chars = 0
    for check in scenario.get("checklist_with_func", []):
        natural.append(str(check.get("check_item", "")))
        code = str(check.get("check_func", ""))
        code_chars += len(code)
        try:
            tree = ast.parse(code)
        except SyntaxError:
            structural.append(re.sub(r"\s+", " ", code))
            continue
        visitor = VerifierVisitor()
        visitor.visit(tree)
        node_count += sum(1 for _ in ast.walk(tree))
        structural.extend(visitor.tokens)
    return " ".join(natural), " ".join(structural), node_count, code_chars


def state_signature(init_config: dict[str, Any], task: str) -> tuple[str, int, int]:
    task_tokens = {
        token
        for token in re.findall(r"[a-z0-9_-]{3,}", task.lower())
        if token not in {"that", "with", "from", "then", "only", "must", "after", "before"}
    }
    parts: list[str] = []
    total_records = 0
    table_count = 0
    for table, raw in sorted(init_config.items()):
        if isinstance(raw, dict):
            records = list(raw.values())
        elif isinstance(raw, list):
            records = raw
        else:
            parts.append(f"scalar_{table}")
            continue
        table_count += 1
        total_records += len(records)
        parts.extend([f"table_{table}", f"table_size_{min(len(records) // 5, 10)}"])
        fields: Counter[str] = Counter()
        enum_values: Counter[str] = Counter()
        ranked: list[tuple[int, int, Any]] = []
        for index, record in enumerate(records):
            if isinstance(record, dict):
                fields.update(map(str, record))
                for key, value in record.items():
                    key_lower = str(key).lower()
                    if any(marker in key_lower for marker in ("status", "state", "type", "role")):
                        normalized = normalize_literal(str(value)).replace(" ", "_")
                        if normalized:
                            enum_values[f"{key_lower}_{normalized}"] += 1
            serialized = json.dumps(record, ensure_ascii=True, sort_keys=True).lower()
            score = sum(token in serialized for token in task_tokens)
            ranked.append((score, -index, record))
        parts.extend(f"field_{field}" for field, _ in fields.most_common(40))
        parts.extend(f"enum_{value}" for value, _ in enum_values.most_common(30))
        for score, _, record in sorted(ranked, reverse=True)[:3]:
            if score <= 0:
                break
            relevant = normalize_literal(json.dumps(record, ensure_ascii=True, sort_keys=True))
            parts.append(f"relevant_{table} {relevant}")
    return " ".join(parts), table_count, total_records
