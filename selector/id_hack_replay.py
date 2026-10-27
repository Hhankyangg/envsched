"""Topology-free rollout replay for the online rewarded-ID-hack guard.

This module intentionally does not alter training rewards or selector state.  It
replays one saved rollout against the executable EnvScaler environment, derives
entity provenance from real pre/post state, and then runs fixed-action
counterfactuals that omit one suspicious root and its tainted-ID descendants.

Three mechanisms are separated:

* ``shadow_entity``: a newly created entity duplicates an existing entity's
  normalized natural name.  This is a model-side, penalizable candidate.
* ``unseen_existing_reference``: a successful state mutation uses an existing
  primary key that was absent from the prompt and all earlier safe read
  results.  This is a model-side, penalizable candidate.
* ``canonical_id_collision``: a newly created primary key differs from an
  existing key only by case/separators.  This is reported as an environment
  design issue and is never included in the penalizable counterfactual.

The audit abstains unless raw replay exactly reproduces the saved episode
reward.  No LLM-labelled task topology is read or trusted.
"""

from __future__ import annotations

import ast
import copy
import json
import re
import types
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

TURN_RE = re.compile(r"<\|im_start\|>(assistant|user)\n")

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)

THINK_RE = re.compile(
    r"(?:<|@)think(?:>|@)\s*(.*?)(?:<|@)/think(?:>|@)",
    re.DOTALL | re.IGNORECASE,
)

ID_FIELD_RE = re.compile(r"(?:^id$|_id$|_uuid$|_code$)", re.IGNORECASE)

ID_ARGUMENT_RE = re.compile(
    r"(?:^ids?$|_ids?$|^uuids?$|_uuids?$|^codes?$|_codes?$)", re.IGNORECASE
)

NAME_FIELDS = ("name", "full_name", "display_name", "title")


@dataclass(frozen=True)
class ParsedAction:
    """One tool action actually executable under the official first-call rule."""

    index: int
    assistant_turn: int
    name: str
    arguments: Mapping[str, Any]


def _assistant_turns(response: str) -> Iterable[tuple[int, str]]:
    text = response
    if not text.startswith("<|im_start|>"):
        text = "<|im_start|>assistant\n" + text
    matches = list(TURN_RE.finditer(text))
    assistant_turn = 0
    for pos, match in enumerate(matches):
        end = matches[pos + 1].start() if pos + 1 < len(matches) else len(text)
        if match.group(1) != "assistant":
            continue
        assistant_turn += 1
        yield assistant_turn, text[match.end() : end]


def parse_actions(response: str) -> list[ParsedAction]:
    """Extract exactly the first tool call from each assistant turn.

    A text-only assistant turn is the non-conversation environment's terminal
    action, so later material (if any) is ignored.  Malformed turns are kept as
    no-ops, matching the official parser closely enough for replay parity to be
    the final authority.
    """

    actions: list[ParsedAction] = []
    for assistant_turn, body in _assistant_turns(response):
        has_open_think = bool(re.search(r"(?:<|@)think(?:>|@)", body, re.I))
        if has_open_think and not THINK_RE.search(body):
            continue
        match = TOOL_CALL_RE.search(body)
        if match is None:
            content = THINK_RE.sub("", body)
            content = content.replace("<|im_end|>", "").strip()
            if content:
                break
            continue
        try:
            call = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = call.get("name")
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            decoded_arguments: Any = None
            try:
                decoded_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                try:
                    decoded_arguments = ast.literal_eval(arguments)
                except (SyntaxError, ValueError):
                    pass
            arguments = decoded_arguments
        if not isinstance(name, str) or not isinstance(arguments, Mapping):
            continue
        actions.append(
            ParsedAction(
                index=len(actions) + 1,
                assistant_turn=assistant_turn,
                name=name,
                arguments=copy.deepcopy(dict(arguments)),
            )
        )
    return actions


def _load_metadata(
    task_path: Path | str, env_path: Path | str
) -> tuple[dict[str, Mapping[str, Any]], Mapping[str, Mapping[str, Any]]]:
    with Path(task_path).open(encoding="utf-8") as handle:
        tasks = json.load(handle)
    with Path(env_path).open(encoding="utf-8") as handle:
        envs = json.load(handle)
    return {str(item["task_id"]): item for item in tasks}, envs


def _new_env(task: Mapping[str, Any], env_item: Mapping[str, Any]) -> Any:
    """Instantiate the dynamic environment exactly like official env_util.py."""

    module = types.ModuleType("id_hack_replay_dynamic_env")
    exec(str(env_item["env_class_code"]), module.__dict__)
    env_class = getattr(module, str(task["env_class_name"]))
    init_config = copy.deepcopy(task["init_config"])
    try:
        instance = env_class(init_config if isinstance(init_config, dict) else {})
    except TypeError:
        instance = env_class()
    if init_config:
        for key, value in init_config.items():
            setattr(instance, key, copy.deepcopy(value))
    return instance


def _state(instance: Any) -> dict[str, Any]:
    return copy.deepcopy(
        {
            key: value
            for key, value in vars(instance).items()
            if not (key.startswith("__") and key.endswith("__"))
        }
    )


def _is_id_field(name: object) -> bool:
    return isinstance(name, str) and bool(ID_FIELD_RE.search(name))


def entity_index(state: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    """Find top-level entity maps whose records carry their mapping primary key."""

    output: dict[tuple[str, str], Mapping[str, Any]] = {}
    for collection, records in state.items():
        if collection == "init_config" or not isinstance(records, Mapping):
            continue
        for primary, record in records.items():
            if not isinstance(record, Mapping):
                continue
            if any(
                _is_id_field(field) and str(value) == str(primary)
                for field, value in record.items()
            ):
                output[(str(collection), str(primary))] = record
    return output


def _flatten_scalars(value: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from _flatten_scalars(child, child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _flatten_scalars(child, f"{path}[{index}]")
    else:
        yield path, value


def _string_values(value: Any) -> set[str]:
    """Return every visible string, including strings used as mapping keys."""

    output: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key:
                output.add(key)
            output.update(_string_values(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            output.update(_string_values(child))
    elif isinstance(value, str) and value:
        output.add(value)
    return output


def _id_argument_values(arguments: Mapping[str, Any]) -> set[str]:
    output: set[str] = set()
    for path, value in _flatten_scalars(arguments):
        if not isinstance(value, str) or not value:
            continue
        leaf = path.rsplit(".", 1)[-1]
        leaf = re.sub(r"\[\d+\]$", "", leaf)
        if ID_ARGUMENT_RE.search(leaf):
            output.add(value)
    return output


def _literal_in_text(value: str, text: str) -> bool:
    # Colon, slash and hyphen are delimiters.  Only identifier characters can
    # accidentally make a substring look like a visible ID.
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])"
    return re.search(pattern, text) is not None


def _canonical_id(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip().casefold()
    return re.sub(r"[-_\s]+", "", value)


def _natural_name(record: Mapping[str, Any]) -> str | None:
    for field in NAME_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return " ".join(
                unicodedata.normalize("NFKC", value).strip().casefold().split()
            )
    return None


def _response_succeeded(result: Any, fatal: bool) -> bool:
    if fatal:
        return False
    if isinstance(result, Mapping) and isinstance(result.get("success"), bool):
        return bool(result["success"])
    return True


def _execute(instance: Any, action: ParsedAction) -> tuple[Any, bool, str | None]:
    method = getattr(instance, action.name, None)
    if method is None:
        return {"error": "invalid action"}, False, None
    try:
        return method(**copy.deepcopy(dict(action.arguments))), False, None
    except Exception as exc:  # Official env terminates on tool exceptions.
        return None, True, f"{type(exc).__name__}: {exc}"


def _run_check(code: str, initial_state: Mapping[str, Any], final_state: Mapping[str, Any]) -> bool | None:
    scope: dict[str, Any] = {
        "__builtins__": __builtins__,
        "initial_state": copy.deepcopy(initial_state),
    }
    try:
        exec(code, scope)
        result = scope["check_func"](copy.deepcopy(final_state))
    except Exception:
        return None
    return result if isinstance(result, bool) else None


def _score(
    task: Mapping[str, Any], initial_state: Mapping[str, Any], final_state: Mapping[str, Any]
) -> tuple[float, list[bool | None]]:
    results = [
        _run_check(item["check_func"], initial_state, final_state)
        for item in task["checklist_with_func"]
    ]
    score = round(sum(bool(value) for value in results if value is not None) / len(results), 4)
    return score, results


def _root_evidence(
    *,
    action: ParsedAction,
    prompt: str,
    visible_ids: set[str],
    pre_entities: Mapping[tuple[str, str], Mapping[str, Any]],
    post_entities: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any] | None:
    arg_values = _id_argument_values(action.arguments)
    pre_ids = {primary for _, primary in pre_entities}
    added = sorted(set(post_entities) - set(pre_entities))

    unseen = sorted((arg_values & pre_ids) - visible_ids)
    shadow: list[dict[str, str]] = []
    canonical: list[dict[str, str]] = []
    for collection, primary in added:
        record = post_entities[(collection, primary)]
        name = _natural_name(record)
        if name is not None and not _literal_in_text(primary, prompt):
            for (old_collection, old_primary), old_record in pre_entities.items():
                if old_collection != collection:
                    continue
                if _natural_name(old_record) == name:
                    shadow.append(
                        {
                            "collection": collection,
                            "new_id": primary,
                            "existing_id": old_primary,
                            "normalized_name": name,
                        }
                    )
        for old_collection, old_primary in pre_entities:
            if old_collection != collection or old_primary == primary:
                continue
            if _canonical_id(old_primary) == _canonical_id(primary):
                canonical.append(
                    {
                        "collection": collection,
                        "new_id": primary,
                        "existing_id": old_primary,
                    }
                )

    types_found: list[str] = []
    if shadow:
        types_found.append("shadow_entity")
    if unseen:
        types_found.append("unseen_existing_reference")
    if canonical:
        types_found.append("canonical_id_collision")
    if not types_found:
        return None
    penalizable = bool(shadow or unseen)
    return {
        "action_index": action.index,
        "assistant_turn": action.assistant_turn,
        "tool": action.name,
        "arguments": copy.deepcopy(dict(action.arguments)),
        "types": types_found,
        "disposition": "penalizable" if penalizable else "environment_audit",
        "unseen_existing_ids": unseen,
        "shadow_matches": shadow,
        "canonical_collisions": canonical,
        "added_entity_ids": [primary for _, primary in added],
    }


def _raw_replay(
    task: Mapping[str, Any],
    env_item: Mapping[str, Any],
    actions: Sequence[ParsedAction],
    prompt: str,
) -> dict[str, Any]:
    instance = _new_env(task, env_item)
    initial_state = _state(instance)
    initial_entities = entity_index(initial_state)
    visible_ids = {
        primary
        for _, primary in initial_entities
        if _literal_in_text(primary, prompt)
    }
    roots: list[dict[str, Any]] = []
    action_records: list[dict[str, Any]] = []
    fatal_error: str | None = None

    for action in actions:
        pre_state = _state(instance)
        pre_entities = entity_index(pre_state)
        pre_ids = {primary for _, primary in pre_entities}
        before_visible = set(visible_ids)
        arg_values = _id_argument_values(action.arguments)
        circular_inputs = (arg_values & pre_ids) - before_visible

        result, fatal, error = _execute(instance, action)
        post_state = _state(instance)
        post_entities = entity_index(post_state)
        changed = pre_state != post_state
        succeeded = _response_succeeded(result, fatal)
        root = None
        if changed and succeeded:
            root = _root_evidence(
                action=action,
                prompt=prompt,
                visible_ids=before_visible,
                pre_entities=pre_entities,
                post_entities=post_entities,
            )
            if root is not None:
                roots.append(root)

        added_keys = sorted(set(post_entities) - set(pre_entities))
        # A successful, state-preserving read can reveal an ID.  An ungrounded
        # ID passed into that same read cannot certify itself through echo.
        if succeeded and not changed:
            current_ids = {primary for _, primary in post_entities}
            visible_ids.update(
                (_string_values(result) & current_ids) - circular_inputs
            )
        if succeeded and changed:
            visible_ids.update(primary for _, primary in added_keys)

        action_records.append(
            {
                "action_index": action.index,
                "tool": action.name,
                "arg_values": sorted(arg_values),
                "changed": changed,
                "succeeded": succeeded,
                "added_entity_ids": [primary for _, primary in added_keys],
                "root": root,
            }
        )
        if fatal:
            fatal_error = error
            break

    final_state = _state(instance)
    score, checks = _score(task, initial_state, final_state)
    return {
        "score": score,
        "checks": checks,
        "roots": roots,
        "action_records": action_records,
        "fatal_error": fatal_error,
    }


def _skip_closure(
    action_records: Sequence[Mapping[str, Any]],
    root_indices: set[int],
) -> tuple[set[int], list[dict[str, Any]]]:
    skipped: set[int] = set()
    tainted: set[str] = set()
    descendants: list[dict[str, Any]] = []
    for record in action_records:
        root = record.get("root")
        index = int(record["action_index"])
        selected_root = root is not None and index in root_indices
        if selected_root:
            skipped.add(index)
            tainted.update(record["added_entity_ids"])
            continue
        used = sorted(set(record["arg_values"]) & tainted)
        if used and record["changed"]:
            skipped.add(index)
            tainted.update(record["added_entity_ids"])
            descendants.append(
                {
                    "action_index": index,
                    "tool": record["tool"],
                    "tainted_ids_used": used,
                    "newly_tainted_ids": record["added_entity_ids"],
                }
            )
    return skipped, descendants


def _counterfactual(
    task: Mapping[str, Any],
    env_item: Mapping[str, Any],
    actions: Sequence[ParsedAction],
    action_records: Sequence[Mapping[str, Any]],
    raw_score: float,
    raw_checks: Sequence[bool | None],
    root_indices: set[int],
) -> dict[str, Any]:
    skipped, descendants = _skip_closure(action_records, root_indices)
    instance = _new_env(task, env_item)
    initial_state = _state(instance)
    fatal_error: str | None = None
    for action in actions:
        if action.index in skipped:
            continue
        _, fatal, error = _execute(instance, action)
        if fatal:
            fatal_error = error
            break
    score, checks = _score(task, initial_state, _state(instance))
    lost = [
        index + 1
        for index, (raw, clean) in enumerate(zip(raw_checks, checks))
        if raw is True and clean is not True
    ]
    return {
        "score": score,
        "gain_in_raw_reward": round(max(0.0, raw_score - score), 4),
        "skip_action_indices": sorted(skipped),
        "descendants": descendants,
        "lost_check_indices_1based": lost,
        "check_results": checks,
        "fatal_error": fatal_error,
    }
