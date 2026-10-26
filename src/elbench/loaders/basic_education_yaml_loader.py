from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any, Iterator

import yaml

from elbench.loaders.base import BaseLoader
from elbench.registry import ResolvedRegistryItem
from elbench.schemas.evaluation import RawRecord, Sample


class BasicEducationYamlLoader(BaseLoader):
    def iter_raw_records(self, item: ResolvedRegistryItem) -> Iterator[RawRecord]:
        template = _read_yaml(item.path)
        task_defs = _expand_tasks(template.get("tasks", {}))
        for task_index, task_payload in enumerate(task_defs, start=1):
            yield RawRecord(
                source_path=str(item.path),
                source_file=item.entry.canonical_name,
                row_index=task_index,
                record={
                    "task_payload": task_payload,
                    "prompt": _extract_prompt(template, task_payload),
                    "task_mode": str(template.get("tasks", {}).get("mode", "iter")).lower(),
                    "multi_turn": _is_multi_turn(template),
                    "agent_count": len(template.get("agents", {})),
                },
            )

    def iter_samples(self, item: ResolvedRegistryItem) -> Iterator[Sample]:
        for raw_record in self.iter_raw_records(item):
            task_payload = raw_record.record["task_payload"]
            yield Sample(
                sample_id=_build_sample_id(item.entry.canonical_name, raw_record.row_index, task_payload),
                source_file=item.entry.canonical_name,
                source_path=str(item.path),
                module=item.entry.module,
                subset=item.entry.subset,
                task=item.entry.task,
                dimension=item.entry.subset,
                prompt=raw_record.record["prompt"],
                reference=None,
                metadata={
                    "row_index": raw_record.row_index,
                    "task_payload": task_payload,
                    "task_mode": raw_record.record["task_mode"],
                    "multi_turn": raw_record.record["multi_turn"],
                    "agent_count": raw_record.record["agent_count"],
                },
                raw_record=task_payload,
            )


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        obj = yaml.safe_load(handle) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"Expected YAML object at {path}, got {type(obj).__name__}")
    return obj


def _expand_tasks(tasks_obj: Any) -> list[dict[str, Any]]:
    if not isinstance(tasks_obj, dict):
        return []
    mode = str(tasks_obj.get("mode", "iter")).lower()
    content = tasks_obj.get("content")
    if mode == "iter":
        if not isinstance(content, list):
            return []
        return [item for item in content if isinstance(item, dict)]
    if mode == "union":
        if not isinstance(content, dict) or not content:
            return []
        keys = list(content.keys())
        values = [value for value in content.values()]
        if not all(isinstance(value, list) for value in values):
            return []
        return [
            {key: combo[index] for index, key in enumerate(keys)}
            for combo in product(*values)
        ]
    return []


def _extract_prompt(template: dict[str, Any], task_payload: dict[str, Any]) -> str:
    start_prompt = template.get("tasks", {}).get("start_prompt")
    prompt = _extract_prompt_from_message(start_prompt, task_payload)
    if prompt:
        return prompt

    agents = template.get("agents", {})
    if isinstance(agents, dict):
        for agent_config in agents.values():
            if not isinstance(agent_config, dict):
                continue
            for message in agent_config.get("prompt", []) or []:
                prompt = _extract_prompt_from_message(message, task_payload)
                if prompt:
                    return prompt

    for key in ("question", "query", "prompt", "instruction"):
        value = task_payload.get(key)
        if value not in (None, ""):
            return str(value)
    return json.dumps(task_payload, ensure_ascii=False)


def _extract_prompt_from_message(message: Any, task_payload: dict[str, Any]) -> str:
    if not isinstance(message, dict):
        return ""
    if message.get("role") != "user":
        return ""
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return ""
    return _render_task_placeholders(content, task_payload).strip()


def _render_task_placeholders(text: str, task_payload: dict[str, Any]) -> str:
    rendered = text
    for key, value in task_payload.items():
        string_value = _stringify(value)
        rendered = rendered.replace(f"{{{key}}}", string_value)
        rendered = rendered.replace(f"{{task.{key}}}", string_value)
    return rendered


def _is_multi_turn(template: dict[str, Any]) -> bool:
    directions = template.get("directions")
    if isinstance(directions, list) and len(directions) > 2:
        return True
    return isinstance(template.get("tasks", {}).get("start_prompt"), dict)


def _build_sample_id(canonical_name: str, row_index: int, task_payload: dict[str, Any]) -> str:
    payload = json.dumps(task_payload, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(f"{canonical_name}::{row_index}::{payload}".encode("utf-8")).hexdigest()[:16]
    safe_stem = canonical_name.replace(".", "_").replace("/", "_").replace("\\", "_")
    return f"{safe_stem}-{digest}"


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)
