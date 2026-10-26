from __future__ import annotations

from typing import Any


def resolve_dimension(name: str | None, record: dict[str, Any]) -> str | None:
    if not name:
        return None
    if name == "highlevel_edu":
        for key in ("dimension", "field", "ability", "tag", "标签", "维度"):
            value = record.get(key)
            if value:
                return str(value)
        return None
    raise ValueError(f"Unknown dimension resolver: {name}")
