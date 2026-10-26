from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any


class CheckpointStore:
    def __init__(self, path: Path, *, flush_interval: int = 1) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._flush_interval = max(1, int(flush_interval))
        self._dirty_operations = 0
        self.completed_ids: set[str] = set()
        self.failed_ids: set[str] = set()
        self.retry_counts: dict[str, int] = {}

    def load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.completed_ids = set(data.get("completed_ids", []))
        self.failed_ids = set(data.get("failed_ids", []))
        self.retry_counts = {str(k): int(v) for k, v in data.get("retry_counts", {}).items()}

    async def mark_completed(self, sample_key: str) -> None:
        self.completed_ids.add(sample_key)
        self.failed_ids.discard(sample_key)
        await self._save_if_needed()

    async def mark_failed(self, sample_key: str) -> None:
        self.failed_ids.add(sample_key)
        await self._save_if_needed()

    async def set_retry_count(self, sample_key: str, retry_count: int) -> None:
        self.retry_counts[sample_key] = retry_count
        await self._save_if_needed()

    def is_completed(self, sample_key: str) -> bool:
        return sample_key in self.completed_ids

    async def flush(self) -> None:
        await self._save_if_needed(force=True)

    async def save(self) -> None:
        async with self._lock:
            payload: dict[str, Any] = {
                "completed_ids": sorted(self.completed_ids),
                "failed_ids": sorted(self.failed_ids),
                "retry_counts": self.retry_counts,
            }
            temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
            temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temp_path.replace(self.path)
            self._dirty_operations = 0

    async def _save_if_needed(self, *, force: bool = False) -> None:
        self._dirty_operations += 1
        if force or self._dirty_operations >= self._flush_interval:
            await self.save()
