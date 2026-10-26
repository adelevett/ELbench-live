from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class OutputPaths:
    raw_path: Path
    judged_path: Path
    failures_path: Path
    retries_path: Path
    summary_path: Path
    checkpoint_path: Path
    log_path: Path

    @classmethod
    def build(cls, output_root: Path, run_id: str, model_id: str) -> "OutputPaths":
        raw_dir = output_root / "raw_responses" / run_id
        judged_dir = output_root / "judged_results" / run_id
        logs_dir = output_root / "logs" / run_id
        summary_dir = output_root / "summaries" / run_id
        for path in (raw_dir, judged_dir, logs_dir, summary_dir):
            path.mkdir(parents=True, exist_ok=True)
        safe_model_id = model_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        return cls(
            raw_path=raw_dir / f"{safe_model_id}.jsonl",
            judged_path=judged_dir / f"{safe_model_id}.jsonl",
            failures_path=logs_dir / f"{safe_model_id}.failures.jsonl",
            retries_path=logs_dir / f"{safe_model_id}.retries.jsonl",
            summary_path=summary_dir / f"{safe_model_id}.summary.json",
            checkpoint_path=logs_dir / f"{safe_model_id}.checkpoint.json",
            log_path=logs_dir / f"{safe_model_id}.log",
        )


class JsonlWriter:
    def __init__(self, path: Path, *, flush_interval: int = 1) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._flush_interval = max(1, int(flush_interval))
        self._buffer: list[str] = []

    async def write(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"
        async with self._lock:
            self._buffer.append(line)
            if len(self._buffer) >= self._flush_interval:
                self._flush_unlocked()

    async def flush(self) -> None:
        async with self._lock:
            self._flush_unlocked()

    def _flush_unlocked(self) -> None:
        if not self._buffer:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.writelines(self._buffer)
        self._buffer.clear()
