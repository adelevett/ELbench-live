"""Optional spending limits for one run, shared with the basic-education runtime subprocess.

Nothing here is active unless a caller installs limits with ``spend_limits``.
The CLI never does, so a normal benchmark run behaves exactly as before.

Each paid request appends one line to a JSONL ledger, so the main process and the
runtime subprocess (which inherits the two environment variables below) count
against the same totals. Requests are checked before they are sent, so a limit
can be overshot by the requests already in flight.
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

LEDGER_ENV = "ELBENCH_SPEND_LEDGER"
LIMITS_ENV = "ELBENCH_SPEND_LIMITS"

ROLE_LABELS = {
    "target": "model under test",
    "judge": "judge",
    "student": "simulated student",
}


class BudgetExceeded(RuntimeError):
    """Raised instead of sending a request once a spending limit is reached.

    ``retryable = False`` keeps ``BenchmarkRunner`` from retrying it, and the message
    says "quota exceeded" so the runner stops the way it does when a key runs out of
    credit: finished items are kept and the run can be resumed with a higher limit.
    """

    retryable = False


@dataclass
class SpendLimits:
    total_usd: float | None = None
    per_role_usd: dict[str, float] = field(default_factory=dict)
    # USD per token, (prompt, completion), used when a response carries no cost.
    prices: dict[str, tuple[float, float]] = field(default_factory=dict)
    roles_by_model_id: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "total_usd": self.total_usd,
                "per_role_usd": self.per_role_usd,
                "prices": {role: list(pair) for role, pair in self.prices.items()},
                "roles_by_model_id": self.roles_by_model_id,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> "SpendLimits":
        data = json.loads(raw)
        return cls(
            total_usd=data.get("total_usd"),
            per_role_usd={k: float(v) for k, v in (data.get("per_role_usd") or {}).items()},
            prices={k: (float(v[0]), float(v[1])) for k, v in (data.get("prices") or {}).items()},
            roles_by_model_id=dict(data.get("roles_by_model_id") or {}),
        )


class SpendLedger:
    def __init__(self, path: Path, limits: SpendLimits) -> None:
        self.path = path
        self.limits = limits
        self._lock = threading.Lock()
        self._offset = 0
        self._spent: dict[str, float] = {}

    def spent(self) -> dict[str, float]:
        with self._lock:
            self._read_new_lines()
            return dict(self._spent)

    def check(self, role: str) -> None:
        spent = self.spent()
        total = sum(spent.values())
        limit = self.limits.total_usd
        if limit is not None and total >= limit:
            raise BudgetExceeded(
                f"Spending limit reached (quota exceeded): ${total:.4f} spent of the ${limit:.2f} total limit."
            )
        role_limit = self.limits.per_role_usd.get(role)
        if role_limit is not None and spent.get(role, 0.0) >= role_limit:
            raise BudgetExceeded(
                f"Spending limit reached (quota exceeded): ${spent.get(role, 0.0):.4f} spent on the "
                f"{ROLE_LABELS.get(role, role)} of its ${role_limit:.2f} limit."
            )

    def record(self, role: str, usage: Any) -> float:
        cost = usage_cost(usage, self.limits.prices.get(role))
        if cost <= 0:
            return 0.0
        line = json.dumps({"role": role, "usd": cost}) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        return cost

    def _read_new_lines(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("rb") as handle:
            handle.seek(self._offset)
            chunk = handle.read()
        end = chunk.rfind(b"\n")
        if end < 0:
            return
        self._offset += end + 1
        for raw in chunk[: end + 1].splitlines():
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            role = str(entry.get("role") or "")
            self._spent[role] = self._spent.get(role, 0.0) + float(entry.get("usd") or 0.0)


def usage_cost(usage: Any, price: tuple[float, float] | None) -> float:
    """USD for one response: OpenRouter's reported ``cost`` when present, else tokens x price."""
    if not isinstance(usage, dict):
        return 0.0
    cost = usage.get("cost")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0:
        return float(cost)
    if price is None:
        return 0.0
    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens")) or 0
    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens")) or 0
    try:
        return float(prompt_tokens) * price[0] + float(completion_tokens) * price[1]
    except (TypeError, ValueError):
        return 0.0


_ACTIVE: SpendLedger | None = None
_ACTIVE_LOCK = threading.Lock()


def active_ledger() -> SpendLedger | None:
    """The installed ledger, or one rebuilt from the environment inside the runtime subprocess."""
    global _ACTIVE
    if _ACTIVE is not None:
        return _ACTIVE
    path = os.environ.get(LEDGER_ENV)
    raw = os.environ.get(LIMITS_ENV)
    if not path or not raw:
        return None
    with _ACTIVE_LOCK:
        if _ACTIVE is None:
            _ACTIVE = SpendLedger(Path(path), SpendLimits.from_json(raw))
    return _ACTIVE


def role_for_model(model_id: str | None) -> str | None:
    ledger = active_ledger()
    if ledger is None or not model_id:
        return None
    return ledger.limits.roles_by_model_id.get(model_id)


def check(role: str | None) -> None:
    ledger = active_ledger()
    if ledger is not None and role:
        ledger.check(role)


def record(role: str | None, usage: Any) -> None:
    ledger = active_ledger()
    if ledger is not None and role:
        ledger.record(role, usage)


@contextmanager
def spend_limits(limits: SpendLimits, ledger_path: Path) -> Iterator[SpendLedger]:
    """Install ``limits`` for this process and any subprocess it starts."""
    global _ACTIVE
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.touch(exist_ok=True)
    ledger = SpendLedger(ledger_path, limits)
    previous = {name: os.environ.get(name) for name in (LEDGER_ENV, LIMITS_ENV)}
    os.environ[LEDGER_ENV] = str(ledger_path)
    os.environ[LIMITS_ENV] = limits.to_json()
    with _ACTIVE_LOCK:
        _ACTIVE = ledger
    try:
        yield ledger
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE = None
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
