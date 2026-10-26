"""Run a visitor selection through BenchmarkRunner, checkpoint every sample, and persist to LeaderboardStore."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from elbench.dashboard.catalog import Catalog
from elbench.dashboard.english import (
    SUMMARY_GROUP_ENGLISH_LABELS,
    english_module,
    english_preflight_message,
    english_summary_name,
)
from elbench.dashboard.leaderboard import (
    LeaderboardStore,
    basic_education_valid_means,
    compute_normalized_scores,
)
from elbench.dashboard.plan import Plan
from elbench.dashboard.protocol import VISITOR_KEY_ENV
from elbench.execution.preflight import PreflightError, PreflightRunner
from elbench.execution.runner import BenchmarkRunner, RunOptions
from elbench.persistence.writers import OutputPaths
from elbench.summary import build_summary
from elbench.utils.spend import SpendLimits, spend_limits

_ENV_LOCK = threading.Lock()


@contextmanager
def visitor_api_key(api_key: str):
    """Hold the only visitor key slot for the whole critical section."""
    secret = (api_key or "").strip()
    if not secret:
        raise ValueError("OpenRouter API key is empty.")
    with _ENV_LOCK:
        previous = os.environ.get(VISITOR_KEY_ENV)
        os.environ[VISITOR_KEY_ENV] = secret
        try:
            yield secret
        finally:
            if previous is None:
                os.environ.pop(VISITOR_KEY_ENV, None)
            else:
                os.environ[VISITOR_KEY_ENV] = previous


def make_run_id(config_hash: str | None = None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    suffix = (config_hash or uuid4().hex)[:8]
    return f"live-{stamp}-{suffix}"


def execute_run(
    catalog: Catalog,
    plan: Plan,
    *,
    max_concurrency: int,
    accept_unpriced: bool,
    run_id: str | None = None,
    resume_previous: bool = True,
    leaderboard_store: LeaderboardStore | None = None,
    persist_to_leaderboard: bool = True,
    total_budget_usd: float | None = None,
    role_budgets_usd: dict[str, float | None] | None = None,
) -> dict[str, object]:
    """Execute ``plan`` with immediate per-sample checkpointing and fault-tolerant recovery."""
    if plan.blocked:
        raise PreflightError(" | ".join(plan.error_messages()) or "Selection is blocked.")
    if plan.needs_price_ack and not accept_unpriced:
        raise PreflightError(
            "OpenRouter returned no token price for this model. Confirm the unknown price before running."
        )

    config = catalog.config.model_copy(deep=True)
    # Flush every single sample immediately so long-running tasks never lose progress
    config.app.default_run.checkpoint_interval = 1

    model = plan.protocol.model_config
    config.models[model.model_id] = model
    if plan.judge_model_config is not None:
        config.models[plan.judge_model_config.model_id] = plan.judge_model_config
        config.judges.default_judge_model_id = plan.judge_model_config.model_id
        config.judges.basic_education_evaluator_model_id = plan.judge_model_config.model_id
        config.judges.basic_education_test_endpoint_model_id = plan.judge_model_config.model_id
    if plan.multiturn_model_config is not None:
        config.models[plan.multiturn_model_config.model_id] = plan.multiturn_model_config
        config.judges.basic_education_test_endpoint_model_id = plan.multiturn_model_config.model_id

    resumed_from_existing = False
    resolved_run_id = (run_id or "").strip()
    if leaderboard_store is not None and resume_previous:
        existing = leaderboard_store.find_resumable_run(
            config_hash=plan.config_hash,
            explicit_run_id=resolved_run_id if resolved_run_id else None,
        )
        if existing is not None and existing.get("run_id"):
            resolved_run_id = str(existing["run_id"])
            resumed_from_existing = True
    if not resolved_run_id:
        resolved_run_id = make_run_id(plan.config_hash)

    output_root = _hosted_output_root(config.project_root, resolved_run_id)
    config.app.output_root = output_root
    paths = OutputPaths.build(output_root, resolved_run_id, model.model_id)

    if leaderboard_store is not None and (resumed_from_existing or resume_previous):
        restored = leaderboard_store.prepare_resume_directory(
            run_id=resolved_run_id,
            model_id=model.model_id,
            output_root=output_root,
        )
        if restored:
            resumed_from_existing = True

    report = PreflightRunner(config).run(
        model_id=model.model_id,
        modules=set(plan.selection.modules),
        subsets=None,
        source_files=set(plan.selection.source_files),
        dimensions=set(plan.selection.dimensions) or None,
        max_samples=plan.selection.max_samples,
        judge_enabled=True,
        require_real_judges=True,
    )
    if not report.ok:
        messages = [english_preflight_message(issue) for issue in report.issues if issue.level == "error"]
        raise PreflightError(" | ".join(messages))

    secret = os.environ.get(VISITOR_KEY_ENV, "")
    basic_root = config.project_root / "outputs" / "basic_education" / resolved_run_id
    interrupted_error: BaseException | None = None
    result: dict[str, Any] = {}

    limits = build_spend_limits(plan, total_budget_usd, role_budgets_usd)
    ledger_path = output_root / "spend.jsonl"
    spent: dict[str, float] = {}
    try:
        with spend_limits(limits, ledger_path) as ledger:
            result = _run_blocking(
                BenchmarkRunner(config).run(
                    RunOptions(
                        model_id=model.model_id,
                        run_id=resolved_run_id,
                        modules=set(plan.selection.modules),
                        subsets=None,
                        source_files=set(plan.selection.source_files),
                        dimensions=set(plan.selection.dimensions) or None,
                        max_samples=plan.selection.max_samples,
                        max_samples_per_file=plan.selection.max_samples_per_file,
                        max_concurrency=max_concurrency,
                        resume=bool(resume_previous),
                        judge_enabled=True,
                        progress_enabled=True,
                    )
                )
            )
            spent = ledger.spent()
    except BaseException as exc:  # noqa: BLE001
        interrupted_error = exc
        if ledger_path.exists():
            spent = _ledger_totals(ledger_path, limits)
    finally:
        scrub_tree(output_root, secret)
        scrub_tree(basic_root, secret)
        if basic_root.exists():
            shutil.rmtree(basic_root, ignore_errors=True)

    has_partial_records = (
        (paths.judged_path.exists() and paths.judged_path.stat().st_size > 0)
        or (paths.failures_path.exists() and paths.failures_path.stat().st_size > 0)
    )
    if interrupted_error is not None and not has_partial_records:
        raise interrupted_error

    if not paths.summary_path.exists() or interrupted_error is not None:
        summary = build_summary(paths.judged_path, paths.failures_path)
        paths.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        summary = json.loads(paths.summary_path.read_text(encoding="utf-8"))

    total_judged = int(summary.get("total_judged") or 0)
    total_failures = int(summary.get("total_failures") or 0)
    quota_exhausted = bool(result.get("quota_exhausted", False))
    spent_total = round(sum(spent.values()), 6)
    budget_stopped = quota_exhausted and _limit_reached(limits, spent)
    if interrupted_error is not None:
        run_status = "interrupted"
    elif quota_exhausted or total_failures > 0:
        run_status = "partial"
    else:
        run_status = "completed"

    normalized_scores = compute_normalized_scores(
        summary,
        basic_valid_means=basic_education_valid_means(paths.judged_path),
    )
    leaderboard_record = {
        "run_id": resolved_run_id,
        "config_hash": plan.config_hash,
        "target_model": plan.protocol.openrouter_model,
        "judge_model": (
            plan.judge_model_config.model_name
            if plan.judge_model_config is not None
            else (plan.judge_model_id or "default")
        ),
        "use_system_one_judge": bool(plan.use_system_one_judge),
        "use_multiturn_chat_model": bool(plan.use_multiturn_chat_model),
        "multiturn_model": (
            plan.multiturn_model_config.model_name
            if plan.multiturn_model_config is not None
            else None
        ),
        "modules": [english_module(m) for m in sorted(plan.selection.modules)],
        "source_files": sorted(plan.selection.source_files),
        "max_samples": plan.selection.max_samples,
        "max_samples_per_file": plan.selection.max_samples_per_file,
        "dimensions": sorted(plan.selection.dimensions),
        "is_full_benchmark": bool(plan.is_full_benchmark),
        "covers_core": bool(plan.covers_core),
        "routing": {role: route.record() for role, route in plan.routes.items()},
        # False when every answer was checked against the answer key, so the run scores the same on every judge's board.
        "judge_used": bool(plan.judge_calls or plan.basic_tasks),
        "status": run_status,
        "resumed": resumed_from_existing,
        "total_judged": total_judged,
        "total_failures": total_failures,
        "quota_exhausted": quota_exhausted,
        "interrupted_error": (
            public_error(interrupted_error, secret) if interrupted_error is not None else None
        ),
        "prompt_usd_per_token": plan.prompt_usd_per_token,
        "completion_usd_per_token": plan.completion_usd_per_token,
        "low_usd": plan.low_usd,
        "high_usd": plan.high_usd,
        "spent_usd": spent_total,
        "spent_usd_by_role": {role: round(value, 6) for role, value in spent.items()},
        "budget_usd": total_budget_usd,
        "role_budgets_usd": {k: v for k, v in (role_budgets_usd or {}).items() if v is not None},
        "stopped_by_budget": budget_stopped,
        "normalized_scores": normalized_scores,
    }

    if persist_to_leaderboard and leaderboard_store is not None and (total_judged > 0 or total_failures > 0):
        leaderboard_store.save_run(
            leaderboard_record,
            output_Paths=paths,
            push_to_hub=True,
        )

    download_paths = [
        str(p)
        for p in (paths.judged_path, paths.summary_path, paths.failures_path)
        if p.exists() and p.stat().st_size > 0
    ]

    return {
        **result,
        "run_id": resolved_run_id,
        "status": run_status,
        "resumed": resumed_from_existing,
        "config_hash": plan.config_hash,
        "spent_usd": spent_total,
        "spent_usd_by_role": leaderboard_record["spent_usd_by_role"],
        "stopped_by_budget": budget_stopped,
        "interrupted_error": leaderboard_record["interrupted_error"],
        "summary_path": str(paths.summary_path),
        "judged_path": str(paths.judged_path),
        "failures_path": str(paths.failures_path),
        "summary": summary,
        "normalized_scores": normalized_scores,
        "leaderboard_record": leaderboard_record,
        "summary_rows": summary_rows(summary),
        "download_paths": download_paths,
        "output_root": str(output_root),
    }


def build_spend_limits(
    plan: Plan,
    total_budget_usd: float | None,
    role_budgets_usd: dict[str, float | None] | None,
) -> SpendLimits:
    """Attribute each request to the model under test, the judge, or the simulated student."""
    roles = {plan.protocol.model_config.model_id: "target"}
    prices: dict[str, tuple[float, float]] = {}
    if plan.prompt_usd_per_token is not None and plan.completion_usd_per_token is not None:
        prices["target"] = (plan.prompt_usd_per_token, plan.completion_usd_per_token)
    if plan.judge_model_config is not None:
        roles[plan.judge_model_config.model_id] = "judge"
    if plan.judge_prices is not None:
        prices["judge"] = plan.judge_prices
    if plan.multiturn_model_config is not None:
        roles[plan.multiturn_model_config.model_id] = "student"
    if plan.student_prices is not None:
        prices["student"] = plan.student_prices
    elif plan.judge_prices is not None:
        # Without a separate student model, the judge model also plays the student.
        prices["student"] = plan.judge_prices
    return SpendLimits(
        total_usd=_positive_or_none(total_budget_usd),
        per_role_usd={
            role: float(value)
            for role, value in (role_budgets_usd or {}).items()
            if _positive_or_none(value) is not None
        },
        prices=prices,
        roles_by_model_id=roles,
    )


def _positive_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _limit_reached(limits: SpendLimits, spent: dict[str, float]) -> bool:
    if limits.total_usd is not None and sum(spent.values()) >= limits.total_usd:
        return True
    return any(spent.get(role, 0.0) >= cap for role, cap in limits.per_role_usd.items())


def _ledger_totals(path: Path, limits: SpendLimits) -> dict[str, float]:
    from elbench.utils.spend import SpendLedger

    return SpendLedger(path, limits).spent()


def judged_progress_path(project_root: Path, run_id: str, model_id: str) -> Path:
    output_root = _hosted_output_root(project_root, run_id)
    paths = OutputPaths.build(output_root, run_id, model_id)
    return paths.judged_path


def failures_progress_path(project_root: Path, run_id: str, model_id: str) -> Path:
    output_root = _hosted_output_root(project_root, run_id)
    paths = OutputPaths.build(output_root, run_id, model_id)
    return paths.failures_path


def summary_rows(summary: dict) -> list[list[object]]:
    rows: list[list[object]] = []
    for section in ("by_module", "by_subset", "by_task", "by_source_file", "by_dimension", "by_scene"):
        block = summary.get(section) or {}
        if not isinstance(block, dict):
            continue
        group_label = SUMMARY_GROUP_ENGLISH_LABELS.get(section, section)
        for name, stats in block.items():
            if not isinstance(stats, dict):
                continue
            avg_score = stats.get("avg_score")
            pass_rate = stats.get("pass_rate")
            rows.append(
                [
                    group_label,
                    english_summary_name(section, str(name)),
                    stats.get("count"),
                    round(float(avg_score), 4) if isinstance(avg_score, (int, float)) else avg_score,
                    round(float(pass_rate), 4) if isinstance(pass_rate, (int, float)) else pass_rate,
                ]
            )
    return rows


def scrub_tree(root: Path, secret: str) -> None:
    if not secret or len(secret) < 8 or not root.exists():
        return
    token = secret.encode("utf-8")
    replacement = b"[redacted]"
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > 50_000_000 or size == 0:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if token not in data:
            continue
        path.write_bytes(data.replace(token, replacement))


def public_error(exc: BaseException, secret: str) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if secret:
        text = text.replace(secret, "[redacted]")
    return text


def _hosted_output_root(project_root: Path, run_id: str) -> Path:
    path = project_root / "outputs" / "hosted" / run_id
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    path.chmod(0o700)
    return path


def _run_blocking(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box: dict[str, object] = {}

    def _target() -> None:
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    thread = threading.Thread(target=_target)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["value"]
