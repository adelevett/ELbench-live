from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import httpx

from elbench.config import load_project_config
from elbench.persistence import OutputPaths
from elbench.schemas.config import ProjectConfig
from elbench.utils.secrets import get_api_key

from .preflight import PreflightRunner
from .runner import BenchmarkRunner, RunOptions


INNOSPARK_API_BASE = "https://api.innospark.cn/v1"
INNOSPARK_AIECNU_API_BASE = "https://innospark-api.aiecnu.net/v1"
EXTERNAL_SUPPORT_API_BASE = "http://35.220.164.252:3888/v1"
GPT_AGENT_API_BASE = "https://gpt-agent.cc/v1"
EXCLUDED_CAMPAIGN_MODELS = {
    "gemini-3-flash-preview",
    "gpt-5.2-pro",
    "gpt-5.4-pro",
    "kimi-k2.6",
}
RESULT_SUBDIRS = ("raw_responses", "judged_results", "logs", "summaries")
SAFETY_TASKS = {
    "safety_refusal",
    "safety_guidance",
    "safety_answer",
    "teaching_harm",
    "adversarial_safety",
}


@dataclass(slots=True)
class ApiPoolPlan:
    pool_id: str
    api_key_env: str
    api_base: str
    model_ids: tuple[str, ...]
    modules: frozenset[str] | None
    api_available: bool = True


@dataclass(slots=True)
class CampaignModelState:
    model_id: str
    run_id: str
    expected_total: int
    total_judged: int
    total_failures: int
    needs_run: bool
    summary_path: Path | None = None


@dataclass(slots=True)
class CampaignModelResult:
    model_id: str
    run_id: str
    status: str
    summary_path: str | None = None
    error: str | None = None


@dataclass(slots=True)
class CampaignResult:
    run_prefix: str
    pools: list[ApiPoolPlan]
    model_results: list[CampaignModelResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "run_prefix": self.run_prefix,
            "pools": [
                {
                    "pool_id": pool.pool_id,
                    "api_key_env": pool.api_key_env,
                    "api_base": pool.api_base,
                    "model_ids": list(pool.model_ids),
                    "modules": sorted(pool.modules) if pool.modules else None,
                    "api_available": pool.api_available,
                }
                for pool in self.pools
            ],
            "model_results": [
                {
                    "model_id": item.model_id,
                    "run_id": item.run_id,
                    "status": item.status,
                    "summary_path": item.summary_path,
                    "error": item.error,
                }
                for item in self.model_results
            ],
        }


def build_default_api_pools(config: ProjectConfig) -> list[ApiPoolPlan]:
    active_modules = tuple(config.app.benchmark_modules.active)
    innospark_modules = _active_modules_without_safety(config)
    external_modules = frozenset(active_modules) if active_modules else None
    innospark = _models_for_api(
        config,
        api_base=INNOSPARK_API_BASE,
        api_key_env="INNOSPARK_RELAY_API_KEY",
        preferred_order=(
            "deepseek-v3.2",
            "doubao-seed-2-0-pro-260215",
            "deepseek-r1-250528",
            "gemini-3.1-pro-preview",
        ),
    )
    innospark_aiecnu = _models_for_api(
        config,
        api_base=INNOSPARK_AIECNU_API_BASE,
        api_key_env="INNOSPARK_AIECNU_API_KEY",
        preferred_order=("innospark-235b",),
    )
    external = _models_for_api(
        config,
        api_base=EXTERNAL_SUPPORT_API_BASE,
        api_key_env="EXTERNAL_SUPPORT_API_KEY",
        preferred_order=(
            "gpt-5.4",
            "claude-sonnet-4-6",
            "claude-opus-4-6",
        ),
    )
    gpt_agent = _models_for_api(
        config,
        api_base=GPT_AGENT_API_BASE,
        api_key_env="GPT_AGENT_API_KEY",
        preferred_order=(
            "deepseek-v4-flash",
            "doubao-seed-2.0-pro",
            "glm-5.1",
            "deepseek-v4-pro",
            "claude-opus-4-8",
        ),
    )
    gpt_agent = tuple(
        model_id
        for model_id in gpt_agent
        if model_id
        in {
            "deepseek-v4-flash",
            "doubao-seed-2.0-pro",
            "glm-5.1",
            "deepseek-v4-pro",
            "claude-opus-4-8",
        }
    )
    return [
        ApiPoolPlan(
            pool_id="innospark_relay",
            api_key_env="INNOSPARK_RELAY_API_KEY",
            api_base=INNOSPARK_API_BASE,
            model_ids=innospark,
            modules=innospark_modules,
        ),
        ApiPoolPlan(
            pool_id="innospark_aiecnu",
            api_key_env="INNOSPARK_AIECNU_API_KEY",
            api_base=INNOSPARK_AIECNU_API_BASE,
            model_ids=innospark_aiecnu,
            modules=innospark_modules,
        ),
        ApiPoolPlan(
            pool_id="external_support",
            api_key_env="EXTERNAL_SUPPORT_API_KEY",
            api_base=EXTERNAL_SUPPORT_API_BASE,
            model_ids=external,
            modules=external_modules,
        ),
        ApiPoolPlan(
            pool_id="gpt_agent",
            api_key_env="GPT_AGENT_API_KEY",
            api_base=GPT_AGENT_API_BASE,
            model_ids=gpt_agent,
            modules=external_modules,
        ),
    ]


async def run_default_campaign(
    *,
    config_root: Path | str = "configs",
    run_prefix: str | None = None,
    max_concurrency: int | None = None,
    modules: set[str] | None = None,
    progress_enabled: bool = False,
) -> CampaignResult:
    config = load_project_config(Path(config_root))
    pools = build_default_api_pools(config)
    prefix = run_prefix or "campaign-official"
    runner = BenchmarkRunner(config)
    result = CampaignResult(run_prefix=prefix, pools=pools)

    for pool in pools:
        if not pool.api_available or not pool.model_ids:
            continue
        pool_available, pool_probe_error = await probe_api_pool(pool)
        if not pool_available:
            pool.api_available = False
            result.model_results.append(
                CampaignModelResult(
                    model_id="*",
                    run_id=prefix,
                    status="pool_unavailable",
                    error=pool_probe_error,
                )
            )
            continue
        for model_id in pool.model_ids:
            if not pool_available:
                break
            selected_modules = modules or (set(pool.modules) if pool.modules is not None else None)
            state = inspect_model_state(
                config=config,
                model_id=model_id,
                modules=selected_modules,
                run_prefix=prefix,
            )
            if not state.needs_run:
                result.model_results.append(
                    CampaignModelResult(
                        model_id=model_id,
                        run_id=state.run_id,
                        status="skipped_complete",
                        summary_path=str(state.summary_path) if state.summary_path else None,
                    )
                )
                continue
            model_concurrency = _model_concurrency(model_id, max_concurrency)
            try:
                run_result = await runner.run(
                    RunOptions(
                        model_id=model_id,
                        run_id=state.run_id,
                        modules=selected_modules,
                        max_concurrency=model_concurrency,
                        resume=True,
                        judge_enabled=True,
                        progress_enabled=progress_enabled,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                if is_quota_exhausted_error(exc):
                    pool_available = False
                    pool.api_available = False
                    result.model_results.append(
                        CampaignModelResult(
                            model_id=model_id,
                            run_id=state.run_id,
                            status="pool_quota_exhausted",
                            error=str(exc),
                        )
                    )
                    break
                result.model_results.append(
                    CampaignModelResult(
                        model_id=model_id,
                        run_id=state.run_id,
                        status="failed",
                        error=str(exc),
                    )
                )
                continue
            if run_result.get("quota_exhausted"):
                pool_available = False
                pool.api_available = False
                result.model_results.append(
                    CampaignModelResult(
                        model_id=model_id,
                        run_id=state.run_id,
                        status="pool_quota_exhausted",
                        summary_path=str(run_result.get("summary_path") or ""),
                    )
                )
                break
            result.model_results.append(
                CampaignModelResult(
                    model_id=model_id,
                    run_id=state.run_id,
                    status="completed",
                    summary_path=str(run_result.get("summary_path") or ""),
                )
            )
    return result


async def probe_api_pool(pool: ApiPoolPlan) -> tuple[bool, str | None]:
    api_key = get_api_key(pool.api_key_env)
    if not api_key:
        return False, f"missing api key env: {pool.api_key_env}"
    url = f"{pool.api_base.rstrip('/')}/models"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
            )
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    if response.status_code == 200:
        return True, None
    if response.status_code == 404:
        return True, f"{url} returned 404; continuing because chat endpoint may still be available"
    body = response.text.strip()
    message = f"{response.status_code}: {body[:500]}"
    if is_quota_exhausted_error(RuntimeError(message)):
        return False, message
    return False, message


def is_quota_exhausted_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    markers = (
        "quota",
        "insufficient quota",
        "balance",
        "\u4f59\u989d",
        "\u989d\u5ea6",
        "credit",
        "payment required",
        "402",
    )
    return any(marker in text for marker in markers)


def discover_campaign_run_ids(output_root: Path, model_ids: Iterable[str]) -> dict[str, list[str]]:
    discovered: dict[str, list[str]] = {model_id: [] for model_id in model_ids}
    summaries_root = output_root / "summaries"
    if not summaries_root.exists():
        return discovered
    for model_id in model_ids:
        safe_model_id = model_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        for summary_path in summaries_root.glob(f"*/{safe_model_id}.summary.json"):
            discovered[model_id].append(summary_path.parent.name)
    return {model_id: sorted(run_ids) for model_id, run_ids in discovered.items()}


def inspect_model_state(
    *,
    config: ProjectConfig,
    model_id: str,
    modules: set[str] | None,
    run_prefix: str,
) -> CampaignModelState:
    expected_total = _expected_total(config=config, model_id=model_id, modules=modules)
    best = _best_summary_for_model(config.app.output_root, model_id, run_prefix=run_prefix)
    if best is None:
        return CampaignModelState(
            model_id=model_id,
            run_id=f"{run_prefix}-{model_id}",
            expected_total=expected_total,
            total_judged=0,
            total_failures=0,
            needs_run=True,
        )

    run_id, summary_path, summary = best
    total_judged = int(summary.get("total_judged") or 0)
    total_failures = int(summary.get("total_failures") or 0)
    return CampaignModelState(
        model_id=model_id,
        run_id=run_id,
        expected_total=expected_total,
        total_judged=total_judged,
        total_failures=total_failures,
        needs_run=total_judged < expected_total or total_failures > 0,
        summary_path=summary_path,
    )


def clean_model_artifacts(output_root: Path, model_ids: Iterable[str]) -> list[Path]:
    deleted: list[Path] = []
    for model_id in model_ids:
        safe_model_id = model_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        for subdir in RESULT_SUBDIRS:
            root = output_root / subdir
            if not root.exists():
                continue
            for path in root.glob(f"*/{safe_model_id}*"):
                if path.is_file():
                    path.unlink()
                    deleted.append(path)
    return deleted


def output_paths_for_campaign_model(output_root: Path, run_id: str, model_id: str) -> OutputPaths:
    return OutputPaths.build(output_root, run_id, model_id)


def run_campaign_sync(**kwargs) -> CampaignResult:
    return asyncio.run(run_default_campaign(**kwargs))


def _models_for_api(
    config: ProjectConfig,
    *,
    api_base: str,
    api_key_env: str,
    preferred_order: tuple[str, ...],
) -> tuple[str, ...]:
    available = {
        model_id
        for model_id, model in config.models.items()
        if (model.api_base or "").rstrip("/") == api_base.rstrip("/")
        and model.api_key_env == api_key_env
        and model_id not in EXCLUDED_CAMPAIGN_MODELS
    }
    ordered = [model_id for model_id in preferred_order if model_id in available]
    extras = sorted(available.difference(ordered))
    return tuple(ordered + extras)


def _active_modules_without_safety(config: ProjectConfig) -> frozenset[str] | None:
    active_modules = tuple(config.app.benchmark_modules.active)
    if not active_modules:
        return None
    safety_modules = {
        entry.module
        for entry in config.file_registry.values()
        if entry.task in SAFETY_TASKS
    }
    return frozenset(module for module in active_modules if module not in safety_modules)


def _expected_total(*, config: ProjectConfig, model_id: str, modules: set[str] | None) -> int:
    report = PreflightRunner(config).run(
        model_id=model_id,
        modules=modules,
        require_real_judges=True,
    )
    return int(report.total_samples)


def _best_summary_for_model(
    output_root: Path,
    model_id: str,
    *,
    run_prefix: str,
) -> tuple[str, Path, dict] | None:
    safe_model_id = model_id.replace("/", "_").replace("\\", "_").replace(":", "_")
    summaries_root = output_root / "summaries"
    if not summaries_root.exists():
        return None

    best: tuple[str, Path, dict] | None = None
    best_key = (-1, -1)
    for summary_path in summaries_root.glob(f"{run_prefix}*/{safe_model_id}.summary.json"):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        judged = int(summary.get("total_judged") or 0)
        failures = int(summary.get("total_failures") or 0)
        candidate_key = (judged, -failures)
        if candidate_key > best_key:
            best_key = candidate_key
            best = (summary_path.parent.name, summary_path, summary)
    return best


def _model_concurrency(model_id: str, requested: int | None) -> int | None:
    if model_id == "kimi-k2.6":
        return 1
    if model_id == "deepseek-r1-250528":
        return min(requested, 2) if requested is not None else 2
    innospark_relay_models = {
        "deepseek-v3.2",
        "doubao-seed-2-0-pro-260215",
        "gemini-3.1-pro-preview",
    }
    if model_id in innospark_relay_models:
        return min(requested, 8) if requested is not None else 8
    if model_id == "innospark-235b":
        return min(requested, 1) if requested is not None else 1
    gpt_agent_models = {
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "glm-5.1",
        "claude-opus-4-8",
        "doubao-seed-2.0-pro",
    }
    if model_id in gpt_agent_models:
        return min(requested, 32) if requested is not None else 32
    return requested
