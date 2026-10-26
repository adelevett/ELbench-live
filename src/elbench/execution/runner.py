from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from elbench.config import load_project_config
from elbench.judges import JudgeRouter
from elbench.loaders import LoaderFactory
from elbench.persistence import CheckpointStore, JsonlWriter, OutputPaths, configure_run_logger
from elbench.providers import ProviderFactory
from elbench.registry import FileRegistry
from elbench.schemas.config import ModelConfig, ProjectConfig, RateLimitConfig, RetryConfig
from elbench.schemas.evaluation import EvalResult, FailureRecord, GenerationRequest, ModelResponse, Sample
from elbench.summary import build_summary

from .basic_education import BasicEducationExecutor, DEFAULT_BASIC_MODULE_NAME
from .selection import resolve_module_selection
from .rate_limit import SlidingWindowRateLimiter
from .response_format import (
    ResponseFormatError,
    attach_format_metadata,
    augment_prompt_for_format,
    build_concise_retry_prompt,
    build_retry_followup,
    get_response_format_spec,
)
from .retry import is_retryable_exception, sleep_before_retry


ANSWER_COMPLETION_TASKS = {"math_500", "aime24", "aime25", "aime26"}


@dataclass(slots=True)
class RunOptions:
    model_id: str
    run_id: str
    modules: set[str] | None = None
    subsets: set[str] | None = None
    source_files: set[str] | None = None
    dimensions: set[str] | None = None
    max_samples: int | None = None
    # Cap per source file (per scenario for basic education), applied before max_samples.
    max_samples_per_file: int | None = None
    max_concurrency: int | None = None
    resume: bool = True
    judge_enabled: bool = True
    progress_enabled: bool = False


class BenchmarkRunner:
    def __init__(self, config: ProjectConfig | None = None) -> None:
        self.config = config or load_project_config()

    async def run(self, options: RunOptions) -> dict[str, object]:
        model_config = self._get_model_config(options.model_id)
        output_paths = OutputPaths.build(self.config.app.output_root, options.run_id, options.model_id)
        logger = configure_run_logger(output_paths.log_path)
        flush_interval = self.config.app.default_run.checkpoint_interval
        checkpoint = CheckpointStore(output_paths.checkpoint_path, flush_interval=flush_interval)
        if options.resume:
            checkpoint.load()

        raw_writer = JsonlWriter(output_paths.raw_path, flush_interval=flush_interval)
        judged_writer = JsonlWriter(output_paths.judged_path, flush_interval=flush_interval)
        failure_writer = JsonlWriter(output_paths.failures_path, flush_interval=flush_interval)
        retry_writer = JsonlWriter(output_paths.retries_path, flush_interval=flush_interval)

        module_selection = resolve_module_selection(self.config, options.modules)
        include_basic_education = module_selection.include_basic_education
        standard_modules = set(module_selection.standard_modules)
        if include_basic_education and standard_modules and options.max_samples is not None:
            logger.warning(
                "Mixed module run with max_samples=%s: standard modules consume the budget first; "
                "basic education receives the remaining budget.",
                options.max_samples,
            )

        standard_loaded = 0
        standard_failures = 0
        if standard_modules:
            standard_stats = await self._run_standard_samples(
                logger=logger,
                checkpoint=checkpoint,
                raw_writer=raw_writer,
                judged_writer=judged_writer,
                failure_writer=failure_writer,
                retry_writer=retry_writer,
                model_config=model_config,
                options=options,
                modules=standard_modules,
            )
            standard_loaded = standard_stats["loaded_samples"]
            standard_failures = standard_stats["failed_samples"]
            standard_quota_exhausted = bool(standard_stats.get("quota_exhausted", 0))
        else:
            standard_quota_exhausted = False

        basic_stats = {
            "scenario_count": 0,
            "loaded_samples": 0,
            "completed_samples": 0,
            "failed_samples": 0,
        }
        remaining_for_basic = options.max_samples
        if remaining_for_basic is not None:
            remaining_for_basic = max(0, remaining_for_basic - standard_loaded)
        if include_basic_education:
            if standard_quota_exhausted:
                logger.info("Skip basic education branch because the target API quota is exhausted.")
            elif remaining_for_basic == 0:
                logger.info("Skip basic education branch because max_samples budget is exhausted.")
            else:
                basic_runner = BasicEducationExecutor(
                    project_config=self.config,
                    model_config=model_config,
                    logger=logger,
                )
                basic_result = await basic_runner.run(
                    run_id=options.run_id,
                    checkpoint=checkpoint,
                    raw_writer=raw_writer,
                    judged_writer=judged_writer,
                    failure_writer=failure_writer,
                    judge_enabled=options.judge_enabled,
                    subsets=options.subsets,
                    source_files=options.source_files,
                    dimensions=options.dimensions,
                    max_samples=remaining_for_basic,
                    max_samples_per_file=options.max_samples_per_file,
                )
                basic_stats = {
                    "scenario_count": basic_result.scenario_count,
                    "loaded_samples": basic_result.loaded_samples,
                    "completed_samples": basic_result.completed_samples,
                    "failed_samples": basic_result.failed_samples,
                }

        await raw_writer.flush()
        await judged_writer.flush()
        await failure_writer.flush()
        await retry_writer.flush()
        await checkpoint.flush()

        summary = build_summary(output_paths.judged_path, output_paths.failures_path)
        output_paths.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Run finished. summary_path=%s", output_paths.summary_path)
        return {
            "run_id": options.run_id,
            "summary_path": str(output_paths.summary_path),
            "raw_path": str(output_paths.raw_path),
            "judged_path": str(output_paths.judged_path),
            "failures_path": str(output_paths.failures_path),
            "standard_loaded_samples": standard_loaded,
            "standard_failed_samples": standard_failures,
            "quota_exhausted": standard_quota_exhausted,
            "basic_education": basic_stats,
        }

    async def _run_standard_samples(
        self,
        *,
        logger: logging.Logger,
        checkpoint: CheckpointStore,
        raw_writer: JsonlWriter,
        judged_writer: JsonlWriter,
        failure_writer: JsonlWriter,
        retry_writer: JsonlWriter,
        model_config: ModelConfig,
        options: RunOptions,
        modules: set[str] | None,
    ) -> dict[str, int]:
        provider_config = self.config.providers[model_config.provider_name]
        registry = FileRegistry(self.config)
        resolved_items = registry.resolve(
            modules=modules,
            subsets=options.subsets,
            source_files=options.source_files,
        )
        samples = list(
            self._load_samples(
                resolved_items, options.dimensions, options.max_samples, options.max_samples_per_file
            )
        )
        logger.info("Resolved %s samples from %s files.", len(samples), len(resolved_items))
        if not samples:
            return {"loaded_samples": 0, "failed_samples": 0}

        judge_router = JudgeRouter(self.config)
        client = ProviderFactory.create(provider_config, model_config)
        merged_rate_limits = self._merge_rate_limits(
            self.config.app.default_run.provider_rate_limits.get(provider_config.provider_name)
            or self.config.app.default_run.provider_rate_limits.get("default")
            or RateLimitConfig(),
            provider_config.rate_limits,
            model_config.rate_limits,
        )
        if options.max_concurrency is not None:
            merged_rate_limits.max_concurrency = options.max_concurrency

        limiter = SlidingWindowRateLimiter(merged_rate_limits)
        queue: asyncio.Queue[Sample | None] = asyncio.Queue()
        for sample in samples:
            await queue.put(sample)

        worker_count = options.max_concurrency or merged_rate_limits.max_concurrency or 100
        worker_count = min(worker_count, max(1, len(samples)))
        for _ in range(worker_count):
            await queue.put(None)

        progress = {"done": 0, "failed": 0, "skipped": 0, "total": len(samples), "quota_exhausted": 0}
        progress_reporter = _ProgressReporter(
            enabled=options.progress_enabled,
            label=model_config.model_id,
            total=len(samples),
        )
        tasks = [
            asyncio.create_task(
                self._worker(
                    queue=queue,
                    logger=logger,
                    checkpoint=checkpoint,
                    raw_writer=raw_writer,
                    judged_writer=judged_writer,
                    failure_writer=failure_writer,
                    retry_writer=retry_writer,
                    client=client,
                    model_config=model_config,
                    limiter=limiter,
                    judge_router=judge_router,
                    judge_enabled=options.judge_enabled,
                    progress=progress,
                    progress_reporter=progress_reporter,
                )
            )
            for _ in range(worker_count)
        ]
        await asyncio.gather(*tasks)
        progress_reporter.close()
        await client.aclose()
        await judge_router.aclose()
        return {
            "loaded_samples": len(samples),
            "failed_samples": progress["failed"],
            "quota_exhausted": progress["quota_exhausted"],
        }

    def _load_samples(
        self,
        resolved_items: Iterable,
        dimensions: set[str] | None,
        max_samples: int | None,
        max_samples_per_file: int | None = None,
    ) -> Iterable[Sample]:
        loaded = 0
        for item in resolved_items:
            loader = LoaderFactory.create(item.entry.loader_name)
            from_file = 0
            for sample in loader.iter_samples(item):
                if dimensions and sample.dimension not in dimensions:
                    continue
                if max_samples_per_file is not None and from_file >= max_samples_per_file:
                    break
                yield sample
                loaded += 1
                from_file += 1
                if max_samples is not None and loaded >= max_samples:
                    return

    async def _worker(
        self,
        queue: asyncio.Queue[Sample | None],
        logger: logging.Logger,
        checkpoint: CheckpointStore,
        raw_writer: JsonlWriter,
        judged_writer: JsonlWriter,
        failure_writer: JsonlWriter,
        retry_writer: JsonlWriter,
        client,
        model_config: ModelConfig,
        limiter: SlidingWindowRateLimiter,
        judge_router: JudgeRouter,
        judge_enabled: bool,
        progress: dict[str, int],
        progress_reporter: "_ProgressReporter",
    ) -> None:
        while True:
            sample = await queue.get()
            if sample is None:
                queue.task_done()
                return
            sample_key = self._sample_key(sample)
            if progress.get("quota_exhausted"):
                progress["skipped"] += 1
                progress_reporter.update(progress)
                queue.task_done()
                continue
            if checkpoint.is_completed(sample_key):
                progress["skipped"] += 1
                progress_reporter.update(progress)
                queue.task_done()
                continue
            try:
                result = await self._process_sample(
                    sample=sample,
                    client=client,
                    model_config=model_config,
                    limiter=limiter,
                    judge_router=judge_router,
                    judge_enabled=judge_enabled,
                    retry_writer=retry_writer,
                    checkpoint=checkpoint,
                )
                await raw_writer.write(
                    {
                        "sample_id": result.sample_id,
                        "source_file": result.source_file,
                        "module": result.module,
                        "subset": result.subset,
                        "task": result.task,
                        "dimension": result.dimension,
                        "reference": result.reference,
                        "provider_name": result.provider_name,
                        "model_id": result.model_id,
                        "model_name": result.model_name,
                        "model_response": result.model_response,
                        "latency_ms": result.latency_ms,
                        "retry_count": result.retry_count,
                        "timestamp": result.timestamp.isoformat(),
                        "judge_metadata": result.judge_metadata,
                        "raw_response": result.raw_response,
                        "metadata": result.metadata,
                    }
                )
                await judged_writer.write(result.model_dump(mode="json"))
                await checkpoint.mark_completed(sample_key)
                progress["done"] += 1
                progress_reporter.update(progress)
                if progress["done"] % self.config.app.default_run.log_every_n == 0:
                    logger.info("Progress: completed=%s failed=%s", progress["done"], progress["failed"])
            except Exception as exc:  # noqa: BLE001
                if _is_quota_exhausted_error(exc):
                    progress["quota_exhausted"] = 1
                    logger.error("Quota exhausted for model %s: %s", model_config.model_id, exc)
                    progress_reporter.update(progress)
                    continue
                progress["failed"] += 1
                failure = FailureRecord(
                    sample_id=sample.sample_id,
                    source_file=sample.source_file,
                    module=sample.module,
                    subset=sample.subset,
                    provider_name=model_config.provider_name,
                    model_id=model_config.model_id,
                    retry_count=checkpoint.retry_counts.get(sample_key, 0),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    timestamp=datetime.now(timezone.utc),
                    metadata={"dimension": sample.dimension},
                )
                await failure_writer.write(failure.model_dump(mode="json"))
                await checkpoint.mark_failed(sample_key)
                progress_reporter.update(progress)
                logger.exception("Sample failed: %s", sample_key)
            finally:
                queue.task_done()

    async def _process_sample(
        self,
        sample: Sample,
        client,
        model_config: ModelConfig,
        limiter: SlidingWindowRateLimiter,
        judge_router: JudgeRouter,
        judge_enabled: bool,
        retry_writer: JsonlWriter,
        checkpoint: CheckpointStore,
    ) -> EvalResult:
        prompt = augment_prompt_for_format(sample, sample.prompt)
        request = GenerationRequest(
            prompt=prompt,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            stream=bool(model_config.supports_stream),
            messages=[{"role": "user", "content": prompt}],
            provider_kwargs={},
        )
        response, retry_count = await self._generate_with_retry(
            sample=sample,
            client=client,
            request=request,
            retry_policy=model_config.retry or self.config.app.default_run.retry,
            limiter=limiter,
            retry_writer=retry_writer,
            checkpoint=checkpoint,
        )
        judge = judge_router.get_judge(sample)
        judged = await judge.judge(sample, response) if judge_enabled else None
        metadata = dict(sample.metadata)
        if response.format_valid is False:
            metadata.update(response.format_metadata)
        return EvalResult(
            sample_id=sample.sample_id,
            source_file=sample.source_file,
            source_path=sample.source_path,
            module=sample.module,
            subset=sample.subset,
            task=sample.task,
            dimension=sample.dimension,
            prompt=sample.prompt,
            reference=sample.reference,
            provider_name=model_config.provider_name,
            model_id=model_config.model_id,
            model_name=model_config.model_name,
            model_response=response.text,
            judge_result=judged.judge_result if judged else None,
            score=judged.score if judged else None,
            judge_reason=judged.judge_reason if judged else None,
            latency_ms=response.latency_ms,
            retry_count=retry_count,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata,
            judge_metadata=judged.judge_metadata if judged else {},
            raw_response=response.raw_payload,
        )

    async def _generate_with_retry(
        self,
        sample: Sample,
        client,
        request: GenerationRequest,
        retry_policy: RetryConfig,
        limiter: SlidingWindowRateLimiter,
        retry_writer: JsonlWriter,
        checkpoint: CheckpointStore,
    ) -> tuple[ModelResponse, int]:
        attempts = retry_policy.max_attempts
        retryable_codes = set(retry_policy.retryable_status_codes)
        sample_key = self._sample_key(sample)
        format_spec = get_response_format_spec(sample)
        base_prompt = request.prompt
        request_messages = list(request.messages) if request.messages else [{"role": "user", "content": base_prompt}]
        base_max_tokens = request.max_tokens
        if sample_key in checkpoint.failed_ids and format_spec is not None:
            concise_prompt = build_concise_retry_prompt(base_prompt, format_spec)
            request_messages = [{"role": "user", "content": concise_prompt}]
            request.prompt = concise_prompt
            request.max_tokens = 64
        elif sample.task in ANSWER_COMPLETION_TASKS and sample_key in checkpoint.failed_ids:
            concise_prompt = _build_concise_answer_retry_prompt(base_prompt)
            request_messages = [{"role": "user", "content": concise_prompt}]
            request.prompt = concise_prompt
            request.max_tokens = _answer_retry_max_tokens(base_max_tokens)
        for attempt in range(1, attempts + 1):
            estimated_tokens = self._estimate_tokens(request.prompt, request.max_tokens)
            await limiter.acquire(estimated_tokens=estimated_tokens)
            try:
                request.messages = request_messages
                response = await client.generate(sample=sample, request=request)
                if _should_retry_truncated_answer(sample, response):
                    await checkpoint.set_retry_count(sample_key, attempt)
                    if attempt >= attempts:
                        response.retry_count = attempt - 1
                        return response, attempt - 1
                    delay = await sleep_before_retry(retry_policy, attempt, None)
                    concise_prompt = _build_concise_answer_retry_prompt(base_prompt)
                    request_messages = [{"role": "user", "content": concise_prompt}]
                    request.prompt = concise_prompt
                    request.max_tokens = _answer_retry_max_tokens(base_max_tokens)
                    await retry_writer.write(
                        {
                            "sample_id": sample.sample_id,
                            "source_file": sample.source_file,
                            "attempt": attempt,
                            "next_delay_seconds": delay,
                            "error_type": "LengthTruncatedResponse",
                            "error_message": "Model response ended because max_tokens was reached before a reliable final answer.",
                            "retry_reason": "answer_length_truncated",
                        }
                    )
                    continue
                response = attach_format_metadata(response, format_spec)
                response.retry_count = attempt - 1
                await checkpoint.set_retry_count(sample_key, attempt - 1)
                return response, attempt - 1
            except ResponseFormatError as exc:
                await checkpoint.set_retry_count(sample_key, attempt)
                if attempt >= attempts:
                    response = locals().get("response")
                    if isinstance(response, ModelResponse):
                        response.format_valid = False
                        response.format_metadata = {
                            **exc.metadata,
                            "failure_category": "model_instruction_following",
                            "failure_type": "response_format",
                            "error_message": str(exc),
                        }
                        response.retry_count = attempt - 1
                        await checkpoint.set_retry_count(sample_key, attempt - 1)
                        return response, attempt - 1
                    raise
                delay = await sleep_before_retry(retry_policy, attempt, exc)
                previous_response_text = ""
                response = locals().get("response")
                if "response" in locals():
                    maybe_response = locals()["response"]
                    if isinstance(maybe_response, ModelResponse):
                        previous_response_text = maybe_response.text or ""
                followup = build_retry_followup(
                    format_spec,
                    response if isinstance(response, ModelResponse) else None,
                )
                if followup.get("retry_reason") == "format_empty_after_reasoning":
                    request_messages = [
                        {
                            "role": "user",
                            "content": build_concise_retry_prompt(base_prompt, format_spec),
                        }
                    ]
                else:
                    request_messages = [{"role": "user", "content": base_prompt}]
                    assistant_content = followup.get("assistant_content")
                    if assistant_content:
                        request_messages.append({"role": "assistant", "content": assistant_content})
                    user_reminder = str(followup.get("user_reminder", "") or "").strip()
                    if user_reminder:
                        request_messages.append({"role": "user", "content": user_reminder})
                request.max_tokens = followup.get("max_tokens") or base_max_tokens
                await retry_writer.write(
                    {
                        "sample_id": sample.sample_id,
                        "source_file": sample.source_file,
                        "attempt": attempt,
                        "next_delay_seconds": delay,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "retry_reason": followup.get("retry_reason", "format_invalid"),
                        "format_metadata": exc.metadata,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                await checkpoint.set_retry_count(sample_key, attempt)
                if attempt >= attempts or not is_retryable_exception(exc, retryable_codes):
                    raise
                delay = await sleep_before_retry(retry_policy, attempt, exc)
                retry_reason = "http_retry"
                if sample.task in ANSWER_COMPLETION_TASKS:
                    concise_prompt = _build_concise_answer_retry_prompt(base_prompt)
                    request_messages = [{"role": "user", "content": concise_prompt}]
                    request.prompt = concise_prompt
                    request.max_tokens = _answer_retry_max_tokens(base_max_tokens)
                    retry_reason = "answer_retry_after_error"
                await retry_writer.write(
                    {
                        "sample_id": sample.sample_id,
                        "source_file": sample.source_file,
                        "attempt": attempt,
                        "next_delay_seconds": delay,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "retry_reason": retry_reason,
                    }
                )
            finally:
                limiter.release()
        raise RuntimeError("Retry loop exited unexpectedly")

    def _get_model_config(self, model_id: str) -> ModelConfig:
        if model_id not in self.config.models:
            available = ", ".join(sorted(self.config.models))
            raise KeyError(f"Unknown model_id={model_id!r}. Available: {available}")
        return self.config.models[model_id]

    def _merge_rate_limits(self, *configs: RateLimitConfig) -> RateLimitConfig:
        merged = RateLimitConfig()
        for config in configs:
            if config.rpm is not None:
                merged.rpm = config.rpm
            if config.tpm is not None:
                merged.tpm = config.tpm
            if config.qps is not None:
                merged.qps = config.qps
            if config.max_concurrency is not None:
                merged.max_concurrency = config.max_concurrency
        return merged

    def _sample_key(self, sample: Sample) -> str:
        return f"{sample.source_file}::{sample.sample_id}"

    def _estimate_tokens(self, prompt: str, max_tokens: int | None) -> int:
        prompt_tokens = max(1, len(prompt) // 4)
        return prompt_tokens + (max_tokens or 0)


class _ProgressReporter:
    def __init__(self, *, enabled: bool, label: str, total: int) -> None:
        self.enabled = enabled
        self.label = label
        self.total = max(1, total)
        self._last_render = ""

    def update(self, progress: dict[str, int]) -> None:
        if not self.enabled:
            return
        completed = progress.get("done", 0) + progress.get("failed", 0) + progress.get("skipped", 0)
        width = 28
        filled = min(width, int(width * completed / self.total))
        bar = "#" * filled + "-" * (width - filled)
        text = (
            f"\r{self.label} [{bar}] {completed}/{self.total} "
            f"ok={progress.get('done', 0)} failed={progress.get('failed', 0)} "
            f"skipped={progress.get('skipped', 0)}"
        )
        if progress.get("quota_exhausted"):
            text += " quota_exhausted"
        if text != self._last_render:
            print(text, end="", file=sys.stderr, flush=True)
            self._last_render = text

    def close(self) -> None:
        if self.enabled and self._last_render:
            print(file=sys.stderr)


def _is_quota_exhausted_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    markers = (
        "insufficient quota",
        "quota exceeded",
        "quota_exceeded",
        "balance",
        "insufficient balance",
        "\u4f59\u989d",
        "\u989d\u5ea6",
        "credit",
        "payment required",
    )
    return any(marker in text for marker in markers)


def _should_retry_truncated_answer(sample: Sample, response: ModelResponse) -> bool:
    if sample.task not in ANSWER_COMPLETION_TASKS:
        return False
    payload = response.raw_payload if isinstance(response.raw_payload, dict) else {}
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    first = choices[0] if isinstance(choices[0], dict) else {}
    finish_reason = str(first.get("finish_reason") or "").strip().lower()
    return finish_reason == "length"


def _answer_retry_max_tokens(base_max_tokens: int | None) -> int:
    return base_max_tokens or 512


def _build_concise_answer_retry_prompt(base_prompt: str) -> str:
    lines = []
    for line in base_prompt.splitlines():
        cleaned = re.sub(
            r"\b(?:please\s+)?(?:reason|think)\s+step\s+by\s+step\b(?:\s+before\s+answering)?[,.]?\s*(?:and\s+)?",
            "",
            line,
            flags=re.I,
        ).strip()
        if cleaned:
            lines.append(cleaned)
    prompt = "\n".join(lines).rstrip()
    return (
        f"{prompt}\n\n"
        "Do not include any reasoning, explanation, or thinking. "
        "Output only the final answer. Put the final answer within \\boxed{}."
    ).strip()

