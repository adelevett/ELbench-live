from __future__ import annotations

import asyncio
import importlib.util
import itertools
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from elbench.persistence import CheckpointStore, JsonlWriter
from elbench.schemas.config import ModelConfig, ProjectConfig
from elbench.schemas.evaluation import EvalResult, FailureRecord
from elbench.utils.secrets import get_api_key
RUNTIME_REQUIRED_MODULES = (
    "aiosqlite",
    "click",
    "langchain",
    "langgraph",
    "polyfactory",
    "tenacity",
    "tqdm",
)


DEFAULT_BASIC_MODULE_NAME = "基本教育"

DEFAULT_MULTI_TURN_RECURSION_LIMIT = 20

DEFAULT_EVALUATOR_MAX_TOKENS = 1024


class BasicEducationScenarioConfig(BaseModel):
    scenario_id: str
    source_file: str
    subset: str
    task: str
    template_path: Path
    expected_tasks: int | None = None
    enabled: bool = True
    multi_turn: bool = False
    dimension: str | None = None
    target_model_keys: list[str] = Field(default_factory=list)
    test_endpoint_model_keys: list[str] = Field(default_factory=list)
    judge_model_keys: list[str] = Field(default_factory=list)
    pass_threshold: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BasicEducationConfig(BaseModel):
    enabled: bool = False
    module_name: str = DEFAULT_BASIC_MODULE_NAME
    runtime_python: str = "python"
    runtime_cli_module: str = "elbench.basic_education_runtime.cli.main"
    command_timeout_seconds: int = 7200
    continue_on_error: bool = False
    workspace_root: Path = Path("outputs/basic_education")
    test_endpoint_model_id: str | None = None
    evaluator_model_id: str | None = None
    default_pass_threshold: float | None = 3.0
    scenarios: list[BasicEducationScenarioConfig] = Field(default_factory=list)


@dataclass(slots=True)
class BasicEducationRunStats:
    scenario_count: int = 0
    loaded_samples: int = 0
    completed_samples: int = 0
    failed_samples: int = 0


def load_basic_education_config(
    config_root: Path,
    project_root: Path,
    data_root: Path | None = None,
) -> BasicEducationConfig:
    config_path = config_root / "basic_education.yaml"
    if not config_path.exists():
        return BasicEducationConfig(enabled=False)

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = BasicEducationConfig.model_validate(raw)

    if not config.workspace_root.is_absolute():
        config.workspace_root = (project_root / config.workspace_root).resolve()

    template_base = data_root or (project_root / "data" / "benchmark_root")
    for scenario in config.scenarios:
        if not scenario.template_path.is_absolute():
            scenario.template_path = (template_base / scenario.template_path).resolve()

    return config


class BasicEducationExecutor:
    def __init__(
        self,
        project_config: ProjectConfig,
        model_config: ModelConfig,
        logger: logging.Logger,
    ) -> None:
        self.project_config = project_config
        self.model_config = model_config
        self.logger = logger
        self.config = load_basic_education_config(
            config_root=self.project_config.config_root,
            project_root=self.project_config.project_root,
            data_root=self.project_config.app.data_root,
        )

    async def run(
        self,
        *,
        run_id: str,
        checkpoint: CheckpointStore,
        raw_writer: JsonlWriter,
        judged_writer: JsonlWriter,
        failure_writer: JsonlWriter,
        judge_enabled: bool,
        subsets: set[str] | None,
        source_files: set[str] | None,
        dimensions: set[str] | None,
        max_samples: int | None,
        max_samples_per_file: int | None = None,
    ) -> BasicEducationRunStats:
        if not self.config.enabled:
            raise ValueError(
                "Basic education integration is disabled. "
                "Enable it in configs/basic_education.yaml first."
            )
        self._ensure_runtime_dependencies()

        scenarios = self._select_scenarios(
            subsets=subsets,
            source_files=source_files,
            dimensions=dimensions,
        )
        stats = BasicEducationRunStats(scenario_count=len(scenarios))
        if not scenarios:
            self.logger.info("No basic education scenarios selected after filters.")
            return stats

        max_remaining = max_samples
        for scenario in scenarios:
            if max_remaining is not None and max_remaining <= 0:
                self.logger.info("Reached max_samples limit for basic education branch.")
                break
            scenario_cap = max_remaining
            if max_samples_per_file is not None:
                scenario_cap = max_samples_per_file if scenario_cap is None else min(scenario_cap, max_samples_per_file)
            scenario_stats = await self._run_one_scenario(
                scenario=scenario,
                run_id=run_id,
                checkpoint=checkpoint,
                raw_writer=raw_writer,
                judged_writer=judged_writer,
                failure_writer=failure_writer,
                judge_enabled=judge_enabled,
                max_samples=scenario_cap,
            )
            stats.loaded_samples += scenario_stats.loaded_samples
            stats.completed_samples += scenario_stats.completed_samples
            stats.failed_samples += scenario_stats.failed_samples
            if max_remaining is not None:
                max_remaining = max(0, max_remaining - scenario_stats.loaded_samples)
        return stats

    def _select_scenarios(
        self,
        *,
        subsets: set[str] | None,
        source_files: set[str] | None,
        dimensions: set[str] | None,
    ) -> list[BasicEducationScenarioConfig]:
        scenarios: list[BasicEducationScenarioConfig] = []
        for scenario in self.config.scenarios:
            if not scenario.enabled:
                continue
            if subsets and scenario.subset not in subsets and scenario.scenario_id not in subsets:
                continue
            if source_files and scenario.source_file not in source_files:
                continue
            scenario_dimension = scenario.dimension or scenario.scenario_id
            if dimensions and scenario_dimension not in dimensions:
                continue
            scenarios.append(scenario)
        return scenarios

    def _ensure_runtime_dependencies(self) -> None:
        missing = [
            module_name
            for module_name in RUNTIME_REQUIRED_MODULES
            if importlib.util.find_spec(module_name) is None
        ]
        if not missing:
            return
        missing_display = ", ".join(missing)
        raise RuntimeError(
            "Basic education runtime dependencies are missing: "
            f"{missing_display}. Install them with `pip install -e .[basic-education]`."
        )

    async def _run_one_scenario(
        self,
        *,
        scenario: BasicEducationScenarioConfig,
        run_id: str,
        checkpoint: CheckpointStore,
        raw_writer: JsonlWriter,
        judged_writer: JsonlWriter,
        failure_writer: JsonlWriter,
        judge_enabled: bool,
        max_samples: int | None,
    ) -> BasicEducationRunStats:
        scenario_checkpoint_key = f"basic_education::scenario::{scenario.scenario_id}"
        if checkpoint.is_completed(scenario_checkpoint_key):
            self.logger.info("Scenario already completed by checkpoint: %s", scenario.scenario_id)
            return BasicEducationRunStats(scenario_count=1)

        runtime_root = self.config.workspace_root / run_id / self.model_config.model_id / scenario.scenario_id
        runtime_root.mkdir(parents=True, exist_ok=True)
        memory_path = runtime_root / "memory"
        rendered_config_path = runtime_root / "basic_education_runtime.yaml"

        try:
            template_data = self._read_yaml_file(scenario.template_path)
            declared_task_count = count_runtime_tasks(template_data)
            if scenario.expected_tasks is not None and declared_task_count != scenario.expected_tasks:
                self.logger.warning(
                    "Scenario task-count mismatch: scenario=%s expected=%s declared=%s",
                    scenario.scenario_id,
                    scenario.expected_tasks,
                    declared_task_count,
                )

            rendered = self._render_runtime_template(
                template_data=template_data,
                scenario=scenario,
                memory_path=memory_path,
                max_samples=max_samples,
            )
            rendered_config_path.write_text(
                yaml.safe_dump(rendered, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            await self._write_scenario_failure(
                scenario=scenario,
                failure_writer=failure_writer,
                message=f"Scenario prepare failed: {exc}",
                checkpoint=checkpoint,
            )
            if not self.config.continue_on_error:
                raise
            return BasicEducationRunStats(scenario_count=1, failed_samples=1)

        try:
            self._reset_runtime_artifacts(memory_path)
            if judge_enabled:
                await self._run_runtime_command(["pipeline", "--config", str(rendered_config_path)])
            else:
                await self._run_runtime_command(
                    ["generate", "--config", str(rendered_config_path)],
                    memory_path=memory_path,
                    expected_samples=max_samples,
                )
                await self._run_runtime_command(["export", "json", "--config", str(rendered_config_path)])
        except Exception as exc:  # noqa: BLE001
            await self._write_scenario_failure(
                scenario=scenario,
                failure_writer=failure_writer,
                message=f"Basic education runtime command failed: {exc}",
                checkpoint=checkpoint,
            )
            if not self.config.continue_on_error:
                raise
            return BasicEducationRunStats(scenario_count=1, failed_samples=1)

        stats = await self._import_runtime_outputs(
            scenario=scenario,
            memory_path=memory_path,
            checkpoint=checkpoint,
            raw_writer=raw_writer,
            judged_writer=judged_writer,
            failure_writer=failure_writer,
            max_samples=max_samples,
            judge_enabled=judge_enabled,
        )
        await checkpoint.mark_completed(scenario_checkpoint_key)
        return stats

    def _render_runtime_template(
        self,
        *,
        template_data: dict[str, Any],
        scenario: BasicEducationScenarioConfig,
        memory_path: Path,
        max_samples: int | None,
    ) -> dict[str, Any]:
        rendered = json.loads(json.dumps(template_data, ensure_ascii=False))
        globals_section = rendered.setdefault("globals", {})
        globals_section.setdefault("memory", {})
        globals_section["memory"]["path"] = str(memory_path)
        globals_section["concurrency"] = self._runtime_concurrency_limit()
        if self.model_config.timeout is not None:
            globals_section["model_call_timeout_seconds"] = int(self.model_config.timeout)
        if scenario.multi_turn:
            existing_recursion_limit = globals_section.get("recursion_limit")
            try:
                current_recursion_limit = int(existing_recursion_limit)
            except (TypeError, ValueError):
                current_recursion_limit = DEFAULT_MULTI_TURN_RECURSION_LIMIT
            globals_section["recursion_limit"] = min(
                current_recursion_limit,
                DEFAULT_MULTI_TURN_RECURSION_LIMIT,
            )

        models_section = rendered.get("models")
        if not isinstance(models_section, dict) or not models_section:
            raise ValueError(f"Invalid or missing models section for scenario={scenario.scenario_id}")

        target_payload = self._build_runtime_model_payload(self.model_config)
        target_keys = [key for key in scenario.target_model_keys if key in models_section]
        if not target_keys:
            raise ValueError(
                f"Scenario {scenario.scenario_id} has no matching target_model_keys in template models. "
                f"Configured keys={scenario.target_model_keys}, available={list(models_section)}"
            )
        target_payload["role"] = "target"
        for key in target_keys:
            models_section[key] = target_payload

        test_endpoint_keys = [
            key for key in scenario.test_endpoint_model_keys if key in models_section
        ]
        if scenario.test_endpoint_model_keys and not test_endpoint_keys:
            raise ValueError(
                f"Scenario {scenario.scenario_id} has no matching test_endpoint_model_keys in template models. "
                f"Configured keys={scenario.test_endpoint_model_keys}, available={list(models_section)}"
            )
        test_endpoint_model = self._resolve_test_endpoint_model()
        if test_endpoint_keys:
            if test_endpoint_model is None:
                raise ValueError(
                    f"Scenario {scenario.scenario_id} requires a configured basic education test endpoint model "
                    f"for keys={scenario.test_endpoint_model_keys}."
                )
            test_endpoint_payload = self._build_runtime_model_payload(test_endpoint_model)
            test_endpoint_payload["role"] = "student"
            for key in test_endpoint_keys:
                models_section[key] = test_endpoint_payload

        evaluation_section = rendered.get("evaluation")
        evaluator_model = self._resolve_evaluator_model()
        if isinstance(evaluation_section, dict) and evaluator_model is not None:
            judge_payload = self._build_runtime_model_payload(evaluator_model)
            if judge_payload.get("type") != "system_one":
                self._tune_runtime_evaluator_payload(judge_payload)
            judge_payload["role"] = "judge"
            judge_keys = [key for key in scenario.judge_model_keys if key in models_section]
            if not judge_keys:
                eval_model_key = evaluation_section.get("model")
                if isinstance(eval_model_key, str):
                    judge_keys = [eval_model_key]
            if not judge_keys:
                judge_keys = ["elbench_judge_model"]
            for key in judge_keys:
                models_section[key] = judge_payload
            evaluation_section["model"] = judge_keys[0]

        self._limit_runtime_tasks(rendered, max_samples)

        return rendered

    def _limit_runtime_tasks(self, rendered: dict[str, Any], max_samples: int | None) -> None:
        if max_samples is None:
            return
        if max_samples <= 0:
            raise ValueError("max_samples must be positive when limiting runtime tasks.")

        tasks_obj = rendered.get("tasks")
        if not isinstance(tasks_obj, dict):
            return

        mode = str(tasks_obj.get("mode", "iter")).lower()
        content = tasks_obj.get("content")

        if mode == "iter":
            if isinstance(content, list):
                tasks_obj["content"] = content[:max_samples]
            return

        if mode == "union":
            if not isinstance(content, dict) or not content:
                return
            keys = list(content.keys())
            value_lists: list[list[Any]] = []
            for key in keys:
                value = content.get(key)
                if not isinstance(value, list):
                    return
                value_lists.append(value)
            combinations = itertools.islice(itertools.product(*value_lists), max_samples)
            tasks_obj["mode"] = "iter"
            tasks_obj["content"] = [dict(zip(keys, combo)) for combo in combinations]

    def _runtime_concurrency_limit(self) -> int:
        limits = [self.project_config.app.default_run.max_concurrency]
        provider_config = self.project_config.providers.get(self.model_config.provider_name)
        if provider_config is not None:
            limits.append(provider_config.rate_limits.max_concurrency)
        limits.append(self.model_config.rate_limits.max_concurrency)
        positive_limits = [int(value) for value in limits if value is not None and int(value) > 0]
        if not positive_limits:
            return 1
        return max(1, min(positive_limits))

    def _resolve_test_endpoint_model(self) -> ModelConfig | None:
        model_id = (
            self.project_config.judges.basic_education_test_endpoint_model_id
            or self.config.test_endpoint_model_id
            or self.config.evaluator_model_id
            or self.project_config.judges.default_judge_model_id
        )
        if model_id is None:
            return None
        if model_id not in self.project_config.models:
            raise KeyError(f"Unknown basic education test endpoint model_id={model_id!r}")
        return self.project_config.models[model_id]

    def _resolve_evaluator_model(self) -> ModelConfig | None:
        model_id = (
            self.project_config.judges.basic_education_evaluator_model_id
            or self.config.evaluator_model_id
            or self.project_config.judges.basic_education_test_endpoint_model_id
            or self.config.test_endpoint_model_id
            or self.project_config.judges.default_judge_model_id
        )
        if model_id is None:
            return None
        if model_id not in self.project_config.models:
            raise KeyError(f"Unknown basic education evaluator model_id={model_id!r}")
        return self.project_config.models[model_id]

    def _build_runtime_model_payload(self, model_config: ModelConfig) -> dict[str, Any]:
        provider_kwargs = dict(model_config.provider_kwargs)
        is_system_one = bool(provider_kwargs.pop("system_one", False)) or (
            model_config.provider_name == "system_one"
        )
        runtime_type = str(provider_kwargs.pop("runtime_type", "")).strip()
        if not runtime_type:
            if model_config.provider_name == "mock":
                runtime_type = "mock"
            elif is_system_one:
                runtime_type = "system_one"
            else:
                runtime_type = "openai"

        runtime_model_name = str(
            provider_kwargs.pop("runtime_model_name", model_config.model_name)
        ).strip()
        runtime_api_base = provider_kwargs.pop("runtime_api_base", model_config.api_base)

        if runtime_type != "mock" and not runtime_api_base:
            raise ValueError(
                f"Model {model_config.model_id} requires api_base (or provider_kwargs.runtime_api_base) "
                "for basic education runtime integration."
            )
        api_key = ""
        if model_config.api_key_env:
            api_key = get_api_key(model_config.api_key_env)
            if not api_key:
                raise ValueError(
                    f"Environment variable {model_config.api_key_env} is not set "
                    f"for model {model_config.model_id}."
                )

        payload: dict[str, Any] = {
            "type": runtime_type,
            "api_key": api_key,
            "api_base": runtime_api_base,
            "model": runtime_model_name,
        }
        kargs: dict[str, Any] = {}
        if model_config.temperature is not None:
            kargs["temperature"] = model_config.temperature
        if model_config.timeout is not None:
            kargs["timeout"] = int(model_config.timeout)
            kargs["request_timeout"] = int(model_config.timeout)
        runtime_max_tokens = provider_kwargs.pop("runtime_max_tokens", None)
        if runtime_max_tokens is not None:
            kargs["max_tokens"] = int(runtime_max_tokens)
        elif model_config.max_tokens is not None:
            kargs["max_tokens"] = model_config.max_tokens
        if provider_kwargs:
            kargs.update(provider_kwargs)
        if runtime_type == "openai":
            self._normalize_langchain_openai_kargs(kargs)
        if kargs:
            payload["kargs"] = kargs
        return payload

    def _normalize_langchain_openai_kargs(self, kargs: dict[str, Any]) -> None:
        """LangChain's OpenAI wrapper requires nonstandard request fields in extra_body."""
        extra_body = kargs.get("extra_body")
        if not isinstance(extra_body, dict):
            extra_body = {}

        # "provider" is OpenRouter's host-routing field.
        for key in ("chat_template_kwargs", "served_model_name", "provider"):
            if key not in kargs:
                continue
            value = kargs.pop(key)
            extra_body.setdefault(key, value)

        if extra_body:
            kargs["extra_body"] = extra_body

    def _tune_runtime_evaluator_payload(self, payload: dict[str, Any]) -> None:
        kargs = payload.setdefault("kargs", {})
        # A card that sets a limit gets at least DEFAULT_EVALUATOR_MAX_TOKENS; a judge with no limit keeps none.
        current_max_tokens = kargs.get("max_tokens")
        if current_max_tokens is not None:
            try:
                max_tokens = int(current_max_tokens)
            except (TypeError, ValueError):
                max_tokens = 0
            if max_tokens < DEFAULT_EVALUATOR_MAX_TOKENS:
                kargs["max_tokens"] = DEFAULT_EVALUATOR_MAX_TOKENS
        extra_body = kargs.setdefault("extra_body", {})
        if isinstance(extra_body, dict):
            chat_template_kwargs = extra_body.setdefault("chat_template_kwargs", {})
            if isinstance(chat_template_kwargs, dict):
                chat_template_kwargs.setdefault("enable_thinking", False)

    def _reset_runtime_artifacts(self, memory_path: Path) -> None:
        if memory_path.exists():
            shutil.rmtree(memory_path)
        memory_path.mkdir(parents=True, exist_ok=True)

    async def _run_runtime_command(
        self,
        cli_args: list[str],
        *,
        memory_path: Path | None = None,
        expected_samples: int | None = None,
    ) -> None:
        # "python" means this interpreter, so the runtime sees the same installed packages even when
        # no bare `python` is on PATH (a venv started by full path, macOS with only python3).
        runtime_python = sys.executable if self.config.runtime_python == "python" else self.config.runtime_python
        command = [runtime_python, "-m", self.config.runtime_cli_module, *cli_args]
        env = os.environ.copy()
        cwd = str(self.project_config.project_root)
        project_src = self.project_config.project_root / "src"
        current_pythonpath = env.get("PYTHONPATH")
        pythonpath_parts = [str(project_src)]
        if current_pythonpath:
            pythonpath_parts.append(current_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

        self.logger.info("Running basic education runtime command: %s", " ".join(command))
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        completed_via_termination = False
        try:
            if memory_path is not None and any(arg == "generate" for arg in cli_args):
                stdout, stderr, completed_via_termination = await self._wait_for_generate_completion(
                    process=process,
                    memory_path=memory_path,
                    expected_samples=expected_samples,
                )
            else:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.command_timeout_seconds,
                )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise TimeoutError(
                f"Basic education runtime command timeout after {self.config.command_timeout_seconds}s"
            )

        stdout_text = stdout.decode("utf-8", errors="ignore").strip()
        stderr_text = stderr.decode("utf-8", errors="ignore").strip()
        if stdout_text:
            self.logger.info("Basic education runtime stdout:\n%s", stdout_text)
        if stderr_text:
            self.logger.warning("Basic education runtime stderr:\n%s", stderr_text)
        if completed_via_termination:
            self.logger.info(
                "Basic education runtime command completed based on exported conversation state; "
                "ignore process returncode=%s after forced termination.",
                process.returncode,
            )
            return
        if process.returncode != 0:
            raise RuntimeError(
                f"Basic education runtime command failed with exit code {process.returncode}"
            )

    async def _wait_for_generate_completion(
        self,
        *,
        process: asyncio.subprocess.Process,
        memory_path: Path,
        expected_samples: int | None,
    ) -> tuple[bytes, bytes, bool]:
        deadline = asyncio.get_running_loop().time() + self.config.command_timeout_seconds
        communicate_task = asyncio.create_task(process.communicate())
        while True:
            try:
                stdout, stderr = await asyncio.wait_for(asyncio.shield(communicate_task), timeout=2)
                return stdout, stderr, False
            except asyncio.TimeoutError:
                if self._is_generation_complete(memory_path, expected_samples):
                    process.terminate()
                    try:
                        stdout, stderr = await asyncio.wait_for(asyncio.shield(communicate_task), timeout=15)
                        return stdout, stderr, True
                    except asyncio.TimeoutError:
                        process.kill()
                        stdout, stderr = await communicate_task
                        return stdout, stderr, True
                if asyncio.get_running_loop().time() >= deadline:
                    raise

    def _is_generation_complete(self, memory_path: Path, expected_samples: int | None) -> bool:
        db_files = sorted(memory_path.glob("*.db"))
        if not db_files:
            return False
        if expected_samples is not None and len(db_files) < expected_samples:
            return False
        completed = 0
        for db_file in db_files:
            try:
                obj = self._export_runtime_db(db_file)
            except Exception:
                return False
            messages = obj.get("messages", [])
            if not isinstance(messages, list) or not messages:
                return False
            last_message = messages[-1]
            if not isinstance(last_message, dict):
                return False
            content = str(last_message.get("content", "") or "")
            role = str(last_message.get("role", "") or "")
            if role == "teacher" and "<end>" in content:
                completed += 1
        return completed == len(db_files)

    def _export_runtime_db(self, input_path: Path) -> dict[str, Any]:
        import sqlite3
        from langgraph.checkpoint.base import Checkpoint
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        conn = sqlite3.connect(input_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "select checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint from checkpoints"
            )
            results = cursor.fetchall()
            if not results:
                return {"task": {}, "messages": []}
            _, _, _, raw_checkpoint = results[-1]
            checkpoint: Checkpoint = JsonPlusSerializer().loads_typed(("msgpack", raw_checkpoint))
            messages = []
            for message in checkpoint.get("channel_values", {}).get("messages", []):
                name = getattr(message, "name", None)
                if name is None:
                    continue
                content = self._stringify_runtime_message_content(getattr(message, "content", None))
                messages.append({"role": name, "content": content})

            cursor.execute("select key, value from task")
            task = {key: value for key, value in cursor.fetchall()}
            return {"task": task, "messages": messages}
        finally:
            conn.close()

    def _stringify_runtime_message_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    block_type = str(item.get("type", "")).strip().lower()
                    if block_type == "reasoning":
                        continue
                    text = item.get("text")
                    if text not in (None, ""):
                        parts.append(str(text))
                        continue
                    nested = item.get("summary")
                    nested_text = self._stringify_runtime_message_content(nested)
                    if nested_text:
                        parts.append(nested_text)
                        continue
                else:
                    item_text = self._stringify_runtime_message_content(item)
                    if item_text:
                        parts.append(item_text)
            return "\n".join(part for part in parts if part).strip()
        if isinstance(content, dict):
            if "text" in content:
                return self._stringify_runtime_message_content(content.get("text"))
            if "content" in content:
                return self._stringify_runtime_message_content(content.get("content"))
            if "summary" in content:
                return self._stringify_runtime_message_content(content.get("summary"))
            return ""
        return str(content)

    async def _import_runtime_outputs(
        self,
        *,
        scenario: BasicEducationScenarioConfig,
        memory_path: Path,
        checkpoint: CheckpointStore,
        raw_writer: JsonlWriter,
        judged_writer: JsonlWriter,
        failure_writer: JsonlWriter,
        max_samples: int | None,
        judge_enabled: bool,
    ) -> BasicEducationRunStats:
        stats = BasicEducationRunStats(scenario_count=1)
        if not memory_path.exists():
            await self._write_scenario_failure(
                scenario=scenario,
                failure_writer=failure_writer,
                message=f"Basic education runtime memory path not found: {memory_path}",
                checkpoint=checkpoint,
            )
            stats.failed_samples += 1
            return stats

        conversation_files = sorted(memory_path.glob("*.json"))
        eval_dir = memory_path / "eval"
        eval_map: dict[str, dict[str, Any]] = {}
        if eval_dir.exists():
            for eval_file in eval_dir.glob("*.json"):
                eval_map[eval_file.stem] = self._read_json_file(eval_file)

        for conv_file in conversation_files:
            if max_samples is not None and stats.loaded_samples >= max_samples:
                break

            task_id = conv_file.stem
            sample_key = f"basic_education::{scenario.scenario_id}::{task_id}"
            if checkpoint.is_completed(sample_key):
                continue

            stats.loaded_samples += 1
            try:
                conv_obj = self._read_json_file(conv_file)
                eval_obj = eval_map.get(task_id, {}) if judge_enabled else {}
                score, score_items = aggregate_numeric_score(eval_obj)
                threshold = (
                    scenario.pass_threshold
                    if scenario.pass_threshold is not None
                    else self.config.default_pass_threshold
                )
                judge_result, judge_reason = self._resolve_judge_result(
                    score=score,
                    threshold=threshold,
                    judge_enabled=judge_enabled,
                )
                prompt = self._extract_prompt(conv_obj.get("task", {}))
                message_items = conv_obj.get("messages", [])
                response_text = dialog_from_messages(message_items)
                sample_id = f"{scenario.scenario_id}-{task_id}"
                dimension = scenario.dimension or scenario.scenario_id
                metadata = {
                    **scenario.metadata,
                    "_scene": scenario.scenario_id,
                    "multi_turn": scenario.multi_turn,
                    "basic_education_runtime_task_id": task_id,
                    "message_count": len(message_items) if isinstance(message_items, list) else 0,
                    "source": "basic_education_runtime",
                }
                judge_metadata = {
                    "score_items": score_items,
                    "raw_eval": eval_obj,
                }

                result = EvalResult(
                    sample_id=sample_id,
                    source_file=scenario.source_file,
                    source_path=str(scenario.template_path),
                    module=self.config.module_name,
                    subset=scenario.subset,
                    task=scenario.task,
                    dimension=dimension,
                    prompt=prompt,
                    reference=None,
                    provider_name=self.model_config.provider_name,
                    model_id=self.model_config.model_id,
                    model_name=self.model_config.model_name,
                    model_response=response_text,
                    judge_result=judge_result,
                    score=score,
                    judge_reason=judge_reason,
                    latency_ms=None,
                    retry_count=0,
                    timestamp=datetime.now(timezone.utc),
                    metadata=metadata,
                    judge_metadata=judge_metadata,
                    raw_response={
                        "basic_education_runtime_task": conv_obj.get("task", {}),
                        "basic_education_runtime_messages": message_items,
                    },
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
                stats.completed_samples += 1
            except Exception as exc:  # noqa: BLE001
                stats.failed_samples += 1
                failure = FailureRecord(
                    sample_id=f"{scenario.scenario_id}-{task_id}",
                    source_file=scenario.source_file,
                    module=self.config.module_name,
                    subset=scenario.subset,
                    provider_name=self.model_config.provider_name,
                    model_id=self.model_config.model_id,
                    retry_count=0,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    timestamp=datetime.now(timezone.utc),
                    metadata={
                        "scenario_id": scenario.scenario_id,
                        "source": "basic_education_runtime_import",
                    },
                )
                await failure_writer.write(failure.model_dump(mode="json"))
                await checkpoint.mark_failed(sample_key)
                if not self.config.continue_on_error:
                    raise
        return stats

    async def _write_scenario_failure(
        self,
        *,
        scenario: BasicEducationScenarioConfig,
        failure_writer: JsonlWriter,
        message: str,
        checkpoint: CheckpointStore,
    ) -> None:
        failure = FailureRecord(
            sample_id=f"{scenario.scenario_id}-scenario",
            source_file=scenario.source_file,
            module=self.config.module_name,
            subset=scenario.subset,
            provider_name=self.model_config.provider_name,
            model_id=self.model_config.model_id,
            retry_count=0,
            error_type="BasicEducationScenarioError",
            error_message=message,
            timestamp=datetime.now(timezone.utc),
            metadata={"scenario_id": scenario.scenario_id},
        )
        await failure_writer.write(failure.model_dump(mode="json"))
        await checkpoint.mark_failed(f"basic_education::scenario::{scenario.scenario_id}")

    def _extract_prompt(self, task_obj: dict[str, Any]) -> str:
        for key in ("question", "query", "prompt", "instruction"):
            value = task_obj.get(key)
            if value not in (None, ""):
                return str(value)
        if task_obj:
            return json.dumps(task_obj, ensure_ascii=False)
        return ""

    def _resolve_judge_result(
        self,
        *,
        score: float | None,
        threshold: float | None,
        judge_enabled: bool,
    ) -> tuple[str | None, str | None]:
        if not judge_enabled:
            return None, "Judge disabled for this run."
        if score is None:
            return "unknown", "No numeric score found in basic education runtime eval output."
        if threshold is None:
            return "pass", "Scored result imported from basic education runtime (no pass threshold configured)."
        if score >= threshold:
            return "pass", f"Score {score:.4f} >= threshold {threshold:.4f}."
        return "fail", f"Score {score:.4f} < threshold {threshold:.4f}."

    def _read_yaml_file(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            obj = yaml.safe_load(handle) or {}
        if not isinstance(obj, dict):
            raise ValueError(f"Expected YAML object at {path}, got {type(obj).__name__}")
        return obj

    def _read_json_file(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            obj = json.load(handle)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object at {path}, got {type(obj).__name__}")
        return obj


def count_runtime_tasks(config_obj: dict[str, Any]) -> int:
    tasks_obj = config_obj.get("tasks")
    if not isinstance(tasks_obj, dict):
        return 0
    mode = str(tasks_obj.get("mode", "iter")).lower()
    content = tasks_obj.get("content")
    if mode == "iter":
        return len(content) if isinstance(content, list) else 0
    if mode == "union":
        if not isinstance(content, dict) or not content:
            return 0
        lengths = [len(value) for value in content.values() if isinstance(value, list)]
        if not lengths or len(lengths) != len(content):
            return 0
        return reduce(mul, lengths, 1)
    return 0


def aggregate_numeric_score(eval_obj: dict[str, Any]) -> tuple[float | None, dict[str, float]]:
    numeric_items: dict[str, float] = {}

    def visit(value: Any, prefix: str) -> None:
        if isinstance(value, bool):
            numeric_items[prefix] = float(value)
            return
        if isinstance(value, (int, float)):
            numeric_items[prefix] = float(value)
            return
        if isinstance(value, dict):
            for key, child in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                visit(child, child_prefix)
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                child_prefix = f"{prefix}[{index}]"
                visit(child, child_prefix)

    visit(eval_obj, "")
    if not numeric_items:
        return None, {}
    avg = sum(numeric_items.values()) / len(numeric_items)
    cleaned = {key.lstrip("."): value for key, value in numeric_items.items()}
    return avg, cleaned


def dialog_from_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    lines: list[str] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "unknown"))
        content = str(item.get("content", ""))
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
