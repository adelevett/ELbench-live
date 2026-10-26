from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from elbench.loaders import LoaderFactory
from elbench.registry import FileRegistry
from elbench.schemas.config import ModelConfig, ProjectConfig
from elbench.utils.secrets import get_api_key

from .basic_education import DEFAULT_BASIC_MODULE_NAME, load_basic_education_config
from .selection import ModuleSelection, resolve_module_selection


LLM_JUDGE_TASKS = {
    "safety_refusal",
    "safety_guidance",
    "safety_answer",
    "adversarial_safety",
    "highlevel_edu",
}


class PreflightError(RuntimeError):
    pass


@dataclass(slots=True)
class PreflightIssue:
    level: str
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PreflightReport:
    ok: bool
    selection: ModuleSelection
    total_samples: int
    by_module: dict[str, int]
    by_source_file: dict[str, int]
    llm_judge_tasks: list[str]
    issues: list[PreflightIssue]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "selected_modules": sorted(self.selection.selected_modules),
            "standard_modules": sorted(self.selection.standard_modules),
            "include_basic_education": self.selection.include_basic_education,
            "total_samples": self.total_samples,
            "by_module": self.by_module,
            "by_source_file": self.by_source_file,
            "llm_judge_tasks": self.llm_judge_tasks,
            "issues": [
                {
                    "level": issue.level,
                    "code": issue.code,
                    "message": issue.message,
                    "details": issue.details,
                }
                for issue in self.issues
            ],
        }


class PreflightRunner:
    def __init__(self, config: ProjectConfig) -> None:
        self._config = config

    def run(
        self,
        *,
        model_id: str,
        modules: set[str] | None = None,
        subsets: set[str] | None = None,
        source_files: set[str] | None = None,
        dimensions: set[str] | None = None,
        max_samples: int | None = None,
        judge_enabled: bool = True,
        require_real_judges: bool = True,
    ) -> PreflightReport:
        issues: list[PreflightIssue] = []
        model_config = self._get_model_config(model_id)
        selection = resolve_module_selection(self._config, modules)

        if not judge_enabled:
            issues.append(
                PreflightIssue(
                    level="error",
                    code="judge_disabled",
                    message="judge_enabled=False 会导致结果无法产出完整评测分数。",
                )
            )

        total_samples = 0
        by_module: dict[str, int] = {}
        by_source_file: dict[str, int] = {}
        llm_judge_tasks: set[str] = set()

        registry = FileRegistry(self._config)
        if selection.standard_modules:
            resolved_items = registry.resolve(
                modules=set(selection.standard_modules),
                subsets=subsets,
                source_files=source_files,
            )
            loaded_total = 0
            for item in resolved_items:
                loader = LoaderFactory.create(item.entry.loader_name)
                count = 0
                for sample in loader.iter_samples(item):
                    if dimensions and sample.dimension not in dimensions:
                        continue
                    count += 1
                    loaded_total += 1
                    if max_samples is not None and loaded_total >= max_samples:
                        break
                by_module[item.entry.module] = by_module.get(item.entry.module, 0) + count
                by_source_file[item.entry.canonical_name] = count
                total_samples += count
                task = item.entry.task or ""
                if task in LLM_JUDGE_TASKS:
                    llm_judge_tasks.add(task)
                if max_samples is not None and loaded_total >= max_samples:
                    break

        if selection.include_basic_education:
            basic_config = load_basic_education_config(
                config_root=self._config.config_root,
                project_root=self._config.project_root,
                data_root=self._config.app.data_root,
            )
            if not basic_config.enabled:
                issues.append(
                    PreflightIssue(
                        level="error",
                        code="basic_education_disabled",
                        message="基本教育模块已被选择，但 configs/basic_education.yaml 未启用。",
                    )
                )
            scenario_count = 0
            for scenario in basic_config.scenarios:
                if not scenario.enabled:
                    continue
                if subsets and scenario.subset not in subsets and scenario.scenario_id not in subsets:
                    continue
                if source_files and scenario.source_file not in source_files:
                    continue
                scenario_dimension = scenario.dimension or scenario.scenario_id
                if dimensions and scenario_dimension not in dimensions:
                    continue
                count = int(scenario.expected_tasks or 0)
                scenario_count += count
                total_samples += count
                by_module[DEFAULT_BASIC_MODULE_NAME] = (
                    by_module.get(DEFAULT_BASIC_MODULE_NAME, 0) + count
                )
                by_source_file[scenario.source_file] = count

            if scenario_count == 0:
                issues.append(
                    PreflightIssue(
                        level="error",
                        code="basic_education_empty",
                        message="基本教育模块已被选择，但当前筛选条件下没有任何可执行场景。",
                    )
                )

            self._validate_basic_education_test_endpoint(
                issues=issues,
                require_real_judges=require_real_judges,
                fallback_model_id=next(iter(llm_judge_tasks), None),
            )

        if total_samples <= 0:
            issues.append(
                PreflightIssue(
                    level="error",
                    code="no_samples",
                    message="当前筛选条件下没有解析出任何样本。",
                )
            )

        if llm_judge_tasks:
            self._validate_llm_judge_model(
                issues=issues,
                tasks=sorted(llm_judge_tasks),
                require_real_judges=require_real_judges,
            )

        self._validate_target_model(
            issues=issues,
            model_config=model_config,
            require_real_model=True,
        )

        ok = not any(issue.level == "error" for issue in issues)
        return PreflightReport(
            ok=ok,
            selection=selection,
            total_samples=total_samples,
            by_module=by_module,
            by_source_file=by_source_file,
            llm_judge_tasks=sorted(llm_judge_tasks),
            issues=issues,
        )

    def ensure_ok(self, **kwargs: Any) -> PreflightReport:
        report = self.run(**kwargs)
        if report.ok:
            return report
        messages = [issue.message for issue in report.issues if issue.level == "error"]
        raise PreflightError(" | ".join(messages))

    def _validate_target_model(
        self,
        *,
        issues: list[PreflightIssue],
        model_config: ModelConfig,
        require_real_model: bool,
    ) -> None:
        if require_real_model and model_config.provider_name == "mock":
            issues.append(
                PreflightIssue(
                    level="warning",
                    code="mock_target_model",
                    message=f"当前目标模型 `{model_config.model_id}` 是 mock，仅适合本地冒烟，不适合正式评测。",
                )
            )
        self._validate_api_key(issues=issues, model_config=model_config, issue_code="target_model_key_missing")

    def _validate_llm_judge_model(
        self,
        *,
        issues: list[PreflightIssue],
        tasks: list[str],
        require_real_judges: bool,
    ) -> None:
        judge_model_id = self._config.judges.default_judge_model_id
        if not judge_model_id:
            issues.append(
                PreflightIssue(
                    level="error",
                    code="judge_model_missing",
                    message=f"以下任务依赖 LLM judge，但 default_judge_model_id 未配置：{', '.join(tasks)}",
                    details={"tasks": tasks},
                )
            )
            return
        if judge_model_id not in self._config.models:
            issues.append(
                PreflightIssue(
                    level="error",
                    code="judge_model_unknown",
                    message=f"default_judge_model_id={judge_model_id!r} 未在 models.yaml 中注册。",
                    details={"judge_model_id": judge_model_id, "tasks": tasks},
                )
            )
            return
        judge_model = self._config.models[judge_model_id]
        if require_real_judges and judge_model.provider_name == "mock":
            issues.append(
                PreflightIssue(
                    level="error",
                    code="judge_model_mock",
                    message=f"以下任务依赖真实 LLM judge，但当前 judge 模型仍是 `{judge_model_id}`（mock）。",
                    details={"tasks": tasks, "judge_model_id": judge_model_id},
                )
            )
        self._validate_api_key(issues=issues, model_config=judge_model, issue_code="judge_model_key_missing")

    def _validate_basic_education_test_endpoint(
        self,
        *,
        issues: list[PreflightIssue],
        require_real_judges: bool,
        fallback_model_id: str | None,
    ) -> None:
        basic_config = load_basic_education_config(
            config_root=self._config.config_root,
            project_root=self._config.project_root,
            data_root=self._config.app.data_root,
        )
        test_endpoint_model_id = (
            self._config.judges.basic_education_test_endpoint_model_id
            or basic_config.test_endpoint_model_id
            or self._config.judges.basic_education_evaluator_model_id
            or basic_config.evaluator_model_id
            or self._config.judges.default_judge_model_id
        )
        if not test_endpoint_model_id:
            issues.append(
                PreflightIssue(
                    level="error",
                    code="basic_education_test_endpoint_missing",
                    message=(
                        "基本教育模块需要测试端点模型，但当前未配置 test_endpoint_model_id，"
                        "也没有 evaluator_model_id 或 default_judge_model_id 可回退。"
                    ),
                )
            )
            return
        if test_endpoint_model_id not in self._config.models:
            issues.append(
                PreflightIssue(
                    level="error",
                    code="basic_education_test_endpoint_unknown",
                    message=f"基本教育测试端点模型 `{test_endpoint_model_id}` 未在 models.yaml 中注册。",
                )
            )
            return
        evaluator_model = self._config.models[test_endpoint_model_id]
        if require_real_judges and evaluator_model.provider_name == "mock":
            issues.append(
                PreflightIssue(
                    level="error",
                    code="basic_education_test_endpoint_mock",
                    message=f"基本教育测试端点当前会落到 `{test_endpoint_model_id}`（mock），无法产出正式分数。",
                    details={
                        "test_endpoint_model_id": test_endpoint_model_id,
                        "fallback": fallback_model_id,
                    },
                )
            )
        self._validate_api_key(
            issues=issues,
            model_config=evaluator_model,
            issue_code="basic_education_test_endpoint_key_missing",
        )
        explicit_eval_id = self._config.judges.basic_education_evaluator_model_id
        if explicit_eval_id and explicit_eval_id != test_endpoint_model_id:
            if explicit_eval_id not in self._config.models:
                issues.append(
                    PreflightIssue(
                        level="error",
                        code="basic_education_evaluator_unknown",
                        message=f"基本教育评估模型 `{explicit_eval_id}` 未在 models.yaml 中注册。",
                    )
                )
                return
            eval_only_model = self._config.models[explicit_eval_id]
            self._validate_api_key(
                issues=issues,
                model_config=eval_only_model,
                issue_code="basic_education_evaluator_key_missing",
            )

    def _validate_api_key(
        self,
        *,
        issues: list[PreflightIssue],
        model_config: ModelConfig,
        issue_code: str,
    ) -> None:
        if model_config.provider_name == "mock":
            return
        if not model_config.api_base:
            issues.append(
                PreflightIssue(
                    level="error",
                    code=f"{issue_code}_api_base",
                    message=f"模型 `{model_config.model_id}` 缺少 api_base 配置。",
                )
            )
        if not model_config.api_key_env:
            issues.append(
                PreflightIssue(
                    level="error",
                    code=f"{issue_code}_env",
                    message=f"模型 `{model_config.model_id}` 缺少 api_key_env 配置。",
                )
            )
            return
        if not get_api_key(model_config.api_key_env):
            issues.append(
                PreflightIssue(
                    level="error",
                    code=issue_code,
                    message=(
                        f"模型 `{model_config.model_id}` 需要的密钥 `{model_config.api_key_env}` 未读取到。"
                    ),
                    details={"api_key_env": model_config.api_key_env},
                )
            )

    def _get_model_config(self, model_id: str) -> ModelConfig:
        if model_id not in self._config.models:
            available = ", ".join(sorted(self._config.models))
            raise PreflightError(f"Unknown model_id={model_id!r}. Available: {available}")
        return self._config.models[model_id]
