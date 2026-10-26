"""Benchmark file catalog built from the registry, loaders, and judge config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from elbench.config import load_project_config
from elbench.dashboard.english import english_file, english_module, english_subset
from elbench.execution.basic_education import (
    DEFAULT_BASIC_MODULE_NAME,
    count_runtime_tasks,
    load_basic_education_config,
)
from elbench.execution.preflight import LLM_JUDGE_TASKS
from elbench.loaders import LoaderFactory
from elbench.registry import FileRegistry
from elbench.schemas.config import ProjectConfig


@dataclass(frozen=True, slots=True)
class CatalogFile:
    canonical_name: str
    module: str
    subset: str
    task: str
    scoring_mode: str
    item_count: int
    expected_tasks: int | None
    dimensions: tuple[str, ...]
    multi_turn: bool
    preflight_llm_task: bool
    recursion_limit: int | None

    @property
    def label(self) -> str:
        score = self.scoring_mode
        if self.preflight_llm_task and self.scoring_mode != "llm":
            score = f"{self.scoring_mode}, preflight requires judge key"
        turn = ", multi-turn" if self.multi_turn else ""
        return (
            f"{english_module(self.module)} / {english_subset(self.subset)} / "
            f"{english_file(self.canonical_name)} · {score} · {self.item_count} items{turn}"
        )


@dataclass(slots=True)
class Catalog:
    config: ProjectConfig
    files: tuple[CatalogFile, ...]

    def by_name(self) -> dict[str, CatalogFile]:
        return {entry.canonical_name: entry for entry in self.files}

    def modules(self) -> list[str]:
        seen: list[str] = []
        for entry in self.files:
            if entry.module not in seen:
                seen.append(entry.module)
        return seen

    def files_in_modules(self, modules: set[str]) -> list[CatalogFile]:
        return [entry for entry in self.files if entry.module in modules]

    def default_modules(self) -> list[str]:
        modules: list[str] = []
        for entry in self.default_files():
            if entry.module not in modules:
                modules.append(entry.module)
        return modules

    def default_files(self) -> list[CatalogFile]:
        """Rule-scored files that the stock preflight does not hold for a judge key."""
        selected: list[CatalogFile] = []
        for entry in self.files:
            if entry.module == DEFAULT_BASIC_MODULE_NAME:
                continue
            if entry.scoring_mode != "rule":
                continue
            if entry.preflight_llm_task:
                continue
            selected.append(entry)
        return selected


def load_catalog(config_root: Path | str | None = None) -> Catalog:
    config = load_project_config(Path(config_root) if config_root is not None else None)
    registry = FileRegistry(config)
    resolved = registry.resolve()
    basic = load_basic_education_config(
        config_root=config.config_root,
        project_root=config.project_root,
        data_root=config.app.data_root,
    )
    scenarios = {scenario.source_file: scenario for scenario in basic.scenarios}

    files: list[CatalogFile] = []
    for item in resolved:
        task = item.entry.task or ""
        scenario = scenarios.get(item.entry.canonical_name)
        if item.entry.module == DEFAULT_BASIC_MODULE_NAME and scenario is not None:
            template = _read_yaml(scenario.template_path)
            globals_section = template.get("globals") if isinstance(template, dict) else None
            recursion_limit = None
            if isinstance(globals_section, dict) and globals_section.get("recursion_limit") is not None:
                recursion_limit = int(globals_section["recursion_limit"])
            dimensions = (scenario.dimension or scenario.scenario_id,)
            files.append(
                CatalogFile(
                    canonical_name=item.entry.canonical_name,
                    module=item.entry.module,
                    subset=item.entry.subset,
                    task=task,
                    scoring_mode="runtime",
                    item_count=count_runtime_tasks(template),
                    expected_tasks=scenario.expected_tasks,
                    dimensions=dimensions,
                    multi_turn=bool(scenario.multi_turn),
                    preflight_llm_task=False,
                    recursion_limit=recursion_limit,
                )
            )
            continue

        loader = LoaderFactory.create(item.entry.loader_name)
        dimensions: set[str] = set()
        count = 0
        for sample in loader.iter_samples(item):
            count += 1
            if sample.dimension:
                dimensions.add(sample.dimension)
        task_config = config.judges.tasks.get(task)
        scoring_mode = task_config.mode if task_config is not None else "placeholder"
        files.append(
            CatalogFile(
                canonical_name=item.entry.canonical_name,
                module=item.entry.module,
                subset=item.entry.subset,
                task=task,
                scoring_mode=scoring_mode,
                item_count=count,
                expected_tasks=None,
                dimensions=tuple(sorted(dimensions)),
                multi_turn=False,
                preflight_llm_task=task in LLM_JUDGE_TASKS,
                recursion_limit=None,
            )
        )
    return Catalog(config=config, files=tuple(files))


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML object at {path}")
    return loaded
