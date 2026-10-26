from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from elbench.schemas.config import (
    AppConfig,
    FieldMappingConfig,
    FileRegistryEntry,
    JudgeConfig,
    ModelConfig,
    ModuleEntry,
    ProjectConfig,
    ProviderConfig,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_project_config(config_root: Path | None = None) -> ProjectConfig:
    resolved_config_root = Path(config_root or "configs").resolve()
    project_root = resolved_config_root.parent

    app = AppConfig.model_validate(_read_yaml(resolved_config_root / "app.yaml"))
    modules_data = _read_yaml(resolved_config_root / "module_registry.yaml").get("modules", [])
    providers_data = _read_yaml(resolved_config_root / "providers.yaml").get("providers", [])
    models_data = _read_yaml(resolved_config_root / "models.yaml").get("models", [])
    judges_data = _read_yaml(resolved_config_root / "judges.yaml")
    files_data = _read_yaml(resolved_config_root / "file_registry.yaml").get("files", [])
    mappings_data = _read_yaml(resolved_config_root / "field_mappings.yaml").get("mappings", {})

    app.data_root = (project_root / app.data_root).resolve()
    app.output_root = (project_root / app.output_root).resolve()

    modules = {item["module_name"]: ModuleEntry.model_validate(item) for item in modules_data}
    providers = {item["provider_name"]: ProviderConfig.model_validate(item) for item in providers_data}
    models = {item["model_id"]: ModelConfig.model_validate(item) for item in models_data}
    judges = JudgeConfig.model_validate(judges_data)
    file_registry = {
        item["canonical_name"]: FileRegistryEntry.model_validate(item) for item in files_data
    }
    field_mappings = {
        key: FieldMappingConfig.model_validate(value) for key, value in mappings_data.items()
    }

    return ProjectConfig(
        project_root=project_root,
        config_root=resolved_config_root,
        app=app,
        modules=modules,
        providers=providers,
        models=models,
        judges=judges,
        file_registry=file_registry,
        field_mappings=field_mappings,
    )

