from __future__ import annotations

from dataclasses import dataclass

from elbench.schemas.config import ProjectConfig

from .basic_education import DEFAULT_BASIC_MODULE_NAME


@dataclass(slots=True, frozen=True)
class ModuleSelection:
    requested_modules: frozenset[str] | None
    selected_modules: frozenset[str]
    standard_modules: frozenset[str]
    include_basic_education: bool


def resolve_module_selection(
    project_config: ProjectConfig,
    requested_modules: set[str] | None,
) -> ModuleSelection:
    requested = frozenset(requested_modules) if requested_modules else None
    if requested is None:
        active_modules = [
            module_name
            for module_name in project_config.app.benchmark_modules.active
            if module_name in project_config.modules
            and project_config.modules[module_name].enabled
        ]
        if active_modules:
            selected = frozenset(active_modules)
        else:
            selected = frozenset(
                module_name
                for module_name, module_entry in project_config.modules.items()
                if module_entry.enabled
            )
    else:
        selected = requested

    include_basic_education = DEFAULT_BASIC_MODULE_NAME in selected
    standard_modules = frozenset(
        module_name for module_name in selected if module_name != DEFAULT_BASIC_MODULE_NAME
    )
    return ModuleSelection(
        requested_modules=requested,
        selected_modules=selected,
        standard_modules=standard_modules,
        include_basic_education=include_basic_education,
    )
