from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from elbench.schemas.config import FieldMappingConfig, FileRegistryEntry, ProjectConfig


@dataclass(slots=True)
class ResolvedRegistryItem:
    entry: FileRegistryEntry
    mapping: FieldMappingConfig
    path: Path


class FileRegistry:
    def __init__(self, config: ProjectConfig) -> None:
        self._config = config
        self._file_cache: list[Path] | None = None

    def resolve(
        self,
        modules: set[str] | None = None,
        subsets: set[str] | None = None,
        source_files: set[str] | None = None,
    ) -> list[ResolvedRegistryItem]:
        resolved: list[ResolvedRegistryItem] = []
        for entry in self._config.file_registry.values():
            if modules and entry.module not in modules:
                continue
            if subsets and entry.subset not in subsets:
                continue
            if source_files and entry.canonical_name not in source_files:
                continue
            path = self._find_path(entry)
            mapping = self._config.field_mappings[entry.mapping_key]
            resolved.append(ResolvedRegistryItem(entry=entry, mapping=mapping, path=path))
        return resolved

    def _find_path(self, entry: FileRegistryEntry) -> Path:
        candidate_names = [entry.canonical_name, *entry.aliases]
        for hint in entry.path_hints:
            candidate = (self._config.app.data_root / hint).resolve()
            if candidate.exists():
                return candidate
        for path in self._iter_files():
            if path.name in candidate_names:
                return path
        for path in self._iter_files():
            if any(fnmatch.fnmatch(path.name, pattern) for pattern in entry.filename_patterns):
                return path
        raise FileNotFoundError(
            f"Cannot resolve file for registry entry {entry.canonical_name!r} under {self._config.app.data_root}"
        )

    def _iter_files(self) -> Iterable[Path]:
        if self._file_cache is None:
            self._file_cache = [path for path in self._config.app.data_root.rglob("*") if path.is_file()]
        return iter(self._file_cache)

