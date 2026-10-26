from __future__ import annotations

import hashlib
import json
from typing import Any

from elbench.loaders.resolvers import resolve_dimension
from elbench.registry import ResolvedRegistryItem
from elbench.schemas.evaluation import RawRecord, Sample


class RecordNormalizer:
    def normalize(self, item: ResolvedRegistryItem, raw_record: RawRecord) -> Sample:
        record = raw_record.record
        mapping = item.mapping
        prompt = self._build_prompt(mapping, record)
        sample_id = self._build_sample_id(item, raw_record)
        dimension = self._extract_dimension(mapping, record)
        reference = self._extract_reference(mapping.reference_fields, record)
        metadata = self._extract_metadata(mapping.metadata_fields, raw_record)
        return Sample(
            sample_id=sample_id,
            source_file=item.entry.canonical_name,
            source_path=str(item.path),
            module=item.entry.module,
            subset=item.entry.subset,
            task=item.entry.task,
            dimension=dimension,
            prompt=prompt,
            reference=reference,
            metadata=metadata,
            raw_record=record,
        )

    def _build_prompt(self, mapping: Any, record: dict[str, Any]) -> str:
        if mapping.prompt_template:
            rendered = mapping.prompt_template
            for key, value in record.items():
                rendered = rendered.replace(f"{{{{ {key} }}}}", self._stringify(value))
            return rendered.strip()
        for key in mapping.prompt_fields:
            value = record.get(key)
            if value not in (None, ""):
                return self._stringify(value)
        raise ValueError("Prompt fields are empty for record.")

    def _build_sample_id(self, item: ResolvedRegistryItem, raw_record: RawRecord) -> str:
        parts = []
        for key in item.mapping.sample_id_fields:
            value = raw_record.record.get(key)
            if value not in (None, ""):
                parts.append(self._stringify(value))
        if not parts:
            parts = [item.entry.canonical_name, str(raw_record.row_index), json.dumps(raw_record.record, ensure_ascii=False)]
        digest = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:16]
        return f"{_safe_stem(item.entry.canonical_name)}-{digest}"

    def _extract_dimension(self, mapping: Any, record: dict[str, Any]) -> str | None:
        for key in mapping.dimension_fields:
            value = record.get(key)
            if value not in (None, ""):
                return self._stringify(value)
        return resolve_dimension(mapping.dimension_resolver, record)

    def _extract_reference(self, fields: list[str], record: dict[str, Any]) -> dict[str, Any] | None:
        reference = {key: record.get(key) for key in fields if key in record}
        return reference or None

    def _extract_metadata(self, fields: list[str], raw_record: RawRecord) -> dict[str, Any]:
        metadata = {key: raw_record.record.get(key) for key in fields if key in raw_record.record}
        metadata["row_index"] = raw_record.row_index
        return metadata

    def _stringify(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)


def _safe_stem(filename: str) -> str:
    return filename.replace(".", "_").replace("/", "_").replace("\\", "_")

