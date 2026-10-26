from __future__ import annotations

import json
import re
from typing import Any, Iterator

from elbench.loaders.base import BaseLoader
from elbench.registry import ResolvedRegistryItem
from elbench.schemas.evaluation import RawRecord, Sample


class MMLUProJsonlLoader(BaseLoader):
    def iter_raw_records(self, item: ResolvedRegistryItem) -> Iterator[RawRecord]:
        with item.path.open("r", encoding="utf-8") as handle:
            for row_index, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                yield RawRecord(
                    source_path=str(item.path),
                    source_file=item.entry.canonical_name,
                    row_index=row_index,
                    record=json.loads(line),
                )

    def iter_samples(self, item: ResolvedRegistryItem) -> Iterator[Sample]:
        for raw_record in self.iter_raw_records(item):
            record = raw_record.record
            target = self._string_or_none(record.get("target"))
            choices = self._choices(record.get("choices"))
            metadata = self._metadata(record, raw_record.row_index)
            dimension = self._dimension(record, metadata)
            yield Sample(
                sample_id=self._sample_id(
                    item.entry.canonical_name,
                    record,
                    raw_record.row_index,
                    namespace=dimension,
                ),
                source_file=item.entry.canonical_name,
                source_path=str(item.path),
                module=item.entry.module,
                subset=item.entry.subset,
                task=item.entry.task,
                dimension=dimension,
                prompt=self._build_zero_shot_prompt(record, choices, item.entry.task),
                reference={"target": target},
                metadata=metadata,
                raw_record=record,
            )

    def _build_zero_shot_prompt(self, record: dict[str, Any], choices: list[str], task: str | None) -> str:
        question = self._extract_question(record)
        option_lines = [f"{chr(65 + index)}) {choice}" for index, choice in enumerate(choices)]
        letters = ",".join(chr(65 + index) for index in range(len(choices)))
        if task == "ceval":
            return "\n".join(
                [
                    "请回答以下单项选择题。",
                    "请先逐步思考，再给出答案。",
                    f"你回答的最后一行必须严格使用格式：答案：[LETTER]，其中 [LETTER] 是 {letters} 中的一个。",
                    "",
                    "问题：",
                    question,
                    "选项：",
                    *option_lines,
                ]
            )
        return "\n".join(
            [
                "Answer the following multiple choice question.",
                "Think step by step before answering.",
                f"The last line of your response must be exactly: ANSWER: [LETTER], where [LETTER] is one of {letters}.",
                "",
                "Question:",
                question,
                "Options:",
                *option_lines,
            ]
        )

    def _extract_question(self, record: dict[str, Any]) -> str:
        input_messages = record.get("input")
        if isinstance(input_messages, list):
            for message in reversed(input_messages):
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if not isinstance(content, str):
                    continue
                matches = re.findall(r"Question:\n(.*?)\nOptions:", content, flags=re.S)
                if not matches:
                    matches = re.findall(r"问题[：:]\s*(.*?)\n选项[：:]", content, flags=re.S)
                if matches:
                    return matches[-1].strip()
        raise ValueError("Cannot extract multiple-choice question from input messages.")

    def _metadata(self, record: dict[str, Any], row_index: int) -> dict[str, Any]:
        metadata = {}
        raw_metadata = record.get("metadata")
        if isinstance(raw_metadata, dict):
            metadata.update(raw_metadata)
        metadata["choices"] = self._choices(record.get("choices"))
        metadata["subset_key"] = self._string_or_none(record.get("subset_key"))
        metadata["row_index"] = row_index
        return metadata

    def _dimension(self, record: dict[str, Any], metadata: dict[str, Any]) -> str | None:
        return (
            self._string_or_none(record.get("subset_key"))
            or self._string_or_none(metadata.get("subject"))
            or self._string_or_none(metadata.get("id"))
        )

    def _sample_id(
        self,
        canonical_name: str,
        record: dict[str, Any],
        row_index: int,
        namespace: str | None = None,
    ) -> str:
        value = record.get("id")
        if value in (None, ""):
            value = row_index
        value_part = self._slug_component(value)
        namespace_part = self._slug_component(namespace)
        if namespace_part:
            return f"{canonical_name.replace('.', '_')}-{namespace_part}-{value_part}"
        return f"{canonical_name.replace('.', '_')}-{value_part}"

    def _choices(self, value: Any) -> list[str]:
        if not isinstance(value, list) or not value:
            raise ValueError("Multiple-choice record must contain a non-empty choices list.")
        return [str(choice) for choice in value]

    def _string_or_none(self, value: Any) -> str | None:
        if value in (None, ""):
            return None
        return str(value)

    def _slug_component(self, value: Any) -> str:
        if value in (None, ""):
            return ""
        text = re.sub(r"[^\w]+", "_", str(value)).strip("_")
        return text or str(value)

    def expected_answer_from_sample(self, sample: Sample) -> str | None:
        reference = sample.reference or {}
        if isinstance(reference, dict):
            value = reference.get("target")
        else:
            value = reference
        if value in (None, ""):
            return None
        normalized = str(value).strip().upper()
        return normalized[0] if normalized else None

    def valid_letters_from_sample(self, sample: Sample) -> list[str]:
        choices = sample.metadata.get("choices")
        if isinstance(choices, list) and choices:
            return [chr(65 + index) for index in range(len(choices))]
        return list("ABCDEFGHIJ")


class CEvalJsonlLoader(MMLUProJsonlLoader):
    pass


class IFEvalJsonlLoader(BaseLoader):
    def iter_raw_records(self, item: ResolvedRegistryItem) -> Iterator[RawRecord]:
        with item.path.open("r", encoding="utf-8") as handle:
            for row_index, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                yield RawRecord(
                    source_path=str(item.path),
                    source_file=item.entry.canonical_name,
                    row_index=row_index,
                    record=json.loads(line),
                )

    def iter_samples(self, item: ResolvedRegistryItem) -> Iterator[Sample]:
        for raw_record in self.iter_raw_records(item):
            record = raw_record.record
            metadata = self._metadata(record, raw_record.row_index)
            yield Sample(
                sample_id=self._sample_id(item.entry.canonical_name, record, raw_record.row_index),
                source_file=item.entry.canonical_name,
                source_path=str(item.path),
                module=item.entry.module,
                subset=item.entry.subset,
                task=item.entry.task,
                dimension="default",
                prompt=self._prompt(record, metadata),
                reference={"target": self._string_or_empty(record.get("target"))},
                metadata=metadata,
                raw_record=record,
            )

    def _metadata(self, record: dict[str, Any], row_index: int) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        raw_metadata = record.get("metadata")
        if isinstance(raw_metadata, dict):
            metadata.update(raw_metadata)

        metadata["instruction_id_list"] = self._instruction_ids(metadata.get("instruction_id_list"))
        metadata["kwargs"] = self._kwargs(metadata.get("kwargs"))
        metadata["key"] = self._string_or_empty(metadata.get("key"))
        metadata["prompt"] = self._prompt(record, metadata)
        metadata["row_index"] = row_index
        return metadata

    def _prompt(self, record: dict[str, Any], metadata: dict[str, Any]) -> str:
        prompt = metadata.get("prompt")
        if isinstance(prompt, str) and prompt:
            return prompt
        input_messages = record.get("input")
        if isinstance(input_messages, list):
            for message in input_messages:
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"]
        return ""

    def _kwargs(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        parsed_kwargs = []
        for item in value:
            if isinstance(item, dict):
                parsed_kwargs.append(item)
                continue
            if isinstance(item, str):
                try:
                    parsed = json.loads(item)
                except json.JSONDecodeError:
                    parsed = {}
                parsed_kwargs.append(parsed if isinstance(parsed, dict) else {})
                continue
            parsed_kwargs.append({})
        return parsed_kwargs

    def _instruction_ids(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value]

    def _sample_id(self, canonical_name: str, record: dict[str, Any], row_index: int) -> str:
        value = record.get("id")
        if value in (None, ""):
            value = row_index
        return f"{canonical_name.replace('.', '_')}-{value}"

    def _string_or_empty(self, value: Any) -> str:
        if value in (None, ""):
            return ""
        return str(value)

    def instruction_ids_from_sample(self, sample: Sample) -> list[str]:
        return self._instruction_ids(sample.metadata.get("instruction_id_list"))

    def kwargs_from_sample(self, sample: Sample) -> list[dict[str, Any]]:
        return self._kwargs(sample.metadata.get("kwargs"))

    def prompt_key_from_sample(self, sample: Sample) -> str:
        return self._string_or_empty(sample.metadata.get("key"))


class MATH500JsonlLoader(BaseLoader):
    def iter_raw_records(self, item: ResolvedRegistryItem) -> Iterator[RawRecord]:
        with item.path.open("r", encoding="utf-8") as handle:
            for row_index, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                yield RawRecord(
                    source_path=str(item.path),
                    source_file=item.entry.canonical_name,
                    row_index=row_index,
                    record=json.loads(line),
                )

    def iter_samples(self, item: ResolvedRegistryItem) -> Iterator[Sample]:
        for raw_record in self.iter_raw_records(item):
            record = raw_record.record
            metadata = self._metadata(record, raw_record.row_index)
            yield Sample(
                sample_id=self._sample_id(item.entry.canonical_name, record, raw_record.row_index),
                source_file=item.entry.canonical_name,
                source_path=str(item.path),
                module=item.entry.module,
                subset=item.entry.subset,
                task=item.entry.task,
                dimension=self._string_or_none(record.get("subset_key")) or "default",
                prompt=self._prompt(record),
                reference={"target": self._string_or_empty(record.get("target"))},
                metadata=metadata,
                raw_record=record,
            )

    def _metadata(self, record: dict[str, Any], row_index: int) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        raw_metadata = record.get("metadata")
        if isinstance(raw_metadata, dict):
            metadata.update(raw_metadata)
        metadata["subset_key"] = self._string_or_none(record.get("subset_key"))
        metadata["row_index"] = row_index
        return metadata

    def _prompt(self, record: dict[str, Any]) -> str:
        input_messages = record.get("input")
        if isinstance(input_messages, list):
            for message in input_messages:
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"]
        return ""

    def _sample_id(self, canonical_name: str, record: dict[str, Any], row_index: int) -> str:
        value = record.get("id")
        subset_key = self._string_or_none(record.get("subset_key")) or "default"
        if value in (None, ""):
            value = row_index
        subset_slug = re.sub(r"[^A-Za-z0-9]+", "_", subset_key).strip("_").lower()
        return f"{canonical_name.replace('.', '_')}-{subset_slug}-{value}"

    def _string_or_none(self, value: Any) -> str | None:
        if value in (None, ""):
            return None
        return str(value)

    def _string_or_empty(self, value: Any) -> str:
        if value in (None, ""):
            return ""
        return str(value)

    def expected_answer_from_sample(self, sample: Sample) -> str:
        reference = sample.reference or {}
        if isinstance(reference, dict):
            value = reference.get("target")
        else:
            value = reference
        return "" if value in (None, "") else str(value)


class AIMEJsonlLoader(MATH500JsonlLoader):
    def iter_samples(self, item: ResolvedRegistryItem) -> Iterator[Sample]:
        for raw_record in self.iter_raw_records(item):
            record = raw_record.record
            metadata = self._metadata(record, raw_record.row_index)
            yield Sample(
                sample_id=self._sample_id(item.entry.canonical_name, record, raw_record.row_index),
                source_file=item.entry.canonical_name,
                source_path=str(item.path),
                module=item.entry.module,
                subset=item.entry.subset,
                task=item.entry.task,
                dimension="default",
                prompt=self._prompt(record),
                reference={"target": self._string_or_empty(record.get("target"))},
                metadata=metadata,
                raw_record=record,
            )
