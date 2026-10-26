from __future__ import annotations

import json
from typing import Iterator

from elbench.loaders.base import BaseLoader
from elbench.loaders.normalizer import RecordNormalizer
from elbench.registry import ResolvedRegistryItem
from elbench.schemas.evaluation import RawRecord, Sample


class JsonlLoader(BaseLoader):
    def __init__(self) -> None:
        self._normalizer = RecordNormalizer()

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
            yield self._normalizer.normalize(item, raw_record)

