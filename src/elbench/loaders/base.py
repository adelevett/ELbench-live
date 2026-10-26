from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from elbench.registry import ResolvedRegistryItem
from elbench.schemas.evaluation import RawRecord, Sample


class BaseLoader(ABC):
    @abstractmethod
    def iter_raw_records(self, item: ResolvedRegistryItem) -> Iterator[RawRecord]:
        raise NotImplementedError

    @abstractmethod
    def iter_samples(self, item: ResolvedRegistryItem) -> Iterator[Sample]:
        raise NotImplementedError

