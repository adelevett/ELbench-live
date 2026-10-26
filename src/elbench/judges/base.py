from __future__ import annotations

from abc import ABC, abstractmethod

from elbench.schemas.evaluation import JudgeResult, ModelResponse, Sample


class BaseJudge(ABC):
    @abstractmethod
    async def judge(self, sample: Sample, response: ModelResponse) -> JudgeResult:
        raise NotImplementedError

