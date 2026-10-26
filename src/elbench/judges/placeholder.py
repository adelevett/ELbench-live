from __future__ import annotations

from elbench.judges.base import BaseJudge
from elbench.schemas.evaluation import JudgeResult, ModelResponse, Sample


class PlaceholderJudge(BaseJudge):
    def __init__(self, judge_name: str) -> None:
        self._judge_name = judge_name

    async def judge(self, sample: Sample, response: ModelResponse) -> JudgeResult:
        return JudgeResult(
            judge_name=self._judge_name,
            judge_result="pending_judge",
            score=None,
            judge_reason="Judge placeholder. Stage 1 only reserves interface and output schema.",
            judge_metadata={"module": sample.module, "subset": sample.subset},
        )

