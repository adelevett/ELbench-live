from __future__ import annotations

from typing import Any

from elbench.judges.base import BaseJudge
from elbench.judges.ifeval_rule.utils import process_results
from elbench.loaders.general_capability_loader import IFEvalJsonlLoader
from elbench.schemas.evaluation import JudgeResult, ModelResponse, Sample


class IFEvalJudge(BaseJudge):
    _loader = IFEvalJsonlLoader()

    async def judge(self, sample: Sample, response: ModelResponse) -> JudgeResult:
        text = response.text or ""
        instruction_ids = self._loader.instruction_ids_from_sample(sample)
        kwargs_list = self._loader.kwargs_from_sample(sample)

        doc = {
            "key": self._loader.prompt_key_from_sample(sample),
            "instruction_id_list": instruction_ids,
            "prompt": sample.metadata.get("prompt", ""),
            "kwargs": kwargs_list,
        }
        score = process_results(doc, results=[text])

        prompt_level_strict = score["prompt_level_strict"]
        inst_level_strict = score["inst_level_strict"]
        prompt_level_loose = score["prompt_level_loose"]
        inst_level_loose = score["inst_level_loose"]
        passed = prompt_level_strict == 1.0

        return JudgeResult(
            judge_name="ifeval_rule",
            judge_result="pass" if passed else "fail",
            score=prompt_level_strict,
            judge_reason=(
                "All instructions passed strict checks."
                if passed
                else "One or more instructions failed strict checks."
            ),
            judge_metadata={
                "prompt_level_strict": prompt_level_strict,
                "inst_level_strict": inst_level_strict,
                "prompt_level_loose": prompt_level_loose,
                "inst_level_loose": inst_level_loose,
                "instruction_id_list": instruction_ids,
                "kwargs": kwargs_list,
                "prompt_key": self._loader.prompt_key_from_sample(sample),
            },
        )
