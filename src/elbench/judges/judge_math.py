from __future__ import annotations

from elbench.judges.base import BaseJudge
from elbench.judges.math_rule.grader import grade_answer
from elbench.judges.math_rule.math_parser import (
    extract_answer,
    math_equal,
    process_math500,
    strip_answer_string,
)
from elbench.loaders.general_capability_loader import MATH500JsonlLoader
from elbench.schemas.evaluation import JudgeResult, ModelResponse, Sample


class Math500Judge(BaseJudge):
    _loader = MATH500JsonlLoader()

    async def judge(self, sample: Sample, response: ModelResponse) -> JudgeResult:
        expected_raw = self._loader.expected_answer_from_sample(sample)
        expected_answer = strip_answer_string(expected_raw)
        predicted_answer = extract_answer(response.text or "")
        passed = process_math500(predicted_answer, expected_answer) == 1.0
        return JudgeResult(
            judge_name="math_500_acc",
            judge_result="pass" if passed else "fail",
            score=1.0 if passed else 0.0,
            judge_reason="Math answer matched reference." if passed else "Math answer did not match reference.",
            judge_metadata={
                "predicted_answer": predicted_answer,
                "expected_answer": expected_answer,
                "raw_expected_answer": expected_raw,
                "question_id": sample.metadata.get("question_id"),
                "level": sample.dimension or sample.metadata.get("subset_key"),
            },
        )


class AIMEJudge(Math500Judge):
    async def judge(self, sample: Sample, response: ModelResponse) -> JudgeResult:
        expected_raw = self._loader.expected_answer_from_sample(sample)
        expected_answer = extract_answer(expected_raw, use_last_number=False) or strip_answer_string(expected_raw)
        predicted_answer = extract_answer(response.text or "")
        passed = grade_answer(predicted_answer, expected_answer)
        return JudgeResult(
            judge_name="aime_acc",
            judge_result="pass" if passed else "fail",
            score=1.0 if passed else 0.0,
            judge_reason="AIME answer matched reference." if passed else "AIME answer did not match reference.",
            judge_metadata={
                "predicted_answer": predicted_answer,
                "expected_answer": expected_answer,
                "raw_expected_answer": expected_raw,
            },
        )
