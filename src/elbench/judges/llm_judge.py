from __future__ import annotations

from elbench.judges.base import BaseJudge
from elbench.judges.llm_prompting import build_judge_prompt
from elbench.providers import ProviderFactory
from elbench.schemas.config import JudgeTaskConfig, ProjectConfig
from elbench.schemas.evaluation import GenerationRequest, JudgeResult, ModelResponse, Sample
from elbench.utils import extract_json_object


class LLMJudge(BaseJudge):
    def __init__(self, project_config: ProjectConfig, task_config: JudgeTaskConfig) -> None:
        self._project_config = project_config
        self._task_config = task_config
        self._client = self._build_client()

    async def judge(self, sample: Sample, response: ModelResponse) -> JudgeResult:
        template = self._task_config.template or sample.task or "generic"
        prompt = build_judge_prompt(template, sample, response)
        strict_prompt = (
            "Return ONLY one valid JSON object. Do not include markdown, chain-of-thought, "
            "analysis text, comments, or any text outside JSON. "
            'The first character must be "{" and the last character must be "}".\n\n'
            f"{prompt}\n"
            "Return ONLY the JSON object now."
        )
        judge_response = await self._client.generate(
            sample=sample,
            request=GenerationRequest(
                prompt=strict_prompt,
                system_prompt="Return JSON only. No reasoning text.",
                temperature=0,
                max_tokens=self._judge_max_tokens(),
                messages=[{"role": "user", "content": strict_prompt}],
                provider_kwargs={"response_format": {"type": "json_object"}},
            ),
        )
        parsed = extract_json_object(judge_response.text)
        if not parsed:
            return JudgeResult(
                judge_name=f"llm_judge_{template}",
                judge_result="fail",
                score=0.0,
                judge_reason="Judge model did not return a valid JSON object.",
                judge_metadata={
                    "template": template,
                    "judge_model_id": self._judge_model_id(),
                    "judge_raw_response": judge_response.text,
                },
            )
        return JudgeResult(
            judge_name=f"llm_judge_{template}",
            judge_result=str(parsed.get("judge_result", "fail")),
            score=_coerce_score(parsed.get("score")),
            judge_reason=str(parsed.get("judge_reason", "")),
            judge_metadata={
                "template": template,
                "judge_model_id": self._judge_model_id(),
                "judge_raw_response": judge_response.text,
                **(parsed.get("judge_metadata") or {}),
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _build_client(self):
        model_id = self._judge_model_id()
        model_config = self._project_config.models[model_id]
        provider_config = self._project_config.providers[model_config.provider_name]
        return ProviderFactory.create(provider_config, model_config)

    def _judge_model_id(self) -> str:
        model_id = self._task_config.judge_model_id or self._project_config.judges.default_judge_model_id
        if not model_id:
            raise ValueError("No judge model configured for LLM judge.")
        if model_id not in self._project_config.models:
            raise KeyError(f"Unknown judge model_id={model_id!r}")
        return model_id

    def _judge_max_tokens(self) -> int | None:
        """The judge card's own limit, or none: a reasoning judge cut off mid-thought returns no grade."""
        return self._project_config.models[self._judge_model_id()].max_tokens


def _coerce_score(value) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, score))

