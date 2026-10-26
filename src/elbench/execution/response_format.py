from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from elbench.schemas.evaluation import ModelResponse, Sample
from elbench.utils import extract_choice_answer


class ResponseFormatError(ValueError):
    def __init__(self, message: str, *, metadata: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.metadata = metadata or {}


@dataclass(slots=True)
class ResponseFormatSpec:
    name: str
    instruction: str
    reminder_template: str
    valid_letters: list[str] = field(default_factory=list)
    answer_prefixes: list[str] = field(default_factory=list)
    require_last_line_exact: bool = True
    allow_multiple: bool = False

    def validate(self, response_text: str | None) -> dict[str, Any]:
        text = (response_text or "").strip()
        if not text:
            raise ResponseFormatError(
                "Model response is empty.",
                metadata={"format_name": self.name, "expected_format": self.instruction},
            )
        answer = self._extract_answer(text)
        if answer is None:
            raise ResponseFormatError(
                "Model response does not follow the required answer format.",
                metadata={
                    "format_name": self.name,
                    "expected_format": self.instruction,
                    "response_preview": text[:200],
                },
            )
        return {
            "format_name": self.name,
            "expected_format": self.instruction,
            "parsed_answer": answer,
        }

    def _extract_answer(self, text: str) -> str | None:
        if not self.require_last_line_exact:
            return None
        return extract_choice_answer(
            text,
            self.valid_letters,
            answer_prefixes=self.answer_prefixes,
            allow_multiple=self.allow_multiple,
        )

    def build_reminder(self, response_text: str | None) -> str:
        preview = (response_text or "").strip()
        preview = re.sub(r"\s+", " ", preview)[:200]
        if not preview:
            return (
                "Your previous response was empty or unparsable. "
                f"Reply again and output only the final answer line. {self.instruction} "
                "Do not leave the answer blank. If unsure, provide your best guess in the required final line."
            ).strip()
        return self.reminder_template.format(
            expected_format=self.instruction,
            response_preview=preview,
        ).strip()


def should_force_concise_retry(response: ModelResponse | None) -> bool:
    if response is None:
        return False

    payload = response.raw_payload if isinstance(response.raw_payload, dict) else {}
    finish_reason = _first_choice_finish_reason(payload)
    if finish_reason != "length":
        return False
    if (response.text or "").strip():
        return True

    usage = response.usage if isinstance(response.usage, dict) else {}
    completion_tokens = _as_int(usage.get("completion_tokens"))
    reasoning_tokens = _as_int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))
    return reasoning_tokens > 0 and completion_tokens >= 512


def build_retry_followup(
    spec: ResponseFormatSpec | None,
    response: ModelResponse | None,
) -> dict[str, Any]:
    response_text = response.text if response is not None else ""
    if spec is None:
        return {
            "assistant_content": response_text,
            "user_reminder": "",
            "retry_reason": "format_invalid",
            "max_tokens": None,
        }
    if should_force_concise_retry(response):
        return {
            "assistant_content": None,
            "user_reminder": (
                "Your previous response used up the output budget before giving the final answer. "
                "Do not include any reasoning, explanation, or thinking. "
                f"Reply with exactly one final answer line only. {spec.instruction}"
            ).strip(),
            "retry_reason": "format_empty_after_reasoning",
            "max_tokens": 64,
        }
    return {
        "assistant_content": response_text,
        "user_reminder": spec.build_reminder(response_text),
        "retry_reason": "format_invalid",
        "max_tokens": None,
    }


def build_concise_retry_prompt(base_prompt: str, spec: ResponseFormatSpec | None) -> str:
    if spec is not None and spec.name in {"single_choice_en", "single_choice_cn"}:
        return _build_choice_only_retry_prompt(base_prompt, spec)

    lines = []
    for line in base_prompt.splitlines():
        lowered = line.lower()
        if "think step by step" in lowered or "逐步思考" in line:
            continue
        lines.append(line)
    prompt = "\n".join(lines).rstrip()
    instruction = spec.instruction if spec is not None else "Output only the final answer."
    return (
        "CRITICAL OUTPUT RULE: Your entire response must be exactly one answer line. "
        "Do not include reasoning, explanation, calculations, restatement, markdown, or any extra text.\n"
        f"{instruction}\n\n"
        f"{prompt}\n\n"
        "FINAL REMINDER: Output exactly one answer line and nothing else. "
        f"{instruction}"
    ).strip()


def _build_choice_only_retry_prompt(base_prompt: str, spec: ResponseFormatSpec) -> str:
    prompt = _strip_reasoning_instructions(base_prompt)
    prompt = _last_question_block(prompt)
    valid = ",".join(spec.valid_letters) if spec.valid_letters else "A,B,C,D"
    if spec.name == "single_choice_cn":
        instruction = f"只输出一个大写选项字母（{valid}），不要输出推理、解释、标点或其他文字。"
        final = "你的答案："
    else:
        instruction = (
            f"Answer with exactly one uppercase option letter ({valid}) and nothing else. "
            "No reasoning, explanation, punctuation, markdown, or extra text."
        )
        final = "Your answer:"
    return f"{instruction}\n\n{prompt}\n\n{final}".strip()


def _strip_reasoning_instructions(prompt: str) -> str:
    filtered: list[str] = []
    for line in prompt.splitlines():
        lowered = line.lower()
        if "think step by step" in lowered or "逐步思考" in line or "先逐步思考" in line:
            continue
        filtered.append(line)
    return "\n".join(filtered).strip()


def _last_question_block(prompt: str) -> str:
    markers = ["Answer the following multiple choice question.", "问题：", "Question:"]
    start = -1
    for marker in markers:
        index = prompt.rfind(marker)
        if index > start:
            start = index
    if start >= 0:
        return prompt[start:].strip()
    return prompt.strip()


def get_response_format_spec(sample: Sample) -> ResponseFormatSpec | None:
    if sample.task == "mmlu_pro":
        return ResponseFormatSpec(
            name="single_choice_en",
            instruction="The last line must be exactly: ANSWER: [LETTER], where [LETTER] is a valid option letter.",
            reminder_template=(
                "Your previous response did not follow the required answer format. "
                "Reply again. If you include reasoning, still end with exactly one final line: ANSWER: [LETTER]. "
                "Do not add any extra text after that final answer line."
            ),
            valid_letters=_choice_letters_from_sample(sample),
            answer_prefixes=["ANSWER", "FINAL ANSWER"],
        )
    if sample.task == "ceval":
        return ResponseFormatSpec(
            name="single_choice_cn",
            instruction="最后一行必须严格是：答案：[LETTER]，其中 [LETTER] 是合法选项字母。",
            reminder_template=(
                "你上一轮回复没有遵守规定格式。请重新回答，最后一行必须严格是：答案：[LETTER]。"
                "不要在最后答案行之后添加任何额外文本。"
            ),
            valid_letters=_choice_letters_from_sample(sample),
            answer_prefixes=["答案", "最终答案", "ANSWER", "FINAL ANSWER"],
        )
    if sample.task == "highlevel_omni":
        return ResponseFormatSpec(
            name="choice_omni",
            instruction="最后一行必须严格是：答案：[LETTER(S)]，其中 [LETTER(S)] 是 A、B、C、D 中的一个或多个字母。",
            reminder_template=(
                "你上一轮回复没有遵守规定格式。请重新回答，最后一行必须严格是：答案：[LETTER(S)]。"
                "如果有多个正确选项，请连续写出多个字母，例如：答案：AC。"
                "不要在最后答案行之后添加任何额外文本。"
            ),
            valid_letters=list("ABCD"),
            answer_prefixes=["答案", "最终答案", "ANSWER", "FINAL ANSWER"],
            allow_multiple=True,
        )
    return None


def augment_prompt_for_format(sample: Sample, prompt: str) -> str:
    spec = get_response_format_spec(sample)
    if spec is None:
        return prompt
    if spec.instruction in prompt:
        return prompt
    if sample.task == "mmlu_pro" and "ANSWER: [LETTER]" in prompt:
        return prompt
    if sample.task == "ceval" and "答案：[LETTER]" in prompt:
        return prompt
    if sample.task == "highlevel_omni" and "答案：[LETTER]" in prompt:
        return prompt
    return f"{prompt.rstrip()}\n\n[Answer Format]\n{spec.instruction}"


def attach_format_metadata(response: ModelResponse, spec: ResponseFormatSpec | None) -> ModelResponse:
    if spec is None:
        response.format_valid = None
        return response
    if str(response.error or "").strip().lower() in {"content_filter", "invalid_prompt"}:
        response.format_valid = False
        response.format_metadata = {
            "format_name": spec.name,
            "expected_format": spec.instruction,
            "blocked_by_upstream_filter": True,
            "error": response.error,
        }
        return response
    metadata = spec.validate(response.text)
    response.format_valid = True
    response.format_metadata = metadata
    return response


def _choice_letters_from_sample(sample: Sample) -> list[str]:
    choices = sample.metadata.get("choices")
    if isinstance(choices, list) and choices:
        return [chr(65 + index) for index in range(len(choices))]
    return list("ABCDEFGHIJ")


def _first_choice_finish_reason(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    value = first.get("finish_reason")
    return "" if value in (None, "") else str(value).strip().lower()


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
