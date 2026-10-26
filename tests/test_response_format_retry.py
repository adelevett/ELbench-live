import asyncio
import json
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elbench.config import load_project_config  # noqa: E402
from elbench.execution import BenchmarkRunner, RunOptions  # noqa: E402
from elbench.execution.rate_limit import SlidingWindowRateLimiter  # noqa: E402
from elbench.execution.runner import _build_concise_answer_retry_prompt  # noqa: E402
from elbench.execution.response_format import (  # noqa: E402
    ResponseFormatSpec,
    attach_format_metadata,
    build_concise_retry_prompt,
    build_retry_followup,
    get_response_format_spec,
    should_force_concise_retry,
)
from elbench.persistence import CheckpointStore, JsonlWriter  # noqa: E402
from elbench.schemas.config import RateLimitConfig, RetryConfig  # noqa: E402
from elbench.schemas.evaluation import GenerationRequest, ModelResponse, Sample  # noqa: E402


class ResponseFormatRetryTest(unittest.TestCase):
    def test_empty_length_response_with_reasoning_tokens_uses_concise_retry(self) -> None:
        response = ModelResponse(
            text="",
            raw_payload={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": ""},
                        "finish_reason": "length",
                    }
                ]
            },
            usage={
                "completion_tokens": 8188,
                "completion_tokens_details": {"reasoning_tokens": 7861},
            },
        )
        spec = ResponseFormatSpec(
            name="single_choice_en",
            instruction="The last line must be exactly: ANSWER: [LETTER].",
            reminder_template="unused",
            valid_letters=list("ABCD"),
            answer_prefixes=["ANSWER"],
        )

        self.assertTrue(should_force_concise_retry(response))

        followup = build_retry_followup(spec, response)
        self.assertEqual(followup["retry_reason"], "format_empty_after_reasoning")
        self.assertEqual(followup["max_tokens"], 64)
        self.assertIsNone(followup["assistant_content"])
        self.assertIn("Do not include any reasoning", followup["user_reminder"])
        self.assertIn("ANSWER: [LETTER]", followup["user_reminder"])

    def test_concise_retry_prompt_removes_step_by_step_instruction(self) -> None:
        base_prompt = "\n".join(
            [
                "Question:",
                "Example question?",
                "Options:",
                "A) example",
                "ANSWER: A",
                "",
                "Answer the following multiple choice question.",
                "Think step by step before answering.",
                "The last line of your response must be exactly: ANSWER: [LETTER].",
                "",
                "Question:",
                "What is 2 + 2?",
                "Options:",
                "A) 3",
                "B) 4",
            ]
        )
        spec = ResponseFormatSpec(
            name="single_choice_en",
            instruction="The last line must be exactly: ANSWER: [LETTER].",
            reminder_template="unused",
            valid_letters=list("AB"),
            answer_prefixes=["ANSWER"],
        )

        retry_prompt = build_concise_retry_prompt(base_prompt, spec)

        self.assertNotIn("Think step by step", retry_prompt)
        self.assertIn("exactly one uppercase option letter", retry_prompt)
        self.assertIn("Question:", retry_prompt)
        self.assertIn("What is 2 + 2?", retry_prompt)
        self.assertNotIn("Example question?", retry_prompt)
        self.assertIn("Your answer:", retry_prompt)

    def test_nonempty_response_keeps_normal_format_retry(self) -> None:
        response = ModelResponse(
            text="I think the answer is A.",
            raw_payload={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "I think the answer is A."},
                        "finish_reason": "stop",
                    }
                ]
            },
            usage={"completion_tokens": 12},
        )
        spec = ResponseFormatSpec(
            name="single_choice_en",
            instruction="The last line must be exactly: ANSWER: [LETTER].",
            reminder_template="Your previous response did not follow the required answer format.",
            valid_letters=list("ABCD"),
            answer_prefixes=["ANSWER"],
        )

        self.assertFalse(should_force_concise_retry(response))

        followup = build_retry_followup(spec, response)
        self.assertEqual(followup["retry_reason"], "format_invalid")
        self.assertEqual(followup["assistant_content"], "I think the answer is A.")
        self.assertIn("did not follow the required answer format", followup["user_reminder"])

    def test_nonempty_length_response_uses_concise_retry(self) -> None:
        response = ModelResponse(
            text="Long reasoning that used the full output budget before the final answer.",
            raw_payload={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Long reasoning that used the full output budget before the final answer.",
                        },
                        "finish_reason": "length",
                    }
                ]
            },
            usage={"completion_tokens": 1024},
        )
        spec = ResponseFormatSpec(
            name="single_choice_en",
            instruction="The last line must be exactly: ANSWER: [LETTER].",
            reminder_template="unused",
            valid_letters=list("ABCD"),
            answer_prefixes=["ANSWER"],
        )

        self.assertTrue(should_force_concise_retry(response))

        followup = build_retry_followup(spec, response)
        self.assertEqual(followup["retry_reason"], "format_empty_after_reasoning")
        self.assertEqual(followup["max_tokens"], 64)
        self.assertIsNone(followup["assistant_content"])

    def test_failed_choice_checkpoint_retries_with_short_concise_prompt(self) -> None:
        class ConciseChoiceClient:
            def __init__(self) -> None:
                self.requests = []

            async def generate(self, sample, request):
                self.requests.append(request)
                return ModelResponse(
                    text="ANSWER: A",
                    raw_payload={
                        "choices": [
                            {
                                "message": {"role": "assistant", "content": "ANSWER: A"},
                                "finish_reason": "stop",
                            }
                        ]
                    },
                    usage={"completion_tokens": 3},
                )

        temp_dir = ROOT / "tmp_test_artifacts" / "failed_choice_checkpoint"
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            runner = BenchmarkRunner(load_project_config(ROOT / "configs"))
            client = ConciseChoiceClient()
            sample = Sample(
                sample_id="mmlu1",
                source_file="mmlu_pro_sampled.jsonl",
                source_path="mmlu_pro_sampled.jsonl",
                module="general",
                subset="mmlu_pro",
                task="mmlu_pro",
                prompt=(
                    "Answer the following multiple choice question.\n"
                    "Think step by step before answering.\n"
                    "Options:\nA. one\nB. two"
                ),
                reference={"answer": "A"},
            )
            request = GenerationRequest(
                prompt=sample.prompt,
                temperature=0,
                max_tokens=1024,
                stream=False,
                messages=[{"role": "user", "content": sample.prompt}],
                provider_kwargs={},
            )
            checkpoint = CheckpointStore(temp_dir / "checkpoint.json")
            checkpoint.failed_ids.add("mmlu_pro_sampled.jsonl::mmlu1")

            response, retry_count = asyncio.run(
                runner._generate_with_retry(
                    sample=sample,
                    client=client,
                    request=request,
                    retry_policy=RetryConfig(
                        max_attempts=2,
                        initial_delay_seconds=0,
                        max_delay_seconds=0,
                        jitter_ratio=0,
                    ),
                    limiter=SlidingWindowRateLimiter(RateLimitConfig(max_concurrency=1)),
                    retry_writer=JsonlWriter(temp_dir / "retries.jsonl"),
                    checkpoint=checkpoint,
                )
            )

            self.assertEqual(retry_count, 0)
            self.assertEqual(response.text, "ANSWER: A")
            self.assertEqual(len(client.requests), 1)
            self.assertEqual(client.requests[0].max_tokens, 64)
            self.assertNotIn("Think step by step", client.requests[0].prompt)
            self.assertIn("exactly one uppercase option letter", client.requests[0].prompt)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_math_length_response_retries_with_concise_prompt(self) -> None:
        class TruncatedMathClient:
            def __init__(self) -> None:
                self.requests = []

            async def generate(self, sample, request):
                self.requests.append(request)
                if len(self.requests) == 1:
                    return ModelResponse(
                        text="Long derivation without a final boxed answer.",
                        raw_payload={
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": "Long derivation without a final boxed answer.",
                                    },
                                    "finish_reason": "length",
                                }
                            ]
                        },
                        usage={"completion_tokens": 1024},
                    )
                return ModelResponse(
                    text="\\boxed{42}",
                    raw_payload={
                        "choices": [
                            {
                                "message": {"role": "assistant", "content": "\\boxed{42}"},
                                "finish_reason": "stop",
                            }
                        ]
                    },
                    usage={"completion_tokens": 8},
                )

        temp_dir = ROOT / "tmp_test_artifacts" / "math_length_retry"
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            runner = BenchmarkRunner(load_project_config(ROOT / "configs"))
            client = TruncatedMathClient()
            sample = Sample(
                sample_id="math1",
                source_file="math_500_sampled.jsonl",
                source_path="math_500_sampled.jsonl",
                module="general",
                subset="math_500",
                task="math_500",
                prompt="Solve this. Please reason step by step, and put your final answer within \\boxed{}.",
                reference={"target": "42"},
            )
            request = GenerationRequest(
                prompt=sample.prompt,
                temperature=0,
                max_tokens=1024,
                stream=False,
                messages=[{"role": "user", "content": sample.prompt}],
                provider_kwargs={},
            )

            response, retry_count = asyncio.run(
                runner._generate_with_retry(
                    sample=sample,
                    client=client,
                    request=request,
                    retry_policy=RetryConfig(
                        max_attempts=2,
                        initial_delay_seconds=0,
                        max_delay_seconds=0,
                        jitter_ratio=0,
                    ),
                    limiter=SlidingWindowRateLimiter(RateLimitConfig(max_concurrency=1)),
                    retry_writer=JsonlWriter(temp_dir / "retries.jsonl"),
                    checkpoint=CheckpointStore(temp_dir / "checkpoint.json"),
                )
            )

            self.assertEqual(retry_count, 1)
            self.assertEqual(response.text, "\\boxed{42}")
            self.assertEqual(len(client.requests), 2)
            self.assertIn("Do not include any reasoning", client.requests[1].prompt)
            self.assertEqual(client.requests[1].max_tokens, 1024)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_math_retryable_error_retries_with_concise_prompt(self) -> None:
        class FailingMathClient:
            def __init__(self) -> None:
                self.requests = []

            async def generate(self, sample, request):
                self.requests.append(request)
                if len(self.requests) == 1:
                    raise RuntimeError("transient gateway timeout")
                return ModelResponse(
                    text="\\boxed{42}",
                    raw_payload={
                        "choices": [
                            {
                                "message": {"role": "assistant", "content": "\\boxed{42}"},
                                "finish_reason": "stop",
                            }
                        ]
                    },
                    usage={"completion_tokens": 8},
                )

        temp_dir = ROOT / "tmp_test_artifacts" / "math_error_retry"
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            runner = BenchmarkRunner(load_project_config(ROOT / "configs"))
            client = FailingMathClient()
            sample = Sample(
                sample_id="math1",
                source_file="math_500_sampled.jsonl",
                source_path="math_500_sampled.jsonl",
                module="general",
                subset="math_500",
                task="math_500",
                prompt="Solve this. Please reason step by step, and put your final answer within \\boxed{}.",
                reference={"target": "42"},
            )
            request = GenerationRequest(
                prompt=sample.prompt,
                temperature=0,
                max_tokens=1024,
                stream=False,
                messages=[{"role": "user", "content": sample.prompt}],
                provider_kwargs={},
            )

            response, retry_count = asyncio.run(
                runner._generate_with_retry(
                    sample=sample,
                    client=client,
                    request=request,
                    retry_policy=RetryConfig(
                        max_attempts=2,
                        initial_delay_seconds=0,
                        max_delay_seconds=0,
                        jitter_ratio=0,
                    ),
                    limiter=SlidingWindowRateLimiter(RateLimitConfig(max_concurrency=1)),
                    retry_writer=JsonlWriter(temp_dir / "retries.jsonl"),
                    checkpoint=CheckpointStore(temp_dir / "checkpoint.json"),
                )
            )

            self.assertEqual(retry_count, 1)
            self.assertEqual(response.text, "\\boxed{42}")
            self.assertEqual(len(client.requests), 2)
            self.assertIn("Do not include any reasoning", client.requests[1].prompt)
            self.assertEqual(client.requests[1].max_tokens, 1024)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_concise_math_retry_prompt_preserves_question_when_instruction_is_inline(self) -> None:
        prompt = (
            "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. "
            "Please reason step by step, and put your final answer within \\boxed{}."
        )

        retry_prompt = _build_concise_answer_retry_prompt(prompt)

        self.assertIn("Convert the point", retry_prompt)
        self.assertNotIn("reason step by step", retry_prompt.lower())
        self.assertIn("\\boxed{}", retry_prompt)

    def test_highlevel_omni_format_accepts_multiple_choice_answer(self) -> None:
        sample = Sample(
            sample_id="o_multi",
            source_file="高阶育人-omni.jsonl",
            source_path="高阶育人-omni.jsonl",
            module="高阶育人",
            subset="omni",
            task="highlevel_omni",
            prompt="dummy",
            reference={"answer": "AC", "reasonableness": 1},
        )
        response = ModelResponse(text="答案：AC")

        response = attach_format_metadata(response, get_response_format_spec(sample))

        self.assertTrue(response.format_valid)
        self.assertEqual(response.format_metadata["parsed_answer"], "AC")

    def test_single_choice_invalid_format_triggers_retry_and_recovers(self) -> None:
        temp_dir = ROOT / "tmp_test_artifacts" / "elbench_format_retry"
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            config = load_project_config(ROOT / "configs")
            config.app.output_root = temp_dir / "outputs"
            model = config.models["mock.default"].model_copy(deep=True)
            model.provider_kwargs["mode"] = "format_retry_probe"
            model.retry.max_attempts = 2
            config.models["mock.default"] = model

            runner = BenchmarkRunner(config)
            result = asyncio.run(
                runner.run(
                    RunOptions(
                        model_id="mock.default",
                        run_id="format-retry-probe",
                        modules={"通用模型"},
                        source_files={"mmlu_pro_sampled.jsonl"},
                        max_samples=1,
                    )
                )
            )

            output_root = temp_dir / "outputs"
            retries_path = output_root / "logs" / "format-retry-probe" / "mock.default.retries.jsonl"
            judged_path = Path(result["judged_path"])

            retries = [
                json.loads(line)
                for line in retries_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            judged = [
                json.loads(line)
                for line in judged_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(len(retries), 1)
            self.assertEqual(retries[0]["retry_reason"], "format_invalid")
            self.assertEqual(len(judged), 1)
            self.assertEqual(judged[0]["retry_count"], 1)
            self.assertEqual(judged[0]["judge_result"], "fail")
            self.assertEqual(judged[0]["model_response"], "ANSWER: A")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_final_format_failure_is_counted_as_model_wrong_answer(self) -> None:
        temp_dir = ROOT / "tmp_test_artifacts" / "elbench_format_final_failure"
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            config = load_project_config(ROOT / "configs")
            config.app.output_root = temp_dir / "outputs"
            model = config.models["mock.default"].model_copy(deep=True)
            model.provider_kwargs["mode"] = "empty"
            model.retry.max_attempts = 2
            config.models["mock.default"] = model

            runner = BenchmarkRunner(config)
            result = asyncio.run(
                runner.run(
                    RunOptions(
                        model_id="mock.default",
                        run_id="format-final-failure-probe",
                        modules={"通用模型"},
                        source_files={"mmlu_pro_sampled.jsonl"},
                        max_samples=1,
                    )
                )
            )

            judged_path = Path(result["judged_path"])
            failures_path = Path(result["failures_path"])
            judged = [
                json.loads(line)
                for line in judged_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            failure_lines = failures_path.read_text(encoding="utf-8").splitlines() if failures_path.exists() else []
            failures = [json.loads(line) for line in failure_lines if line.strip()]

            self.assertEqual(len(judged), 1)
            self.assertEqual(len(failures), 0)
            self.assertEqual(judged[0]["judge_result"], "fail")
            self.assertEqual(judged[0]["score"], 0)
            self.assertEqual(judged[0]["metadata"]["failure_category"], "model_instruction_following")
            self.assertEqual(judged[0]["metadata"]["failure_type"], "response_format")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
