import asyncio
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elbench.judges.judge_highlevel import HighLevelJudge
from elbench.judges.judge_ifeval import IFEvalJudge
from elbench.judges.judge_math import AIMEJudge, Math500Judge
from elbench.judges.judge_mmlu import MMLUProJudge
from elbench.judges.judge_teaching_harm import TeachingHarmJudge
from elbench.judges.llm_judge import LLMJudge
from elbench.judges.router import JudgeRouter
from elbench.judges.ifeval_rule.instructions_registry import INSTRUCTION_DICT
from elbench.config import load_project_config
from elbench.registry import FileRegistry
from elbench.schemas.config import JudgeTaskConfig
from elbench.schemas.evaluation import ModelResponse, Sample
from elbench.utils import extract_json_object


class JudgeSmokeTest(unittest.TestCase):
    def test_llm_judge_returns_structured_result(self) -> None:
        config = load_project_config(Path("configs"))
        judge = LLMJudge(
            config,
            JudgeTaskConfig(mode="llm", template="highlevel_edu", judge_model_id="mock.judge"),
        )
        sample = Sample(
            sample_id="l1",
            source_file="高阶育人-edu.jsonl",
            source_path="高阶育人-edu.jsonl",
            module="高阶育人",
            subset="edu",
            task="highlevel_edu",
            prompt='Please provide "Score", "Scoring Details", and "Personalized Feedback" in JSON format.',
            metadata={"_scene": "AG"},
        )
        response = ModelResponse(text='{"Score": "4/5", "Scoring Details": "ok", "Personalized Feedback": "ok"}')
        result = asyncio.run(judge.judge(sample, response))
        self.assertEqual(result.judge_result, "pass")
        self.assertEqual(result.score, 1.0)

    def test_router_uses_llm_judge_for_subjective_tasks(self) -> None:
        config = load_project_config(Path("configs"))
        router = JudgeRouter(config)
        sample = Sample(
            sample_id="r1",
            source_file="安全拒答.jsonl",
            source_path="安全拒答.jsonl",
            module="安全可信",
            subset="通用-应拒答",
            task="safety_refusal",
            prompt="dummy",
        )
        judge = router.get_judge(sample)
        self.assertIsInstance(judge, LLMJudge)

    def test_router_uses_rule_judge_for_highlevel_edu(self) -> None:
        config = load_project_config(Path("configs"))
        router = JudgeRouter(config)
        sample = Sample(
            sample_id="edu1",
            source_file="高阶育人-edu.jsonl",
            source_path="高阶育人-edu.jsonl",
            module="高阶育人",
            subset="edu",
            task="highlevel_edu",
            prompt="dummy",
        )

        judge = router.get_judge(sample)

        self.assertIsInstance(judge, HighLevelJudge)

    def test_satas_exact_match(self) -> None:
        judge = TeachingHarmJudge()
        sample = Sample(
            sample_id="s1",
            source_file="SATAs.xlsx",
            source_path="SATAs.xlsx",
            module="安全可信",
            subset="教育-教学安全",
            task="teaching_harm",
            prompt="dummy",
            reference={"Answer": "A,B,E"},
        )
        response = ModelResponse(text="A, B, E")
        result = asyncio.run(judge.judge(sample, response))
        self.assertEqual(result.judge_result, "pass")
        self.assertEqual(result.score, 1.0)
        self.assertFalse(result.judge_metadata["partial_credit"])

    def test_highlevel_omni_exact_match(self) -> None:
        judge = HighLevelJudge()
        sample = Sample(
            sample_id="o1",
            source_file="高阶育人-omni.jsonl",
            source_path="高阶育人-omni.jsonl",
            module="高阶育人",
            subset="omni",
            task="highlevel_omni",
            prompt="dummy",
            reference={"answer": "C", "reasonableness": 1},
        )
        response = ModelResponse(text="答案：C")
        result = asyncio.run(judge.judge(sample, response))
        self.assertEqual(result.judge_result, "pass")
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.judge_metadata["predicted_answer"], "C")

    def test_satas_missing_option_receives_partial_credit(self) -> None:
        judge = TeachingHarmJudge()
        sample = Sample(
            sample_id="s2",
            source_file="SATAs.xlsx",
            source_path="SATAs.xlsx",
            module="安全可信",
            subset="教育-教学安全",
            task="teaching_harm",
            prompt="dummy",
            reference={"Answer": "A,B,E"},
        )
        response = ModelResponse(text="A, E")

        result = asyncio.run(judge.judge(sample, response))

        self.assertEqual(result.judge_result, "fail")
        self.assertEqual(result.score, 0.5)
        self.assertTrue(result.judge_metadata["partial_credit"])

    def test_satas_wrong_option_receives_zero_credit(self) -> None:
        judge = TeachingHarmJudge()
        sample = Sample(
            sample_id="s3",
            source_file="SATAs.xlsx",
            source_path="SATAs.xlsx",
            module="安全可信",
            subset="教育-教学安全",
            task="teaching_harm",
            prompt="dummy",
            reference={"Answer": "A,B,E"},
        )
        response = ModelResponse(text="A, C")

        result = asyncio.run(judge.judge(sample, response))

        self.assertEqual(result.judge_result, "fail")
        self.assertEqual(result.score, 0.0)
        self.assertFalse(result.judge_metadata["partial_credit"])

    def test_highlevel_omni_multiple_choice_exact_match(self) -> None:
        judge = HighLevelJudge()
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
        result = asyncio.run(judge.judge(sample, response))
        self.assertEqual(result.judge_result, "pass")
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.judge_metadata["predicted_answer"], "AC")
        self.assertEqual(result.judge_metadata["expected_answer"], "AC")

    def test_llm_judge_respects_model_max_tokens_cap(self) -> None:
        config = load_project_config(Path("configs"))
        judge = LLMJudge(
            config,
            JudgeTaskConfig(mode="llm", template="safety_refusal", judge_model_id="qwen-test"),
        )

        class DummyClient:
            def __init__(self) -> None:
                self.request = None

            async def generate(self, sample, request):
                self.request = request
                return ModelResponse(text='{"judge_result": "pass", "score": 1, "judge_reason": "ok"}')

            async def aclose(self) -> None:
                return None

        dummy = DummyClient()
        judge._client = dummy
        sample = Sample(
            sample_id="l2",
            source_file="安全拒答.jsonl",
            source_path="安全拒答.jsonl",
            module="安全可信",
            subset="通用-应拒答",
            task="safety_refusal",
            prompt="dummy",
        )
        response = ModelResponse(text="refuse")

        asyncio.run(judge.judge(sample, response))
        self.assertIsNotNone(dummy.request)
        self.assertEqual(dummy.request.max_tokens, 1024)
        self.assertLessEqual(dummy.request.max_tokens, config.models["qwen-test"].max_tokens)

    def test_highlevel_edu_ag_json_scoring(self) -> None:
        judge = HighLevelJudge()
        sample = Sample(
            sample_id="ag1",
            source_file="高阶育人-edu.jsonl",
            source_path="高阶育人-edu.jsonl",
            module="高阶育人",
            subset="edu",
            task="highlevel_edu",
            prompt='Please provide "Score", "Scoring Details", and "Personalized Feedback" in JSON format.',
            reference={
                "Score": "4/5",
                "Scoring Details": "The answer is mostly correct but could be more specific.",
                "Personalized Feedback": "Good work. Add one concrete example next time.",
            },
            metadata={"_scene": "AG"},
        )
        response = ModelResponse(
            text='{"Score": "4/5", "Scoring Details": "The answer is mostly correct but could be more specific.", "Personalized Feedback": "Good work. Add one concrete example next time."}'
        )
        result = asyncio.run(judge.judge(sample, response))
        self.assertEqual(result.judge_result, "pass")
        self.assertGreaterEqual(result.score or 0, 0.99)

    def test_mmlu_pro_explicit_answer_match(self) -> None:
        judge = MMLUProJudge()
        sample = self._mmlu_sample("D")
        response = ModelResponse(text="Reasoning here.\nANSWER: D")
        result = asyncio.run(judge.judge(sample, response))
        self.assertEqual(result.judge_result, "pass")
        self.assertEqual(result.score, 1.0)

    def test_extract_json_object_handles_nested_fenced_json(self) -> None:
        parsed = extract_json_object(
            '```json\n{"Answer": {"CorrectOption": "C", "Explanation": "ok"}}\n```'
        )

        self.assertEqual(parsed, {"Answer": {"CorrectOption": "C", "Explanation": "ok"}})

    def test_mmlu_pro_uses_last_explicit_answer(self) -> None:
        judge = MMLUProJudge()
        sample = self._mmlu_sample("B")
        response = ModelResponse(text="A seems tempting at first.\nANSWER: B")
        result = asyncio.run(judge.judge(sample, response))
        self.assertEqual(result.judge_result, "pass")
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.judge_metadata["predicted_answer"], "B")

    def test_mmlu_pro_accepts_answer_letter_with_option_text(self) -> None:
        judge = MMLUProJudge()
        sample = self._mmlu_sample("B")
        response = ModelResponse(text="Reasoning here.\nANSWER: B) negative")
        result = asyncio.run(judge.judge(sample, response))
        self.assertEqual(result.judge_result, "pass")
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.judge_metadata["predicted_answer"], "B")

    def test_mmlu_pro_wrong_answer_fails(self) -> None:
        judge = MMLUProJudge()
        sample = self._mmlu_sample("D")
        response = ModelResponse(text="ANSWER: E")
        result = asyncio.run(judge.judge(sample, response))
        self.assertEqual(result.judge_result, "fail")
        self.assertEqual(result.score, 0.0)

    def test_mmlu_pro_supports_j_option(self) -> None:
        judge = MMLUProJudge()
        sample = self._mmlu_sample("J")
        response = ModelResponse(text="ANSWER: J")
        result = asyncio.run(judge.judge(sample, response))
        self.assertEqual(result.judge_result, "pass")
        self.assertEqual(result.score, 1.0)

    def test_ceval_chinese_answer_prefix(self) -> None:
        judge = MMLUProJudge()
        sample = self._mmlu_sample("C", task="ceval", choices="ABCD")
        response = ModelResponse(text="逐步分析后可知，最后答案如下。\n答案：C")
        result = asyncio.run(judge.judge(sample, response))
        self.assertEqual(result.judge_result, "pass")
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.judge_metadata["predicted_answer"], "C")

    def test_ceval_accepts_answer_letter_with_option_text(self) -> None:
        judge = MMLUProJudge()
        sample = self._mmlu_sample("C", task="ceval", choices="ABCD")
        response = ModelResponse(text="逐步分析后可知，最后答案如下。\n答案：C）正确选项")
        result = asyncio.run(judge.judge(sample, response))
        self.assertEqual(result.judge_result, "pass")
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.judge_metadata["predicted_answer"], "C")

    def test_mmlu_pro_requires_explicit_final_answer_format(self) -> None:
        judge = MMLUProJudge()
        sample = self._mmlu_sample("D", choices="ABCD")
        response = ModelResponse(text="Options: A B C D")
        result = asyncio.run(judge.judge(sample, response))
        self.assertEqual(result.judge_result, "fail")
        self.assertIsNone(result.judge_metadata["predicted_answer"])

    def test_mmlu_pro_multiple_letters_without_final_marker_fails(self) -> None:
        judge = MMLUProJudge()
        sample = self._mmlu_sample("D", choices="ABCD")
        response = ModelResponse(text="I considered A, B, C, and D. D seems possible.")
        result = asyncio.run(judge.judge(sample, response))
        self.assertEqual(result.judge_result, "fail")
        self.assertIsNone(result.judge_metadata["predicted_answer"])

    def test_highlevel_omni_requires_explicit_final_answer_format(self) -> None:
        judge = HighLevelJudge()
        sample = Sample(
            sample_id="o2",
            source_file="高阶育人-omni.jsonl",
            source_path="高阶育人-omni.jsonl",
            module="高阶育人",
            subset="omni",
            task="highlevel_omni",
            prompt="dummy",
            reference={"answer": "C", "reasonableness": 1},
        )
        response = ModelResponse(text="A is tempting, but C is ultimately better.")
        result = asyncio.run(judge.judge(sample, response))
        self.assertEqual(result.judge_result, "fail")
        self.assertIsNone(result.judge_metadata["predicted_answer"])

    def test_highlevel_omni_does_not_accept_earlier_correct_letter(self) -> None:
        judge = HighLevelJudge()
        sample = Sample(
            sample_id="o3",
            source_file="高阶育人-omni.jsonl",
            source_path="高阶育人-omni.jsonl",
            module="高阶育人",
            subset="omni",
            task="highlevel_omni",
            prompt="dummy",
            reference={"answer": "C", "reasonableness": 1},
        )
        response = ModelResponse(text="C is one possible answer, but after reconsidering I choose A.")
        result = asyncio.run(judge.judge(sample, response))
        self.assertEqual(result.judge_result, "fail")
        self.assertIsNone(result.judge_metadata["predicted_answer"])

    def test_ifeval_no_comma_passes_and_fails(self) -> None:
        judge = IFEvalJudge()
        sample = self._ifeval_sample(["punctuation:no_comma"], [{}])

        pass_result = asyncio.run(judge.judge(sample, ModelResponse(text="No comma appears here")))
        fail_result = asyncio.run(judge.judge(sample, ModelResponse(text="A comma, appears here")))

        self.assertEqual(pass_result.judge_result, "pass")
        self.assertEqual(pass_result.score, 1.0)
        self.assertEqual(fail_result.judge_result, "fail")
        self.assertEqual(fail_result.score, 0.0)

    def test_ifeval_highlighted_sections_counts_single_and_double_markdown(self) -> None:
        judge = IFEvalJudge()
        sample = self._ifeval_sample(
            ["detectable_format:number_highlighted_sections"],
            [{"num_highlights": 3}],
        )
        response = ModelResponse(text="*one* **two** **   ** *three*")
        result = asyncio.run(judge.judge(sample, response))

        self.assertEqual(result.judge_result, "pass")
        self.assertEqual(result.score, 1.0)

    def test_ifeval_number_words_relations(self) -> None:
        judge = IFEvalJudge()
        at_least = self._ifeval_sample(["length_constraints:number_words"], [{"relation": "at least", "num_words": 3}])
        less_than = self._ifeval_sample(["length_constraints:number_words"], [{"relation": "less than", "num_words": 3}])

        self.assertEqual(asyncio.run(judge.judge(at_least, ModelResponse(text="one two three"))).score, 1.0)
        self.assertEqual(asyncio.run(judge.judge(less_than, ModelResponse(text="one two"))).score, 1.0)
        self.assertEqual(asyncio.run(judge.judge(less_than, ModelResponse(text="one two three"))).score, 0.0)

    def test_ifeval_number_placeholders(self) -> None:
        judge = IFEvalJudge()
        sample = self._ifeval_sample(["detectable_content:number_placeholders"], [{"num_placeholders": 2}])
        result = asyncio.run(judge.judge(sample, ModelResponse(text="Hello [name] at [address].")))

        self.assertEqual(result.judge_result, "pass")
        self.assertEqual(result.score, 1.0)

    def test_ifeval_multi_instruction_averages_instruction_level(self) -> None:
        judge = IFEvalJudge()
        sample = self._ifeval_sample(
            [
                "punctuation:no_comma",
                "detectable_content:number_placeholders",
                "length_constraints:number_words",
            ],
            [{}, {"num_placeholders": 1}, {"relation": "at least", "num_words": 100}],
        )
        result = asyncio.run(judge.judge(sample, ModelResponse(text="No comma here [slot]")))

        self.assertEqual(result.judge_result, "fail")
        self.assertEqual(result.judge_metadata["prompt_level_strict"], 0.0)
        self.assertAlmostEqual(result.judge_metadata["inst_level_strict"], 2 / 3)

    def test_ifeval_loose_mode_tries_trimmed_lines(self) -> None:
        judge = IFEvalJudge()
        sample = self._ifeval_sample(["punctuation:no_comma"], [{}])
        result = asyncio.run(judge.judge(sample, ModelResponse(text="bad, first line\nclean final line")))

        self.assertEqual(result.judge_metadata["prompt_level_strict"], 0.0)
        self.assertEqual(result.judge_metadata["prompt_level_loose"], 1.0)

    def test_ifeval_dataset_instructions_are_all_registered(self) -> None:
        config = load_project_config(Path("configs"))
        registry = FileRegistry(config)
        resolved = registry.resolve(source_files={"ifeval_sampled.jsonl"})
        self.assertEqual(len(resolved), 1)

        instruction_ids: set[str] = set()
        with resolved[0].path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                metadata = record.get("metadata") or {}
                raw_instruction_ids = metadata.get("instruction_id_list") or []
                instruction_ids.update(str(item) for item in raw_instruction_ids)

        self.assertTrue(instruction_ids)
        self.assertTrue(instruction_ids.issubset(set(INSTRUCTION_DICT)))

    def test_math_500_extracts_boxed_answer(self) -> None:
        judge = Math500Judge()
        sample = self._math_sample("42")
        result = asyncio.run(judge.judge(sample, ModelResponse(text="Reasoning here. Therefore, \\boxed{42}.")))

        self.assertEqual(result.judge_result, "pass")
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.judge_metadata["predicted_answer"], "42")

    def test_math_500_extracts_answer_marker_and_last_number(self) -> None:
        judge = Math500Judge()

        marker_result = asyncio.run(judge.judge(self._math_sample("9"), ModelResponse(text="The answer is 9.")))
        fallback_result = asyncio.run(judge.judge(self._math_sample("-50"), ModelResponse(text="After simplification we get -50.")))

        self.assertEqual(marker_result.score, 1.0)
        self.assertEqual(fallback_result.score, 1.0)

    def test_math_500_normalizes_latex_fraction_and_text(self) -> None:
        judge = Math500Judge()

        fraction_result = asyncio.run(judge.judge(self._math_sample("\\frac{14}{3}"), ModelResponse(text="\\boxed{14/3}")))
        text_result = asyncio.run(judge.judge(self._math_sample("\\text{Evelyn}"), ModelResponse(text="final answer is Evelyn")))

        self.assertEqual(fraction_result.score, 1.0)
        self.assertEqual(text_result.score, 1.0)

    def test_math_500_tuple_latex_matches_reference(self) -> None:
        judge = Math500Judge()
        sample = self._math_sample("\\left( 3, \\frac{\\pi}{2} \\right)")
        response = ModelResponse(text="The polar form is \\boxed{\\left(3,\\frac{\\pi}{2}\\right)}.")
        result = asyncio.run(judge.judge(sample, response))

        self.assertEqual(result.judge_result, "pass")
        self.assertEqual(result.score, 1.0)

    def test_math_500_wrong_answer_fails(self) -> None:
        judge = Math500Judge()
        result = asyncio.run(judge.judge(self._math_sample("42"), ModelResponse(text="\\boxed{43}")))

        self.assertEqual(result.judge_result, "fail")
        self.assertEqual(result.score, 0.0)

    def test_aime_extracts_boxed_reference_and_prediction(self) -> None:
        judge = AIMEJudge()
        result = asyncio.run(judge.judge(self._aime_sample("\\boxed{204}"), ModelResponse(text="After solving, \\boxed{204}")))

        self.assertEqual(result.judge_result, "pass")
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.judge_metadata["expected_answer"], "204")
        self.assertEqual(result.judge_metadata["predicted_answer"], "204")

    def test_aime_uses_last_number_fallback_for_prediction(self) -> None:
        judge = AIMEJudge()
        result = asyncio.run(judge.judge(self._aime_sample("\\boxed{113}"), ModelResponse(text="The requested value is 113.")))

        self.assertEqual(result.judge_result, "pass")
        self.assertEqual(result.score, 1.0)

    def test_aime_wrong_answer_fails(self) -> None:
        judge = AIMEJudge()
        result = asyncio.run(judge.judge(self._aime_sample("\\boxed{371}"), ModelResponse(text="\\boxed{370}")))

        self.assertEqual(result.judge_result, "fail")
        self.assertEqual(result.score, 0.0)

    def test_aime25_and_aime26_use_same_rule_judge(self) -> None:
        config = load_project_config(Path("configs"))
        router = JudgeRouter(config)

        aime25 = self._aime_sample("70", task="aime25", source_file="aime25.jsonl")
        aime26 = self._aime_sample("277", task="aime26", source_file="aime26.jsonl")

        self.assertIsInstance(router.get_judge(aime25), AIMEJudge)
        self.assertIsInstance(router.get_judge(aime26), AIMEJudge)

    def _mmlu_sample(self, target: str, task: str = "mmlu_pro", choices: str = "ABCDEFGHIJ") -> Sample:
        return Sample(
            sample_id="mmlu1",
            source_file=f"{task}_sampled.jsonl",
            source_path=f"{task}_sampled.jsonl",
            module="通用模型",
            subset=task,
            task=task,
            dimension="law",
            prompt="dummy",
            reference={"target": target},
            metadata={"choices": [f"choice {letter}" for letter in choices]},
        )

    def _ifeval_sample(self, instruction_ids: list[str], kwargs: list[dict]) -> Sample:
        return Sample(
            sample_id="ifeval1",
            source_file="ifeval_sampled.jsonl",
            source_path="ifeval_sampled.jsonl",
            module="通用模型",
            subset="ifeval",
            task="ifeval",
            dimension="default",
            prompt="dummy",
            reference={"target": ""},
            metadata={
                "instruction_id_list": instruction_ids,
                "kwargs": kwargs,
                "key": 1,
            },
        )

    def _math_sample(self, target: str) -> Sample:
        return Sample(
            sample_id="math1",
            source_file="math_500_sampled.jsonl",
            source_path="math_500_sampled.jsonl",
            module="通用模型",
            subset="math_500",
            task="math_500",
            dimension="Level 2",
            prompt="dummy",
            reference={"target": target},
            metadata={"question_id": "test/example.json", "subset_key": "Level 2"},
        )

    def _aime_sample(self, target: str, task: str = "aime24", source_file: str = "aime24_sampled.jsonl") -> Sample:
        return Sample(
            sample_id="aime1",
            source_file=source_file,
            source_path=source_file,
            module="通用模型",
            subset=task,
            task=task,
            dimension="default",
            prompt="dummy",
            reference={"target": target},
            metadata={},
        )


if __name__ == "__main__":
    unittest.main()
