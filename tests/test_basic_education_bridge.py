import logging
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elbench.config import load_project_config  # noqa: E402
from elbench.basic_education_runtime.cli.eval import _metric_fields_from_evals  # noqa: E402
from elbench.basic_education_runtime.entity import AgentConfig, AgentMemoryConfig, ExportFormat, Prompt  # noqa: E402
from elbench.basic_education_runtime.router import any_keyword_route  # noqa: E402
from elbench.basic_education_runtime.utils import content_to_text, remove_think  # noqa: E402
from elbench.execution.basic_education import (  # noqa: E402
    BasicEducationExecutor,
    aggregate_numeric_score,
    count_runtime_tasks,
    load_basic_education_config,
)


class BasicEducationBridgeTest(unittest.TestCase):
    def test_count_iter_tasks(self) -> None:
        cfg = {
            "tasks": {
                "mode": "iter",
                "content": [{"q": 1}, {"q": 2}, {"q": 3}],
            }
        }
        self.assertEqual(count_runtime_tasks(cfg), 3)

    def test_count_union_tasks(self) -> None:
        cfg = {
            "tasks": {
                "mode": "union",
                "content": {
                    "image": ["a", "b", "c"],
                    "question": ["q1", "q2", "q3", "q4", "q5"],
                },
            }
        }
        self.assertEqual(count_runtime_tasks(cfg), 15)

    def test_numeric_score_aggregation(self) -> None:
        score, details = aggregate_numeric_score(
            {
                "a": 4,
                "b": {"x": 2.0, "y": 3},
                "c": [1, {"z": 5}],
                "d": "skip",
            }
        )
        self.assertIsNotNone(score)
        self.assertAlmostEqual(score or 0, (4 + 2 + 3 + 1 + 5) / 5)
        self.assertEqual(details["a"], 4.0)
        self.assertEqual(details["b.x"], 2.0)
        self.assertEqual(details["b.y"], 3.0)
        self.assertEqual(details["c[0]"], 1.0)
        self.assertEqual(details["c[1].z"], 5.0)

    def test_basic_education_config_declares_45_tasks(self) -> None:
        config = load_basic_education_config(
            config_root=ROOT / "configs",
            project_root=ROOT,
            data_root=ROOT / "data" / "benchmark_root",
        )
        self.assertTrue(config.enabled)
        expected = sum(int(s.expected_tasks or 0) for s in config.scenarios if s.enabled)
        self.assertEqual(expected, 45)
        base_dir = ROOT / "data" / "benchmark_root" / "基本教育"
        for scenario in config.scenarios:
            self.assertIn(str(base_dir), str(scenario.template_path))
            self.assertTrue(scenario.template_path.exists())

    def test_mock_runtime_payload_does_not_require_api_base(self) -> None:
        project_config = load_project_config(ROOT / "configs")
        executor = BasicEducationExecutor(
            project_config=project_config,
            model_config=project_config.models["mock.default"],
            logger=logging.getLogger("test-basic-education"),
        )

        payload = executor._build_runtime_model_payload(project_config.models["mock.default"])

        self.assertEqual(payload["type"], "mock")
        self.assertEqual(payload["model"], "mock-echo")
        self.assertIsNone(payload["api_base"])
        self.assertEqual(payload["kargs"]["prefix"], "[MOCK]")

    def test_guided_scenario_splits_teacher_student_and_eval_models(self) -> None:
        project_config = load_project_config(ROOT / "configs")
        executor = BasicEducationExecutor(
            project_config=project_config,
            model_config=project_config.models["gpt-5.4"],
            logger=logging.getLogger("test-basic-education"),
        )
        executor.config.test_endpoint_model_id = "qwen-test"
        scenario = next(
            item
            for item in executor.config.scenarios
            if item.scenario_id == "guided_problem_solving_teaching"
        )
        template_data = executor._read_yaml_file(scenario.template_path)

        with patch("elbench.execution.basic_education.get_api_key", return_value="test-key"):
            rendered = executor._render_runtime_template(
                template_data=template_data,
                scenario=scenario,
                memory_path=ROOT / "tmp_test_artifacts" / "basic_education_guided_memory",
                max_samples=1,
            )

        models_section = rendered["models"]
        self.assertEqual(models_section["teacher"]["model"], "gpt-5.4")
        self.assertEqual(
            models_section["teacher"]["api_base"],
            project_config.models["gpt-5.4"].api_base,
        )
        self.assertEqual(models_section["stu"]["model"], project_config.models["qwen-test"].model_name)
        self.assertEqual(
            models_section["stu"]["api_base"],
            project_config.models["qwen-test"].api_base,
        )
        self.assertEqual(models_section["eval"]["model"], project_config.models["qwen-test"].model_name)
        self.assertEqual(models_section["stu"]["kargs"]["max_tokens"], 512)
        self.assertEqual(models_section["eval"]["kargs"]["max_tokens"], 1024)
        self.assertEqual(
            models_section["eval"]["kargs"]["extra_body"],
            {
                "chat_template_kwargs": {"enable_thinking": False},
                "served_model_name": "qwen3.6-35B-sft-v1",
            },
        )
        self.assertEqual(rendered["globals"]["recursion_limit"], 20)
        self.assertEqual(rendered["evaluation"]["model"], "eval")

    def test_runtime_template_uses_target_model_concurrency_cap(self) -> None:
        project_config = load_project_config(ROOT / "configs")
        executor = BasicEducationExecutor(
            project_config=project_config,
            model_config=project_config.models["innospark-235b"],
            logger=logging.getLogger("test-basic-education"),
        )
        scenario = next(
            item
            for item in executor.config.scenarios
            if item.scenario_id == "contextualized_question_generation"
        )
        template_data = executor._read_yaml_file(scenario.template_path)

        with patch("elbench.execution.basic_education.get_api_key", return_value="test-key"):
            rendered = executor._render_runtime_template(
                template_data=template_data,
                scenario=scenario,
                memory_path=ROOT / "tmp_test_artifacts" / "basic_education_question_memory",
                max_samples=1,
            )

        self.assertEqual(rendered["globals"]["concurrency"], 1)

    def test_content_to_text_ignores_reasoning_only_blocks(self) -> None:
        content = [{"type": "reasoning", "summary": [], "text": "hidden"}]
        self.assertEqual(content_to_text(content), "")

    def test_content_to_text_extracts_text_blocks(self) -> None:
        content = [
            {"type": "reasoning", "summary": [], "text": "hidden"},
            {"type": "text", "text": "课堂结束 <end>"},
        ]
        self.assertEqual(content_to_text(remove_think(content)), "课堂结束 <end>")

    def test_keyword_route_handles_openai_content_blocks(self) -> None:
        route, _ = any_keyword_route(["<end>"], exists_to="END", else_to="student")
        state = {
            "messages": [
                type("Msg", (), {"content": [{"type": "text", "text": "下课。<end>"}]})(),
            ]
        }
        self.assertTrue(route(state))


    def test_agent_skips_messages_that_become_empty_after_reasoning_cleanup(self) -> None:
        import asyncio
        from langchain_core.messages import AIMessage, HumanMessage
        from elbench.basic_education_runtime import config as runtime_config

        runtime_config.CONFIG = SimpleNamespace(
            globals=SimpleNamespace(
                retry=SimpleNamespace(attempt=1, interval=0),
                model_call_timeout_seconds=300,
            )
        )
        from elbench.basic_education_runtime.agent import _init_agent_from_dict

        class CapturingModel:
            def __init__(self) -> None:
                self.messages = []

            async def ainvoke(self, messages):
                self.messages = messages
                return AIMessage(content="ok")

        model = CapturingModel()
        agent = _init_agent_from_dict(
            AgentConfig(
                model="teacher",
                prompt=[Prompt(role="system", content="Teach.")],
                memory=AgentMemoryConfig(enable=False, keep_turns=3),
            ),
            {"teacher": model},
            "teacher",
        )

        state = {
            "messages": [
                AIMessage(
                    content=[{"type": "reasoning", "summary": [], "text": "hidden"}],
                    name="teacher",
                ),
                HumanMessage(content="Please continue.", name="student"),
            ]
        }

        asyncio.run(agent(state))

        contents = [getattr(message, "content", "") for message in model.messages]
        self.assertEqual(contents, ["Teach.", "Please continue."])


class BasicEducationEvalCliHelpersTest(unittest.TestCase):
    def test_metric_fields_from_evals_ignores_empty_results(self) -> None:
        self.assertEqual(_metric_fields_from_evals([{}, {}]), [])
        self.assertEqual(_metric_fields_from_evals([{}, {"a": 1, "b": 2}]), ["a", "b"])

    def test_evaluation_messages_add_user_prompt_when_template_only_has_system(self) -> None:
        from elbench.basic_education_runtime import config as runtime_config

        runtime_config.CONFIG = SimpleNamespace(
            globals=SimpleNamespace(
                retry=SimpleNamespace(attempt=1, interval=0),
                model_call_timeout_seconds=300,
            )
        )
        from elbench.basic_education_runtime.evaluation import _build_evaluation_messages

        exported = ExportFormat(
            task={"query": "Explain decimals"},
            messages=[Prompt(role="teacher", content="Decimals are parts of one.")],
        )

        messages = _build_evaluation_messages(
            system_prompt="Evaluate this output:\n{messages.as_dialog()}\nTask: {task.query}",
            other_prompts=[],
            exported_result=exported,
        )

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("teacher: Decimals are parts of one.", messages[0]["content"])
        self.assertIn("Task: Explain decimals", messages[0]["content"])

    def test_replace_template_preserves_non_placeholder_braces(self) -> None:
        exported = ExportFormat(
            task={"query": "Solve the set example"},
            messages=[Prompt(role="teacher", content="The answer set is {1}.")],
        )

        rendered = exported.replace_template(
            "Evaluate {task.query}\n{messages.as_dialog()}\nDo not treat {1} as a template."
        )

        self.assertIn("Evaluate Solve the set example", rendered)
        self.assertIn("teacher: The answer set is {1}.", rendered)
        self.assertIn("Do not treat {1} as a template.", rendered)


if __name__ == "__main__":
    unittest.main()
