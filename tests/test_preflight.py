import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elbench.config import load_project_config  # noqa: E402
from elbench.execution import PreflightRunner  # noqa: E402
from elbench.execution.selection import resolve_module_selection  # noqa: E402


class PreflightSmokeTest(unittest.TestCase):
    def test_default_module_selection_includes_all_four_modules(self) -> None:
        config = load_project_config(ROOT / "configs")
        selection = resolve_module_selection(config, None)

        self.assertIn("安全可信", selection.selected_modules)
        self.assertIn("高阶育人", selection.selected_modules)
        self.assertIn("通用模型", selection.selected_modules)
        self.assertIn("基本教育", selection.selected_modules)
        self.assertTrue(selection.include_basic_education)

    def test_preflight_accepts_real_judge_for_official_run(self) -> None:
        config = load_project_config(ROOT / "configs")
        runner = PreflightRunner(config)

        report = runner.run(
            model_id="gpt-5.4",
            modules={"安全可信"},
            source_files={"安全拒答.jsonl"},
            require_real_judges=True,
        )

        self.assertTrue(report.ok)

    def test_preflight_allows_mock_judge_for_local_smoke(self) -> None:
        config = load_project_config(ROOT / "configs")
        runner = PreflightRunner(config)

        report = runner.run(
            model_id="mock.default",
            modules={"通用模型"},
            source_files={"mmlu_pro_sampled.jsonl"},
            max_samples=1,
            require_real_judges=False,
        )

        self.assertTrue(report.ok)
        self.assertEqual(report.total_samples, 1)
        self.assertIn("通用模型", report.by_module)

    def test_preflight_accepts_real_basic_education_test_endpoint_for_official_run(self) -> None:
        config = load_project_config(ROOT / "configs")
        runner = PreflightRunner(config)

        report = runner.run(
            model_id="gpt-5.4",
            modules={"基本教育"},
            source_files={"config_guided_task.yaml"},
            require_real_judges=True,
        )

        self.assertTrue(report.ok)


if __name__ == "__main__":
    unittest.main()
