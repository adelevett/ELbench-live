import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elbench.dashboard.catalog import load_catalog
from elbench.dashboard.plan import build_plan, render_plan
from elbench.dashboard.protocol import (
    UNLISTED_MAX_TOKENS,
    VISITOR_KEY_ENV,
    build_visitor_protocol,
    preview_standard_payload,
)
from elbench.dashboard.run import execute_run, scrub_tree
from elbench.execution.preflight import PreflightError
from elbench.execution.response_format import augment_prompt_for_format
from elbench.execution.runner import BenchmarkRunner
from elbench.providers.openai_compatible import OpenAICompatibleClient
from elbench.registry import FileRegistry
from elbench.schemas.evaluation import GenerationRequest


class DashboardFaithfulnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(ROOT / "configs")

    def setUp(self):
        self._previous_key = os.environ.get(VISITOR_KEY_ENV)
        os.environ[VISITOR_KEY_ENV] = "sk-or-test-key-not-a-secret"

    def tearDown(self):
        if self._previous_key is None:
            os.environ.pop(VISITOR_KEY_ENV, None)
        else:
            os.environ[VISITOR_KEY_ENV] = self._previous_key

    def test_listed_card_copies_generation_settings_and_keeps_openrouter_model_id(self):
        protocol = build_visitor_protocol(self.catalog.config, "openai/gpt-5.4")
        self.assertEqual(protocol.matched_card_id, "gpt-5.4")
        self.assertEqual(protocol.model_config.model_name, "openai/gpt-5.4")
        self.assertEqual(protocol.model_config.max_tokens, 4096)
        self.assertEqual(protocol.model_config.temperature, 0)
        self.assertFalse(protocol.model_config.supports_stream)
        self.assertEqual(protocol.model_config.api_base, "https://openrouter.ai/api/v1")
        self.assertFalse(protocol.request_branch_differences)
        codes = {caveat.code for caveat in protocol.caveats}
        self.assertIn("model_card_copied", codes)
        self.assertIn("openrouter_host", codes)
        self.assertIn("openrouter_provider_routing", codes)
        self.assertNotIn("request_branch_differs", codes)

    def test_routing_variant_keeps_the_slug_and_copies_the_base_card(self):
        protocol = build_visitor_protocol(self.catalog.config, "openai/gpt-5.4:nitro")
        self.assertEqual(protocol.matched_card_id, "gpt-5.4")
        self.assertEqual(protocol.model_config.model_name, "openai/gpt-5.4:nitro")
        self.assertEqual(protocol.model_config.max_tokens, 4096)
        self.assertIn("routing_variant", {caveat.code for caveat in protocol.caveats})

    def test_free_catalog_variant_is_not_rewritten_to_the_paid_card(self):
        protocol = build_visitor_protocol(self.catalog.config, "openai/gpt-5.4:free")
        self.assertIsNone(protocol.matched_card_id)
        self.assertEqual(protocol.model_config.model_name, "openai/gpt-5.4:free")
        self.assertNotIn("routing_variant", {caveat.code for caveat in protocol.caveats})

    def test_responses_api_branch_is_reported_and_not_patched(self):
        protocol = build_visitor_protocol(self.catalog.config, "openai/gpt-5.2-pro")
        self.assertEqual(protocol.matched_card_id, "gpt-5.2-pro")
        self.assertIn("request_branch_differs", {caveat.code for caveat in protocol.caveats})
        self.assertTrue(protocol.card_would_use_responses_api)
        self.assertFalse(protocol.responses_api)

    def test_unlisted_model_uses_modal_settings(self):
        protocol = build_visitor_protocol(self.catalog.config, "acme/unknown-model")
        self.assertIsNone(protocol.matched_card_id)
        self.assertEqual(protocol.model_config.temperature, 0)
        self.assertEqual(protocol.model_config.max_tokens, UNLISTED_MAX_TOKENS)
        self.assertEqual(protocol.model_config.provider_kwargs, {})
        self.assertIn("no_model_card", {caveat.code for caveat in protocol.caveats})

    def test_shared_model_name_does_not_copy_a_card(self):
        protocol = build_visitor_protocol(self.catalog.config, "vendor/qwen")
        self.assertIsNone(protocol.matched_card_id)
        self.assertEqual(protocol.model_config.max_tokens, UNLISTED_MAX_TOKENS)
        self.assertIn("ambiguous_model_card", {caveat.code for caveat in protocol.caveats})

    def test_preview_payload_matches_the_runner_request_fields(self):
        protocol = build_visitor_protocol(self.catalog.config, "openai/gpt-5.4")
        config = self.catalog.config.model_copy(deep=True)
        config.models[protocol.model_config.model_id] = protocol.model_config
        resolved = FileRegistry(config).resolve(
            modules={"通用模型"},
            source_files={"mmlu_pro_sampled.jsonl"},
        )
        sample = next(BenchmarkRunner(config)._load_samples(resolved, None, 1))
        prompt = augment_prompt_for_format(sample, sample.prompt)
        model = protocol.model_config
        request = GenerationRequest(
            prompt=prompt,
            temperature=model.temperature,
            max_tokens=model.max_tokens,
            stream=bool(model.supports_stream),
            messages=[{"role": "user", "content": prompt}],
            provider_kwargs={},
        )
        client = OpenAICompatibleClient(config.providers["openai_compatible"], model)
        try:
            merged = {**model.provider_kwargs, **request.provider_kwargs}
            expected = client._build_payload(
                request,
                provider_kwargs=merged,
                use_responses_api=client._use_responses_api(merged),
            )
        finally:
            _close_client(client)
        actual = preview_standard_payload(config, model, prompt)
        self.assertEqual(actual, expected)
        self.assertEqual(actual["model"], "openai/gpt-5.4")
        self.assertEqual(actual["temperature"], 0)
        self.assertEqual(actual["max_tokens"], 4096)
        self.assertEqual(actual["stream"], False)
        self.assertEqual(actual["messages"], [{"role": "user", "content": prompt}])
        self.assertEqual(prompt, augment_prompt_for_format(sample, sample.prompt))
        self.assertNotIn("/", protocol.model_config.model_id)

    def test_cost_uses_runner_sample_order_and_token_floor(self):
        plan = build_plan(
            self.catalog,
            openrouter_model="openai/gpt-5.4",
            modules={"通用模型"},
            source_files={"mmlu_pro_sampled.jsonl", "ceval_sampled.jsonl"},
            dimensions=set(),
            max_samples=2,
            prompt_usd_per_token=1e-6,
            completion_usd_per_token=2e-6,
            max_concurrency=8,
        )
        config = self.catalog.config
        resolved = FileRegistry(config).resolve(
            modules={"通用模型"},
            source_files={"mmlu_pro_sampled.jsonl", "ceval_sampled.jsonl"},
        )
        samples = list(BenchmarkRunner(config)._load_samples(resolved, None, 2))
        self.assertEqual(plan.standard_requests, len(samples))
        self.assertEqual([name for name, _ in plan.file_counts], [samples[0].source_file])
        low = sum(max(1, len(augment_prompt_for_format(sample, sample.prompt)) // 4) for sample in samples)
        high = sum(max(1, len(augment_prompt_for_format(sample, sample.prompt))) for sample in samples)
        self.assertEqual(plan.standard_low_input_tokens, low)
        self.assertEqual(plan.standard_high_input_tokens, high)
        self.assertEqual(plan.standard_output_cap_tokens, 2 * 4096)
        self.assertAlmostEqual(plan.low_usd, low * 1e-6)
        self.assertAlmostEqual(plan.high_usd, high * 1e-6 + (2 * 4096) * 2e-6)
        codes = {caveat.code for caveat in plan.caveats}
        self.assertIn("not_full_benchmark", codes)
        self.assertIn("max_samples_prefix", codes)
        self.assertNotIn("preflight_judge_model_key_missing", codes)
        self.assertFalse(plan.blocked)
        self.assertIn("Cost range", render_plan(plan))

    def test_llm_judge_tasks_stay_blocked_without_the_official_judge_key(self):
        plan = build_plan(
            self.catalog,
            openrouter_model="acme/unknown-model",
            modules={"安全可信"},
            source_files={"安全拒答.jsonl"},
            dimensions=set(),
            max_samples=1,
            prompt_usd_per_token=0.0,
            completion_usd_per_token=0.0,
            max_concurrency=8,
        )
        self.assertTrue(plan.blocked)
        self.assertTrue(plan.judge_calls >= 1)
        codes = {caveat.code for caveat in plan.caveats}
        self.assertIn("preflight_judge_model_key_missing", codes)
        self.assertIn("judge_calls_not_on_visitor_key", codes)

    def test_rule_scored_highlevel_edu_is_still_held_by_preflight(self):
        plan = build_plan(
            self.catalog,
            openrouter_model="acme/unknown-model",
            modules={"高阶育人"},
            source_files={"高阶育人-edu.jsonl"},
            dimensions=set(),
            max_samples=1,
            prompt_usd_per_token=0.0,
            completion_usd_per_token=0.0,
            max_concurrency=8,
        )
        self.assertEqual(plan.judge_calls, 0)
        self.assertTrue(plan.blocked)
        codes = {caveat.code for caveat in plan.caveats}
        self.assertIn("rule_task_preflight_hold", codes)
        self.assertIn("preflight_judge_model_key_missing", codes)

    def test_free_model_suffix_states_the_documented_daily_cap(self):
        plan = build_plan(
            self.catalog,
            openrouter_model="acme/unknown-model:free",
            modules={"通用模型"},
            source_files={"mmlu_pro_sampled.jsonl"},
            dimensions=set(),
            max_samples=1,
            prompt_usd_per_token=0.0,
            completion_usd_per_token=0.0,
            max_concurrency=8,
        )
        self.assertIn("free_model_rate_limit", {caveat.code for caveat in plan.caveats})
        self.assertFalse(plan.blocked)

    def test_empty_file_selection_does_not_mean_every_file(self):
        plan = build_plan(
            self.catalog,
            openrouter_model="acme/unknown-model",
            modules={"通用模型"},
            source_files=set(),
            dimensions=set(),
            max_samples=None,
            prompt_usd_per_token=1.0,
            completion_usd_per_token=1.0,
            max_concurrency=8,
        )
        self.assertEqual(plan.standard_requests, 0)
        self.assertIn("no_files", {caveat.code for caveat in plan.caveats})
        self.assertTrue(plan.blocked)

    def test_basic_education_bound_uses_runtime_task_count_and_recursion_cap(self):
        plan = build_plan(
            self.catalog,
            openrouter_model="acme/unknown-model",
            modules={"基本教育"},
            source_files={"config_guided_task.yaml"},
            dimensions=set(),
            max_samples=None,
            prompt_usd_per_token=1e-6,
            completion_usd_per_token=2e-6,
            max_concurrency=8,
        )
        guided = self.catalog.by_name()["config_guided_task.yaml"]
        self.assertTrue(guided.multi_turn)
        self.assertEqual(plan.basic_tasks, guided.item_count)
        self.assertEqual(plan.basic_target_calls_low, guided.item_count)
        self.assertEqual(plan.basic_target_calls_high, guided.item_count * 20)
        codes = {caveat.code for caveat in plan.caveats}
        self.assertIn("basic_education_call_bound", codes)
        self.assertIn("basic_education_key_on_disk", codes)
        self.assertIn("basic_education_score_scale", codes)

    def test_default_catalog_files_are_rule_scored(self):
        defaults = self.catalog.default_files()
        self.assertTrue(defaults)
        for entry in defaults:
            self.assertEqual(entry.scoring_mode, "rule")
            self.assertFalse(entry.preflight_llm_task)
            self.assertNotEqual(entry.module, "基本教育")

    def test_execute_refuses_a_blocked_plan(self):
        plan = build_plan(
            self.catalog,
            openrouter_model="acme/unknown-model",
            modules=set(),
            source_files=set(),
            dimensions=set(),
            max_samples=None,
            prompt_usd_per_token=1.0,
            completion_usd_per_token=1.0,
            max_concurrency=8,
        )
        with self.assertRaises(PreflightError):
            execute_run(self.catalog, plan, max_concurrency=8, accept_unpriced=True)

    def test_scrub_removes_the_visitor_key_from_run_files(self):
        secret = "sk-or-test-key-not-a-secret"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "nested" / "basic_education_runtime.yaml"
            target.parent.mkdir()
            target.write_text(f"api_key: {secret}\n", encoding="utf-8")
            scrub_tree(root, secret)
            self.assertNotIn(secret, target.read_text(encoding="utf-8"))
            self.assertIn("[redacted]", target.read_text(encoding="utf-8"))

    def test_visitor_judge_endpoint_unblocks_llm_judge_tasks_on_single_key(self):
        plan = build_plan(
            self.catalog,
            openrouter_model="openai/gpt-5.4",
            modules={"安全可信"},
            source_files={"安全拒答.jsonl"},
            dimensions=set(),
            max_samples=1,
            prompt_usd_per_token=1e-6,
            completion_usd_per_token=2e-6,
            max_concurrency=8,
            judge_openrouter_model="qwen/qwen3-235b-a22b-2507",
            use_system_one_judge=False,
            judge_prompt_usd_per_token=0.5e-6,
            judge_completion_usd_per_token=1.0e-6,
        )
        self.assertFalse(plan.blocked)
        self.assertTrue(plan.preflight_ok)
        self.assertEqual(plan.judge_calls, 1)
        codes = {caveat.code for caveat in plan.caveats}
        self.assertIn("judge_on_visitor_key", codes)
        self.assertNotIn("preflight_judge_model_key_missing", codes)

    def test_system_one_judge_routes_to_system_one_judge_and_requires_multiturn_chat_for_guided_teaching(self):
        from elbench.execution.basic_education import BasicEducationExecutor
        from elbench.judges.router import JudgeRouter
        from elbench.judges.system_one_judge import SystemOneJudge
        from elbench.schemas.evaluation import Sample
        import logging

        plan = build_plan(
            self.catalog,
            openrouter_model="openai/gpt-5.4",
            modules={"安全可信"},
            source_files={"安全拒答.jsonl"},
            dimensions=set(),
            max_samples=1,
            prompt_usd_per_token=1e-6,
            completion_usd_per_token=2e-6,
            max_concurrency=8,
            judge_openrouter_model="liquid/d1-ultra",
            use_system_one_judge=True,
            judge_prompt_usd_per_token=0.042e-6,
            judge_completion_usd_per_token=0.0,
        )
        self.assertFalse(plan.blocked)
        self.assertTrue(plan.use_system_one_judge)
        self.assertIsNotNone(plan.judge_model_config)
        config = self.catalog.config.model_copy(deep=True)
        config.models[plan.judge_model_config.model_id] = plan.judge_model_config
        config.judges.default_judge_model_id = plan.judge_model_config.model_id
        router = JudgeRouter(config)
        sample = Sample(
            sample_id="s1",
            source_file="安全拒答.jsonl",
            source_path="安全拒答.jsonl",
            module="安全可信",
            subset="通用-应拒答",
            task="safety_refusal",
            prompt="test",
        )
        judge = router.get_judge(sample)
        self.assertIsInstance(judge, SystemOneJudge)
        import asyncio
        asyncio.run(router.aclose())

        # Guided multi-turn teaching with System 1 judge requires multi-turn chat endpoint
        blocked_guided = build_plan(
            self.catalog,
            openrouter_model="openai/gpt-5.4",
            modules={"基本教育"},
            source_files={"config_guided_task.yaml"},
            dimensions=set(),
            max_samples=1,
            prompt_usd_per_token=1e-6,
            completion_usd_per_token=2e-6,
            max_concurrency=8,
            judge_openrouter_model="liquid/d1-ultra",
            use_system_one_judge=True,
            use_multiturn_chat_model=False,
        )
        self.assertTrue(blocked_guided.blocked)
        self.assertIn(
            "multiturn_endpoint_required_for_system_one",
            {c.code for c in blocked_guided.caveats},
        )

        # Opting into multi-turn chat endpoint unblocks guided teaching and splits teacher, stu, and eval
        unblocked_guided = build_plan(
            self.catalog,
            openrouter_model="openai/gpt-5.4",
            modules={"基本教育"},
            source_files={"config_guided_task.yaml"},
            dimensions=set(),
            max_samples=1,
            prompt_usd_per_token=1e-6,
            completion_usd_per_token=2e-6,
            max_concurrency=8,
            judge_openrouter_model="liquid/d1-ultra",
            use_system_one_judge=True,
            use_multiturn_chat_model=True,
            multiturn_openrouter_model="liquid/lfm-2.5-24b-a2b",
        )
        self.assertFalse(unblocked_guided.blocked)
        self.assertIn("multiturn_chat_on_visitor_key", {c.code for c in unblocked_guided.caveats})

        run_cfg = self.catalog.config.model_copy(deep=True)
        run_cfg.models[unblocked_guided.protocol.model_config.model_id] = unblocked_guided.protocol.model_config
        run_cfg.models[unblocked_guided.judge_model_config.model_id] = unblocked_guided.judge_model_config
        run_cfg.models[unblocked_guided.multiturn_model_config.model_id] = unblocked_guided.multiturn_model_config
        run_cfg.judges.default_judge_model_id = unblocked_guided.judge_model_config.model_id
        run_cfg.judges.basic_education_evaluator_model_id = unblocked_guided.judge_model_config.model_id
        run_cfg.judges.basic_education_test_endpoint_model_id = unblocked_guided.multiturn_model_config.model_id

        executor = BasicEducationExecutor(
            project_config=run_cfg,
            model_config=unblocked_guided.protocol.model_config,
            logger=logging.getLogger("test"),
        )
        scenario = next(s for s in executor.config.scenarios if s.scenario_id == "guided_problem_solving_teaching")
        template = executor._read_yaml_file(scenario.template_path)
        rendered = executor._render_runtime_template(
            template_data=template,
            scenario=scenario,
            memory_path=ROOT / "tmp_test_artifacts" / "guided_mem",
            max_samples=1,
        )
        self.assertEqual(rendered["models"]["teacher"]["model"], "openai/gpt-5.4")
        self.assertEqual(rendered["models"]["stu"]["model"], "liquid/lfm-2.5-24b-a2b")
        self.assertEqual(rendered["models"]["stu"]["type"], "openai")
        self.assertEqual(rendered["models"]["eval"]["model"], "liquid/d1-ultra")
        self.assertEqual(rendered["models"]["eval"]["type"], "system_one")

    def test_all_ui_labels_and_rendered_plans_are_english_only(self):
        import re
        from elbench.dashboard.english import english_dimension, english_file, english_module, english_subset
        from elbench.dashboard.run import summary_rows

        cjk_re = re.compile(r"[\u4e00-\u9fff]")
        for m in self.catalog.modules():
            self.assertIsNone(cjk_re.search(english_module(m)), f"Non-English module label: {english_module(m)}")
        for entry in self.catalog.files:
            self.assertIsNone(cjk_re.search(entry.label), f"Non-English file label: {entry.label}")
            self.assertIsNone(cjk_re.search(english_file(entry.canonical_name)))
            self.assertIsNone(cjk_re.search(english_subset(entry.subset)))
            for dim in entry.dimensions:
                self.assertIsNone(
                    cjk_re.search(english_dimension(dim)),
                    f"Non-English dimension label: {english_dimension(dim)}",
                )

        # Even when preflight blocks for a missing judge key, rendered plan text must be 100% English
        blocked_plan = build_plan(
            self.catalog,
            openrouter_model="openai/gpt-5.4",
            modules={"安全可信", "高阶育人"},
            source_files={"安全拒答.jsonl", "高阶育人-edu.jsonl"},
            dimensions=set(),
            max_samples=2,
            prompt_usd_per_token=1e-6,
            completion_usd_per_token=2e-6,
            max_concurrency=8,
        )
        rendered_md = render_plan(blocked_plan)
        self.assertIsNone(cjk_re.search(rendered_md), f"Found non-English text in render_plan:\n{rendered_md}")

        rows = summary_rows(
            {
                "by_module": {"安全可信": {"count": 2, "avg_score": 1.0, "pass_rate": 1.0}},
                "by_subset": {"安全可信 | 通用-应拒答": {"count": 2, "avg_score": 1.0, "pass_rate": 1.0}},
                "by_source_file": {"安全拒答.jsonl": {"count": 2, "avg_score": 1.0, "pass_rate": 1.0}},
                "by_dimension": {"批判性思维": {"count": 2, "avg_score": 1.0, "pass_rate": 1.0}},
            }
        )
        for row in rows:
            for cell in row:
                if isinstance(cell, str):
                    self.assertIsNone(cjk_re.search(cell), f"Non-English summary cell: {cell}")

    def test_persistent_leaderboard_comparison_and_checkpoint_recovery(self):
        from elbench.dashboard.leaderboard import (
            LeaderboardStore,
            build_comparison_table,
            compute_normalized_scores,
            official_leaderboard_rows,
        )

        off_rows = official_leaderboard_rows()
        self.assertEqual(len(off_rows), 9)
        self.assertEqual(off_rows[0][1], "DeepSeek-V4-Flash")

        with tempfile.TemporaryDirectory() as tmpdir:
            store = LeaderboardStore(Path(tmpdir), dataset_repo_id="", hf_token="")
            summary_payload = {
                "total_judged": 10,
                "total_failures": 2,
                "by_source_file": {
                    "mmlu_pro_sampled.jsonl": {"count": 5, "avg_score": 0.8, "pass_rate": 0.8},
                    "knowledge.yaml": {"count": 5, "avg_score": 7.5, "pass_rate": 1.0},
                },
            }
            norm = compute_normalized_scores(summary_payload)
            self.assertEqual(norm["general_pct"], 80.0)
            self.assertEqual(norm["basic_education_pct"], 75.0)
            self.assertEqual(norm["overall_pct"], 77.5)

            saved = store.save_run(
                {
                    "run_id": "live-test-run-1",
                    "config_hash": "abc1234567890def",
                    "target_model": "openai/gpt-5.4",
                    "judge_model": "liquid/d1-ultra",
                    "use_system_one_judge": True,
                    "status": "partial",
                    "is_full_benchmark": False,
                    "total_judged": 10,
                    "total_failures": 2,
                    "normalized_scores": norm,
                },
                push_to_hub=False,
            )
            self.assertEqual(saved["run_id"], "live-test-run-1")

            # Resumable run lookup finds partial/failed run by config_hash
            resumable = store.find_resumable_run(config_hash="abc1234567890def")
            self.assertIsNotNone(resumable)
            self.assertEqual(resumable["run_id"], "live-test-run-1")

            # Compare custom run against official GPT-5.4 and DeepSeek-V4-Flash
            runs = store.load_runs()
            headers, comp_rows, note = build_comparison_table(
                "run:live-test-run-1",
                ["official:gpt-5.4", "official:deepseek-v4-flash"],
                "By Source File / Task (Matched Subset)",
                runs,
            )
            self.assertEqual(len(headers), 7)
            self.assertTrue(len(comp_rows) >= 3)
            self.assertIn("Matched Subset Weighted Average", str(comp_rows[0][0]))
            self.assertIn("2", note)

    def test_pareto_frontier_chart_and_table(self):
        from elbench.dashboard.leaderboard import build_pareto_frontier

        # Basic Education has a 3-step Pareto frontier across Official Models:
        # DeepSeek-V4-Flash (84.9%) -> DeepSeek-V4-Pro (88.5%) -> Gemini-3.5-Flash (94.6%)
        fig, rows, md = build_pareto_frontier(
            runs=[],
            prices=None,
            score_metric="basic_education",
            cost_metric="blended_per_1m",
            include_custom_runs=True,
        )
        traces = {trace.name: trace for trace in fig.data}
        self.assertEqual(set(traces), {"Best value line", "Published result"})
        self.assertEqual(fig.layout.xaxis.type, "log")
        self.assertEqual(fig.layout.title.text, "ELBench Teaching: score vs. price")
        published = traces["Published result"]
        self.assertIsNotNone(published.error_y.array)
        # Best-value points are always named on the chart.
        named = {text for text in published.text if text}
        self.assertTrue({"DeepSeek-V4-Flash", "DeepSeek-V4-Pro", "Gemini-3.5-Flash"} <= named)
        pareto_models = [row[1] for row in rows if row[0] == "★ Best value"]
        self.assertEqual(
            pareto_models,
            ["DeepSeek-V4-Flash", "DeepSeek-V4-Pro", "Gemini-3.5-Flash"],
        )
        self.assertIn("DeepSeek-V4-Flash", md)
        self.assertIn("Gemini-3.5-Flash", md)

        # Adding a high-scoring custom run at a low price places the custom run on the Pareto frontier
        from elbench.dashboard.leaderboard import BASIC_FILES, full_file_counts

        full = full_file_counts()
        custom_runs = [
            {
                "run_id": "live-pareto-1",
                "target_model": "qwen/qwen3-235b-a22b-2507",
                "is_full_benchmark": True,
                "total_judged": 2939,
                "prompt_usd_per_token": 0.05e-6,
                "completion_usd_per_token": 0.10e-6,
                "normalized_scores": {
                    "overall_pct": 90.0,
                    "basic_education_pct": 91.0,
                    "per_file": {name: {"count": full[name], "normalized_pct": 91.0} for name in BASIC_FILES},
                },
            }
        ]
        fig2, rows2, md2 = build_pareto_frontier(
            runs=custom_runs,
            prices=None,
            score_metric="basic_education",
            cost_metric="blended_per_1m",
            include_custom_runs=True,
        )
        self.assertIsNotNone(fig2)
        pareto_models_2 = [row[1] for row in rows2 if row[0] == "★ Best value"]
        self.assertTrue(any("qwen/qwen3-235b-a22b-2507" in m for m in pareto_models_2))
        self.assertIn("Gemini-3.5-Flash", pareto_models_2)

    def test_official_per_file_data_reproduces_the_published_table(self):
        from elbench.dashboard.leaderboard import load_official_models, module_scores_from_per_file

        for model in load_official_models():
            scores = module_scores_from_per_file(model["per_file"])
            for key in ("safety", "basic_education", "high_level", "overall"):
                self.assertAlmostEqual(
                    scores[f"{key}_pct"], model[key], delta=0.1, msg=f"{model['display_name']} {key}"
                )
            # The paper's General score also counts 3 GSM8K items this repository no longer ships.
            self.assertAlmostEqual(scores["general_pct"], model["general"], delta=0.35, msg=model["display_name"])
            for name, stats in model["per_file"].items():
                self.assertLessEqual(stats["normalized_pct"], 100.0, msg=f"{model['display_name']} {name}")

    def test_scoring_follows_the_paper_per_module(self):
        from elbench.dashboard.leaderboard import compute_normalized_scores

        norm = compute_normalized_scores(
            {
                "by_source_file": {
                    # Safety: mean score, so partial credit counts.
                    "SATAs.xlsx": {"count": 150, "avg_score": 0.6, "pass_rate": 0.5},
                    "安全拒答.jsonl": {"count": 250, "avg_score": 0.9, "pass_rate": 0.8},
                    # Basic Education: plain mean of scenarios, not weighted by item count.
                    "knowledge.yaml": {"count": 10, "avg_score": 8.0, "pass_rate": 1.0},
                    "config_guided_task.yaml": {"count": 15, "avg_score": 5.0, "pass_rate": 1.0},
                    # High-Level: pass rate.
                    "高阶育人-omni.jsonl": {"count": 500, "avg_score": 0.7, "pass_rate": 0.5},
                }
            }
        )
        self.assertEqual(norm["safety_pct"], round((150 * 60 + 250 * 90) / 400, 2))
        self.assertEqual(norm["basic_education_pct"], 90.0)
        self.assertEqual(norm["high_level_pct"], 50.0)

    def test_basic_education_mean_skips_missing_and_out_of_range_scores(self):
        import json as _json
        from elbench.dashboard.leaderboard import basic_education_valid_means, compute_normalized_scores

        with tempfile.TemporaryDirectory() as tmpdir:
            judged = Path(tmpdir) / "judged.jsonl"
            records = [{"source_file": "question.yaml", "score": 4.0}] * 9
            records += [{"source_file": "question.yaml", "score": 74.07}, {"source_file": "question.yaml", "score": None}]
            judged.write_text("\n".join(_json.dumps(r) for r in records), encoding="utf-8")
            means = basic_education_valid_means(judged)
        self.assertEqual(means, {"question.yaml": 4.0})
        norm = compute_normalized_scores(
            {"by_source_file": {"question.yaml": {"count": 11, "avg_score": 10.37, "pass_rate": 1.0}}},
            basic_valid_means=means,
        )
        self.assertEqual(norm["basic_education_pct"], 80.0)

    def test_price_and_size_filter_hides_models_from_chart_and_best_value_line(self):
        from elbench.dashboard.leaderboard import build_pareto_frontier

        _, all_rows, _ = build_pareto_frontier([], None, score_metric="basic_education")
        _, rows, md = build_pareto_frontier(
            [], None, score_metric="basic_education", keep=lambda c: c["name"] != "DeepSeek-V4-Flash"
        )
        names = [row[1] for row in rows]
        self.assertNotIn("DeepSeek-V4-Flash", names)
        self.assertEqual(len(rows), len(all_rows) - 1)
        # With the cheapest model hidden, the next one becomes the cheapest best-value model.
        self.assertEqual(rows[0][1], "DeepSeek-V4-Pro")
        self.assertEqual(rows[0][0], "★ Best value")
        self.assertIn("1 model hidden by your price or size filter", md)

    def test_model_sizes_from_cache_and_name(self):
        from elbench.dashboard.model_sizes import load_sizes, model_size, size_from_name

        self.assertEqual(size_from_name("qwen/qwen3-235b-a22b-2507").label(), "235B (22B active)")
        self.assertEqual(size_from_name("liquid/lfm-2.5-24b-a2b").label(), "24B (2B active)")
        self.assertEqual(size_from_name("meta-llama/llama-3.1-8b-instruct").total_b, 8.0)
        self.assertIsNone(size_from_name("mistralai/mixtral-8x7b-instruct"))
        self.assertIsNone(size_from_name("openai/gpt-5.4"))
        sizes = load_sizes()
        self.assertGreater(len(sizes), 100)
        # The cached Hugging Face total wins over the name; the name still gives the active size.
        qwen = model_size("qwen/qwen3-235b-a22b-2507", "Qwen/Qwen3-235B-A22B-Instruct-2507", sizes)
        self.assertAlmostEqual(qwen.total_b, 235.09, places=1)
        self.assertEqual(qwen.active_b, 22.0)

    def test_partial_run_is_not_plotted_against_full_runs(self):
        from elbench.dashboard.leaderboard import build_pareto_frontier, full_file_counts

        smoke = {
            "run_id": "smoke-abc",
            "target_model": "qwen/qwen3-235b-a22b-2507",
            "total_judged": 5,
            "normalized_scores": {
                "overall_pct": 100.0,
                "general_pct": 100.0,
                "per_file": {"mmlu_pro_sampled.jsonl": {"count": 5, "normalized_pct": 100.0}},
            },
        }
        prices = {"qwen/qwen3-235b-a22b-2507": (9e-8, 5.5e-7)}
        for metric in ("overall", "general", "file:mmlu_pro_sampled.jsonl"):
            fig, rows, md = build_pareto_frontier([smoke], prices, score_metric=metric)
            self.assertFalse(any("smoke-abc" in str(row[1]) for row in rows), metric)
            self.assertIn("1 visitor run is not shown", md)

        full_mmlu = dict(smoke, run_id="mmlu-full")
        full_mmlu["normalized_scores"] = {
            "per_file": {"mmlu_pro_sampled.jsonl": {"count": full_file_counts()["mmlu_pro_sampled.jsonl"], "normalized_pct": 99.0}}
        }
        fig, rows, _ = build_pareto_frontier([full_mmlu], prices, score_metric="file:mmlu_pro_sampled.jsonl")
        self.assertTrue(any("mmlu-full" in str(row[1]) for row in rows))


    def test_core_scope_drops_the_china_specific_sets_and_keeps_paper_scope_reproducible(self):
        from elbench.dashboard.leaderboard import (
            CHINA_SPECIFIC_FILES,
            SCOPE_CORE,
            SCOPE_PAPER,
            load_official_models,
            module_scores_from_per_file,
            official_leaderboard_rows,
            official_scores,
            scope_item_count,
        )

        self.assertEqual(scope_item_count(SCOPE_PAPER), 2939)
        self.assertEqual(scope_item_count(SCOPE_CORE), 2939 - 250 - 208)
        for model in load_official_models():
            paper = official_scores(model, SCOPE_PAPER)
            self.assertEqual(paper["overall"], model["overall"])
            core = official_scores(model, SCOPE_CORE)
            kept = {fn: stats for fn, stats in model["per_file"].items() if fn not in CHINA_SPECIFIC_FILES}
            self.assertEqual(core["overall"], module_scores_from_per_file(kept)["overall_pct"])
            # High-Level and Basic Education hold no region-specific file, so they don't move.
            self.assertAlmostEqual(core["high_level"], model["high_level"], delta=0.1)
            self.assertAlmostEqual(core["basic_education"], model["basic_education"], delta=0.1)
        core_rows = official_leaderboard_rows(SCOPE_CORE)
        self.assertEqual([row[0] for row in core_rows], list(range(1, len(core_rows) + 1)))
        self.assertEqual(core_rows[0][1], "GPT-5.4")
        self.assertNotIn("[", core_rows[0][2])  # no published range outside the paper's scope
        self.assertIn("[", official_leaderboard_rows(SCOPE_PAPER)[0][2])

    def test_run_without_china_specific_sets_is_complete_and_plotted_in_core_scope_only(self):
        from elbench.dashboard.leaderboard import (
            CHINA_SPECIFIC_FILES,
            SCOPE_CORE,
            SCOPE_PAPER,
            build_pareto_frontier,
            full_file_counts,
            run_coverage_label,
            run_covers_metric,
        )

        counts = full_file_counts()
        run = {
            "run_id": "core-run",
            "target_model": "qwen/qwen3-235b-a22b-2507",
            "normalized_scores": {
                "per_file": {
                    fn: {"count": n, "normalized_pct": 80.0}
                    for fn, n in counts.items()
                    if fn not in CHINA_SPECIFIC_FILES
                }
            },
        }
        self.assertTrue(run_covers_metric(run, "overall", counts, scope=SCOPE_CORE))
        self.assertFalse(run_covers_metric(run, "overall", counts, scope=SCOPE_PAPER))
        self.assertEqual(run_coverage_label(run, counts), "Complete, without the region-specific sets")
        prices = {"qwen/qwen3-235b-a22b-2507": (9e-8, 5.5e-7)}
        _, rows, _ = build_pareto_frontier([run], prices, scope=SCOPE_CORE)
        self.assertTrue(any("core-run" in str(row[1]) for row in rows))
        _, rows, _ = build_pareto_frontier([run], prices, scope=SCOPE_PAPER)
        self.assertFalse(any("core-run" in str(row[1]) for row in rows))

    def test_plan_reports_core_coverage(self):
        from elbench.dashboard.leaderboard import CHINA_SPECIFIC_FILES

        core_files = {e.canonical_name for e in self.catalog.files} - set(CHINA_SPECIFIC_FILES)
        common = dict(
            openrouter_model="openai/gpt-5.4",
            modules=set(self.catalog.modules()),
            dimensions=set(),
            max_samples=None,
            prompt_usd_per_token=1e-6,
            completion_usd_per_token=2e-6,
            max_concurrency=8,
        )
        core = build_plan(self.catalog, source_files=core_files, **common)
        self.assertTrue(core.covers_core)
        self.assertFalse(core.is_full_benchmark)
        capped = build_plan(self.catalog, source_files=core_files, max_samples_per_file=1, **common)
        self.assertFalse(capped.covers_core)

    def test_quick_check_takes_one_question_from_each_task(self):
        files = {"mmlu_pro_sampled.jsonl", "ifeval_sampled.jsonl", "math_500_sampled.jsonl"}
        plan = build_plan(
            self.catalog,
            openrouter_model="openai/gpt-5.4",
            modules={"通用模型"},
            source_files=files,
            dimensions=set(),
            max_samples=None,
            max_samples_per_file=1,
            prompt_usd_per_token=1e-6,
            completion_usd_per_token=2e-6,
            max_concurrency=8,
        )
        self.assertEqual(plan.standard_requests, 3)
        self.assertEqual(sorted(name for name, _ in plan.file_counts), sorted(files))
        self.assertTrue(all(count == 1 for _, count in plan.file_counts))
        resolved = FileRegistry(self.catalog.config).resolve(modules={"通用模型"}, source_files=files)
        samples = list(BenchmarkRunner(self.catalog.config)._load_samples(resolved, None, None, 1))
        self.assertEqual(sorted(s.source_file for s in samples), sorted(files))
        teaching = build_plan(
            self.catalog,
            openrouter_model="openai/gpt-5.4",
            modules={"基本教育"},
            source_files={"knowledge.yaml", "cross.yaml"},
            dimensions=set(),
            max_samples=None,
            max_samples_per_file=1,
            prompt_usd_per_token=1e-6,
            completion_usd_per_token=2e-6,
            max_concurrency=8,
        )
        self.assertEqual(teaching.basic_tasks, 2)

    def test_visitor_judge_and_student_have_no_length_limit(self):
        from elbench.dashboard.protocol import build_judge_model_config, build_multiturn_chat_model_config
        from elbench.execution.basic_education import BasicEducationExecutor

        config = self.catalog.config
        # A listed card sets max_tokens for answers; the judge and student copies must not inherit a cap.
        judge = build_judge_model_config(config, "openai/gpt-5.4")
        student = build_multiturn_chat_model_config(config, "openai/gpt-5.4")
        self.assertIsNone(judge.max_tokens)
        self.assertIsNone(student.max_tokens)
        payload = {"kargs": {}}
        BasicEducationExecutor._tune_runtime_evaluator_payload(None, payload)
        self.assertNotIn("max_tokens", payload["kargs"])
        capped = {"kargs": {"max_tokens": 512}}
        BasicEducationExecutor._tune_runtime_evaluator_payload(None, capped)
        self.assertEqual(capped["kargs"]["max_tokens"], 1024)  # a card's own limit keeps its floor

    def test_partial_visitor_run_shows_no_headline_score(self):
        from elbench.dashboard.leaderboard import SCOPE_CORE, SCOPE_PAPER, custom_leaderboard_rows

        quick = {
            "run_id": "quick",
            "target_model": "meta/muse-spark-1.3-contributor",
            "total_judged": 5,
            "normalized_scores": {
                "overall_pct": 21.84,
                "safety_pct": 21.84,
                "per_file": {"安全拒答.jsonl": {"count": 5, "normalized_pct": 21.84}},
            },
        }
        for scope in (SCOPE_CORE, SCOPE_PAPER):
            row = custom_leaderboard_rows([quick], scope)[0]
            self.assertEqual(row[5], "Part of the benchmark")
            self.assertEqual(row[6], 5)
            self.assertEqual(row[8:], [None] * 5, scope)

    def test_route_keeps_the_highest_precision_and_sorts_by_price(self):
        from elbench.dashboard.routing import route_from_endpoints

        def host(name, quant, prompt, tier=None):
            return {
                "provider_name": name,
                "quantization": quant,
                "pricing": {"prompt": str(prompt), "completion": str(prompt * 4)},
                "service_tier": tier,
            }

        mixed = route_from_endpoints(
            "z-ai/glm-5.3-flash",
            [host("A", "fp4", 1e-8), host("B", "unknown", 2e-8), host("C", "fp8", 5e-8), host("D", "fp8", 4e-8)],
        )
        self.assertEqual(mixed.provider, {"sort": "price", "quantizations": ["fp8"]})
        self.assertEqual(mixed.cheapest.name, "D")
        self.assertEqual(sorted(h.name for h in mixed.hosts), ["C", "D"])
        half = route_from_endpoints("m", [host("A", "fp16", 3e-8), host("B", "bf16", 2e-8), host("C", "fp8", 1e-8)])
        self.assertEqual(half.provider["quantizations"], ["bf16", "fp16"])
        # Closed models report no precision: price order only, and opt-in priority tiers are never counted.
        closed = route_from_endpoints(
            "openai/gpt-6-luna", [host("OpenAI", "unknown", 1e-7), host("OpenAI", "unknown", 2e-8, "priority")]
        )
        self.assertEqual(closed.provider, {"sort": "price"})
        self.assertEqual(closed.cheapest.prompt_usd_per_token, 1e-7)
        self.assertNotIn("sort", route_from_endpoints("m:nitro", [host("A", "fp8", 1e-8)]).provider)

    def test_plan_sends_the_route_with_every_chat_model(self):
        from elbench.dashboard.routing import route_from_endpoints
        from elbench.execution.basic_education import BasicEducationExecutor

        fp8 = [{"provider_name": "A", "quantization": "fp8", "pricing": {"prompt": "1e-8", "completion": "4e-8"}}]
        routes = {role: route_from_endpoints(f"{role}/model", fp8) for role in ("target", "judge", "student")}
        plan = build_plan(
            self.catalog,
            openrouter_model="openai/gpt-5.4",
            modules={"基本教育"},
            source_files={"config_guided_task.yaml"},
            dimensions=set(),
            max_samples=None,
            max_samples_per_file=1,
            prompt_usd_per_token=1e-6,
            completion_usd_per_token=2e-6,
            max_concurrency=8,
            judge_openrouter_model="z-ai/glm-5.3-flash",
            use_multiturn_chat_model=True,
            multiturn_openrouter_model="liquid/lfm-2.5-24b-a2b",
            routes=routes,
        )
        expected = {"sort": "price", "quantizations": ["fp8"]}
        self.assertEqual(set(plan.routes), {"target", "judge", "student"})
        body = preview_standard_payload(self.catalog.config, plan.protocol.model_config, "hi")
        self.assertEqual(body["provider"], expected)
        judge_body = preview_standard_payload(self.catalog.config, plan.judge_model_config, "hi")
        self.assertEqual(judge_body["provider"], expected)
        # The teaching runtime calls LangChain's OpenAI client, which needs the field inside extra_body.
        executor = BasicEducationExecutor.__new__(BasicEducationExecutor)
        runtime = executor._build_runtime_model_payload(plan.multiturn_model_config)
        self.assertEqual(runtime["kargs"]["extra_body"]["provider"], expected)
        self.assertNotIn("provider", runtime["kargs"])
        self.assertTrue(any(c.code == "openrouter_host_choice" for c in plan.caveats))

        decision = build_plan(
            self.catalog,
            openrouter_model="openai/gpt-5.4",
            modules={"安全可信"},
            source_files={"安全拒答.jsonl"},
            dimensions=set(),
            max_samples=1,
            prompt_usd_per_token=1e-6,
            completion_usd_per_token=2e-6,
            max_concurrency=8,
            judge_openrouter_model="liquid/d1-ultra",
            use_system_one_judge=True,
            routes=routes,
        )
        self.assertNotIn("judge", decision.routes)
        self.assertNotIn("provider", decision.judge_model_config.provider_kwargs)

    def test_each_judge_has_its_own_board(self):
        from elbench.dashboard.leaderboard import (
            CHINA_SPECIFIC_FILES,
            PAPER_JUDGE_SLUG,
            PUBLISHED_SOURCE,
            SCOPE_CORE,
            build_comparison_table,
            build_pareto_frontier,
            full_file_counts,
            judge_board_choices,
        )

        counts = full_file_counts()
        per_file = {fn: {"count": n, "normalized_pct": 80.0} for fn, n in counts.items() if fn not in CHINA_SPECIFIC_FILES}

        def run(rid, judge, **extra):
            return {
                "run_id": rid,
                "target_model": "openai/gpt-6-luna",
                "judge_model": judge,
                "normalized_scores": {"per_file": per_file},
                **extra,
            }

        default = run("default-run", PAPER_JUDGE_SLUG)
        glm = run("glm-run", "z-ai/glm-5.3-flash")
        no_judge = run("auto-run", "z-ai/glm-5.3-flash", judge_used=False)
        decision = run("d1-run", "liquid/d1-ultra", use_system_one_judge=True)
        runs = [default, glm, no_judge, decision]
        prices = {"openai/gpt-6-luna": (1e-7, 5e-7)}

        def names(board):
            _, rows, md = build_pareto_frontier(runs, prices, scope=SCOPE_CORE, judge=board)
            return {str(row[1]).split("(")[-1].rstrip(")") for row in rows if row[2] != PUBLISHED_SOURCE}, rows, md

        shown, rows, _ = names(PAPER_JUDGE_SLUG)
        self.assertEqual(shown, {"default-run", "auto-run"})
        self.assertTrue(any(row[2] == PUBLISHED_SOURCE for row in rows))
        shown, rows, md = names("z-ai/glm-5.3-flash")
        self.assertEqual(shown, {"glm-run", "auto-run"})
        self.assertFalse(any(row[2] == PUBLISHED_SOURCE for row in rows))
        self.assertIn("Graded by z-ai/glm-5.3-flash", md)
        shown, _, _ = names("decision:liquid/d1-ultra")
        self.assertEqual(shown, {"d1-run", "auto-run"})

        keys = [key for _, key in judge_board_choices(runs)]
        self.assertEqual(keys, [PAPER_JUDGE_SLUG, "decision:liquid/d1-ultra", "z-ai/glm-5.3-flash"])
        _, _, note = build_comparison_table("run:glm-run", ["official:gpt-5.4"], "By Module (0–100 Normalized)", runs)
        self.assertIn("graded by different judges", note)
        _, _, note = build_comparison_table("run:default-run", ["official:gpt-5.4"], "By Module (0–100 Normalized)", runs)
        self.assertNotIn("different judges", note)


def _close_client(client: OpenAICompatibleClient) -> None:
    import asyncio

    asyncio.run(client.aclose())


if __name__ == "__main__":
    unittest.main()
