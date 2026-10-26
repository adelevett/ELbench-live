import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import httpx

from elbench.dashboard.catalog import load_catalog
from elbench.dashboard.leaderboard import LeaderboardStore
from elbench.dashboard.plan import build_plan
from elbench.dashboard.protocol import VISITOR_KEY_ENV
from elbench.dashboard.run import build_spend_limits, execute_run
from elbench.execution.retry import is_retryable_exception
from elbench.execution.runner import _is_quota_exhausted_error
from elbench.utils import spend
from elbench.utils.spend import BudgetExceeded, SpendLedger, SpendLimits, spend_limits, usage_cost

MODULE = "通用模型"
MMLU = "mmlu_pro_sampled.jsonl"


class SpendLedgerTest(unittest.TestCase):
    def test_total_and_per_role_limits_stop_before_the_next_request(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            limits = SpendLimits(total_usd=1.0, per_role_usd={"judge": 0.1})
            ledger = SpendLedger(Path(tmpdir) / "spend.jsonl", limits)
            ledger.record("judge", {"cost": 0.1})
            ledger.check("target")
            with self.assertRaises(BudgetExceeded) as judge_stop:
                ledger.check("judge")
            self.assertIn("judge", str(judge_stop.exception))
            ledger.record("target", {"cost": 0.95})
            with self.assertRaises(BudgetExceeded):
                ledger.check("target")

    def test_budget_stop_uses_the_runner_quota_path_and_is_not_retried(self):
        exc = BudgetExceeded("Spending limit reached (quota exceeded): $1 spent of the $1 total limit.")
        self.assertTrue(_is_quota_exhausted_error(exc))
        self.assertFalse(is_retryable_exception(exc, {429, 500}))

    def test_cost_prefers_openrouter_cost_then_tokens_times_price(self):
        self.assertEqual(usage_cost({"cost": 0.002, "prompt_tokens": 10**6}, (1.0, 1.0)), 0.002)
        self.assertAlmostEqual(usage_cost({"prompt_tokens": 1000, "completion_tokens": 100}, (1e-6, 2e-6)), 0.0012)
        self.assertAlmostEqual(usage_cost({"input_tokens": 1000, "output_tokens": 100}, (1e-6, 2e-6)), 0.0012)
        self.assertEqual(usage_cost(None, (1e-6, 2e-6)), 0.0)

    def test_subprocess_sees_the_same_ledger_through_the_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "spend.jsonl"
            with spend_limits(SpendLimits(total_usd=0.5), path):
                self.assertEqual(os.environ[spend.LEDGER_ENV], str(path))
                # What the runtime subprocess does: rebuild from the environment, record, check.
                child = SpendLedger(Path(os.environ[spend.LEDGER_ENV]), SpendLimits.from_json(os.environ[spend.LIMITS_ENV]))
                child.record("student", {"cost": 0.5})
                with self.assertRaises(BudgetExceeded):
                    spend.check("target")
            self.assertNotIn(spend.LEDGER_ENV, os.environ)
            self.assertIsNone(spend.active_ledger())

    def test_checks_are_inert_without_installed_limits(self):
        self.assertIsNone(spend.active_ledger())
        spend.check("target")
        spend.record("target", {"cost": 99})


class BudgetStopsARunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(ROOT / "configs")

    def setUp(self):
        self._previous_key = os.environ.get(VISITOR_KEY_ENV)
        os.environ[VISITOR_KEY_ENV] = "sk-or-test-key-not-a-secret"
        self.requests = 0

    def tearDown(self):
        if self._previous_key is None:
            os.environ.pop(VISITOR_KEY_ENV, None)
        else:
            os.environ[VISITOR_KEY_ENV] = self._previous_key

    def _plan(self, max_samples):
        return build_plan(
            self.catalog,
            openrouter_model="openai/gpt-5.4",
            modules={MODULE},
            source_files={MMLU},
            dimensions=set(),
            max_samples=max_samples,
            prompt_usd_per_token=1e-6,
            completion_usd_per_token=2e-6,
            max_concurrency=1,
            judge_openrouter_model="qwen/qwen3-235b-a22b-2507",
        )

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        body = {
            "choices": [{"message": {"role": "assistant", "content": "Reasoning.\nANSWER: A"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "cost": 0.01},
        }
        return httpx.Response(200, json=body)

    def _execute(self, plan, project_root, store, budget):
        real_client = httpx.AsyncClient

        def fake_client(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(self._handler)
            return real_client(*args, **kwargs)

        config = self.catalog.config.model_copy(deep=True)
        config.project_root = project_root
        catalog = type(self.catalog)(config=config, files=self.catalog.files)
        with mock.patch("elbench.providers.openai_compatible.httpx.AsyncClient", fake_client):
            return execute_run(
                catalog,
                plan,
                max_concurrency=1,
                accept_unpriced=False,
                run_id="live-budget-test",
                resume_previous=True,
                leaderboard_store=store,
                persist_to_leaderboard=True,
                total_budget_usd=budget,
            )

    def test_run_stops_at_the_limit_keeps_progress_and_resumes(self):
        plan = self._plan(max_samples=4)
        self.assertEqual(plan.standard_requests, 4)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = LeaderboardStore(root, dataset_repo_id="", hf_token="")
            first = self._execute(plan, root, store, budget=0.02)
            self.assertEqual(self.requests, 2)
            self.assertEqual(first["status"], "partial")
            self.assertTrue(first["stopped_by_budget"])
            self.assertAlmostEqual(first["spent_usd"], 0.02)
            self.assertEqual(first["summary"]["total_judged"], 2)
            saved = store.load_runs()[0]
            self.assertTrue(saved["stopped_by_budget"])
            self.assertEqual(saved["spent_usd_by_role"], {"target": 0.02})

            # Raising the limit resumes the same run and only sends the remaining items.
            second = self._execute(plan, root, store, budget=0.10)
            self.assertEqual(self.requests, 4)
            self.assertEqual(second["status"], "completed")
            self.assertFalse(second["stopped_by_budget"])
            self.assertEqual(second["summary"]["total_judged"], 4)
            self.assertAlmostEqual(second["spent_usd"], 0.04)

    def test_roles_follow_model_ids_even_when_judge_and_target_share_a_slug(self):
        plan = build_plan(
            self.catalog,
            openrouter_model="qwen/qwen3-235b-a22b-2507",
            modules={MODULE},
            source_files={MMLU},
            dimensions=set(),
            max_samples=1,
            prompt_usd_per_token=1e-7,
            completion_usd_per_token=5e-7,
            max_concurrency=1,
            judge_openrouter_model="qwen/qwen3-235b-a22b-2507",
            judge_prompt_usd_per_token=1e-7,
            judge_completion_usd_per_token=5e-7,
        )
        limits = build_spend_limits(plan, 1.0, {"judge": 0.5, "student": None})
        self.assertEqual(
            limits.roles_by_model_id,
            {plan.protocol.model_config.model_id: "target", plan.judge_model_config.model_id: "judge"},
        )
        self.assertEqual(limits.per_role_usd, {"judge": 0.5})
        self.assertEqual(limits.prices["student"], limits.prices["judge"])
        self.assertEqual(json.loads(limits.to_json())["total_usd"], 1.0)


if __name__ == "__main__":
    unittest.main()
