import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
except ImportError:  # the basic-education extra is optional
    GenericFakeChatModel = None

from elbench.utils.spend import BudgetExceeded, SpendLimits, spend_limits


@unittest.skipIf(GenericFakeChatModel is None, "langchain-core is not installed")
class RuntimeSpendCallbackTest(unittest.TestCase):
    def _model(self):
        from elbench.basic_education_runtime.spend_callback import SpendCallback

        replies = iter([AIMessage(content="hi", usage_metadata={"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100})] * 3)
        model = GenericFakeChatModel(messages=replies)
        model.callbacks = [SpendCallback("student")]
        return model

    def test_runtime_calls_are_recorded_and_stopped_at_the_role_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            limits = SpendLimits(per_role_usd={"student": 0.002}, prices={"student": (1e-6, 2e-6)})
            with spend_limits(limits, Path(tmpdir) / "spend.jsonl") as ledger:
                model = self._model()
                model.invoke("hello")
                self.assertAlmostEqual(ledger.spent()["student"], 0.0012)
                model.invoke("hello")
                with self.assertRaises(BudgetExceeded):
                    model.invoke("hello")


if __name__ == "__main__":
    unittest.main()
