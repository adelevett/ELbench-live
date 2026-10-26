import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elbench.execution.retry import is_retryable_exception, sleep_before_retry  # noqa: E402
from elbench.schemas.config import RetryConfig  # noqa: E402


class RetryDelayTest(unittest.TestCase):
    def test_retryable_false_marker_disables_retry(self) -> None:
        exc = RuntimeError("known non-retryable failure")
        exc.retryable = False

        self.assertFalse(is_retryable_exception(exc, retryable_status_codes={500, 504}))

    def test_429_waits_for_minute_window_when_retry_after_is_absent(self) -> None:
        request = httpx.Request("POST", "https://api.innospark.cn/v1/chat/completions")
        response = httpx.Response(429, request=request)
        exc = httpx.HTTPStatusError("too many requests", request=request, response=response)
        policy = RetryConfig(
            initial_delay_seconds=1,
            max_delay_seconds=20,
            backoff_multiplier=2,
            jitter_ratio=0,
        )

        async def run() -> float:
            with patch("asyncio.sleep", new=AsyncMock()) as sleep:
                delay = await sleep_before_retry(policy, 1, exc)
                sleep.assert_awaited_once_with(60.0)
                return delay

        self.assertEqual(asyncio.run(run()), 60.0)

    def test_429_honors_retry_after_when_longer_than_default_window(self) -> None:
        request = httpx.Request("POST", "https://api.innospark.cn/v1/chat/completions")
        response = httpx.Response(429, headers={"Retry-After": "90"}, request=request)
        exc = httpx.HTTPStatusError("too many requests", request=request, response=response)
        policy = RetryConfig(jitter_ratio=0)

        async def run() -> float:
            with patch("asyncio.sleep", new=AsyncMock()) as sleep:
                delay = await sleep_before_retry(policy, 1, exc)
                sleep.assert_awaited_once_with(90.0)
                return delay

        self.assertEqual(asyncio.run(run()), 90.0)


if __name__ == "__main__":
    unittest.main()
