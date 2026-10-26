from __future__ import annotations

import asyncio
import random
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone

from elbench.schemas.config import RetryConfig


def is_retryable_exception(exc: Exception, retryable_status_codes: set[int]) -> bool:
    if getattr(exc, "retryable", None) is False:
        return False
    response = getattr(exc, "response", None)
    if response is None:
        return True
    status_code = getattr(response, "status_code", None)
    if status_code is None:
        return True
    return status_code in retryable_status_codes


async def sleep_before_retry(policy: RetryConfig, attempt_index: int, exc: Exception | None = None) -> float:
    base = policy.initial_delay_seconds * (policy.backoff_multiplier ** max(0, attempt_index - 1))
    delay = min(base, policy.max_delay_seconds)
    retry_after_delay = _retry_after_delay(exc)
    if retry_after_delay is not None:
        delay = max(delay, retry_after_delay)
    elif _status_code(exc) == 429:
        delay = max(delay, 60.0)
    jitter = delay * policy.jitter_ratio * random.random()
    final_delay = delay + jitter
    await asyncio.sleep(final_delay)
    return final_delay


def _status_code(exc: Exception | None) -> int | None:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def _retry_after_delay(exc: Exception | None) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    retry_after = headers.get("Retry-After")
    if not retry_after:
        return None
    try:
        return max(0.0, float(retry_after))
    except ValueError:
        try:
            retry_time = parsedate_to_datetime(retry_after)
        except (TypeError, ValueError):
            return None
        if retry_time.tzinfo is None:
            retry_time = retry_time.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_time - datetime.now(timezone.utc)).total_seconds())

