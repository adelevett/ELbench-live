from __future__ import annotations

import asyncio
from collections import deque
from time import monotonic

from elbench.schemas.config import RateLimitConfig


class SlidingWindowRateLimiter:
    def __init__(self, config: RateLimitConfig) -> None:
        self._config = config
        self._lock = asyncio.Lock()
        self._request_events_1s: deque[float] = deque()
        self._request_events_60s: deque[float] = deque()
        self._token_events_60s: deque[tuple[float, int]] = deque()
        concurrency = config.max_concurrency or 100
        self._semaphore = asyncio.Semaphore(concurrency)

    async def acquire(self, estimated_tokens: int = 0) -> None:
        await self._semaphore.acquire()
        await self._wait_for_slot(estimated_tokens)

    def release(self) -> None:
        self._semaphore.release()

    async def _wait_for_slot(self, estimated_tokens: int) -> None:
        while True:
            async with self._lock:
                now = monotonic()
                self._prune(now)
                delay = self._compute_delay(now, estimated_tokens)
                if delay <= 0:
                    self._request_events_1s.append(now)
                    self._request_events_60s.append(now)
                    if estimated_tokens > 0:
                        self._token_events_60s.append((now, estimated_tokens))
                    return
            await asyncio.sleep(delay)

    def _compute_delay(self, now: float, estimated_tokens: int) -> float:
        delays: list[float] = []
        if self._config.qps and len(self._request_events_1s) >= self._config.qps:
            delays.append(max(0.0, 1.0 - (now - self._request_events_1s[0])))
        if self._config.rpm and len(self._request_events_60s) >= self._config.rpm:
            delays.append(max(0.0, 60.0 - (now - self._request_events_60s[0])))
        if self._config.tpm:
            used_tokens = sum(tokens for _, tokens in self._token_events_60s)
            if used_tokens + estimated_tokens > self._config.tpm and self._token_events_60s:
                delays.append(max(0.0, 60.0 - (now - self._token_events_60s[0][0])))
        return max(delays) if delays else 0.0

    def _prune(self, now: float) -> None:
        while self._request_events_1s and now - self._request_events_1s[0] >= 1.0:
            self._request_events_1s.popleft()
        while self._request_events_60s and now - self._request_events_60s[0] >= 60.0:
            self._request_events_60s.popleft()
        while self._token_events_60s and now - self._token_events_60s[0][0] >= 60.0:
            self._token_events_60s.popleft()

