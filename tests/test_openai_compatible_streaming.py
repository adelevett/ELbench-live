import asyncio
import sys
import unittest
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elbench.providers.openai_compatible import OpenAICompatibleClient  # noqa: E402
from elbench.schemas.config import ModelConfig, ProviderConfig  # noqa: E402
from elbench.schemas.evaluation import GenerationRequest  # noqa: E402


class OpenAICompatibleStreamingTest(unittest.TestCase):
    def test_streaming_chat_completion_accumulates_reasoning_and_content(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/v1/chat/completions")
            body = b"".join(
                [
                    b'data: {"id":"stream-1","object":"chat.completion.chunk","model":"deepseek-r1",'
                    b'"choices":[{"index":0,"delta":{"role":"assistant","content":"","reasoning_content":"think "},'
                    b'"finish_reason":null}]}\n\n',
                    b'data: {"id":"stream-1","object":"chat.completion.chunk","model":"deepseek-r1",'
                    b'"choices":[{"index":0,"delta":{"content":"","reasoning_content":"steps"},'
                    b'"finish_reason":null}]}\n\n',
                    b'data: {"id":"stream-1","object":"chat.completion.chunk","model":"deepseek-r1",'
                    b'"choices":[{"index":0,"delta":{"content":"Final answer is \\\\boxed{204}"},'
                    b'"finish_reason":null}]}\n\n',
                    b'data: {"id":"stream-1","object":"chat.completion.chunk","model":"deepseek-r1",'
                    b'"usage":{"prompt_tokens":10,"completion_tokens":20,'
                    b'"completion_tokens_details":{"reasoning_tokens":12}},'
                    b'"choices":[{"index":0,"delta":{"content":""},"finish_reason":"stop"}]}\n\n',
                    b"data: [DONE]\n\n",
                ]
            )
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=body)

        client = self._client(handler)
        try:
            response = asyncio.run(
                client.generate(
                    sample=None,
                    request=GenerationRequest(
                        prompt="Solve.",
                        messages=[{"role": "user", "content": "Solve."}],
                        max_tokens=8192,
                        stream=True,
                    ),
                )
            )
        finally:
            asyncio.run(client.aclose())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "Final answer is \\boxed{204}")
        self.assertEqual(response.usage["completion_tokens_details"]["reasoning_tokens"], 12)
        self.assertEqual(
            response.raw_payload["choices"][0]["message"]["reasoning_content"],
            "think steps",
        )
        self.assertEqual(response.raw_payload["choices"][0]["finish_reason"], "stop")
        self.assertTrue(response.raw_payload["stream"])

    def test_streaming_chat_completion_requires_done_marker(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            body = (
                b'data: {"id":"stream-1","object":"chat.completion.chunk","model":"deepseek-r1",'
                b'"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}\n\n'
            )
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=body)

        client = self._client(handler)
        try:
            with self.assertRaises(RuntimeError):
                asyncio.run(
                    client.generate(
                        sample=None,
                        request=GenerationRequest(
                            prompt="Solve.",
                            messages=[{"role": "user", "content": "Solve."}],
                            max_tokens=8192,
                            stream=True,
                        ),
                    )
                )
        finally:
            asyncio.run(client.aclose())

    def test_streaming_chat_completion_respects_total_timeout(self) -> None:
        class SlowSseStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield (
                    b'data: {"id":"stream-1","object":"chat.completion.chunk","model":"deepseek-r1",'
                    b'"choices":[{"index":0,"delta":{"reasoning_content":"still thinking"},'
                    b'"finish_reason":null}]}\n\n'
                )
                await asyncio.sleep(0.05)
                yield b"data: [DONE]\n\n"

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                stream=SlowSseStream(),
            )

        client = self._client(handler, timeout=0.01)
        try:
            with self.assertRaises(httpx.TimeoutException):
                asyncio.run(
                    client.generate(
                        sample=None,
                        request=GenerationRequest(
                            prompt="Solve.",
                            messages=[{"role": "user", "content": "Solve."}],
                            max_tokens=8192,
                            stream=True,
                        ),
                    )
                )
        finally:
            asyncio.run(client.aclose())

    def _client(self, handler, *, timeout: float | None = None) -> OpenAICompatibleClient:
        client = OpenAICompatibleClient(
            ProviderConfig(provider_name="openai_compatible", adapter="openai_compatible"),
            ModelConfig(
                model_id="deepseek-r1-250528",
                provider_name="openai_compatible",
                model_name="deepseek-r1-250528",
                api_base="https://api.innospark.cn/v1",
                api_key_env=None,
                timeout=timeout,
            ),
        )
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return client


if __name__ == "__main__":
    unittest.main()
