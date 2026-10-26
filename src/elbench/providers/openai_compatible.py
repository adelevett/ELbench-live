from __future__ import annotations

import asyncio
import json
from time import perf_counter
from typing import Any

import httpx

from elbench.providers.base import ModelClient
from elbench.schemas.evaluation import GenerationRequest, ModelResponse
from elbench.utils import spend
from elbench.utils.secrets import get_api_key


RESPONSES_API_ONLY_PREFIXES = (
    "gpt-5-pro",
    "gpt-5.2-pro",
    "gpt-5.4-pro",
)


class StreamingResponseTimeout(httpx.TimeoutException):
    retryable = False


class OpenAICompatibleClient(ModelClient):
    def __init__(self, provider_config, model_config) -> None:
        super().__init__(provider_config, model_config)
        timeout = model_config.timeout or provider_config.default_timeout
        self._timeout_seconds = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def generate(self, sample, request: GenerationRequest) -> ModelResponse:
        role = spend.role_for_model(self.model_config.model_id)
        spend.check(role)
        response = await self._generate(sample, request)
        spend.record(role, response.usage)
        return response

    async def _generate(self, sample, request: GenerationRequest) -> ModelResponse:
        if not self.model_config.api_base:
            raise ValueError(f"Model {self.model_config.model_id} missing api_base")

        headers = dict(self.provider_config.headers)
        api_key_env = self.model_config.api_key_env
        if api_key_env:
            api_key = get_api_key(api_key_env)
            if not api_key:
                raise ValueError(f"Environment variable {api_key_env} is not set")
            headers["Authorization"] = f"Bearer {api_key}"
        headers.setdefault("Content-Type", "application/json")

        merged_provider_kwargs = {
            **self.model_config.provider_kwargs,
            **request.provider_kwargs,
        }
        use_responses_api = self._use_responses_api(merged_provider_kwargs)
        payload = self._build_payload(
            request,
            provider_kwargs=merged_provider_kwargs,
            use_responses_api=use_responses_api,
        )

        start = perf_counter()
        if request.stream:
            return await self._generate_streaming(
                headers=headers,
                payload=payload,
                start=start,
                use_responses_api=use_responses_api,
            )

        response = await self._client.post(
            self._build_url(use_responses_api=use_responses_api),
            headers=headers,
            json=payload,
        )
        latency_ms = (perf_counter() - start) * 1000

        if (
            response.status_code == 400
            and not use_responses_api
            and self._should_retry_with_responses(response)
        ):
            use_responses_api = True
            payload = self._build_payload(
                request,
                provider_kwargs=merged_provider_kwargs,
                use_responses_api=True,
            )
            start = perf_counter()
            response = await self._client.post(
                self._build_url(use_responses_api=True),
                headers=headers,
                json=payload,
            )
            latency_ms = (perf_counter() - start) * 1000

        if (
            response.status_code == 400
            and use_responses_api
            and self._should_retry_with_chat_completions(response)
        ):
            use_responses_api = False
            payload = self._build_payload(
                request,
                provider_kwargs=merged_provider_kwargs,
                use_responses_api=False,
            )
            start = perf_counter()
            response = await self._client.post(
                self._build_url(use_responses_api=False),
                headers=headers,
                json=payload,
            )
            latency_ms = (perf_counter() - start) * 1000

        blocked_response = self._coerce_filtered_prompt_response(response, latency_ms=latency_ms)
        if blocked_response is not None:
            return blocked_response

        self._raise_for_status_with_body(response)
        data = response.json()
        return ModelResponse(
            text=self._extract_text(data, use_responses_api=use_responses_api),
            raw_payload=data,
            usage=data.get("usage", {}),
            latency_ms=latency_ms,
            status_code=response.status_code,
        )

    async def _generate_streaming(
        self,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        start: float,
        use_responses_api: bool,
    ) -> ModelResponse:
        url = self._build_url(use_responses_api=use_responses_api)
        try:
            async with asyncio.timeout(float(self._timeout_seconds)):
                async with self._client.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code >= 400:
                        await response.aread()
                        latency_ms = (perf_counter() - start) * 1000
                        blocked_response = self._coerce_filtered_prompt_response(response, latency_ms=latency_ms)
                        if blocked_response is not None:
                            return blocked_response
                        self._raise_for_status_with_body(response)

                    data = await self._consume_chat_completion_stream(response)
        except TimeoutError as exc:
            elapsed_ms = (perf_counter() - start) * 1000
            raise StreamingResponseTimeout(
                f"Streaming response exceeded total timeout of {self._timeout_seconds}s "
                f"after {elapsed_ms:.0f}ms"
            ) from exc

        latency_ms = (perf_counter() - start) * 1000
        return ModelResponse(
            text=self._extract_text(data, use_responses_api=False),
            raw_payload=data,
            usage=data.get("usage", {}),
            latency_ms=latency_ms,
            status_code=response.status_code,
        )

    async def _consume_chat_completion_stream(self, response: httpx.Response) -> dict[str, Any]:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: dict[str, Any] = {}
        role = "assistant"
        finish_reason: str | None = None
        chunk_count = 0
        done = False
        envelope: dict[str, Any] = {
            "id": None,
            "object": "chat.completion",
            "created": None,
            "model": self.model_config.model_name,
        }

        async for line in response.aiter_lines():
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done = True
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid streaming JSON chunk: {payload[:200]}") from exc
            if not isinstance(chunk, dict):
                continue

            chunk_count += 1
            for key in ("id", "created", "model"):
                if chunk.get(key) is not None:
                    envelope[key] = chunk.get(key)
            if chunk.get("object"):
                envelope["object"] = "chat.completion"
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]

            choices = chunk.get("choices") or []
            if not choices or not isinstance(choices[0], dict):
                continue
            choice = choices[0]
            if choice.get("finish_reason") is not None:
                finish_reason = choice.get("finish_reason")
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                continue
            if delta.get("role"):
                role = str(delta.get("role"))
            content = self._stringify_content(delta.get("content"))
            if content:
                content_parts.append(content)
            reasoning = self._stringify_content(delta.get("reasoning_content"))
            if not reasoning:
                reasoning = self._stringify_content(delta.get("reasoning"))
            if reasoning:
                reasoning_parts.append(reasoning)

        if not done:
            raise RuntimeError("Streaming response ended before [DONE] marker")

        content_text = "".join(content_parts)
        reasoning_text = "".join(reasoning_parts)
        message: dict[str, Any] = {"role": role, "content": content_text}
        if reasoning_text:
            message["reasoning_content"] = reasoning_text
        envelope["choices"] = [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ]
        envelope["usage"] = usage
        envelope["stream"] = True
        envelope["stream_chunk_count"] = chunk_count
        return envelope

    async def aclose(self) -> None:
        await self._client.aclose()

    def _build_url(self, *, use_responses_api: bool) -> str:
        path = "/responses" if use_responses_api else self.provider_config.endpoint_path
        return self.model_config.api_base.rstrip("/") + path

    def _build_payload(
        self,
        request: GenerationRequest,
        *,
        provider_kwargs: dict[str, Any],
        use_responses_api: bool,
    ) -> dict[str, Any]:
        provider_kwargs = dict(provider_kwargs)
        provider_kwargs.pop("use_responses_api", None)

        messages: list[dict[str, Any]] = []
        supports_system = (
            self.model_config.supports_system_prompt
            if self.model_config.supports_system_prompt is not None
            else self.provider_config.capabilities.supports_system_prompt
        )
        if request.system_prompt and supports_system:
            messages.append({"role": "system", "content": request.system_prompt})
        if request.messages:
            messages.extend(request.messages)
        else:
            messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {"model": self.model_config.model_name}
        if use_responses_api:
            payload["input"] = self._build_responses_input(messages)
            if request.system_prompt and supports_system:
                payload["instructions"] = request.system_prompt
            if request.stream:
                payload["stream"] = True
            if request.temperature is not None:
                payload["temperature"] = request.temperature
            if request.max_tokens is not None:
                payload["max_output_tokens"] = request.max_tokens
            if request.reasoning is not None:
                payload["reasoning"] = request.reasoning
            payload.update(provider_kwargs)
            self._normalize_responses_payload(payload)
            return payload

        payload["messages"] = messages
        payload["stream"] = request.stream
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.reasoning is not None:
            payload["reasoning"] = request.reasoning
        payload.update(provider_kwargs)
        return payload

    def _build_responses_input(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        response_messages: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "user")).strip().lower() or "user"
            if role == "system":
                continue
            response_messages.append(
                {
                    "role": role,
                    "content": self._build_responses_content(
                        role=role,
                        content=message.get("content", ""),
                    ),
                }
            )
        return response_messages

    def _build_responses_content(self, *, role: str, content: Any) -> list[dict[str, Any]]:
        normalized_type = "output_text" if role == "assistant" else "input_text"
        if isinstance(content, list):
            result: list[dict[str, Any]] = []
            for block in content:
                if isinstance(block, dict):
                    block_type = str(block.get("type", "")).strip().lower()
                    if block_type in {"input_text", "output_text"} and "text" in block:
                        result.append(
                            {
                                "type": block_type,
                                "text": str(block.get("text", "") or ""),
                            }
                        )
                        continue
                    if block_type == "text" and "text" in block:
                        result.append(
                            {
                                "type": normalized_type,
                                "text": str(block.get("text", "") or ""),
                            }
                        )
                        continue
                text = self._stringify_content(block)
                if text:
                    result.append({"type": normalized_type, "text": text})
            return result

        return [{"type": normalized_type, "text": self._stringify_content(content)}]

    def _normalize_responses_payload(self, payload: dict[str, Any]) -> None:
        model_name = str(payload.get("model", "") or "")
        reasoning = payload.get("reasoning") or {}
        reasoning_effort = ""
        if isinstance(reasoning, dict):
            reasoning_effort = str(reasoning.get("effort", "")).strip().lower()
        if (
            model_name.startswith("gpt-5")
            and "chat" not in model_name
            and reasoning_effort != "none"
        ):
            payload.pop("temperature", None)

    def _use_responses_api(self, provider_kwargs: dict[str, Any]) -> bool:
        use_responses_api = provider_kwargs.get("use_responses_api")
        if isinstance(use_responses_api, bool):
            return use_responses_api
        model_name = self.model_config.model_name or ""
        return model_name.startswith(RESPONSES_API_ONLY_PREFIXES) or "codex" in model_name

    def _should_retry_with_responses(self, response: httpx.Response) -> bool:
        body = response.text.lower()
        return "requested operation is unsupported" in body or "unsupported" in body

    def _should_retry_with_chat_completions(self, response: httpx.Response) -> bool:
        body = response.text.lower()
        return "unsupported" in body or "responses" in body

    def _raise_for_status_with_body(self, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body_preview = response.text.strip()
            if body_preview:
                body_preview = body_preview[:1000]
                message = f"{exc}. Response body: {body_preview}"
            else:
                message = str(exc)
            raise httpx.HTTPStatusError(
                message,
                request=exc.request,
                response=exc.response,
            ) from exc

    def _coerce_filtered_prompt_response(
        self,
        response: httpx.Response,
        *,
        latency_ms: float,
    ) -> ModelResponse | None:
        if response.status_code not in {400, 403}:
            return None
        try:
            data = response.json()
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        error = data.get("error")
        if not isinstance(error, dict):
            return None

        code = str(error.get("code", "") or "").strip().lower()
        message = str(error.get("message", "") or "").strip()
        lowered = message.lower()
        if code == "invalid_request_error" and "content exists risk" in lowered:
            text = f"[UPSTREAM_FILTERED:{code}] {message}" if message else f"[UPSTREAM_FILTERED:{code}]"
            return ModelResponse(
                text=text,
                raw_payload=data,
                usage=data.get("usage", {}),
                latency_ms=latency_ms,
                status_code=response.status_code,
                error=code,
            )
        if code == "1301":
            text = f"[UPSTREAM_FILTERED:{code}] {message}" if message else f"[UPSTREAM_FILTERED:{code}]"
            return ModelResponse(
                text=text,
                raw_payload=data,
                usage=data.get("usage", {}),
                latency_ms=latency_ms,
                status_code=response.status_code,
                error=code,
            )
        if code not in {"content_filter", "invalid_prompt", "modelarts.81011", "1301"}:
            return None
        if not any(
            marker in lowered
            for marker in (
                "filtered",
                "safety",
                "content management policy",
                "limited access",
                "sensitive information",
                "不安全",
                "安全",
                "敏感",
                "审核",
            )
        ):
            return None

        text = f"[UPSTREAM_FILTERED:{code}] {message}" if message else f"[UPSTREAM_FILTERED:{code}]"
        return ModelResponse(
            text=text,
            raw_payload=data,
            usage=data.get("usage", {}),
            latency_ms=latency_ms,
            status_code=response.status_code,
            error=code,
        )

    def _extract_text(self, payload: dict[str, Any], *, use_responses_api: bool) -> str:
        if use_responses_api:
            return self._extract_text_from_responses(payload)

        choices = payload.get("choices") or []
        if not choices:
            return ""
        first_choice = choices[0] if isinstance(choices[0], dict) else {}
        if "text" in first_choice:
            choice_text = self._stringify_content(first_choice.get("text"))
            if choice_text.strip():
                return choice_text
        message = first_choice.get("message", {})
        content = message.get("content", "")
        if content is None:
            content = ""
        if isinstance(content, list):
            text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
            if text.strip():
                return text
        elif str(content).strip():
            return str(content)

        reasoning_content = message.get("reasoning_content", "")
        if isinstance(reasoning_content, list):
            text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in reasoning_content
            )
            if text.strip():
                return text
        elif str(reasoning_content).strip():
            return str(reasoning_content)

        reasoning = message.get("reasoning", "")
        if isinstance(reasoning, list):
            text = "".join(
                self._stringify_content(part)
                for part in reasoning
            )
            if text.strip():
                return text
        elif isinstance(reasoning, dict):
            text = self._stringify_content(reasoning)
            if text.strip():
                return text
        elif str(reasoning).strip():
            return str(reasoning)
        return ""

    def _extract_text_from_responses(self, payload: dict[str, Any]) -> str:
        outputs = payload.get("output") or []
        response_parts: list[str] = []
        reasoning_parts: list[str] = []
        for item in outputs:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).strip().lower()
            if item_type == "message":
                for block in item.get("content") or []:
                    if not isinstance(block, dict):
                        continue
                    block_type = str(block.get("type", "")).strip().lower()
                    if block_type in {"output_text", "text"}:
                        text = str(block.get("text", "") or "")
                        if text:
                            response_parts.append(text)
                    elif block_type == "refusal":
                        refusal_text = str(block.get("refusal", "") or "")
                        if refusal_text:
                            response_parts.append(refusal_text)
            elif item_type == "reasoning":
                for block in item.get("summary") or []:
                    text = self._stringify_content(block)
                    if text:
                        reasoning_parts.append(text)

        if response_parts:
            return "\n".join(response_parts).strip()
        if reasoning_parts:
            return "\n".join(reasoning_parts).strip()
        return ""

    def _stringify_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            if "text" in content:
                return self._stringify_content(content.get("text"))
            if "content" in content:
                return self._stringify_content(content.get("content"))
            return ""
        if isinstance(content, list):
            parts = [self._stringify_content(item) for item in content]
            return "\n".join(part for part in parts if part).strip()
        return str(content)
