"""LangChain hook that applies the run's spending limits to basic-education runtime calls."""

from __future__ import annotations

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from elbench.utils import spend


class SpendCallback(BaseCallbackHandler):
    """Checks the run's spending limit before each call and records what the call cost."""

    raise_error = True

    def __init__(self, role: str) -> None:
        self.role = role

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        spend.check(self.role)

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        usage = (response.llm_output or {}).get("token_usage")
        if not usage:
            for generations in response.generations:
                for generation in generations:
                    message = getattr(generation, "message", None)
                    usage = getattr(message, "usage_metadata", None)
                    if usage:
                        break
        spend.record(self.role, dict(usage) if usage else None)
