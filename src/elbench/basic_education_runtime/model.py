from __future__ import annotations

from typing import Any, Dict

from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import AIMessage

from elbench.basic_education_runtime.config import CONFIG
from elbench.basic_education_runtime.entity import ModelConfig
from elbench.basic_education_runtime.spend_callback import SpendCallback
from elbench.utils import spend


class MockRuntimeChatModel:
    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        prefix = ""
        if self.model_config.kargs:
            prefix = str(self.model_config.kargs.get("prefix", "")).strip()

        last_content = ""
        for item in reversed(messages):
            content = getattr(item, "content", None)
            if isinstance(content, str) and content.strip():
                last_content = content.strip()
                break
            if isinstance(item, dict):
                dict_content = item.get("content")
                if isinstance(dict_content, str) and dict_content.strip():
                    last_content = dict_content.strip()
                    break

        summary = last_content[:120]
        content = "占位回答：我们一步一步来。<end>"
        if summary:
            content = f"{prefix} 占位回答：已收到问题。{summary}\n我们一步一步来。<end>".strip()
        elif prefix:
            content = f"{prefix} {content}".strip()
        return AIMessage(content=content)

    def bind_tools(self, tools: list[Any], tool_choice: Any = None) -> "MockRuntimeChatModel":
        return self


class SystemOneRuntimeModel:
    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config


def init_chat_model_from_dict(mc: ModelConfig) -> BaseChatModel:
    if mc.type == "mock":
        return MockRuntimeChatModel(mc)  # type: ignore[return-value]
    if mc.type == "system_one":
        return SystemOneRuntimeModel(mc)  # type: ignore[return-value]
    if mc.kargs is not None:
        llm = init_chat_model(
            model=f"{mc.type}:{mc.model}",
            api_key=mc.api_key,
            base_url=mc.api_base,
            **mc.kargs,
        )
    else:
        llm = init_chat_model(
            model=f"{mc.type}:{mc.model}",
            api_key=mc.api_key,
            base_url=mc.api_base,
        )
    if mc.role and spend.active_ledger() is not None:
        llm.callbacks = [*(llm.callbacks or []), SpendCallback(mc.role)]
    return llm


def init_model_map_from_dict() -> Dict[str, BaseChatModel]:
    cfg = CONFIG.models
    result = {}
    for k, v in cfg.items():
        result[k] = init_chat_model_from_dict(v)
    return result


init_model_map = init_model_map_from_dict

if __name__ == "__main__":
    a = init_model_map_from_dict()
    print(a)
