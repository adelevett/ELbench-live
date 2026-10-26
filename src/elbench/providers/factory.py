from __future__ import annotations

from elbench.providers.base import ModelClient
from elbench.providers.mock import MockModelClient
from elbench.providers.openai_compatible import OpenAICompatibleClient
from elbench.schemas.config import ModelConfig, ProviderConfig


class ProviderFactory:
    _PROVIDERS: dict[str, type[ModelClient]] = {
        "mock": MockModelClient,
        "openai_compatible": OpenAICompatibleClient,
    }

    @classmethod
    def create(cls, provider_config: ProviderConfig, model_config: ModelConfig) -> ModelClient:
        adapter = provider_config.adapter
        if adapter not in cls._PROVIDERS:
            raise ValueError(f"Unsupported provider adapter: {adapter}")
        return cls._PROVIDERS[adapter](provider_config=provider_config, model_config=model_config)

