from __future__ import annotations

from abc import ABC, abstractmethod

from elbench.schemas.config import ModelConfig, ProviderConfig
from elbench.schemas.evaluation import GenerationRequest, ModelResponse, Sample


class ModelClient(ABC):
    def __init__(self, provider_config: ProviderConfig, model_config: ModelConfig) -> None:
        self.provider_config = provider_config
        self.model_config = model_config

    @abstractmethod
    async def generate(self, sample: Sample, request: GenerationRequest) -> ModelResponse:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None

