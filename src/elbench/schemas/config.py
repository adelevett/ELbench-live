from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RetryConfig(BaseModel):
    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 20.0
    backoff_multiplier: float = 2.0
    jitter_ratio: float = 0.2
    retryable_status_codes: list[int] = Field(default_factory=list)


class RateLimitConfig(BaseModel):
    rpm: int | None = None
    tpm: int | None = None
    qps: float | None = None
    max_concurrency: int | None = None


class ProviderCapabilities(BaseModel):
    supports_system_prompt: bool = True
    supports_stream: bool = False
    supports_reasoning: bool = False


class ProviderConfig(BaseModel):
    provider_name: str
    adapter: str
    capabilities: ProviderCapabilities = Field(default_factory=ProviderCapabilities)
    default_timeout: float = 60.0
    endpoint_path: str = "/chat/completions"
    headers: dict[str, str] = Field(default_factory=dict)
    rate_limits: RateLimitConfig = Field(default_factory=RateLimitConfig)


class ModelConfig(BaseModel):
    model_id: str
    provider_name: str
    model_name: str
    api_base: str | None = None
    api_key_env: str | None = None
    temperature: float | None = 0
    max_tokens: int | None = None
    timeout: float | None = None
    retry: RetryConfig = Field(default_factory=RetryConfig)
    rate_limits: RateLimitConfig = Field(default_factory=RateLimitConfig)
    supports_system_prompt: bool | None = None
    supports_stream: bool | None = None
    supports_reasoning: bool | None = None
    provider_kwargs: dict[str, Any] = Field(default_factory=dict)


class ModuleEntry(BaseModel):
    module_name: str
    enabled: bool = True
    description: str | None = None
    owner: str | None = None


class FileRegistryEntry(BaseModel):
    canonical_name: str
    module: str
    subset: str
    file_type: str
    task: str | None = None
    loader_name: str
    mapping_key: str
    path_hints: list[str] = Field(default_factory=list)
    filename_patterns: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)


class FieldMappingConfig(BaseModel):
    sample_id_fields: list[str] = Field(default_factory=list)
    prompt_fields: list[str] = Field(default_factory=list)
    prompt_template: str | None = None
    dimension_fields: list[str] = Field(default_factory=list)
    dimension_resolver: str | None = None
    reference_fields: list[str] = Field(default_factory=list)
    metadata_fields: list[str] = Field(default_factory=list)
    sheet_name: str | None = None


class AppRunConfig(BaseModel):
    max_concurrency: int = 100
    checkpoint_interval: int = 25
    log_every_n: int = 25
    judge_enabled: bool = True
    summary_enabled: bool = True
    resume: bool = True
    retry: RetryConfig = Field(default_factory=RetryConfig)
    provider_rate_limits: dict[str, RateLimitConfig] = Field(default_factory=dict)


class BenchmarkModulesConfig(BaseModel):
    active: list[str] = Field(default_factory=list)
    reserved: list[str] = Field(default_factory=list)


class AppConfig(BaseModel):
    project_name: str = "ELBench"
    data_root: Path = Path("data/benchmark_root")
    output_root: Path = Path("outputs")
    default_run: AppRunConfig = Field(default_factory=AppRunConfig)
    benchmark_modules: BenchmarkModulesConfig = Field(default_factory=BenchmarkModulesConfig)


class JudgeTaskConfig(BaseModel):
    mode: str = "placeholder"
    template: str | None = None
    judge_model_id: str | None = None


class JudgeConfig(BaseModel):
    default_judge_model_id: str | None = None
    basic_education_test_endpoint_model_id: str | None = None
    basic_education_evaluator_model_id: str | None = None
    tasks: dict[str, JudgeTaskConfig] = Field(default_factory=dict)


class ProjectConfig(BaseModel):
    project_root: Path
    config_root: Path
    app: AppConfig
    modules: dict[str, ModuleEntry]
    providers: dict[str, ProviderConfig]
    models: dict[str, ModelConfig]
    judges: JudgeConfig
    file_registry: dict[str, FileRegistryEntry]
    field_mappings: dict[str, FieldMappingConfig]
