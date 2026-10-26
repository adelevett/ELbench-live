from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RawRecord(BaseModel):
    source_path: str
    source_file: str
    row_index: int
    record: dict[str, Any]


class Sample(BaseModel):
    sample_id: str
    source_file: str
    source_path: str
    module: str
    subset: str
    task: str | None = None
    dimension: str | None = None
    prompt: str
    reference: dict[str, Any] | str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_record: dict[str, Any] = Field(default_factory=dict)


class GenerationRequest(BaseModel):
    prompt: str
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    reasoning: dict[str, Any] | None = None
    messages: list[dict[str, str]] = Field(default_factory=list)
    provider_kwargs: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    text: str | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float | None = None
    retry_count: int = 0
    status_code: int | None = None
    error: str | None = None
    format_valid: bool | None = None
    format_metadata: dict[str, Any] = Field(default_factory=dict)


class JudgeResult(BaseModel):
    judge_name: str
    judge_result: str
    score: float | None = None
    judge_reason: str | None = None
    judge_metadata: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    sample_id: str
    source_file: str
    source_path: str
    module: str
    subset: str
    task: str | None = None
    dimension: str | None = None
    prompt: str
    reference: dict[str, Any] | str | None = None
    provider_name: str
    model_id: str
    model_name: str
    model_response: str | None = None
    judge_result: str | None = None
    score: float | None = None
    judge_reason: str | None = None
    latency_ms: float | None = None
    retry_count: int = 0
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    judge_metadata: dict[str, Any] = Field(default_factory=dict)
    raw_response: dict[str, Any] = Field(default_factory=dict)


class FailureRecord(BaseModel):
    sample_id: str
    source_file: str
    module: str
    subset: str
    provider_name: str
    model_id: str
    retry_count: int
    error_type: str
    error_message: str
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
