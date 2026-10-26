"""Map an OpenRouter model id onto a source ModelConfig.

The HTTP body is still built by ``OpenAICompatibleClient``. When the id matches
one published card, generation settings are copied from that card. The ``model``
string stays the OpenRouter id, because that is the id OpenRouter routes.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

from elbench.providers.openai_compatible import OpenAICompatibleClient
from elbench.schemas.config import ModelConfig, ProjectConfig, RateLimitConfig, RetryConfig
from elbench.schemas.evaluation import GenerationRequest

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
VISITOR_KEY_ENV = "ELBENCH_VISITOR_OPENROUTER_API_KEY"
# Routing variants are not rows in GET /api/v1/models. :free and :batch are.
ROUTING_VARIANT_SUFFIXES = (":nitro", ":floor")

DEFAULT_SYSTEM_ONE_MODELS: tuple[tuple[str, str, float | None], ...] = (
    ("LiquidAI: d1-ultra", "liquid/d1-ultra", 0.042e-6),
    ("LiquidAI: d1", "liquid/d1", 0.020e-6),
    ("TypeSafe: Jev 1.14", "typesafe/jev-1.14", 0.042e-6),
    ("TypeSafe: Jev Latest", "~typesafe/jev-latest", 0.042e-6),
    ("OpenAI: GPT-6 Luna Decisions", "openai/gpt-6-luna-decisions", 0.100e-6),
    ("Perplexity: Decider V1.2", "perplexity/pplx-decider-v1.2", 0.020e-6),
    ("Perplexity: Decider Flash V1.2", "perplexity/pplx-decider-flash-v1.2", 0.010e-6),
    ("Microsoft: Microsoft-Decision-1", "microsoft/microsoft-decision-1", 0.042e-6),
    ("Cloudflare: Clef", "cloudflare/clef", 0.042e-6),
    ("Cloudflare: Clef Flash", "cloudflare/clef-flash", 0.019e-6),
    ("Cloudflare: Clef Omni", "cloudflare/clef-omni", 0.150e-6),
    ("Inception: Mercury Decide", "inception/mercury-decide", 0.020e-6),
    ("Inception: Mercury Decide (free)", "inception/mercury-decide:free", 0.0),
    ("Upstage: Solar Decide", "upstage/solar-decide", 0.050e-6),
    ("Nace.AI: Drex Max", "nace-ai/drex-max", 0.060e-6),
    ("ScaleDown: Classify-1", "scaledown/classify-1", 0.040e-6),
    ("Respan: Span-02", "respan/span-02", 0.010e-6),
    ("Baseten: Mammoth 1 (free)", "baseten/mammoth-1:free", 0.0),
)

# Modal generation settings among the 13 published openai-compatible cards that
# have empty provider_kwargs and max_tokens 8192. Temperature 0 is unanimous
# across every non-mock card. The other fields are the mode of that group, not
# a value the source defines for unlisted models.
UNLISTED_TEMPERATURE = 0.0
UNLISTED_MAX_TOKENS = 8192
UNLISTED_TIMEOUT = 240.0
UNLISTED_STREAM = False
UNLISTED_RETRY = RetryConfig(
    max_attempts=2,
    initial_delay_seconds=1.0,
    max_delay_seconds=20.0,
    backoff_multiplier=2.0,
    jitter_ratio=0.2,
    retryable_status_codes=[],
)
UNLISTED_RATE_LIMITS = RateLimitConfig(max_concurrency=100)


@dataclass(slots=True)
class Caveat:
    level: str
    code: str
    message: str


@dataclass(slots=True)
class VisitorProtocol:
    openrouter_model: str
    model_config: ModelConfig
    matched_card_id: str | None
    matched_card_model_name: str | None
    responses_api: bool
    card_would_use_responses_api: bool | None
    request_branch_differences: list[str] = field(default_factory=list)
    caveats: list[Caveat] = field(default_factory=list)


def catalog_slug(openrouter_model: str) -> str:
    """Slug whose card and catalog price apply.

    ``:nitro`` and ``:floor`` only change provider sort. ``:free`` and ``:batch``
    are separate catalog rows and stay intact.
    """
    model = openrouter_model.strip()
    for suffix in ROUTING_VARIANT_SUFFIXES:
        if model.endswith(suffix):
            return model[: -len(suffix)]
    return model


def safe_model_id(openrouter_model: str) -> str:
    """Filesystem-safe id. The basic-education runtime uses model_id as a path segment."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", openrouter_model).strip("._-")
    if not cleaned:
        cleaned = "model"
    return f"openrouter_{cleaned[:80]}"


def match_model_card(models: dict[str, ModelConfig], openrouter_model: str) -> ModelConfig | None:
    published = [model for model in models.values() if model.provider_name != "mock"]
    exact = [model for model in published if model.model_name == openrouter_model]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return None
    suffix = [model for model in published if openrouter_model.endswith("/" + model.model_name)]
    if len(suffix) == 1:
        return suffix[0]
    return None


def card_match_is_ambiguous(models: dict[str, ModelConfig], openrouter_model: str) -> bool:
    published = [model for model in models.values() if model.provider_name != "mock"]
    exact = [model for model in published if model.model_name == openrouter_model]
    if len(exact) > 1:
        return True
    if len(exact) == 1:
        return False
    suffix = [model for model in published if openrouter_model.endswith("/" + model.model_name)]
    return len(suffix) > 1


def build_visitor_protocol(config: ProjectConfig, openrouter_model: str) -> VisitorProtocol:
    model_name = openrouter_model.strip()
    if not model_name:
        raise ValueError("OpenRouter model id is empty.")

    lookup_name = catalog_slug(model_name)
    card = match_model_card(config.models, lookup_name)
    ambiguous = card is None and card_match_is_ambiguous(config.models, lookup_name)
    caveats: list[Caveat] = []
    if lookup_name != model_name:
        caveats.append(
            Caveat(
                level="warning",
                code="routing_variant",
                message=(
                    f"{model_name} keeps that suffix in the model field. "
                    f":nitro and :floor are routing variants, not rows in GET /api/v1/models. "
                    f"The card and the catalog price are looked up on {lookup_name}. "
                    ":nitro sorts providers by throughput and :floor sorts them by price. "
                    ":free and :batch are real catalog rows and are not rewritten."
                ),
            )
        )

    if card is None:
        model_config = ModelConfig(
            model_id=safe_model_id(model_name),
            provider_name="openai_compatible",
            model_name=model_name,
            api_base=OPENROUTER_API_BASE,
            api_key_env=VISITOR_KEY_ENV,
            temperature=UNLISTED_TEMPERATURE,
            max_tokens=UNLISTED_MAX_TOKENS,
            timeout=UNLISTED_TIMEOUT,
            retry=UNLISTED_RETRY.model_copy(deep=True),
            rate_limits=UNLISTED_RATE_LIMITS.model_copy(deep=True),
            supports_system_prompt=True,
            supports_stream=UNLISTED_STREAM,
            supports_reasoning=True,
            provider_kwargs={},
        )
        if ambiguous:
            caveats.append(
                Caveat(
                    level="warning",
                    code="ambiguous_model_card",
                    message=(
                        f"{model_name} matches more than one model_name in configs/models.yaml. "
                        "No card was copied. Generation settings are the unlisted-model modal "
                        f"values: temperature {UNLISTED_TEMPERATURE:g}, max_tokens {UNLISTED_MAX_TOKENS}, "
                        f"timeout {UNLISTED_TIMEOUT:g}s, stream {str(UNLISTED_STREAM).lower()}, "
                        f"retry max_attempts {UNLISTED_RETRY.max_attempts}, empty provider_kwargs."
                    ),
                )
            )
        else:
            caveats.append(
                Caveat(
                    level="warning",
                    code="no_model_card",
                    message=(
                        f"{model_name} has no model card in configs/models.yaml. "
                        "Published cards do not share one max_tokens. Unlisted models use the modal "
                        "settings of the 13 empty-provider_kwargs cards at max_tokens 8192: "
                        f"temperature {UNLISTED_TEMPERATURE:g} (this value is on every non-mock card), "
                        f"max_tokens {UNLISTED_MAX_TOKENS}, timeout {UNLISTED_TIMEOUT:g}s, "
                        f"stream {str(UNLISTED_STREAM).lower()}, retry max_attempts {UNLISTED_RETRY.max_attempts}, "
                        "empty provider_kwargs. A listed model such as gpt-5.4 uses its own card instead."
                    ),
                )
            )
    else:
        model_config = card.model_copy(
            update={
                "model_id": safe_model_id(model_name),
                "model_name": model_name,
                "api_base": OPENROUTER_API_BASE,
                "api_key_env": VISITOR_KEY_ENV,
            },
            deep=True,
        )
        caveats.append(
            Caveat(
                level="info",
                code="model_card_copied",
                message=(
                    f"Generation settings were copied from model card {card.model_id} "
                    f"(model_name {card.model_name}): temperature {model_config.temperature}, "
                    f"max_tokens {model_config.max_tokens}, timeout {model_config.timeout}, "
                    f"stream {str(bool(model_config.supports_stream)).lower()}, "
                    f"retry max_attempts {model_config.retry.max_attempts}, "
                    f"provider_kwargs keys {sorted(model_config.provider_kwargs) or '[]'}."
                ),
            )
        )
        if card.api_base and card.api_base.rstrip("/") != OPENROUTER_API_BASE:
            caveats.append(
                Caveat(
                    level="warning",
                    code="openrouter_host",
                    message=(
                        f"Card {card.model_id} calls {card.api_base}. This run calls {OPENROUTER_API_BASE}. "
                        "The JSON body uses the same client and the copied generation settings, but the "
                        f"model field is {model_name}, not {card.model_name}."
                    ),
                )
            )

    if card is None:
        caveats.append(
            Caveat(
                level="info",
                code="openrouter_host",
                message=(
                    f"Requests go to {OPENROUTER_API_BASE} through OpenAICompatibleClient. "
                    "The published runs used the api_base values in configs/models.yaml."
                ),
            )
        )

    caveats.append(
        Caveat(
            level="warning",
            code="openrouter_provider_routing",
            message=(
                "OpenRouter serves this model slug from several hosts. The run limits them to the "
                "highest precision any host reports and tries them cheapest first (see the host note). "
                "allow_fallbacks defaults to true, so a host error is retried on another host at the "
                "same precision. require_parameters defaults to false, so a host that does not implement "
                "temperature or max_tokens still accepts the request and ignores those fields. "
                "The model field is the slug above. "
                "POST /api/v1/chat/completions authenticates with a Bearer token. "
                "Reported token counts use the model's native tokenizer."
            ),
        )
    )

    provider = config.providers["openai_compatible"]
    responses_api, _ = _preview_payload(provider, model_config, "protocol-check")
    card_responses: bool | None = None
    differences: list[str] = []
    if card is not None:
        card_model = model_config.model_copy(update={"model_name": card.model_name}, deep=True)
        card_responses, card_payload = _preview_payload(provider, card_model, "protocol-check")
        _, visitor_payload = _preview_payload(provider, model_config, "protocol-check")
        differences = _payload_differences(card_payload, visitor_payload)
        if differences:
            caveats.append(
                Caveat(
                    level="warning",
                    code="request_branch_differs",
                    message=(
                        "OpenAICompatibleClient branches on the model string. "
                        f"The card name {card.model_name} and the OpenRouter id {model_name} "
                        f"do not produce the same body. Differing fields: {', '.join(differences)}. "
                        "The run sends the OpenRouter id, because OpenRouter routes on that id. "
                        "The client is not patched."
                    ),
                )
            )

    return VisitorProtocol(
        openrouter_model=model_name,
        model_config=model_config,
        matched_card_id=None if card is None else card.model_id,
        matched_card_model_name=None if card is None else card.model_name,
        responses_api=responses_api,
        card_would_use_responses_api=card_responses,
        request_branch_differences=differences,
        caveats=caveats,
    )


def build_judge_model_config(
    config: ProjectConfig,
    openrouter_judge_model: str,
    *,
    system_one: bool = False,
) -> ModelConfig:
    """Build a judge ModelConfig authenticated by the single visitor OpenRouter key."""
    model_name = openrouter_judge_model.strip()
    if not model_name:
        raise ValueError("Judge OpenRouter model id is empty.")
    if system_one:
        return ModelConfig(
            model_id=f"judge_s1_{safe_model_id(model_name)}",
            provider_name="openai_compatible",
            model_name=model_name,
            api_base=OPENROUTER_API_BASE,
            api_key_env=VISITOR_KEY_ENV,
            temperature=0.0,
            max_tokens=0,
            timeout=UNLISTED_TIMEOUT,
            retry=UNLISTED_RETRY.model_copy(deep=True),
            rate_limits=UNLISTED_RATE_LIMITS.model_copy(deep=True),
            supports_system_prompt=False,
            supports_stream=False,
            supports_reasoning=False,
            provider_kwargs={"system_one": True},
        )
    protocol = build_visitor_protocol(config, model_name)
    return protocol.model_config.model_copy(
        update={
            "model_id": f"judge_{safe_model_id(model_name)}",
            "temperature": 0.0,
            # No length limit: a reasoning judge thinks before it grades, and a cap cuts off the grade.
            "max_tokens": None,
        },
        deep=True,
    )


def build_multiturn_chat_model_config(
    config: ProjectConfig,
    openrouter_multiturn_model: str,
) -> ModelConfig:
    """Build a multi-turn student chat ModelConfig authenticated by the visitor OpenRouter key."""
    model_name = openrouter_multiturn_model.strip()
    if not model_name:
        raise ValueError("Multi-turn OpenRouter model id is empty.")
    protocol = build_visitor_protocol(config, model_name)
    return protocol.model_config.model_copy(
        update={
            "model_id": f"multiturn_{safe_model_id(model_name)}",
            "temperature": 0.0,
            # No length limit, for the same reason as the judge: a reasoning student needs room to think.
            "max_tokens": None,
        },
        deep=True,
    )


def preview_standard_payload(config: ProjectConfig, model_config: ModelConfig, prompt: str) -> dict:
    """Build the JSON body the runner's first request would send for this prompt."""
    provider = config.providers[model_config.provider_name]
    _, payload = _preview_payload(provider, model_config, prompt)
    return payload


def _preview_payload(
    provider,
    model_config: ModelConfig,
    prompt: str,
) -> tuple[bool, dict]:
    client = OpenAICompatibleClient(provider, model_config)

    async def _build() -> tuple[bool, dict]:
        try:
            request = GenerationRequest(
                prompt=prompt,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                stream=bool(model_config.supports_stream),
                messages=[{"role": "user", "content": prompt}],
                provider_kwargs={},
            )
            merged = {**model_config.provider_kwargs, **request.provider_kwargs}
            use_responses_api = client._use_responses_api(merged)
            payload = client._build_payload(
                request,
                provider_kwargs=merged,
                use_responses_api=use_responses_api,
            )
            return use_responses_api, payload
        finally:
            await client.aclose()

    return asyncio.run(_build())


def _payload_differences(card_payload: dict, visitor_payload: dict) -> list[str]:
    differences: list[str] = []
    keys = sorted(set(card_payload) | set(visitor_payload))
    for key in keys:
        if key == "model":
            continue
        if key in {"messages", "input"}:
            continue
        if card_payload.get(key) != visitor_payload.get(key):
            differences.append(key)
        elif key not in card_payload or key not in visitor_payload:
            differences.append(key)
    return differences
