"""Pick OpenRouter hosts for a model: the highest precision on offer, then the lowest price.

OpenRouter lists every host of a model with its quantization and price at
``GET /api/v1/models/{slug}/endpoints`` (no key needed). The request body then
carries a ``provider`` object that limits routing to that precision and sorts
the remaining hosts by price, so fallbacks stay at the same precision.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{slug}/endpoints"
CACHE_SECONDS = 900.0

# Bits per weight, floating point ahead of integer at the same width. "unknown" is not ranked:
# when any host reports a precision, hosts that report none are left out.
PRECISION_RANK: dict[str, float] = {
    "fp32": 32,
    "bf16": 16,
    "fp16": 16,
    "fp8": 8,
    "int8": 7.5,
    "fp6": 6,
    "fp4": 4,
    "nvfp4": 4,
    "mxfp4": 4,
    "int4": 3.5,
}
# Hosts OpenRouter only uses when the request asks for that tier.
OPT_IN_SERVICE_TIERS = {"priority"}


@dataclass(slots=True)
class Host:
    name: str
    quantization: str
    prompt_usd_per_token: float | None
    completion_usd_per_token: float | None
    service_tier: str | None = None


@dataclass(slots=True)
class Route:
    slug: str
    # The OpenRouter ``provider`` request field; empty when the hosts could not be read.
    provider: dict[str, Any] = field(default_factory=dict)
    precisions: list[str] = field(default_factory=list)
    hosts: list[Host] = field(default_factory=list)
    error: str | None = None

    @property
    def cheapest(self) -> Host | None:
        priced = [h for h in self.hosts if h.prompt_usd_per_token is not None and h.completion_usd_per_token is not None]
        return min(priced, key=lambda h: (h.prompt_usd_per_token, h.completion_usd_per_token), default=None)

    def describe(self) -> str:
        """One plain sentence for the estimate."""
        if self.error:
            return f"`{self.slug}`: host list unavailable ({self.error}), so the cheapest host is used at any precision."
        host = self.cheapest
        where = f", starting with {host.name}" if host else ""
        count = len(self.hosts)
        hosts = f"{count} host{'s' if count != 1 else ''}"
        if self.precisions:
            return f"`{self.slug}`: {' / '.join(self.precisions)} only ({hosts}), cheapest first{where}."
        return f"`{self.slug}`: no host reports its precision, so the cheapest of {hosts} is used{where}."

    def record(self) -> dict[str, Any]:
        """What the run record keeps, so a result says which precision produced it."""
        host = self.cheapest
        return {
            "provider": dict(self.provider),
            "precisions": list(self.precisions),
            "cheapest_host": host.name if host else None,
        }


def _price(value: Any) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if price >= 0 else None


def route_from_endpoints(slug: str, endpoints: list[dict[str, Any]]) -> Route:
    hosts = [
        Host(
            name=str(e.get("provider_name") or e.get("tag") or "?"),
            quantization=str(e.get("quantization") or "unknown").lower(),
            prompt_usd_per_token=_price((e.get("pricing") or {}).get("prompt")),
            completion_usd_per_token=_price((e.get("pricing") or {}).get("completion")),
            service_tier=e.get("service_tier"),
        )
        for e in endpoints
        if isinstance(e, dict)
    ]
    hosts = [h for h in hosts if h.service_tier not in OPT_IN_SERVICE_TIERS]
    # ":nitro" already asks OpenRouter to sort by speed; every other id is sorted by price.
    provider: dict[str, Any] = {} if slug.endswith(":nitro") else {"sort": "price"}
    ranked = [h for h in hosts if h.quantization in PRECISION_RANK]
    if not ranked:
        return Route(slug=slug, provider=provider, hosts=hosts)
    best = max(PRECISION_RANK[h.quantization] for h in ranked)
    precisions = sorted({h.quantization for h in ranked if PRECISION_RANK[h.quantization] == best})
    return Route(
        slug=slug,
        provider={**provider, "quantizations": precisions},
        precisions=precisions,
        hosts=[h for h in ranked if h.quantization in precisions],
    )


_cache: dict[str, tuple[float, Route]] = {}
_lock = threading.Lock()


def resolve_route(slug: str, *, client: httpx.Client | None = None) -> Route:
    """Route for ``slug``, cached for ``CACHE_SECONDS``. A failed lookup still sorts by price."""
    slug = slug.strip()
    now = time.monotonic()
    with _lock:
        cached = _cache.get(slug)
        if cached and now - cached[0] < CACHE_SECONDS:
            return cached[1]
    path_slug = slug[: -len(":nitro")] if slug.endswith(":nitro") else slug
    path_slug = path_slug[: -len(":floor")] if path_slug.endswith(":floor") else path_slug
    try:
        if client is None:
            with httpx.Client(timeout=10.0) as own:
                response = own.get(ENDPOINTS_URL.format(slug=path_slug))
        else:
            response = client.get(ENDPOINTS_URL.format(slug=path_slug))
        response.raise_for_status()
        endpoints = ((response.json() or {}).get("data") or {}).get("endpoints") or []
        route = route_from_endpoints(slug, endpoints)
    except (httpx.HTTPError, ValueError) as exc:
        reason = f"HTTP {exc.response.status_code}" if isinstance(exc, httpx.HTTPStatusError) else type(exc).__name__
        return Route(slug=slug, provider={} if slug.endswith(":nitro") else {"sort": "price"}, error=reason)
    with _lock:
        _cache[slug] = (now, route)
    return route
