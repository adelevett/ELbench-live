"""Model sizes (billions of parameters) for the dashboard's size filter.

OpenRouter lists no parameter counts. For open-weight models it gives a Hugging Face
repo id, and the Hugging Face API reports the parameter total from the safetensors
metadata. Those totals are cached in ``model_sizes.json`` next to this file so pages
never wait on Hugging Face; refresh the cache with::

    python -m elbench.dashboard.model_sizes

Models with no repo id fall back to the size in their name (``qwen3-235b-a22b`` is
235B total, 22B active). Closed models have no size at all.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx

SIZES_PATH = Path(__file__).with_name("model_sizes.json")
HF_MODEL_API = "https://huggingface.co/api/models/{repo}?expand[]=safetensors"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# "-235b" or "/8b" but not "a22b" or "8x7b"; "-a22b" is the active size of a mixture-of-experts model.
_TOTAL_IN_NAME = re.compile(r"(?:^|[-_/.])(\d+(?:\.\d+)?)b(?=$|[-_:.])", re.IGNORECASE)
_ACTIVE_IN_NAME = re.compile(r"(?:^|[-_/.])a(\d+(?:\.\d+)?)b(?=$|[-_:.])", re.IGNORECASE)


@dataclass(frozen=True)
class ModelSize:
    total_b: float
    active_b: float | None = None

    def label(self) -> str:
        text = f"{_billions(self.total_b)}B"
        if self.active_b is not None and self.active_b < self.total_b:
            text += f" ({_billions(self.active_b)}B active)"
        return text


def _billions(value: float) -> str:
    return f"{value:.0f}" if value >= 10 else f"{value:.1f}".rstrip("0").rstrip(".")


def size_from_name(model_id: str) -> ModelSize | None:
    name = str(model_id or "").split("/", 1)[-1]
    total = _TOTAL_IN_NAME.search(name)
    if total is None:
        return None
    active = _ACTIVE_IN_NAME.search(name)
    return ModelSize(float(total.group(1)), float(active.group(1)) if active else None)


def load_sizes(path: Path = SIZES_PATH) -> dict[str, float]:
    """Cached totals: Hugging Face repo id -> billions of parameters."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(repo): float(total) for repo, total in (data.get("total_b") or {}).items()}


def model_size(model_id: str, hugging_face_id: str | None, sizes: dict[str, float]) -> ModelSize | None:
    named = size_from_name(model_id)
    total = sizes.get(hugging_face_id or "")
    if total is not None:
        return ModelSize(total, named.active_b if named else None)
    return named


def fetch_total_b(repo: str, client: httpx.Client) -> float | None:
    response = client.get(HF_MODEL_API.format(repo=repo))
    if response.status_code != 200:
        return None
    total = ((response.json() or {}).get("safetensors") or {}).get("total")
    return float(total) / 1e9 if isinstance(total, (int, float)) and total > 0 else None


def refresh_sizes(repos: list[str], *, token: str | None = None, path: Path = SIZES_PATH) -> dict[str, float]:
    """Fetch totals for ``repos`` not yet cached, then rewrite the cache file."""
    sizes = load_sizes(path)
    missing = sorted({repo for repo in repos if repo and repo not in sizes})
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with httpx.Client(headers=headers, timeout=20, follow_redirects=True) as client:

        def fetch(repo: str) -> tuple[str, float | None]:
            try:
                return repo, fetch_total_b(repo, client)
            except httpx.HTTPError:
                return repo, None

        with ThreadPoolExecutor(max_workers=8) as pool:
            for repo, total in pool.map(fetch, missing):
                if total is not None:
                    sizes[repo] = round(total, 3)
    path.write_text(
        json.dumps(
            {
                "source": "Hugging Face API safetensors.total, for OpenRouter models with a hugging_face_id",
                "updated": time.strftime("%Y-%m-%d", time.gmtime()),
                "total_b": dict(sorted(sizes.items())),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return sizes


def main() -> int:
    catalog = httpx.get(OPENROUTER_MODELS_URL, timeout=45).json()
    repos = [str(item.get("hugging_face_id") or "") for item in catalog.get("data") or []]
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
    sizes = refresh_sizes([repo for repo in repos if repo], token=token)
    print(f"{len(sizes)} model sizes cached in {SIZES_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
