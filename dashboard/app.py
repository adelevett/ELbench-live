"""ELBench-live: the published ELBench results, and a hosted runner for any OpenRouter model.

Runs go through ``BenchmarkRunner``, ``OpenAICompatibleClient``, and ``JudgeRouter``, so
prompts, retries, and scoring are the benchmark's own.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

try:
    import spaces  # type: ignore[import-not-found]
except ImportError:
    spaces = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gradio as gr
import httpx

from elbench.dashboard.catalog import Catalog, CatalogFile, load_catalog
from elbench.dashboard.english import AREA_LABELS, area_label, english_dimension, plain_file
from elbench.dashboard.leaderboard import (
    PAPER_JUDGE_LABEL,
    PAPER_JUDGE_SLUG,
    judge_board,
    judge_board_choices,
    judge_board_label,
    CHINA_SPECIFIC_FILES,
    SCOPE_CORE,
    SCOPE_PAPER,
    LeaderboardStore,
    build_comparison_choices,
    build_comparison_table,
    build_pareto_frontier,
    custom_leaderboard_rows,
    load_official_models,
    official_leaderboard_rows,
    official_per_file_rows,
    pareto_cost_metric_choices,
    pareto_score_metric_choices,
    scope_item_count,
    seed_official_openrouter_prices,
)
from elbench.dashboard.model_sizes import ModelSize, load_sizes, model_size
from elbench.dashboard.plan import JUDGE_HIGH_OUTPUT_TOKENS, Plan, build_plan, render_plan
from elbench.dashboard.protocol import DEFAULT_SYSTEM_ONE_MODELS, catalog_slug
from elbench.dashboard.routing import Route, resolve_route
from elbench.dashboard.run import (
    execute_run,
    failures_progress_path,
    judged_progress_path,
    make_run_id,
    public_error,
    visitor_api_key,
)
from elbench.utils.spend import ROLE_LABELS, SpendLedger, SpendLimits

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_DECISION_MODELS_URL = "https://openrouter.ai/api/v1/models?output_modalities=decisions"
OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"
OPENROUTER_KEYS_PAGE = "https://openrouter.ai/settings/keys"
DEFAULT_TARGET_SLUG = "openai/gpt-5.4"
DEFAULT_CHAT_JUDGE_SLUG = PAPER_JUDGE_SLUG
DEFAULT_SYSTEM_ONE_JUDGE_SLUG = "liquid/d1-ultra"
DEFAULT_MULTITURN_CHAT_SLUG = "liquid/lfm-2.5-24b-a2b"
DEFAULT_BUDGET_USD = 5.0

ELBENCH_PAPER_URL = "https://arxiv.org/abs/2608.09548"
ELBENCH_GITHUB_URL = "https://github.com/ZeroLoss-Lab/ELBench"
AI_FOR_EDUCATION_LEADERBOARD_URL = "https://benchmarks.ai-for-education.org/"
ELBENCH_DATASET_URL = "https://huggingface.co/datasets/ZeroLoss-Lab/ELBench"
ELBENCH_RESULTS_URL = "https://huggingface.co/datasets/ZeroLoss-Lab/ELBench-results"

PRESET_QUICK = "quick"
PRESET_AUTO = "auto"
PRESET_FULL = "full"
QUICK_CHECK_PER_TASK = 1

# Filter sliders: the right-hand end means "no limit".
PRICE_FILTER_MAX = 30.0
SIZE_FILTER_MAX = 1000.0
MODEL_SIZES = load_sizes()

FOCUS_CHOICES = [(label, key) for label, key in pareto_score_metric_choices() if not key.startswith("file:")]
TASK_CHOICES = [("None, use the area above", "")] + [
    (label, key) for label, key in pareto_score_metric_choices() if key.startswith("file:")
]


def _load_catalog() -> Catalog | None:
    try:
        return load_catalog(ROOT / "configs")
    except Exception as exc:  # noqa: BLE001
        _load_catalog.error = str(exc)  # type: ignore[attr-defined]
        return None


_load_catalog.error = ""  # type: ignore[attr-defined]
CATALOG = _load_catalog()
LEADERBOARD_STORE = LeaderboardStore(ROOT)
try:
    LEADERBOARD_STORE.sync_from_hub()
except Exception:  # noqa: BLE001
    pass


# ---------------------------------------------------------------------------
# OpenRouter catalog and prices (one cached snapshot per process)
# ---------------------------------------------------------------------------

CATALOG_SNAPSHOT_TTL_SECONDS = 6 * 3600
_SNAPSHOT: dict[str, Any] = {"value": None, "fetched_at": 0.0}
_SNAPSHOT_LOCK = threading.Lock()


def _nonnegative_price(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if price < 0:
        return None
    return price


def _fetch_model_catalog(api_key: str, url: str = OPENROUTER_MODELS_URL, timeout: float = 45) -> dict:
    headers = {"Accept": "application/json"}
    secret = (api_key or "").strip()
    if secret:
        headers["Authorization"] = f"Bearer {secret}"
    response = httpx.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("OpenRouter /api/v1/models did not return an object.")
    return payload


def _per_million(price: float | None) -> str:
    if price is None:
        return "?"
    value = price * 1_000_000
    return f"${value:.2f}" if value >= 0.1 or value == 0 else f"${value:.3f}"


def _parse_catalog_rows(payload: dict) -> tuple[list[dict[str, Any]], dict[str, tuple[float | None, float | None]]]:
    models: list[dict[str, Any]] = []
    prices: dict[str, tuple[float | None, float | None]] = {}
    for item in payload.get("data") or []:
        model_id = str(item.get("id") or "").strip()
        if not model_id:
            continue
        pricing = item.get("pricing") or {}
        prompt = _nonnegative_price(pricing.get("prompt"))
        completion = _nonnegative_price(pricing.get("completion"))
        prices[model_id] = (prompt, completion)
        models.append(
            {
                "id": model_id,
                "name": str(item.get("name") or model_id),
                "prompt": prompt,
                "completion": completion,
                "hugging_face_id": item.get("hugging_face_id") or None,
            }
        )
    models.sort(key=lambda m: m["name"].lower())
    return models, prices


def _model_choices(models: list[dict[str, Any]]) -> list[tuple[str, str]]:
    choices = []
    for m in models:
        if m["prompt"] is None:
            price = "no price listed"
        else:
            price = f"{_per_million(m['prompt'])} in / {_per_million(m['completion'])} out per 1M tokens"
        size = model_size(m["id"], m.get("hugging_face_id"), MODEL_SIZES)
        choices.append((f"{m['name']}  ·  {price}" + (f"  ·  {size.label()}" if size else ""), m["id"]))
    return choices


def _blended_per_1m(prompt: float | None, completion: float | None) -> float | None:
    """The chart's default price measure: 3 parts input to 1 part output, per million tokens."""
    if prompt is None or completion is None:
        return None
    return (3 * prompt + completion) / 4 * 1_000_000


def passes_filters(
    price_per_1m: float | None, size: ModelSize | None, max_price, max_size, include_undisclosed
) -> bool:
    if max_price is not None and float(max_price) < PRICE_FILTER_MAX:
        if price_per_1m is None or price_per_1m > float(max_price):
            return False
    if size is None:
        return bool(include_undisclosed)
    return max_size is None or float(max_size) >= SIZE_FILTER_MAX or size.total_b <= float(max_size)


def _size_of(slug: str, snapshot: dict[str, Any]) -> ModelSize | None:
    hf_ids = snapshot.get("hf_ids") or {}
    return model_size(slug, hf_ids.get(slug) or hf_ids.get(catalog_slug(slug)), MODEL_SIZES)


def _filtered_chat_models(snapshot: dict[str, Any], max_price, max_size, include_undisclosed) -> list[dict[str, Any]]:
    return [
        m
        for m in snapshot["chat_models"]
        if passes_filters(
            _blended_per_1m(m["prompt"], m["completion"]),
            model_size(m["id"], m.get("hugging_face_id"), MODEL_SIZES),
            max_price,
            max_size,
            include_undisclosed,
        )
    ]


def _default_system_one_models() -> list[dict[str, Any]]:
    return [
        {"id": slug, "name": name, "prompt": price, "completion": 0.0, "hugging_face_id": None}
        for name, slug, price in DEFAULT_SYSTEM_ONE_MODELS
    ]


def _default_prices() -> dict[str, tuple[float | None, float | None]]:
    table: dict[str, tuple[float | None, float | None]] = seed_official_openrouter_prices()
    for _, slug, prompt_price in DEFAULT_SYSTEM_ONE_MODELS:
        table[slug] = (prompt_price, 0.0)
    return table


def openrouter_snapshot() -> dict[str, Any]:
    """Chat models, decision models, and prices from OpenRouter, cached for a few hours."""
    with _SNAPSHOT_LOCK:
        cached = _SNAPSHOT["value"]
        if cached is not None and time.time() - _SNAPSHOT["fetched_at"] < CATALOG_SNAPSHOT_TTL_SECONDS:
            return cached
        prices = _default_prices()
        try:
            chat_models, live_prices = _parse_catalog_rows(_fetch_model_catalog("", OPENROUTER_MODELS_URL, timeout=15))
        except Exception:  # noqa: BLE001
            if cached is not None:
                return cached
            return {
                "hf_ids": {},
                "chat_models": [],
                "decision_models": _default_system_one_models(),
                "prices": prices,
                "label": "Prices stored with the published results (OpenRouter could not be reached).",
                "live": False,
            }
        prices.update(live_prices)
        decision_models = _default_system_one_models()
        try:
            fetched, decision_prices = _parse_catalog_rows(
                _fetch_model_catalog("", OPENROUTER_DECISION_MODELS_URL, timeout=15)
            )
            if fetched:
                decision_models = fetched
                prices.update(decision_prices)
        except Exception:  # noqa: BLE001
            pass
        value = {
            "hf_ids": {m["id"]: m["hugging_face_id"] for m in chat_models if m["hugging_face_id"]},
            "chat_models": chat_models,
            "decision_models": decision_models,
            "prices": prices,
            "label": f"Prices from OpenRouter, {time.strftime('%d %b %Y %H:%M UTC', time.gmtime())}.",
            "live": True,
        }
        _SNAPSHOT.update(value=value, fetched_at=time.time())
        return value


def price_snapshot() -> tuple[dict[str, tuple[float | None, float | None]], str]:
    snapshot = openrouter_snapshot()
    return dict(snapshot["prices"]), snapshot["label"]


def _price_pair(prices: dict, model_id: str) -> tuple[float | None, float | None]:
    key = str(model_id or "").strip()
    raw = (prices or {}).get(key) or (prices or {}).get(catalog_slug(key))
    if not raw:
        return None, None
    return raw[0], raw[1]


def _routed_price(route: Route | None, fallback: tuple[float | None, float | None]) -> tuple[float | None, float | None]:
    """Price of the host the request goes to first, else the catalog price."""
    host = route.cheapest if route is not None else None
    if host is None:
        return fallback
    return host.prompt_usd_per_token, host.completion_usd_per_token


def _pick_preferred(choices: list[tuple[str, str]], preferred: str) -> str:
    slugs = [slug for _, slug in choices]
    if preferred in slugs or not slugs:
        return preferred
    return slugs[0]


def check_key(api_key: str) -> tuple[bool, str]:
    """Free call to OpenRouter's key endpoint: is the key valid, and how much credit is left."""
    try:
        response = httpx.get(OPENROUTER_KEY_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=15)
    except Exception:  # noqa: BLE001
        return True, "Could not check the key with OpenRouter just now; the run will tell you if it is rejected."
    if response.status_code in {401, 403}:
        return False, "OpenRouter did not accept this API key. Check that it was copied in full."
    if response.status_code != 200:
        return True, "Could not check the key with OpenRouter just now; the run will tell you if it is rejected."
    data = (response.json() or {}).get("data") or {}
    remaining = data.get("limit_remaining")
    if isinstance(remaining, (int, float)):
        return True, f"Key accepted. This key has ${remaining:,.2f} of credit left."
    return True, "Key accepted."


# ---------------------------------------------------------------------------
# Run scope
# ---------------------------------------------------------------------------


def _module_choices() -> list[tuple[str, str]]:
    if CATALOG is None:
        return []
    return [(area_label(m), m) for m in CATALOG.modules()]


def _file_label(entry: CatalogFile) -> str:
    grading = "graded by the judge model" if entry.scoring_mode in {"llm", "runtime"} else "checked automatically"
    return f"{area_label(entry.module)} · {plain_file(entry.canonical_name)} · {entry.item_count} questions, {grading}"


def _file_choices(files: list[CatalogFile]) -> list[tuple[str, str]]:
    return [(_file_label(entry), entry.canonical_name) for entry in files]


def _dimension_choices(files: list[CatalogFile]) -> list[tuple[str, str]]:
    seen: list[str] = []
    for entry in files:
        for dimension in entry.dimensions:
            if dimension not in seen:
                seen.append(dimension)
    return [(english_dimension(d), d) for d in seen]


def _selected_entries(names: list[str] | None) -> list[CatalogFile]:
    if CATALOG is None:
        return []
    wanted = set(names or [])
    return [entry for entry in CATALOG.files if entry.canonical_name in wanted]


def _scope(include_china: bool) -> str:
    return SCOPE_PAPER if include_china else SCOPE_CORE


def _in_scope(files: list[CatalogFile], include_china: bool) -> list[CatalogFile]:
    if include_china:
        return files
    return [entry for entry in files if entry.canonical_name not in CHINA_SPECIFIC_FILES]


def _preset_files(preset: str, include_china: bool) -> list[CatalogFile]:
    if CATALOG is None:
        return []
    if preset == PRESET_AUTO:
        return _in_scope(CATALOG.default_files(), include_china)
    return _in_scope(list(CATALOG.files), include_china)


def _preset_choices(include_china: bool = False) -> list[tuple[str, str]]:
    if CATALOG is None:
        return []
    quick_items = len(_preset_files(PRESET_QUICK, include_china)) * QUICK_CHECK_PER_TASK
    auto_items = sum(entry.item_count for entry in _preset_files(PRESET_AUTO, include_china))
    full_items = sum(entry.item_count for entry in _preset_files(PRESET_FULL, include_china))
    return [
        (f"Quick check · {quick_items} questions, one from each task, to see that everything works", PRESET_QUICK),
        (f"Automatically checked set · {auto_items:,} questions, no judge model needed", PRESET_AUTO),
        (f"Complete benchmark · all {full_items:,} questions · appears on the leaderboard chart", PRESET_FULL),
    ]


def apply_preset(preset: str, include_china: bool = False):
    if CATALOG is None:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    files = _preset_files(preset, include_china)
    modules: list[str] = []
    for entry in files:
        if entry.module not in modules:
            modules.append(entry.module)
    return (
        gr.update(value=modules),
        gr.update(value=[entry.canonical_name for entry in files]),
        gr.update(choices=_dimension_choices(files), value=[]),
        gr.update(value=None),
        gr.update(value=QUICK_CHECK_PER_TASK if preset == PRESET_QUICK else None),
    )


def on_include_china(preset: str, include_china: bool):
    return (gr.update(choices=_preset_choices(include_china), value=preset), *apply_preset(preset, include_china))


def on_modules(modules: list[str]):
    if CATALOG is None:
        return gr.update(), gr.update()
    chosen = CATALOG.files_in_modules(set(modules or []))
    return (
        gr.update(value=[entry.canonical_name for entry in chosen]),
        gr.update(choices=_dimension_choices(chosen), value=[]),
    )


def on_files(files: list[str]):
    return gr.update(choices=_dimension_choices(_selected_entries(files)), value=[])


def _max_samples(value) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _budget(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


# ---------------------------------------------------------------------------
# Model pickers
# ---------------------------------------------------------------------------


def _models_shown_note(shown: int, total: int) -> str:
    return f"Showing {shown:,} of {total:,} OpenRouter models." if total else ""


def load_model_pickers(use_system_one: bool, max_price, max_size, include_undisclosed):
    snapshot = openrouter_snapshot()
    chat = _model_choices(snapshot["chat_models"])
    targets = _model_choices(_filtered_chat_models(snapshot, max_price, max_size, include_undisclosed))
    judge_choices = _model_choices(snapshot["decision_models"]) if use_system_one else chat
    judge_default = DEFAULT_SYSTEM_ONE_JUDGE_SLUG if use_system_one else DEFAULT_CHAT_JUDGE_SLUG
    return (
        gr.update(choices=targets, value=_pick_preferred(targets, DEFAULT_TARGET_SLUG)),
        gr.update(choices=judge_choices, value=_pick_preferred(judge_choices, judge_default)),
        gr.update(choices=chat, value=_pick_preferred(chat, DEFAULT_MULTITURN_CHAT_SLUG)),
        dict(snapshot["prices"]),
        _models_shown_note(len(targets), len(chat)),
    )


def filter_target_models(current, max_price, max_size, include_undisclosed):
    snapshot = openrouter_snapshot()
    targets = _model_choices(_filtered_chat_models(snapshot, max_price, max_size, include_undisclosed))
    slugs = [slug for _, slug in targets]
    value = current if current in slugs else None
    return (
        gr.update(choices=targets, value=value),
        _models_shown_note(len(targets), len(snapshot["chat_models"])),
    )


def on_toggle_system_one(use_system_one: bool):
    snapshot = openrouter_snapshot()
    if use_system_one:
        choices = _model_choices(snapshot["decision_models"])
        return gr.update(
            label="Judge (decision model)",
            choices=choices,
            value=_pick_preferred(choices, DEFAULT_SYSTEM_ONE_JUDGE_SLUG),
            info="Decision models return a verdict with no written output, which is much cheaper per answer.",
        )
    choices = _model_choices(snapshot["chat_models"])
    return gr.update(
        label="Judge model",
        choices=choices,
        value=_pick_preferred(choices, DEFAULT_CHAT_JUDGE_SLUG),
        info="Grades open-ended answers. The default is the closest OpenRouter match to the paper's judge.",
    )


def on_toggle_multiturn(use_multiturn: bool):
    return gr.update(visible=bool(use_multiturn), interactive=bool(use_multiturn))


# ---------------------------------------------------------------------------
# Estimate and run
# ---------------------------------------------------------------------------


def _usd(value: float | None) -> str:
    if value is None:
        return "?"
    if value >= 0.995:
        return f"${value:,.2f}"
    if value >= 0.01:
        return f"${value:.2f}"
    if value == 0:
        return "$0"
    return "$" + f"{value:.6f}".rstrip("0")


def _make_plan(
    *,
    model_id,
    judge_model_id,
    use_system_one_judge,
    use_multiturn_chat,
    multiturn_model_id,
    modules,
    files,
    dimensions,
    max_samples,
    per_task,
    concurrency,
    prices,
) -> Plan:
    slugs = {
        "target": str(model_id).strip(),
        "judge": "" if use_system_one_judge else str(judge_model_id).strip(),
        "student": str(multiturn_model_id or "").strip() if use_multiturn_chat else "",
    }
    routes = {role: resolve_route(slug) for role, slug in slugs.items() if slug}
    prompt_price, completion_price = _routed_price(routes.get("target"), _price_pair(prices, slugs["target"]))
    judge_prompt_price, judge_completion_price = _routed_price(
        routes.get("judge"), _price_pair(prices, str(judge_model_id).strip())
    )
    mt_prompt_price, mt_completion_price = _routed_price(
        routes.get("student"), _price_pair(prices, str(multiturn_model_id or "").strip())
    )
    return build_plan(
        CATALOG,
        openrouter_model=str(model_id).strip(),
        modules=set(modules or []),
        source_files=set(files or []),
        dimensions=set(dimensions or []),
        max_samples=_max_samples(max_samples),
        max_samples_per_file=_max_samples(per_task),
        prompt_usd_per_token=prompt_price,
        completion_usd_per_token=completion_price,
        max_concurrency=int(concurrency),
        judge_openrouter_model=str(judge_model_id).strip(),
        use_system_one_judge=bool(use_system_one_judge),
        judge_prompt_usd_per_token=judge_prompt_price,
        judge_completion_usd_per_token=judge_completion_price,
        use_multiturn_chat_model=bool(use_multiturn_chat),
        multiturn_openrouter_model=str(multiturn_model_id or "").strip() if use_multiturn_chat else None,
        multiturn_prompt_usd_per_token=mt_prompt_price,
        multiturn_completion_usd_per_token=mt_completion_price,
        routes=routes,
    )


def _plan_board(plan: Plan) -> str | None:
    """The judge board this plan's run will be ranked on."""
    if not (plan.judge_calls or plan.basic_tasks) or plan.judge_model_config is None:
        return None
    return judge_board(
        {"judge_model": plan.judge_model_config.model_name, "use_system_one_judge": plan.use_system_one_judge}
    )


def budget_problems(plan: Plan, total: float | None, per_role: dict[str, float | None]) -> list[str]:
    """Limits that the lowest estimate already exceeds, so the run could not finish."""
    problems: list[str] = []
    if total is not None and plan.low_usd is not None and plan.low_usd > total:
        problems.append(
            f"Your spending limit ({_usd(total)}) is below the lowest estimate for this run "
            f"({_usd(plan.low_usd)}). Raise the limit or choose fewer questions."
        )
    for role, cap in per_role.items():
        low = (plan.component_usd.get(role) or (None, None))[0]
        if cap is not None and low is not None and low > cap:
            problems.append(
                f"The limit for the {ROLE_LABELS[role]} ({_usd(cap)}) is below its lowest estimate ({_usd(low)})."
            )
    return problems


def plain_estimate(
    plan: Plan,
    *,
    total_budget: float | None,
    role_budgets: dict[str, float | None],
    key_note: str,
    resumable: dict | None,
) -> str:
    lines: list[str] = [f"*{key_note}*", ""] if key_note else []
    errors = plan.error_messages()
    if errors:
        lines.append("**This run can't start yet:**")
        lines.extend(f"- {message}" for message in errors)
        return "\n".join(lines)

    questions = plan.standard_requests + plan.basic_tasks
    lines.append(f"### {questions:,} questions for `{plan.protocol.openrouter_model}`")
    graded = plan.judge_calls + plan.basic_tasks
    if graded and plan.judge_model_config is not None and graded >= questions:
        lines.append(f"Every answer is graded by `{plan.judge_model_config.model_name}`.")
    elif graded and plan.judge_model_config is not None:
        lines.append(
            f"{graded:,} answers are graded by `{plan.judge_model_config.model_name}`; "
            "the rest are checked automatically against the answer key."
        )
    else:
        lines.append("Every answer is checked automatically against the answer key.")
    board = _plan_board(plan)
    if board is not None and board != PAPER_JUDGE_SLUG:
        lines.append("")
        lines.append(
            f"**Your own leaderboard:** results graded by `{plan.judge_model_config.model_name}` are ranked with "
            "other runs that used the same judge (pick it under *Graded by* on the chart), not with the published "
            f"results. Keep the default judge, {PAPER_JUDGE_LABEL}, to be ranked next to them."
        )
    lines.append("")

    if plan.low_usd is None or plan.high_usd is None:
        lines.append(
            "**No cost estimate:** OpenRouter lists no price for this model. To run it anyway, tick "
            "*Run even though this model has no listed price* under Advanced settings."
        )
    else:
        lines.append(
            f"**Estimated cost: {_usd(plan.low_usd)} to {_usd(plan.high_usd)}.** The low end counts the "
            "questions only; the high end assumes every answer runs to the length limit, so real runs "
            "land well below it."
            + (
                f" The judge has no length limit, so its high end assumes {JUDGE_HIGH_OUTPUT_TOKENS:,} tokens "
                "per grade, thinking included."
                if "judge" in plan.component_usd and not plan.use_system_one_judge
                else ""
            )
        )
        for role in ("target", "judge"):
            if role in plan.component_usd:
                low, high = plan.component_usd[role]
                cap = role_budgets.get(role)
                limit = f" · limit {_usd(cap)}" if cap is not None else ""
                lines.append(f"- {ROLE_LABELS[role].capitalize()}: {_usd(low)} to {_usd(high)}{limit}")
        if role_budgets.get("student") is not None:
            lines.append(f"- Simulated student: not estimated · limit {_usd(role_budgets['student'])}")
        if plan.basic_tasks:
            lines.append(
                "- Teaching tasks are multi-turn conversations whose length isn't known in advance; "
                "their judge and simulated-student calls are not in this estimate but do count toward your limits."
            )
    lines.append("")

    if plan.routes:
        lines.append(
            "**Hosts:** each model runs at the highest precision any OpenRouter host offers, on the cheapest "
            "such host. If that host fails, the next cheapest at the same precision takes over."
        )
        names = {"target": "Model under test", "judge": "Judge", "student": "Simulated student"}
        lines.extend(f"- {names[role]}: {route.describe()}" for role, route in plan.routes.items())
        lines.append("")

    problems = budget_problems(plan, total_budget, role_budgets)
    if problems:
        lines.append("**Spending limit too low:**")
        lines.extend(f"- {problem}" for problem in problems)
    elif total_budget is None:
        lines.append("No spending limit is set. The run continues until it finishes or your key runs out of credit.")
    elif plan.high_usd is not None and plan.high_usd <= total_budget:
        lines.append(f"Your limit of {_usd(total_budget)} covers even the high estimate.")
    else:
        lines.append(
            f"**Spending limit {_usd(total_budget)}.** If spending reaches it, the run pauses and keeps every "
            "finished answer. Press **Run** again with a higher limit to carry on from where it stopped."
        )
    if total_budget is not None or any(cap is not None for cap in role_budgets.values()):
        lines.append(
            "Questions already sent when a limit is reached still finish, so the final total can go slightly over it."
        )
    lines.append("")

    if plan.covers_core:
        lines.append("This is a complete run: once it finishes, it appears on the leaderboard chart next to the published models.")
        if not plan.is_full_benchmark:
            lines.append(
                "It leaves out the two region-specific sets, so it is scored like the chart's default view. "
                "Tick *Include the region-specific sets* to also appear in the paper's original scoring."
            )
    else:
        lines.append(
            "This is a partial run. Its results are saved with the visitor runs, but the chart only shows "
            "models scored on every question in an area, so the comparison stays fair."
        )
    if plan.protocol.openrouter_model.endswith(":free"):
        lines.append("Free models on OpenRouter are limited to about 20 requests a minute, so this will be slow.")
    if resumable is not None:
        lines.append(
            f"\nAn earlier run with these exact settings stopped after {resumable.get('total_judged', 0)} "
            "answers. Running again picks up where it left off."
        )
    return "\n".join(lines)


def _role_budgets(budget_target, budget_judge, budget_student) -> dict[str, float | None]:
    return {"target": _budget(budget_target), "judge": _budget(budget_judge), "student": _budget(budget_student)}


def estimate(
    api_key,
    model_id,
    judge_model_id,
    use_system_one_judge,
    use_multiturn_chat,
    multiturn_model_id,
    modules,
    files,
    dimensions,
    max_samples,
    per_task,
    concurrency,
    prices,
    total_budget,
    budget_target,
    budget_judge,
    budget_student,
):
    if CATALOG is None:
        return f"The benchmark data did not load: {_load_catalog.error}", ""
    secret = (api_key or "").strip()
    if not secret:
        return f"Paste your OpenRouter API key first. You can create one at [openrouter.ai]({OPENROUTER_KEYS_PAGE}).", ""
    if not str(model_id or "").strip():
        return "Choose the model you want to test.", ""
    if not str(judge_model_id or "").strip():
        return "Choose a judge model under Advanced settings.", ""
    key_ok, key_note = check_key(secret)
    if not key_ok:
        return key_note, ""
    try:
        with visitor_api_key(secret):
            plan = _make_plan(
                model_id=model_id,
                judge_model_id=judge_model_id,
                use_system_one_judge=use_system_one_judge,
                use_multiturn_chat=use_multiturn_chat,
                multiturn_model_id=multiturn_model_id,
                modules=modules,
                files=files,
                dimensions=dimensions,
                max_samples=max_samples,
                per_task=per_task,
                concurrency=concurrency,
                prices=prices,
            )
    except Exception as exc:  # noqa: BLE001
        return public_error(exc, secret), ""
    resumable = LEADERBOARD_STORE.find_resumable_run(config_hash=plan.config_hash)
    summary = plain_estimate(
        plan,
        total_budget=_budget(total_budget),
        role_budgets=_role_budgets(budget_target, budget_judge, budget_student),
        key_note=key_note,
        resumable=resumable,
    )
    return summary, render_plan(plan)


def _spent_so_far(run_id: str) -> float:
    if CATALOG is None or not run_id:
        return 0.0
    path = CATALOG.config.project_root / "outputs" / "hosted" / run_id / "spend.jsonl"
    if not path.exists():
        return 0.0
    return sum(SpendLedger(path, SpendLimits()).spent().values())


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def _area_scores_line(norm: dict) -> str:
    parts = []
    for module, key in (("通用模型", "general_pct"), ("安全可信", "safety_pct"), ("基本教育", "basic_education_pct"), ("高阶育人", "high_level_pct")):
        value = norm.get(key)
        if isinstance(value, (int, float)):
            parts.append(f"{area_label(module)} **{value:.1f}**")
    return " · ".join(parts)


def run_evaluation(
    api_key,
    model_id,
    judge_model_id,
    use_system_one_judge,
    use_multiturn_chat,
    multiturn_model_id,
    modules,
    files,
    dimensions,
    max_samples,
    per_task,
    concurrency,
    prices,
    total_budget,
    budget_target,
    budget_judge,
    budget_student,
    confirmed,
    accept_unpriced,
    resume_previous=True,
    persist_to_leaderboard=True,
):
    empty_table: list[list[Any]] = []
    if CATALOG is None:
        yield f"The benchmark data did not load: {_load_catalog.error}", empty_table, None
        return
    secret = (api_key or "").strip()
    if not secret:
        yield "Paste your OpenRouter API key first.", empty_table, None
        return
    if not confirmed:
        yield "Tick the box above to confirm the run can be charged to your OpenRouter key.", empty_table, None
        return
    if not str(model_id or "").strip():
        yield "Choose the model you want to test.", empty_table, None
        return
    if not str(judge_model_id or "").strip():
        yield "Choose a judge model under Advanced settings.", empty_table, None
        return
    try:
        sample_cap = _max_samples(max_samples)
        worker_concurrency = int(concurrency)
    except (TypeError, ValueError) as exc:
        yield public_error(exc, secret), empty_table, None
        return
    total_limit = _budget(total_budget)
    role_limits = _role_budgets(budget_target, budget_judge, budget_student)

    holder: dict[str, object] = {"status": "starting", "done": False, "error": "", "result": None, "run_id": ""}
    model_holder: dict[str, str] = {}

    def _worker() -> None:
        try:
            with visitor_api_key(secret):
                plan = _make_plan(
                    model_id=model_id,
                    judge_model_id=judge_model_id,
                    use_system_one_judge=use_system_one_judge,
                    use_multiturn_chat=use_multiturn_chat,
                    multiturn_model_id=multiturn_model_id,
                    modules=modules,
                    files=files,
                    dimensions=dimensions,
                    max_samples=sample_cap,
                    per_task=per_task,
                    concurrency=worker_concurrency,
                    prices=prices,
                )
                model_holder["model_id"] = plan.protocol.model_config.model_id
                problems = plan.error_messages() + budget_problems(plan, total_limit, role_limits)
                if plan.needs_price_ack and not accept_unpriced:
                    problems.append(
                        "OpenRouter lists no price for this model. Tick *Run even though this model has no "
                        "listed price* under Advanced settings to run it anyway."
                    )
                if problems:
                    holder["error"] = "**This run can't start yet:**\n" + "\n".join(f"- {p}" for p in problems)
                    return
                resumable = LEADERBOARD_STORE.find_resumable_run(config_hash=plan.config_hash) if resume_previous else None
                active_run_id = (
                    str(resumable["run_id"])
                    if resumable is not None and resumable.get("run_id")
                    else make_run_id(plan.config_hash)
                )
                holder["run_id"] = active_run_id
                holder["status"] = "Continuing" if resumable is not None else "Running"
                holder["result"] = execute_run(
                    CATALOG,
                    plan,
                    max_concurrency=worker_concurrency,
                    accept_unpriced=bool(accept_unpriced),
                    run_id=active_run_id,
                    resume_previous=bool(resume_previous),
                    leaderboard_store=LEADERBOARD_STORE,
                    persist_to_leaderboard=bool(persist_to_leaderboard),
                    total_budget_usd=total_limit,
                    role_budgets_usd=role_limits,
                )
        except Exception as exc:  # noqa: BLE001
            holder["error"] = public_error(exc, secret)
        finally:
            holder["done"] = True

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    while not holder["done"]:
        model_key = model_holder.get("model_id", "")
        active_id = str(holder.get("run_id") or "")
        judged = failed = 0
        if model_key and active_id:
            judged = _count_lines(judged_progress_path(CATALOG.config.project_root, active_id, model_key))
            failed = _count_lines(failures_progress_path(CATALOG.config.project_root, active_id, model_key))
        spent = _spent_so_far(active_id)
        limit = f" of your {_usd(total_limit)} limit" if total_limit is not None else ""
        yield (
            f"**{holder['status']}…** {judged:,} answers scored"
            + (f", {failed:,} hit errors" if failed else "")
            + f". Spent so far: {_usd(spent)}{limit}. Every finished answer is saved as it completes.",
            empty_table,
            None,
        )
        time.sleep(1)
    thread.join(timeout=1)

    if holder.get("error"):
        yield str(holder["error"]), empty_table, None
        return
    result = holder.get("result") or {}
    summary = result.get("summary") or {}
    norm = result.get("normalized_scores") or {}
    judged_total = int(summary.get("total_judged") or 0)
    failures = int(summary.get("total_failures") or 0)
    status_label = str(result.get("status") or "completed")
    spent_text = _usd(float(result.get("spent_usd") or 0.0))

    if result.get("stopped_by_budget"):
        heading = "### Paused at your spending limit"
        body = (
            f"{judged_total:,} answers scored for {spent_text}. Everything finished so far is saved. "
            "Raise the limit and press **Run** again to carry on from here."
        )
    elif status_label == "completed":
        heading = "### Finished"
        body = f"{judged_total:,} answers scored for {spent_text}."
    elif status_label == "interrupted":
        heading = "### Stopped early"
        body = (
            f"{judged_total:,} answers scored for {spent_text} before the run stopped"
            + (f": {result['interrupted_error']}" if result.get("interrupted_error") else ".")
            + " Press **Run** again to continue."
        )
    else:
        heading = "### Finished with some errors"
        body = (
            f"{judged_total:,} answers scored for {spent_text}; {failures:,} hit errors, usually rate limits "
            "or timeouts. Press **Run** again to retry only those."
        )
    scores = _area_scores_line(norm)
    lines = [heading, "", body]
    if scores:
        lines += ["", f"Scores on the questions that ran: {scores}"]
        if not (result.get("leaderboard_record") or {}).get("covers_core"):
            lines[-1] += ". A partial run is too small to compare with the leaderboard."
    if persist_to_leaderboard:
        lines += ["", "Saved to the visitor-run leaderboard."]
        record = result.get("leaderboard_record") or {}
        if status_label == "completed" and record.get("covers_core"):
            board = judge_board(record)
            where = (
                "the leaderboard chart"
                if board in (None, PAPER_JUDGE_SLUG)
                else f"the leaderboard chart under *Graded by {judge_board_label(board)}*"
            )
            lines[-1] += f" It now appears on {where}; press **Refresh** there to see it."
    else:
        lines += ["", "Not saved to the leaderboard (you opted out)."]
    lines += ["", f"<sub>Run id `{result.get('run_id')}`</sub>"]
    yield "\n".join(lines), result.get("summary_rows") or empty_table, result.get("download_paths")


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def _metric(focus: str, single_task: str) -> str:
    return single_task or focus or "overall"


def leaderboard_view(
    focus,
    single_task,
    cost_metric,
    include_visitor_runs,
    max_price,
    max_size,
    include_undisclosed,
    judge,
    include_china,
    prices,
):
    runs = LEADERBOARD_STORE.load_runs()
    snapshot = openrouter_snapshot()
    merged, label = price_snapshot()
    if isinstance(prices, dict):
        merged.update(prices)

    def keep(candidate: dict[str, Any]) -> bool:
        return passes_filters(
            candidate["cost"], _size_of(candidate["slug"], snapshot), max_price, max_size, include_undisclosed
        )

    fig, rows, md = build_pareto_frontier(
        runs,
        merged,
        score_metric=_metric(focus, single_task),
        cost_metric=cost_metric or "blended_per_1m",
        include_custom_runs=bool(include_visitor_runs),
        keep=keep,
        scope=_scope(include_china),
        judge=judge or PAPER_JUDGE_SLUG,
    )
    return fig, rows, f"{md}\n\n<sub>{_scope_note(include_china)} {label}</sub>"


def _scope_note(include_china: bool) -> str:
    if include_china:
        return (
            f"Paper's original scoring: all {scope_item_count(SCOPE_PAPER):,} questions, including the two "
            "Region-specific sets."
        )
    return (
        f"Scored on {scope_item_count(SCOPE_CORE):,} questions, without the two region-specific sets "
        "(see About). Published 95% ranges cover the paper's original scoring only, so none are shown here."
    )


def scope_tables(include_china, official_model):
    scope = _scope(include_china)
    runs = LEADERBOARD_STORE.load_runs()
    return (
        official_leaderboard_rows(scope),
        official_per_file_rows(official_model, scope),
        custom_leaderboard_rows(runs, scope),
    )


def refresh_leaderboards(
    focus,
    single_task,
    cost_metric,
    include_visitor_runs,
    max_price,
    max_size,
    include_undisclosed,
    judge,
    include_china,
    primary_sel,
    baseline_sels,
    granularity,
    prices,
):
    runs = LEADERBOARD_STORE.sync_from_hub()
    scope = _scope(include_china)
    fig, rows, md = leaderboard_view(
        focus,
        single_task,
        cost_metric,
        include_visitor_runs,
        max_price,
        max_size,
        include_undisclosed,
        judge,
        include_china,
        prices,
    )
    primary_choices, baseline_choices = build_comparison_choices(runs, scope)
    keys = [value for _, value in primary_choices]
    resolved_primary = primary_sel if primary_sel in keys else (keys[0] if keys else None)
    resolved_baselines = [b for b in (baseline_sels or []) if b in keys] or [
        key for key in ("official:deepseek-v4-flash", "official:gpt-5.4") if key in keys and key != resolved_primary
    ]
    headers, comp_rows, comp_note = build_comparison_table(
        resolved_primary, resolved_baselines, granularity, runs, scope
    )
    return (
        fig,
        rows,
        md,
        custom_leaderboard_rows(runs, scope),
        f"{len(runs)} visitor run(s). {LEADERBOARD_STORE.sync_status}",
        _judge_board_update(runs, judge),
        gr.update(choices=primary_choices, value=resolved_primary),
        gr.update(choices=baseline_choices, value=resolved_baselines),
        gr.Dataframe(headers=headers, value=comp_rows, wrap=True),
        comp_note,
    )


def _judge_board_update(runs: list[dict], judge: str | None):
    choices = judge_board_choices(runs)
    keys = [key for _, key in choices]
    return gr.update(choices=choices, value=judge if judge in keys else PAPER_JUDGE_SLUG)


def rescope_leaderboards(
    focus,
    single_task,
    cost_metric,
    include_visitor_runs,
    max_price,
    max_size,
    include_undisclosed,
    judge,
    include_china,
    primary_sel,
    baseline_sels,
    granularity,
    official_model,
    prices,
):
    """Switch every leaderboard view between the default scope and the paper's original scoring."""
    scope = _scope(include_china)
    runs = LEADERBOARD_STORE.load_runs()
    fig, rows, md = leaderboard_view(
        focus,
        single_task,
        cost_metric,
        include_visitor_runs,
        max_price,
        max_size,
        include_undisclosed,
        judge,
        include_china,
        prices,
    )
    primary_choices, baseline_choices = build_comparison_choices(runs, scope)
    headers, comp_rows, comp_note = build_comparison_table(primary_sel, baseline_sels, granularity, runs, scope)
    official_rows, official_files, visitor_rows = scope_tables(include_china, official_model)
    return (
        fig,
        rows,
        md,
        official_rows,
        official_files,
        visitor_rows,
        gr.update(choices=primary_choices, value=primary_sel),
        gr.update(choices=baseline_choices, value=baseline_sels),
        gr.Dataframe(headers=headers, value=comp_rows, wrap=True),
        comp_note,
    )


def update_comparison(primary_sel, baseline_sels, granularity, include_china=False):
    headers, comp_rows, comp_note = build_comparison_table(
        primary_sel, baseline_sels, granularity, LEADERBOARD_STORE.load_runs(), _scope(include_china)
    )
    return gr.Dataframe(headers=headers, value=comp_rows, wrap=True), comp_note


def _about_markdown() -> str:
    counts: dict[str, int] = {}
    if CATALOG is not None:
        for entry in _in_scope(list(CATALOG.files), False):
            counts[entry.module] = counts.get(entry.module, 0) + entry.item_count
    rows = "\n".join(
        f"| **{label}** | {counts.get(module, 0):,} | {description} |"
        for module, (label, description) in AREA_LABELS.items()
    )
    return f"""## What ELBench measures

ELBench asks a model the kinds of things a teacher, tutor or school assistant has to get right, and scores
the answers in four areas. A single overall number never tells the whole story, so every result is shown
area by area too.

| Area | Questions | What it covers |
|:-----|:---------:|:---------------|
{rows}

## Two region-specific sets, left out by default

ELBench was developed in the People's Republic of China. Two of its sets test agreement with region-specific rules,
which are excluded by default. Enable them by ticking the *Include the region-specific sets* option.
*Include the region-specific sets*:

- **Refusal under region-specific content rules** (250 safety questions). Only regionally aligned models are 
  expected to refuse 212 of the questions
- **C-Eval** (208 knowledge questions) from region-specific national exams reflecting core values.

Without them the benchmark has {scope_item_count(SCOPE_CORE):,} questions. With them it has all
{scope_item_count(SCOPE_PAPER):,}. The published results are given per task,
so a whole task is the smallest part that can be removed while still re-scoring the published models.

## How the scores work

- Each area is scored from 0 to 100. **Overall** is the average of the four areas.
- Questions with a single right answer are checked automatically. Open-ended answers (safety guidance,
  teaching, character and thinking skills) are graded by a judge model against a rubric.
- The published results include a 95% range from 10,000 bootstrap resamples, for the paper's original scoring
  only. When two models' ranges overlap, the difference between them could disappear on a re-run, and the
  chart says so.
- **Best value** models score higher than every model that costs less. Price is the OpenRouter price per
  million tokens, weighting input three times as heavily as output, a common way to compare models.

## Visitor runs

- Runs use the same questions, prompts, retries and scoring code as the paper.
- Models are called through OpenRouter. Each one runs at the highest precision any of its hosts offers
  (for example fp8 rather than fp4), on the cheapest such host. Models whose hosts don't say, such as most
  closed models, go to the cheapest host.
- The default judge is Qwen3-235B on OpenRouter. The paper used its own hosted Qwen judge, so judged scores
  from visitor runs can differ from the published ones.
- Judges disagree: the same answers can score very differently under two judge models. So each judge gets
  its own leaderboard (*Graded by* on the chart), and only runs graded by the default judge are ranked with
  the published results. Questions with a single right answer don't need a judge, so those scores carry
  across every board.
- Only runs that answered every question in an area appear on the chart for that area, so partial runs never
  compete with complete ones.

## Your key and your data

- Your OpenRouter key is used only while your run is going and is removed from every file the run writes.
  It is never saved or shown on the leaderboard.
- If you leave *Add my results to the public leaderboard* ticked, the model name, settings, scores and
  answers are published in a public Hugging Face dataset.

## Learn more

- [Paper (arXiv:2608.09548)]({ELBENCH_PAPER_URL}) · [Code]({ELBENCH_GITHUB_URL}) ·
  [Questions]({ELBENCH_DATASET_URL}) · [Published results and audits]({ELBENCH_RESULTS_URL}) ·
  [AI for Education leaderboard]({AI_FOR_EDUCATION_LEADERBOARD_URL})
"""


def _zerogpu_noop() -> str:
    """No-op handler so Hugging Face ZeroGPU passes its startup check without burning GPU quota."""
    return "ok"


if spaces is not None and hasattr(spaces, "GPU"):
    _zerogpu_noop = spaces.GPU(duration=1)(_zerogpu_noop)


def _filter_controls():
    with gr.Row():
        max_price = gr.Slider(
            label="Highest price",
            minimum=0.1,
            maximum=PRICE_FILTER_MAX,
            step=0.1,
            value=PRICE_FILTER_MAX,
            info="Dollars per million tokens. All the way right shows every price.",
        )
        max_size = gr.Slider(
            label="Largest model size",
            minimum=1,
            maximum=SIZE_FILTER_MAX,
            step=1,
            value=SIZE_FILTER_MAX,
            info="Billions of parameters. All the way right shows every size.",
        )
        include_undisclosed = gr.Checkbox(
            label="Include models that don't publish their size",
            value=True,
            info="Most closed models, such as GPT, Claude and Gemini, don't.",
        )
    return max_price, max_size, include_undisclosed


def build_demo() -> gr.Blocks:
    if CATALOG is None:
        with gr.Blocks(title="ELBench-live") as demo:
            gr.Markdown(f"ELBench-live data did not load.\n\n```\n{_load_catalog.error}\n```")
            with gr.Row(visible=False):
                _noop_btn = gr.Button("noop")
                _noop_out = gr.Textbox()
                _noop_btn.click(_zerogpu_noop, [], [_noop_out], api_name="healthz")
        return demo

    initial_runs = LEADERBOARD_STORE.load_runs()
    initial_prices, _ = price_snapshot()
    init_fig, init_rows, init_md = leaderboard_view(
        "overall", "", "blended_per_1m", True, PRICE_FILTER_MAX, SIZE_FILTER_MAX, True, PAPER_JUDGE_SLUG, False,
        initial_prices,
    )
    primary_choices, baseline_choices = build_comparison_choices(initial_runs, SCOPE_CORE)
    default_primary = primary_choices[0][1] if primary_choices else "official:gpt-5.4"
    default_baselines = [k for k in ("official:deepseek-v4-flash", "official:gpt-5.4") if k != default_primary]
    granularity_default = "By Source File / Task (Matched Subset)"
    init_comp_headers, init_comp_rows, init_comp_note = build_comparison_table(
        default_primary, default_baselines, granularity_default, initial_runs, SCOPE_CORE
    )
    default_files = _preset_files(PRESET_QUICK, False)

    with gr.Blocks(title="ELBench-live") as demo:
        prices = gr.State(initial_prices)

        with gr.Tabs() as tabs:
            with gr.Tab("Leaderboard", id="leaderboard"):
                gr.Markdown(
                    "# How well do AI models handle education?\n"
                    f"ELBench scores AI models on {scope_item_count(SCOPE_CORE):,} questions in four areas: "
                    "**teaching**, **safety**, **knowledge & reasoning**, and **character & thinking skills**. "
                    "Each dot is a model: higher scores better, further left costs less."
                )
                focus = gr.Radio(label="Area", choices=FOCUS_CHOICES, value="overall")
                include_china = gr.Checkbox(
                    label="Include the region-specific sets (the paper's original scoring)",
                    value=False,
                    info="Adds two sets that test region-specific content rules and national exams. See About.",
                )
                judge_board_pick = gr.Dropdown(
                    label="Graded by",
                    choices=judge_board_choices(initial_runs),
                    value=PAPER_JUDGE_SLUG,
                    info="Each judge model has its own leaderboard. The published results were graded by the "
                    "paper's judge, so they appear on the default one.",
                )
                chart_max_price, chart_max_size, chart_undisclosed = _filter_controls()
                chart = gr.Plot(value=init_fig, show_label=False)
                takeaway = gr.Markdown(value=init_md)
                with gr.Row():
                    go_test = gr.Button("Test your own model →", variant="primary")
                    refresh = gr.Button("Refresh", variant="secondary")

                with gr.Accordion("More ways to look at the results", open=False):
                    with gr.Row():
                        single_task = gr.Dropdown(
                            label="One task only",
                            choices=TASK_CHOICES,
                            value="",
                            filterable=True,
                            info="Show the scores for a single task instead of a whole area.",
                        )
                        cost_metric = gr.Dropdown(
                            label="Price measure",
                            choices=pareto_cost_metric_choices(),
                            value="blended_per_1m",
                        )
                        include_visitor_runs = gr.Checkbox(label="Show visitor runs", value=True)
                    value_table = gr.Dataframe(
                        headers=["", "Model", "Source", "OpenRouter id", "Input $/1M", "Output $/1M", "Price", "Score"],
                        value=init_rows,
                        wrap=True,
                        show_label=False,
                    )

                with gr.Accordion("Published results", open=False):
                    official_table = gr.Dataframe(
                        headers=[
                            "Rank",
                            "Model",
                            "Overall [95% range]",
                            "Knowledge & reasoning",
                            "Safety",
                            "Teaching",
                            "Character & thinking",
                            "Questions",
                        ],
                        value=official_leaderboard_rows(SCOPE_CORE),
                        wrap=True,
                        show_label=False,
                    )
                    official_model_filter = gr.Dropdown(
                        label="Task-by-task scores for",
                        choices=[("All models", "All Official Models")]
                        + [m["display_name"] for m in load_official_models()],
                        value="GPT-5.4",
                    )
                    official_file_table = gr.Dataframe(
                        headers=["Model", "Area", "Task", "Questions", "Score", "Pass rate (%)", "Mean raw score"],
                        value=official_per_file_rows("GPT-5.4", SCOPE_CORE),
                        wrap=True,
                        show_label=False,
                    )

                with gr.Accordion("Visitor runs", open=False):
                    runs_status = gr.Markdown(f"{len(initial_runs)} visitor run(s). {LEADERBOARD_STORE.sync_status}")
                    runs_table = gr.Dataframe(
                        headers=[
                            "Run id",
                            "Updated (UTC)",
                            "Model",
                            "Judge",
                            "Status",
                            "Scope",
                            "Answers",
                            "Errors",
                            "Overall",
                            "Knowledge & reasoning",
                            "Safety",
                            "Teaching",
                            "Character & thinking",
                        ],
                        value=custom_leaderboard_rows(initial_runs, SCOPE_CORE),
                        wrap=True,
                        show_label=False,
                    )

                with gr.Accordion("Compare results side by side", open=False):
                    with gr.Row():
                        comp_primary = gr.Dropdown(
                            label="Result", choices=primary_choices, value=default_primary, filterable=True
                        )
                        comp_baselines = gr.Dropdown(
                            label="Compare with",
                            choices=baseline_choices,
                            value=default_baselines,
                            multiselect=True,
                            filterable=True,
                        )
                        comp_granularity = gr.Radio(
                            label="Detail",
                            choices=[
                                ("Task by task (only tasks the first result covers)", "By Source File / Task (Matched Subset)"),
                                ("By area", "By Module (0–100 Normalized)"),
                            ],
                            value=granularity_default,
                        )
                    comp_note = gr.Markdown(value=init_comp_note)
                    comp_table = gr.Dataframe(
                        headers=init_comp_headers, value=init_comp_rows, wrap=True, show_label=False
                    )

            with gr.Tab("Test a model", id="test"):
                gr.Markdown(
                    "# Test a model on ELBench\n"
                    "Bring an [OpenRouter API key](" + OPENROUTER_KEYS_PAGE + "), pick a model, and see how it "
                    "does. You'll see the expected cost before anything is charged, and the run stops at your "
                    "spending limit."
                )
                api_key = gr.Textbox(
                    label="OpenRouter API key",
                    type="password",
                    placeholder="sk-or-...",
                    info="Used only for this run. Never saved or shown.",
                )
                model = gr.Dropdown(
                    label="Model to test",
                    choices=[],
                    value=DEFAULT_TARGET_SLUG,
                    allow_custom_value=True,
                    filterable=True,
                    info="Type to search, or paste any OpenRouter model id.",
                )
                with gr.Accordion("Narrow the list by price or size", open=False):
                    pick_max_price, pick_max_size, pick_undisclosed = _filter_controls()
                    models_shown = gr.Markdown()
                preset = gr.Radio(label="How much to run", choices=_preset_choices(False), value=PRESET_QUICK)
                run_include_china = gr.Checkbox(
                    label="Include the region-specific sets",
                    value=False,
                    info="Needed only to appear in the paper's original scoring. See About for details on region-specific sets.",
                )
                with gr.Row():
                    total_budget = gr.Number(
                        label="Spending limit (USD)",
                        value=DEFAULT_BUDGET_USD,
                        minimum=0,
                        info="The run pauses here and keeps its progress. Leave empty for no limit.",
                    )
                    persist_to_leaderboard = gr.Checkbox(
                        label="Add my results to the public leaderboard",
                        value=True,
                        info="Shares the model name, settings, answers and scores. Never your key.",
                    )

                with gr.Accordion("Advanced settings", open=False):
                    gr.Markdown("**Judge and simulated student**")
                    with gr.Row():
                        judge_model = gr.Dropdown(
                            label="Judge model",
                            choices=[],
                            value=DEFAULT_CHAT_JUDGE_SLUG,
                            allow_custom_value=True,
                            filterable=True,
                            info="Grades open-ended answers. The default is the closest OpenRouter match to the paper's "
                            "judge; runs graded by another judge get their own leaderboard.",
                        )
                        use_system_one_judge = gr.Checkbox(
                            label="Use a decision model as the judge",
                            value=False,
                            info="Cheaper verdict-only models (System 1). Not the paper's setup.",
                        )
                    with gr.Row():
                        use_multiturn_chat = gr.Checkbox(
                            label="Use a separate model as the simulated student",
                            value=False,
                            info="The tutoring task needs a student to talk to. By default the judge model plays the student.",
                        )
                        multiturn_model = gr.Dropdown(
                            label="Simulated student",
                            choices=[],
                            value=DEFAULT_MULTITURN_CHAT_SLUG,
                            visible=False,
                            interactive=False,
                            allow_custom_value=True,
                            filterable=True,
                        )
                    gr.Markdown("**Separate spending limits** (optional, in USD; they apply within the overall limit)")
                    with gr.Row():
                        budget_target = gr.Number(label="Model under test", value=None, minimum=0)
                        budget_judge = gr.Number(label="Judge", value=None, minimum=0)
                        budget_student = gr.Number(label="Simulated student", value=None, minimum=0)
                    gr.Markdown("**Questions**")
                    modules = gr.CheckboxGroup(
                        label="Areas",
                        choices=_module_choices(),
                        value=list(dict.fromkeys(entry.module for entry in default_files)),
                        info="Choosing an area selects all of its tasks.",
                    )
                    files = gr.CheckboxGroup(
                        label="Tasks",
                        choices=_file_choices(list(CATALOG.files)),
                        value=[entry.canonical_name for entry in default_files],
                    )
                    dimensions = gr.CheckboxGroup(
                        label="Only these skills (leave empty for all)",
                        choices=_dimension_choices(default_files),
                        value=[],
                    )
                    with gr.Row():
                        per_task = gr.Number(
                            label="Questions per task",
                            value=QUICK_CHECK_PER_TASK,
                            precision=0,
                            info="Takes the first questions of each task. Leave empty for all of them.",
                        )
                        max_samples = gr.Number(
                            label="Stop after this many questions",
                            value=None,
                            precision=0,
                            info="Leave empty to run every selected question.",
                        )
                        concurrency = gr.Slider(
                            label="Requests at once",
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=8,
                            info="Higher is faster but more likely to hit rate limits.",
                        )
                    with gr.Row():
                        resume_previous = gr.Checkbox(
                            label="Continue an unfinished run with the same settings", value=True
                        )
                        accept_unpriced = gr.Checkbox(
                            label="Run even though this model has no listed price", value=False
                        )

                estimate_button = gr.Button("Estimate cost", variant="secondary")
                report = gr.Markdown()
                with gr.Accordion("Technical details", open=False):
                    report_details = gr.Markdown()
                confirmed = gr.Checkbox(label="Charge my OpenRouter key for this run, up to my spending limit")
                run_button = gr.Button("Run", variant="primary")
                status = gr.Markdown()
                with gr.Accordion("Detailed results", open=False):
                    table = gr.Dataframe(
                        headers=["Group", "Name", "Count", "Mean score", "Pass rate"], wrap=True, show_label=False
                    )
                    download = gr.File(label="Download answers, scores and errors", file_count="multiple")

            with gr.Tab("About", id="about"):
                gr.Markdown(_about_markdown())

        # Leaderboard events
        view_controls = [
            focus,
            single_task,
            cost_metric,
            include_visitor_runs,
            chart_max_price,
            chart_max_size,
            chart_undisclosed,
            judge_board_pick,
            include_china,
        ]
        view_outputs = [chart, value_table, takeaway]
        for control in view_controls[:-1]:
            control.change(leaderboard_view, view_controls + [prices], view_outputs)
        include_china.change(
            rescope_leaderboards,
            view_controls + [comp_primary, comp_baselines, comp_granularity, official_model_filter, prices],
            view_outputs
            + [official_table, official_file_table, runs_table, comp_primary, comp_baselines, comp_table, comp_note],
        )
        refresh.click(
            refresh_leaderboards,
            view_controls + [comp_primary, comp_baselines, comp_granularity, prices],
            [
                chart,
                value_table,
                takeaway,
                runs_table,
                runs_status,
                judge_board_pick,
                comp_primary,
                comp_baselines,
                comp_table,
                comp_note,
            ],
        )
        go_test.click(lambda: gr.Tabs(selected="test"), None, tabs)
        official_model_filter.change(
            lambda model_name, china: official_per_file_rows(model_name, _scope(china)),
            [official_model_filter, include_china],
            [official_file_table],
        )
        for control in (comp_primary, comp_baselines, comp_granularity):
            control.change(
                update_comparison,
                [comp_primary, comp_baselines, comp_granularity, include_china],
                [comp_table, comp_note],
            )

        # Test-a-model events
        pick_filters = [pick_max_price, pick_max_size, pick_undisclosed]
        demo.load(
            load_model_pickers,
            [use_system_one_judge] + pick_filters,
            [model, judge_model, multiturn_model, prices, models_shown],
        )
        for control in pick_filters:
            control.change(filter_target_models, [model] + pick_filters, [model, models_shown])
        use_system_one_judge.change(on_toggle_system_one, [use_system_one_judge], [judge_model])
        use_multiturn_chat.change(on_toggle_multiturn, [use_multiturn_chat], [multiturn_model])
        preset.change(apply_preset, [preset, run_include_china], [modules, files, dimensions, max_samples, per_task])
        run_include_china.change(
            on_include_china,
            [preset, run_include_china],
            [preset, modules, files, dimensions, max_samples, per_task],
        )
        modules.input(on_modules, [modules], [files, dimensions])
        files.input(on_files, [files], [dimensions])
        run_inputs = [
            api_key,
            model,
            judge_model,
            use_system_one_judge,
            use_multiturn_chat,
            multiturn_model,
            modules,
            files,
            dimensions,
            max_samples,
            per_task,
            concurrency,
            prices,
            total_budget,
            budget_target,
            budget_judge,
            budget_student,
        ]
        estimate_button.click(estimate, run_inputs, [report, report_details])
        run_button.click(
            run_evaluation,
            run_inputs + [confirmed, accept_unpriced, resume_previous, persist_to_leaderboard],
            [status, table, download],
        )
        with gr.Row(visible=False):
            _noop_btn = gr.Button("noop")
            _noop_out = gr.Textbox()
            _noop_btn.click(_zerogpu_noop, [], [_noop_out], api_name="healthz")
    return demo


demo = build_demo()
# Spaces imports this module and calls launch() itself, so the queue has to be set here.
demo.queue(default_concurrency_limit=1)


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
