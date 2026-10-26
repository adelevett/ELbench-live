"""Selection caveats and a cost range for one visitor run.

Sample order, prompt text, and the token floor follow the runner:
``FileRegistry.resolve``, ``LoaderFactory``, ``augment_prompt_for_format``,
and ``len(prompt) // 4``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from elbench.execution.basic_education import (
    DEFAULT_BASIC_MODULE_NAME,
    DEFAULT_MULTI_TURN_RECURSION_LIMIT,
    BasicEducationExecutor,
    count_runtime_tasks,
)
from elbench.execution.preflight import PreflightRunner
from elbench.execution.response_format import augment_prompt_for_format
from elbench.execution.runner import BenchmarkRunner
from elbench.execution.selection import resolve_module_selection
from elbench.loaders import LoaderFactory
from elbench.registry import FileRegistry
from elbench.schemas.config import ProjectConfig
from elbench.schemas.evaluation import Sample

from elbench.dashboard.catalog import Catalog, CatalogFile
from elbench.dashboard.english import english_file, english_module, english_preflight_message
from elbench.dashboard.leaderboard import CHINA_SPECIFIC_FILES, compute_plan_fingerprint
from elbench.dashboard.protocol import (
    VISITOR_KEY_ENV,
    Caveat,
    VisitorProtocol,
    build_judge_model_config,
    build_multiturn_chat_model_config,
    build_visitor_protocol,
)
from elbench.dashboard.routing import Route
from elbench.schemas.config import ModelConfig
from elbench.utils.secrets import get_api_key


# High-end assumption for one judge grade, thinking included. Not sent as a limit.
JUDGE_HIGH_OUTPUT_TOKENS = 8192


@dataclass(frozen=True, slots=True)
class RunSelection:
    modules: frozenset[str]
    source_files: frozenset[str]
    dimensions: frozenset[str]
    max_samples: int | None
    max_samples_per_file: int | None = None


@dataclass(slots=True)
class Plan:
    protocol: VisitorProtocol
    selection: RunSelection
    caveats: list[Caveat]
    standard_requests: int
    standard_low_input_tokens: int
    standard_high_input_tokens: int
    standard_output_cap_tokens: int
    judge_calls: int
    judge_model_id: str | None
    judge_api_base: str | None
    basic_tasks: int
    basic_target_calls_low: int
    basic_target_calls_high: int
    max_tokens: int
    prompt_usd_per_token: float | None
    completion_usd_per_token: float | None
    low_usd: float | None
    high_usd: float | None
    preflight_ok: bool
    blocked: bool
    needs_price_ack: bool
    file_counts: list[tuple[str, int]] = field(default_factory=list)
    judge_model_config: ModelConfig | None = None
    use_system_one_judge: bool = False
    multiturn_model_config: ModelConfig | None = None
    use_multiturn_chat_model: bool = False
    config_hash: str = ""
    is_full_benchmark: bool = False
    # Every question outside the region-specific sets is selected (the default "core" scope).
    covers_core: bool = False
    judge_prices: tuple[float, float] | None = None
    student_prices: tuple[float, float] | None = None
    # (low, high) USD per component: "target", "judge". Missing means no estimate.
    component_usd: dict[str, tuple[float, float]] = field(default_factory=dict)
    # Host choice per role ("target", "judge", "student") that the request bodies carry.
    routes: dict[str, Route] = field(default_factory=dict)

    def error_messages(self) -> list[str]:
        return [caveat.message for caveat in self.caveats if caveat.level == "error"]


def build_plan(
    catalog: Catalog,
    *,
    openrouter_model: str,
    modules: set[str],
    source_files: set[str],
    dimensions: set[str] | None,
    max_samples: int | None,
    prompt_usd_per_token: float | None,
    completion_usd_per_token: float | None,
    max_concurrency: int,
    max_samples_per_file: int | None = None,
    judge_openrouter_model: str | None = None,
    use_system_one_judge: bool = False,
    judge_prompt_usd_per_token: float | None = None,
    judge_completion_usd_per_token: float | None = None,
    use_multiturn_chat_model: bool = False,
    multiturn_openrouter_model: str | None = None,
    multiturn_prompt_usd_per_token: float | None = None,
    multiturn_completion_usd_per_token: float | None = None,
    routes: dict[str, Route] | None = None,
) -> Plan:
    protocol = build_visitor_protocol(catalog.config, openrouter_model)
    config = catalog.config.model_copy(deep=True)
    config.models[protocol.model_config.model_id] = protocol.model_config
    selection = RunSelection(
        modules=frozenset(modules),
        source_files=frozenset(source_files),
        dimensions=frozenset(dimensions or ()),
        max_samples=max_samples,
        max_samples_per_file=max_samples_per_file,
    )
    caveats = list(protocol.caveats)

    judge_model_config: ModelConfig | None = None
    cleaned_judge = (judge_openrouter_model or "").strip()
    if cleaned_judge:
        judge_model_config = build_judge_model_config(
            config,
            cleaned_judge,
            system_one=bool(use_system_one_judge),
        )
        config.models[judge_model_config.model_id] = judge_model_config
        config.judges.default_judge_model_id = judge_model_config.model_id
        config.judges.basic_education_evaluator_model_id = judge_model_config.model_id
        config.judges.basic_education_test_endpoint_model_id = judge_model_config.model_id

    multiturn_model_config: ModelConfig | None = None
    cleaned_multiturn = (multiturn_openrouter_model or "").strip()
    if use_multiturn_chat_model:
        if not cleaned_multiturn:
            caveats.append(
                Caveat(
                    level="error",
                    code="multiturn_model_missing",
                    message=(
                        "Multi-turn chat model endpoint is enabled, but no OpenRouter model id "
                        "was selected for the multi-turn endpoint."
                    ),
                )
            )
        else:
            multiturn_model_config = build_multiturn_chat_model_config(config, cleaned_multiturn)
            config.models[multiturn_model_config.model_id] = multiturn_model_config
            config.judges.basic_education_test_endpoint_model_id = multiturn_model_config.model_id
            caveats.append(
                Caveat(
                    level="info",
                    code="multiturn_chat_on_visitor_key",
                    message=(
                        f"Multi-turn student dialogue (`stu`) uses OpenRouter chat model "
                        f"`{multiturn_model_config.model_name}` via the visitor's single OpenRouter key."
                    ),
                )
            )
    elif (
        use_system_one_judge
        and DEFAULT_BASIC_MODULE_NAME in selection.modules
        and "config_guided_task.yaml" in selection.source_files
    ):
        caveats.append(
            Caveat(
                level="error",
                code="multiturn_endpoint_required_for_system_one",
                message=(
                    "Scenario `config_guided_task.yaml` is multi-turn and requires a conversational "
                    "student model (`stu`). System 1 decision models have 0 output tokens and cannot "
                    "generate student dialogue turns. Check 'Opt into multi-turn chat model endpoint' "
                    "and select an OpenRouter chat model."
                ),
            )
        )
    if protocol.openrouter_model.endswith(":free"):
        caveats.append(
            Caveat(
                level="warning",
                code="free_model_rate_limit",
                message=(
                    "This id ends in :free. OpenRouter caps those variants at 20 requests per minute, "
                    "and at 50 requests per UTC day until the account has purchased at least 10 credits, "
                    "then 1,000 per day. Paid slugs are not under that cap. They are limited by credit "
                    "balance, the in-flight spending budget, the upstream provider, and DDoS protection. "
                    "GET /api/v1/key reports limit_remaining and free_model_daily_requests."
                ),
            )
        )
    applied_routes = _apply_routes(
        routes or {},
        target=protocol.model_config,
        judge=None if use_system_one_judge else judge_model_config,
        student=multiturn_model_config,
    )
    if applied_routes:
        caveats.append(
            Caveat(
                level="info",
                code="openrouter_host_choice",
                message=" ".join(route.describe() for route in applied_routes.values()),
            )
        )
    caveats.extend(_selection_caveats(catalog, selection, max_concurrency))

    samples = iter_standard_samples(config, selection)
    file_counts: dict[str, int] = {}
    low_input = 0
    high_input = 0
    judge_calls = 0
    for sample in samples:
        file_counts[sample.source_file] = file_counts.get(sample.source_file, 0) + 1
        prompt = augment_prompt_for_format(sample, sample.prompt)
        chars = len(prompt)
        low_input += max(1, chars // 4)
        high_input += max(1, chars)
        if _task_mode(config, sample.task) == "llm":
            judge_calls += 1

    basic_tasks, basic_low, basic_high, basic_caveats = _basic_education_bounds(
        config,
        protocol.model_config,
        selection,
        standard_loaded=len(samples),
    )
    caveats.extend(basic_caveats)

    max_tokens = int(protocol.model_config.max_tokens or 0)
    output_cap = len(samples) * max_tokens
    judge_model_id, judge_api_base = _judge_endpoint(config)

    price_ok = (
        prompt_usd_per_token is not None
        and completion_usd_per_token is not None
        and prompt_usd_per_token >= 0
        and completion_usd_per_token >= 0
    )
    low_usd = None
    high_usd = None
    component_usd: dict[str, tuple[float, float]] = {}
    if price_ok:
        low_usd = (low_input * prompt_usd_per_token) + (basic_low * max_tokens * completion_usd_per_token)
        high_usd = (high_input * prompt_usd_per_token) + (
            (output_cap + (basic_high * max_tokens)) * completion_usd_per_token
        )
        component_usd["target"] = (low_usd, high_usd)
        caveats.append(
            Caveat(
                level="info",
                code="cost_range",
                message=(
                    "The low dollar figure prices input tokens with the runner's estimator, "
                    "len(prompt) // 4, and prices no standard-module output. "
                    "The high figure prices one token per prompt character and output at the "
                    "request max_tokens cap. OpenRouter's usage object counts the model's native tokenizer, "
                    "not this estimator. Paid requests in flight are also held against an in-flight spending "
                    "budget estimated from max_tokens. "
                    "Format retries, answer-length retries, and HTTP retries are extra and are not included. "
                    "Basic-education input tokens are not included, because that runtime assembles its own prompts."
                ),
            )
        )
    else:
        caveats.append(
            Caveat(
                level="warning",
                code="price_unknown",
                message=(
                    f"OpenRouter did not return a token price for {protocol.openrouter_model}. "
                    "There is no dollar estimate. Running requires an explicit confirmation."
                ),
            )
        )

    if judge_calls:
        if judge_model_config is not None:
            endpoint_suffix = "/systemone" if use_system_one_judge else "/chat/completions"
            judge_kind = "System 1 decision-model" if use_system_one_judge else "OpenRouter chat"
            caveats.append(
                Caveat(
                    level="info",
                    code="judge_on_visitor_key",
                    message=(
                        f"This selection makes {judge_calls} {judge_kind} judge calls through JudgeRouter "
                        f"to `{judge_model_config.model_name}` at `{judge_model_config.api_base}{endpoint_suffix}` "
                        "using the visitor's single OpenRouter API key."
                    ),
                )
            )
            if (
                low_usd is not None
                and high_usd is not None
                and judge_prompt_usd_per_token is not None
                and judge_prompt_usd_per_token >= 0
            ):
                # Each judge call sends state + rubric (~300 tokens low, ~1024 tokens high). The judge has no
                # length limit, so the high end assumes JUDGE_HIGH_OUTPUT_TOKENS of output, thinking included.
                judge_low = judge_calls * 300 * judge_prompt_usd_per_token
                judge_out_cap = 0 if use_system_one_judge else JUDGE_HIGH_OUTPUT_TOKENS
                judge_comp_price = max(0.0, float(judge_completion_usd_per_token or 0.0))
                judge_high = (judge_calls * 1024 * judge_prompt_usd_per_token) + (
                    judge_calls * judge_out_cap * judge_comp_price
                )
                component_usd["judge"] = (judge_low, judge_high)
                low_usd += judge_low
                high_usd += judge_high
        else:
            caveats.append(
                Caveat(
                    level="info",
                    code="judge_calls_not_on_visitor_key",
                    message=(
                        f"This selection makes {judge_calls} LLM-judge calls through JudgeRouter. "
                        f"The judge model is {judge_model_id} at {judge_api_base}. "
                        "Those calls are not billed to the visitor OpenRouter key, and the judge model "
                        "is not replaced."
                    ),
                )
            )

    caveats.extend(_preflight_caveats(config, protocol, selection))
    blocked = any(caveat.level == "error" for caveat in caveats)
    preflight_ok = not any(caveat.code.startswith("preflight_") and caveat.level == "error" for caveat in caveats)
    is_full = any(caveat.code == "full_item_set" for caveat in caveats)
    covers_core = _covers_core(catalog, selection)
    cfg_hash = compute_plan_fingerprint(
        openrouter_model=protocol.openrouter_model,
        judge_openrouter_model=cleaned_judge,
        use_system_one_judge=bool(use_system_one_judge),
        use_multiturn_chat_model=bool(use_multiturn_chat_model),
        multiturn_openrouter_model=cleaned_multiturn,
        modules=selection.modules,
        source_files=selection.source_files,
        dimensions=selection.dimensions,
        max_samples=selection.max_samples,
        max_samples_per_file=selection.max_samples_per_file,
    )
    return Plan(
        protocol=protocol,
        selection=selection,
        caveats=caveats,
        standard_requests=len(samples),
        standard_low_input_tokens=low_input,
        standard_high_input_tokens=high_input,
        standard_output_cap_tokens=output_cap,
        judge_calls=judge_calls,
        judge_model_id=judge_model_id,
        judge_api_base=judge_api_base,
        basic_tasks=basic_tasks,
        basic_target_calls_low=basic_low,
        basic_target_calls_high=basic_high,
        max_tokens=max_tokens,
        prompt_usd_per_token=prompt_usd_per_token if price_ok else None,
        completion_usd_per_token=completion_usd_per_token if price_ok else None,
        low_usd=low_usd,
        high_usd=high_usd,
        preflight_ok=preflight_ok and not blocked,
        blocked=blocked,
        needs_price_ack=not price_ok,
        file_counts=sorted(file_counts.items()),
        judge_model_config=judge_model_config,
        use_system_one_judge=bool(use_system_one_judge),
        multiturn_model_config=multiturn_model_config,
        use_multiturn_chat_model=bool(use_multiturn_chat_model),
        config_hash=cfg_hash,
        is_full_benchmark=is_full,
        covers_core=covers_core,
        judge_prices=_price_tuple(judge_prompt_usd_per_token, judge_completion_usd_per_token),
        student_prices=(
            _price_tuple(multiturn_prompt_usd_per_token, multiturn_completion_usd_per_token)
            if multiturn_model_config is not None
            else None
        ),
        component_usd=component_usd,
        routes=applied_routes,
    )


def _apply_routes(routes: dict[str, Route], **configs: ModelConfig | None) -> dict[str, Route]:
    """Put each role's OpenRouter ``provider`` object into its request body. Decision models are not routed."""
    applied: dict[str, Route] = {}
    for role, model_config in configs.items():
        route = routes.get(role)
        if model_config is None or route is None or not route.provider:
            continue
        model_config.provider_kwargs = {**model_config.provider_kwargs, "provider": dict(route.provider)}
        applied[role] = route
    return applied


def _covers_core(catalog: Catalog, selection: RunSelection) -> bool:
    """True when every question outside the region-specific sets is selected, uncapped and unfiltered."""
    active = [
        name
        for name in catalog.config.app.benchmark_modules.active
        if name in catalog.config.modules and catalog.config.modules[name].enabled
    ]
    core = {entry.canonical_name for entry in catalog.files} - set(CHINA_SPECIFIC_FILES)
    return (
        set(selection.modules) == set(active)
        and core <= set(selection.source_files)
        and not selection.dimensions
        and selection.max_samples is None
        and selection.max_samples_per_file is None
    )


def _price_tuple(prompt: float | None, completion: float | None) -> tuple[float, float] | None:
    if prompt is None or prompt < 0:
        return None
    return float(prompt), max(0.0, float(completion or 0.0))


def iter_standard_samples(config: ProjectConfig, selection: RunSelection) -> list[Sample]:
    """Same sample sequence ``BenchmarkRunner._load_samples`` produces."""
    if not selection.modules or not selection.source_files:
        return []
    module_selection = resolve_module_selection(config, set(selection.modules))
    if not module_selection.standard_modules:
        return []
    resolved = FileRegistry(config).resolve(
        modules=set(module_selection.standard_modules),
        subsets=None,
        source_files=set(selection.source_files),
    )
    dimensions = set(selection.dimensions) or None
    runner = BenchmarkRunner(config)
    return list(runner._load_samples(resolved, dimensions, selection.max_samples, selection.max_samples_per_file))


def render_plan(plan: Plan) -> str:
    lines = ["### Protocol", ""]
    model = plan.protocol.model_config
    lines.append(f"- Model field: `{model.model_name}`")
    lines.append(f"- Endpoint: `{model.api_base}` + the provider `endpoint_path`")
    lines.append(
        f"- temperature `{model.temperature}`, max_tokens `{model.max_tokens}`, "
        f"stream `{str(bool(model.supports_stream)).lower()}`, "
        f"responses API `{str(plan.protocol.responses_api).lower()}`"
    )
    if plan.protocol.matched_card_id:
        lines.append(f"- Copied card: `{plan.protocol.matched_card_id}`")
    else:
        lines.append("- Copied card: none")
    if plan.judge_model_config is not None:
        judge_mode = "System 1 (`/systemone`)" if plan.use_system_one_judge else "OpenRouter Chat (`/chat/completions`)"
        lines.append(f"- Judge endpoint: `{plan.judge_model_config.model_name}` ({judge_mode})")
    if plan.multiturn_model_config is not None:
        lines.append(f"- Multi-turn chat endpoint (`stu`): `{plan.multiturn_model_config.model_name}`")
    lines.append("")
    lines.append("### Selection")
    lines.append("")
    lines.append(f"- Standard requests: {plan.standard_requests}")
    if plan.file_counts:
        for name, count in plan.file_counts:
            lines.append(f"  - `{english_file(name)}`: {count}")
    lines.append(
        f"- Input tokens, low / high: {plan.standard_low_input_tokens} / {plan.standard_high_input_tokens}"
    )
    lines.append(f"- Standard output cap: {plan.standard_output_cap_tokens} tokens")
    lines.append(f"- LLM-judge calls: {plan.judge_calls}")
    lines.append(f"- Basic-education tasks: {plan.basic_tasks}")
    lines.append(
        "- Basic-education target-model calls, low / high: "
        f"{plan.basic_target_calls_low} / {plan.basic_target_calls_high}"
    )
    if plan.config_hash:
        lines.append(f"- Configuration checkpoint fingerprint: `{plan.config_hash}`")
    lines.append("")
    lines.append("### Cost range charged to the OpenRouter key")
    lines.append("")
    if plan.low_usd is None or plan.high_usd is None:
        lines.append("No price is available for this model id.")
    else:
        lines.append(f"- Low: {_usd(plan.low_usd)}")
        lines.append(f"- High: {_usd(plan.high_usd)}")
        lines.append(
            f"- Prices: prompt {_per_million(plan.prompt_usd_per_token)} / 1M, "
            f"completion {_per_million(plan.completion_usd_per_token)} / 1M"
        )
    lines.append("")
    lines.append("### Caveats")
    lines.append("")
    if not plan.caveats:
        lines.append("None.")
    else:
        for caveat in plan.caveats:
            lines.append(f"- **{caveat.level}** `{caveat.code}`: {caveat.message}")
    if plan.blocked:
        lines.append("")
        lines.append("The run is blocked until every error above is cleared.")
    return "\n".join(lines)


def _selection_caveats(catalog: Catalog, selection: RunSelection, max_concurrency: int) -> list[Caveat]:
    caveats: list[Caveat] = []
    known = catalog.by_name()
    if not selection.modules:
        caveats.append(Caveat("error", "no_modules", "Select at least one module."))
    if not selection.source_files:
        caveats.append(Caveat("error", "no_files", "Select at least one source file."))
    unknown = sorted(name for name in selection.source_files if name not in known)
    if unknown:
        caveats.append(
            Caveat(
                "error",
                "unknown_files",
                "These source files are not in the registry: " + ", ".join(unknown),
            )
        )
    if selection.max_samples is not None and selection.max_samples <= 0:
        caveats.append(
            Caveat(
                "error",
                "bad_max_samples",
                "Max samples must be empty (every selected item) or a positive integer.",
            )
        )
    if max_concurrency < 1 or max_concurrency > 100:
        caveats.append(
            Caveat(
                "error",
                "bad_concurrency",
                "Max concurrency must be from 1 to 100. 100 is the cap in configs/app.yaml.",
            )
        )

    active = [
        name
        for name in catalog.config.app.benchmark_modules.active
        if name in catalog.config.modules and catalog.config.modules[name].enabled
    ]
    all_files = {entry.canonical_name for entry in catalog.files}
    full_modules = set(selection.modules) == set(active)
    full_files = set(selection.source_files) == all_files
    full_items = (
        full_modules
        and full_files
        and not selection.dimensions
        and selection.max_samples is None
        and selection.max_samples_per_file is None
    )
    if not full_items:
        reasons: list[str] = []
        if not full_modules:
            missing = [english_module(name) for name in active if name not in selection.modules]
            reasons.append("modules omitted: " + ", ".join(missing))
        if not full_files:
            missing_files = [name for name in catalog.files if name.canonical_name not in selection.source_files]
            reasons.append(f"{len(missing_files)} registry files omitted")
        if selection.dimensions:
            reasons.append("dimension filter is set")
        if selection.max_samples is not None:
            reasons.append(
                f"max_samples={selection.max_samples} keeps a prefix in registry order, "
                "the same order BenchmarkRunner uses, not a random or stratified sample"
            )
        if selection.max_samples_per_file is not None:
            reasons.append(f"max_samples_per_file={selection.max_samples_per_file} keeps a prefix of each file")
        caveats.append(
            Caveat(
                "warning",
                "not_full_benchmark",
                "This selection is not the full 2,939-item benchmark ("
                + "; ".join(reasons)
                + "). Module averages from build_summary describe only the items that ran. "
                "They are not comparable to the published leaderboard, and this repository "
                "does not compute that table's bootstrap interval.",
            )
        )
    else:
        caveats.append(
            Caveat(
                "info",
                "full_item_set",
                "Every active module and registry file is selected, with no dimension filter "
                "and no max_samples cap. build_summary will average the full item set. "
                "That average is still not the published 0–100 leaderboard: the host is OpenRouter, "
                "and the bootstrap interval is not implemented in this repository.",
            )
        )

    if selection.max_samples is not None and selection.max_samples > 0:
        caveats.append(
            Caveat(
                "warning",
                "max_samples_prefix",
                "max_samples stops after that many standard samples in file-registry order. "
                "When basic education is also selected, standard modules consume the budget first "
                "and basic education receives the remainder. That is the runner's rule.",
            )
        )

    orphan_files = [
        name
        for name in selection.source_files
        if name in known and known[name].module not in selection.modules
    ]
    if orphan_files:
        caveats.append(
            Caveat(
                "warning",
                "file_module_mismatch",
                "These files are checked, but their module is not selected, so the runner will skip them: "
                + ", ".join(sorted(orphan_files)),
            )
        )

    selected_entries = [
        known[name]
        for name in selection.source_files
        if name in known and known[name].module in selection.modules
    ]
    for entry in selected_entries:
        if entry.scoring_mode == "rule" and entry.preflight_llm_task:
            caveats.append(
                Caveat(
                    "warning",
                    "rule_task_preflight_hold",
                    f"{english_file(entry.canonical_name)} is scored by the rule judge "
                    f"(configs/judges.yaml task {entry.task}, mode rule). "
                    "Preflight still lists this task with the LLM-judge tasks and will not "
                    "start the run without that judge model's API key. The dashboard keeps that check.",
                )
            )
    _scale_caveats(caveats, selected_entries)

    if max_concurrency != 100:
        caveats.append(
            Caveat(
                "info",
                "concurrency",
                f"max_concurrency is {max_concurrency}. The source default in configs/app.yaml is 100. "
                "Concurrency changes scheduling only. Each request is still built by OpenAICompatibleClient "
                "and scored by JudgeRouter. The hosted default is 8 so a 2-vCPU Space does not turn "
                "timeouts into failed samples.",
            )
        )
    if not get_api_key(VISITOR_KEY_ENV):
        caveats.append(
            Caveat(
                "error",
                "visitor_key_missing",
                "Enter an OpenRouter API key. It is read from the process environment for the run "
                "and is not written to configs or key.txt.",
            )
        )
    return caveats


def _scale_caveats(caveats: list[Caveat], entries: list[CatalogFile]) -> None:
    if any(entry.module == DEFAULT_BASIC_MODULE_NAME for entry in entries):
        caveats.append(
            Caveat(
                "warning",
                "basic_education_score_scale",
                "Basic-education scores are the runtime's mean of numeric rubric fields. "
                "The pass threshold in configs/basic_education.yaml is 3.0. "
                "Other modules' rule and LLM judges write scores on a 0–1 scale "
                "(teaching-harm can also be 0.5). build_summary averages each group separately. "
                "This dashboard does not rescale either group onto the published 0–100 leaderboard.",
            )
        )
    if any(entry.scoring_mode == "placeholder" for entry in entries):
        caveats.append(
            Caveat(
                "error",
                "placeholder_judge",
                "A selected task has no judge in configs/judges.yaml. The placeholder judge does not emit a score.",
            )
        )


def _basic_education_bounds(
    config: ProjectConfig,
    model_config,
    selection: RunSelection,
    *,
    standard_loaded: int,
) -> tuple[int, int, int, list[Caveat]]:
    caveats: list[Caveat] = []
    if DEFAULT_BASIC_MODULE_NAME not in selection.modules:
        return 0, 0, 0, caveats
    if not selection.source_files:
        return 0, 0, 0, caveats

    executor = BasicEducationExecutor(config, model_config, _SilentLogger())
    scenarios = executor._select_scenarios(
        subsets=None,
        source_files=set(selection.source_files),
        dimensions=set(selection.dimensions) or None,
    )
    remaining = selection.max_samples
    if remaining is not None:
        remaining = max(0, remaining - standard_loaded)
        if standard_loaded and DEFAULT_BASIC_MODULE_NAME in selection.modules:
            caveats.append(
                Caveat(
                    "warning",
                    "mixed_max_samples",
                    "Standard modules and basic education share one max_samples budget. "
                    "Standard samples are counted first. Basic education is skipped when nothing remains.",
                )
            )

    tasks = 0
    low_calls = 0
    high_calls = 0
    for scenario in scenarios:
        if remaining is not None and remaining <= 0:
            break
        template = executor._read_yaml_file(scenario.template_path)
        declared = count_runtime_tasks(template)
        if scenario.expected_tasks is not None and declared != scenario.expected_tasks:
            caveats.append(
                Caveat(
                    "warning",
                    "basic_education_task_count_mismatch",
                    f"{scenario.scenario_id} declares {declared} runtime tasks but "
                    f"configs/basic_education.yaml expected_tasks is {scenario.expected_tasks}. "
                    "The executor logs this and runs the template.",
                )
            )
        take = declared if remaining is None else min(declared, remaining)
        if selection.max_samples_per_file is not None:
            take = min(take, selection.max_samples_per_file)
        globals_section = template.get("globals") if isinstance(template, dict) else {}
        raw_limit = globals_section.get("recursion_limit") if isinstance(globals_section, dict) else None
        try:
            recursion_limit = int(raw_limit)
        except (TypeError, ValueError):
            recursion_limit = 25
            caveats.append(
                Caveat(
                    "info",
                    "recursion_limit_default",
                    f"{scenario.scenario_id} has no globals.recursion_limit. "
                    "The runtime entity default is 25.",
                )
            )
        if scenario.multi_turn:
            recursion_limit = min(recursion_limit, DEFAULT_MULTI_TURN_RECURSION_LIMIT)
        tasks += take
        low_calls += take
        high_calls += take * max(1, recursion_limit)
        if remaining is not None:
            remaining -= take

    if scenarios:
        caveats.append(
            Caveat(
                "warning",
                "basic_education_call_bound",
                "Basic-education prompts are assembled by basic_education_runtime, not by the "
                "standard prompt loader. The low call bound is one target-model call per task. "
                "The high bound is one target-model call per graph step, capped by the recursion "
                "limit the executor writes (20 on multi-turn scenarios, otherwise the template value). "
                "Basic-education judge calls and the guided-teaching simulated student are not in the "
                "dollar range; with a visitor judge they are billed to the visitor key and count toward "
                "the judge and student spending limits. Input tokens for these calls are also omitted.",
            )
        )
        caveats.append(
            Caveat(
                "warning",
                "basic_education_key_on_disk",
                "The basic-education executor writes the target API key into "
                "outputs/basic_education/<run_id>/<model_id>/<scenario>/basic_education_runtime.yaml "
                "because the runtime reads it from that file. The dashboard deletes that run directory "
                "when the run finishes. A killed process can leave the file behind.",
            )
        )
    return tasks, low_calls, high_calls, caveats


def _preflight_caveats(config: ProjectConfig, protocol: VisitorProtocol, selection: RunSelection) -> list[Caveat]:
    if not selection.modules or not selection.source_files:
        return []
    if selection.max_samples is not None and selection.max_samples <= 0:
        return []
    report = PreflightRunner(config).run(
        model_id=protocol.model_config.model_id,
        modules=set(selection.modules),
        subsets=None,
        source_files=set(selection.source_files),
        dimensions=set(selection.dimensions) or None,
        max_samples=selection.max_samples,
        judge_enabled=True,
        require_real_judges=True,
    )
    caveats: list[Caveat] = []
    for issue in report.issues:
        caveats.append(
            Caveat(
                level=issue.level,
                code=f"preflight_{issue.code}",
                message=english_preflight_message(issue),
            )
        )
    return caveats


def _task_mode(config: ProjectConfig, task: str | None) -> str:
    if not task:
        return "placeholder"
    task_config = config.judges.tasks.get(task)
    if task_config is None:
        return "placeholder"
    return task_config.mode


def _judge_endpoint(config: ProjectConfig) -> tuple[str | None, str | None]:
    model_id = config.judges.default_judge_model_id
    if not model_id or model_id not in config.models:
        return model_id, None
    return model_id, config.models[model_id].api_base


def _usd(value: float) -> str:
    if value >= 0.01:
        return f"${value:,.2f}"
    if value == 0:
        return "$0"
    return f"${value:.4f}"


def _per_million(price_per_token: float | None) -> str:
    if price_per_token is None:
        return "n/a"
    return f"${price_per_token * 1_000_000:.4f}"


class _SilentLogger:
    def info(self, *args, **kwargs) -> None:
        return None

    def warning(self, *args, **kwargs) -> None:
        return None

    def error(self, *args, **kwargs) -> None:
        return None

    def exception(self, *args, **kwargs) -> None:
        return None
