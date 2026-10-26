# ELBench Architecture

This document describes the engineering design of ELBench: the registry, the
provider abstraction, the execution model, and how to extend the framework.
For an overview of the benchmark and the leaderboard, see the top-level
[README](../README.md).

ELBench is built as an engineering system, not a one-off script. Four benchmark
modules are kept in one consistent framework:

- `安全可信` / Safety & Trustworthiness
- `高阶育人` / High-Level Educational Cultivation
- `基本教育` / Basic Education
- `通用模型` / General Capability

## Principles

- Registry-driven loading instead of hardcoded file logic
- Provider adapters instead of provider-specific branches in the runner
- Rule-based judging for objective tasks
- `LLM-as-a-Judge` for subjective and open-ended tasks
- Stable machine-readable outputs, logs, checkpoints, and summaries
- Resume support after interruption
- Configurable concurrency with a default global cap of `100`

## What Is Actually Wired Today

### Registered benchmark files

`configs/file_registry.yaml` currently resolves these files:

- `安全拒答.jsonl`
- `安全引导.jsonl`
- `安全回答.jsonl`
- `SATAs.xlsx`
- `adversarial_prompts.xlsx`
- `高阶育人-edu.jsonl`
- `高阶育人-omni.jsonl`
- `knowledge.yaml`
- `question.yaml`
- `cross.yaml`
- `config_guided_task.yaml`
- `mmlu_pro_sampled.jsonl`
- `ceval_sampled.jsonl`
- `ifeval_sampled.jsonl`
- `math_500_sampled.jsonl`
- `aime24_sampled.jsonl`
- `aime25.jsonl`
- `aime26.jsonl`
- `gsm8k_sampled.jsonl`

Files are only considered active benchmark inputs after they are registered.
Data files not present in `configs/file_registry.yaml` are not part of the
runnable benchmark.

### Current judge strategy

Objective tasks use rule or reference matching:

- `SATAs.xlsx`
- `高阶育人-omni.jsonl`
- `mmlu_pro_sampled.jsonl`
- `ceval_sampled.jsonl`
- `ifeval_sampled.jsonl`
- `math_500_sampled.jsonl`
- `aime24_sampled.jsonl`
- `aime25.jsonl`
- `aime26.jsonl`
- `gsm8k_sampled.jsonl`

Subjective or open-ended tasks use `LLM-as-a-Judge`:

- `安全拒答.jsonl`
- `安全引导.jsonl`
- `安全回答.jsonl`
- `adversarial_prompts.xlsx`
- `高阶育人-edu.jsonl`

The repository still treats external provider integration as pluggable.
`mock.default` and `mock.judge` are included so the full pipeline can be
validated locally without real API access.

## Repository Layout

```text
.
├── configs/
│   ├── app.yaml
│   ├── basic_education.yaml
│   ├── field_mappings.yaml
│   ├── file_registry.yaml
│   ├── judges.yaml
│   ├── models.yaml
│   ├── module_registry.yaml
│   └── providers.yaml
├── data/
│   └── benchmark_root/
│       ├── 安全可信/
│       ├── 高阶育人/
│       ├── 基本教育/
│       └── 通用/
├── docs/
│   └── TEAM_MAINTENANCE_GUIDE.zh-CN.md
├── results/
├── scripts/
│   └── run_benchmark.py
├── src/
│   └── elbench/
│       ├── basic_education_runtime/
│       ├── cli.py
│       ├── config/
│       ├── execution/
│       ├── judges/
│       ├── loaders/
│       ├── persistence/
│       ├── providers/
│       ├── registry/
│       ├── schemas/
│       ├── summary/
│       └── utils/
└── tests/
```

`scripts/run_benchmark.py` is intentionally a thin entrypoint. The actual
benchmark logic lives under `src/elbench/`.

## Core Architecture

### Registry and schema adaptation

- `configs/file_registry.yaml` maps files to module, subset, task, loader, and
  canonical name.
- `configs/field_mappings.yaml` maps raw dataset fields into ELBench's unified
  internal schema.
- JSONL, XLSX, and basic-education YAML inputs are normalized into the same
  `Sample` structure.

### Unified sample/result schema

Loaded records are normalized into a stable schema with fields such as:

- `sample_id`
- `source_file`
- `module`
- `subset`
- `task`
- `dimension`
- `prompt`
- `reference`
- `metadata`

Execution output includes:

- `provider_name`
- `model_id`
- `model_name`
- `model_response`
- `judge_result`
- `score`
- `judge_reason`
- `judge_metadata`
- `latency_ms`
- `retry_count`
- `timestamp`

### Provider abstraction

- `ModelClient` is the common interface for model execution.
- Provider-specific API differences are isolated under `src/elbench/providers/`.
- The main runner does not care whether a model comes from OpenAI, Anthropic,
  Gemini, or another vendor once the adapter exists.

### Execution, concurrency, and recovery

The runner supports:

- global concurrency control
- provider/model-level concurrency limits
- retry with exponential backoff
- failure logging
- checkpoint-based resume
- raw-response persistence
- judged-result persistence
- summary generation

Default global max concurrency is `100` in `configs/app.yaml`.

### Basic Education runtime

`基本教育` differs from the other modules only in execution style, not in data
governance:

- benchmark data stays under `data/benchmark_root/基本教育/`
- file discovery still goes through `configs/file_registry.yaml`
- runtime execution is handled by `src/elbench/execution/basic_education.py`
- the internal runtime implementation lives under
  `src/elbench/basic_education_runtime/`

This internal runtime was adapted from the ELMES project, but it is now packaged
and maintained as part of ELBench itself.

## Output Layout

```text
outputs/
├── raw_responses/
├── judged_results/
├── summaries/
└── logs/
```

These directories are runtime artifacts and are intentionally ignored by Git.
Curated leaderboards and audit summaries are committed under `results/`.

## Configuration Surface

### `configs/models.yaml`

Concrete evaluated-model and judge-model instances, including timeout, retry,
rate limits, and provider kwargs.

### `configs/providers.yaml`

Provider adapter registry plus default provider capabilities. Current built-in
adapters:

- `mock`
- `openai_compatible`

### `configs/judges.yaml`

Task-to-judge routing:

- `rule`
- `llm`

### `configs/basic_education.yaml`

Scenario-level runtime settings for the internal basic education runtime. This
file does not replace the registry; it complements it.

## Extending ELBench

### Add a new provider

1. Implement an adapter under `src/elbench/providers/`.
2. Register it in `configs/providers.yaml`.
3. Add concrete models in `configs/models.yaml`.

### Add a new benchmark file

1. Put the data file under `data/benchmark_root/`.
2. Register it in `configs/file_registry.yaml`.
3. Add field mappings in `configs/field_mappings.yaml`.
4. Route judging in `configs/judges.yaml`.
5. Add or reuse tests.

### Add a future module

ELBench already reserves space for further modules. The intended path remains:

1. add module registration
2. add file registration
3. add field mappings
4. add judge routing
5. add task-specific logic only if configuration cannot express the behavior

No main-runner rewrite should be required.

## Maintenance

The canonical team maintenance guide is
[`docs/TEAM_MAINTENANCE_GUIDE.zh-CN.md`](TEAM_MAINTENANCE_GUIDE.zh-CN.md). That
document explains what each maintained directory is for, which paths are
runtime-only, and what should or should not be committed.

## Status

Implemented:

- config system
- file registry
- JSONL/XLSX/YAML loaders
- unified sample/result schema
- provider abstraction
- concurrent runner
- retries and checkpoint resume
- raw/judged/summary/log outputs
- rule judges for objective tasks
- `LLM-as-a-Judge` execution path for subjective tasks
- built-in basic-education bridge on top of the internal runtime module

Still intentionally unfinished:

- production-grade adapters for every external vendor
- finalized subjective judge rubrics and prompts
- complete future-module integration beyond the current registered files
