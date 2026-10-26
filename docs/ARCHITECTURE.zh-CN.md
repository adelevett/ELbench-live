# ELBench 架构说明

本文档描述 ELBench 的工程设计：注册表、provider 抽象、执行模型，以及如何扩展框架。基准与排行榜的总览见顶层 [README](../README.zh-CN.md)。

ELBench 不是临时脚本，而是按正式工程维护的评测系统。当前仓库在一个统一框架下承载四个模块：

- `安全可信` / Safety & Trustworthiness
- `高阶育人` / High-Level Educational Cultivation
- `基本教育` / Basic Education
- `通用模型` / General Capability

`基本教育` 已作为 ELBench 的正式模块接入。它的 benchmark 数据放在 `data/benchmark_root/基本教育/`，多轮执行由项目内部的 `basic_education_runtime` 模块负责，代码位于 `src/elbench/basic_education_runtime/`。这个内部运行时来源于 ELMES 项目，但现在已经按 ELBench 的内部模块方式接管和维护。

## 设计原则

- 评测流程由注册表和配置驱动，不在主流程里写死文件逻辑
- 模型调用统一走 provider adapter，不让厂商差异污染主 runner
- 客观题优先规则判分
- 主观和开放式任务采用 `LLM-as-a-Judge`
- 稳定的机器可读输出、日志、检查点与汇总
- 中断后支持续跑
- 可配置并发，默认全局上限 `100`

## 当前实际接入的内容

### 已注册的 benchmark 文件

`configs/file_registry.yaml` 当前解析以下文件：

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

文件只有在注册之后才会被视为有效的 benchmark 输入。没有出现在 `configs/file_registry.yaml` 中的数据文件，不属于可运行基准的一部分。

### 当前判分策略

客观题用规则或参考答案匹配：

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

主观或开放式任务用 `LLM-as-a-Judge`：

- `安全拒答.jsonl`
- `安全引导.jsonl`
- `安全回答.jsonl`
- `adversarial_prompts.xlsx`
- `高阶育人-edu.jsonl`

仓库把外部 provider 接入设计为可插拔。内置 `mock.default` 和 `mock.judge`，可在没有真实 API 的情况下本地验证完整流水线。

## 仓库结构

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

`scripts/run_benchmark.py` 刻意只是一个薄入口，真正的评测逻辑都在 `src/elbench/` 下。

## 核心架构

### 注册表与 schema 适配

- `configs/file_registry.yaml` 把文件映射到模块、子集、任务、loader 与规范名。
- `configs/field_mappings.yaml` 把原始数据集字段映射进 ELBench 的统一内部 schema。
- JSONL、XLSX、基本教育 YAML 输入都被规范化为同一个 `Sample` 结构。

### 统一的 sample/result schema

加载后的记录被规范化为稳定 schema，字段包括：

- `sample_id`
- `source_file`
- `module`
- `subset`
- `task`
- `dimension`
- `prompt`
- `reference`
- `metadata`

执行输出包括：

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

### Provider 抽象

- `ModelClient` 是模型执行的统一接口。
- 厂商 API 差异都隔离在 `src/elbench/providers/` 下。
- 一旦 adapter 存在，主 runner 不关心模型来自 OpenAI、Anthropic、Gemini 还是其他厂商。

### 执行、并发与恢复

runner 支持：

- 全局并发控制
- provider/模型级并发上限
- 指数退避重试
- 失败日志
- 基于检查点的续跑
- 原始响应持久化
- 判分结果持久化
- 汇总生成

`configs/app.yaml` 中默认全局最大并发为 `100`。

### 基本教育运行时

`基本教育` 与其他模块的差别只在执行方式，不在数据治理：

- benchmark 数据仍放在 `data/benchmark_root/基本教育/`
- 文件发现仍走 `configs/file_registry.yaml`
- 运行时执行由 `src/elbench/execution/basic_education.py` 负责
- 内部运行时实现位于 `src/elbench/basic_education_runtime/`

## 输出结构

```text
outputs/
├── raw_responses/
├── judged_results/
├── summaries/
└── logs/
```

这些目录是运行时产物，刻意被 Git 忽略。整理后的排行榜与审计汇总提交在 `results/` 下。

## 配置面

### `configs/models.yaml`

具体的被测模型与裁判模型实例，含超时、重试、限速与 provider 参数。

### `configs/providers.yaml`

provider adapter 注册表与默认 provider 能力。当前内置 adapter：

- `mock`
- `openai_compatible`

### `configs/judges.yaml`

任务到裁判的路由：

- `rule`
- `llm`

### `configs/basic_education.yaml`

基本教育内部运行时的场景级设置。该文件不替代注册表，而是对其补充。

## 扩展 ELBench

### 新增 provider

1. 在 `src/elbench/providers/` 下实现 adapter。
2. 在 `configs/providers.yaml` 中注册。
3. 在 `configs/models.yaml` 中加入具体模型。

### 新增 benchmark 文件

1. 把数据文件放到 `data/benchmark_root/` 下。
2. 在 `configs/file_registry.yaml` 中注册。
3. 在 `configs/field_mappings.yaml` 中加字段映射。
4. 在 `configs/judges.yaml` 中路由判分。
5. 新增或复用测试。

### 新增未来模块

ELBench 已为更多模块预留空间，预期路径仍是：

1. 加模块注册
2. 加文件注册
3. 加字段映射
4. 加裁判路由
5. 仅当配置无法表达行为时才加任务专属逻辑

不需要重写主 runner。

## 维护

权威的团队维护指南是 [`docs/TEAM_MAINTENANCE_GUIDE.zh-CN.md`](TEAM_MAINTENANCE_GUIDE.zh-CN.md)，它说明每个维护目录的用途、哪些路径仅运行时、以及哪些该提交哪些不该提交。

## 状态

已实现：

- 配置系统
- 文件注册表
- JSONL/XLSX/YAML loader
- 统一 sample/result schema
- provider 抽象
- 并发 runner
- 重试与检查点续跑
- 原始/判分/汇总/日志输出
- 客观题规则裁判
- 主观题 `LLM-as-a-Judge` 执行路径
- 基于内部运行时模块的基本教育桥接

仍刻意未完成：

- 面向每个外部厂商的生产级 adapter
- 最终化的主观裁判 rubric 与 prompt
- 超出当前已注册文件的完整未来模块接入
