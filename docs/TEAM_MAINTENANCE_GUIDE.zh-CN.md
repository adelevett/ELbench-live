# ELBench 团队维护指南

本指南面向 ELBench 团队维护者，目标是把“什么应该保留、什么应该删、什么地方负责什么”说清楚，减少仓库熵，保证多人协作和大模型协作时都不会把项目越改越乱。

## 1. 总原则

- 配置优先。新增数据、模型、judge 时，优先改注册表和配置，不直接侵入主流程。
- 边界清晰。加载、执行、判分、持久化、汇总必须分层，不要混写。
- 输出稳定。`outputs/` 结构、结果字段、日志字段不能随意改。
- 数据归位。正式 benchmark 数据统一放在 `data/benchmark_root/` 下，不能散落到根目录或外挂目录。
- 数据集不删。任何 benchmark 数据集文件都不得因为“暂未接入”“暂未注册”“暂时无测试”而删除；未接入的数据集只能保留并标记为待接入。
- 轻仓库。缓存、临时分析文档、历史脚本、系统垃圾文件不要留在主仓库。

一句话：**优先走“数据目录 -> 注册表 -> 字段映射 -> judge 路由 -> 边界层代码 -> 测试验证”这条路径。**

## 2. 当前清理与重构结论

这次整理后，以下判断作为当前团队共识：

| 路径 | 结论 | 原因 |
|---|---|---|
| `tests/` | 保留 | 这是正式回归保障，不是临时目录 |
| `src/elbench/basic_education_runtime/` | 保留 | `基本教育` 多轮桥接的内置运行时 |
| `third_party/` | 删除 | 基本教育运行时已内收进项目内部，不再作为外挂目录存在 |
| `run.sh` / `setup.sh` | 删除 | 过时、未被文档和主流程使用 |
| `docs/*analysis*.md` | 删除 | 历史分析稿，且已过时 |
| `docs/sample_command.md` | 删除 | 使用了已废弃的模型配置 |
| `edu_eval_spec_for_codex.md` | 删除 | 早期实现规格，已被当前代码和 README 替代 |
| `.DS_Store`、`pytest-cache-files-*`、`__pycache__`、`outputs/` | 不提交，按需本地删除 | 运行时或系统垃圾文件 |

额外说明：

- ELBench 当前只有 `src/elbench/` 是项目源码根。
- `基本教育` 的内部运行时模块位于 `src/elbench/basic_education_runtime/`。
- 这个内部运行时来源于 ELMES，但现在已经按 ELBench 自有模块维护，不再以外挂目录的形式存在。

## 3. 顶层目录/文件职责表

| 路径 | 是否应长期保留 | 作用 | 什么时候改 |
|---|---|---|---|
| `README.md` | 保留 | 英文默认首页，对外说明项目能力、结构、安装与运行方法 | 仓库能力、结构、安装方式变化时 |
| `README.zh-CN.md` | 保留 | 中文说明 | 与英文 README 同步更新时 |
| `docs/TEAM_MAINTENANCE_GUIDE.zh-CN.md` | 保留 | 团队内部维护规范和目录职责表 | 维护规范变化时 |
| `pyproject.toml` | 保留 | 包定义、依赖、可选 extras、CLI 入口 | 增加依赖、调整安装方式时 |
| `.gitignore` | 保留 | 忽略缓存、输出和系统垃圾文件 | 新增运行时垃圾目录时 |
| `configs/` | 保留 | 全部配置驱动入口 | 新增模块、文件、模型、judge、并发规则时 |
| `data/benchmark_root/` | 保留 | 正式 benchmark 数据根目录 | 新增、整理 benchmark 数据时 |
| `docs/` | 保留，但只留活文档 | 团队文档目录，不放一次性分析稿 | 文档新增或重写时 |
| `scripts/run_benchmark.py` | 保留 | 运行入口薄封装 | 很少改；只改入口行为 |
| `src/elbench/` | 保留 | ELBench 正式源码 | 根据职责分层修改 |
| `tests/` | 保留 | 回归测试与 smoke test | 新增/修改功能后同步补测试 |
| `results/` | 保留 | 正式结果包目录，只放审核后的可提交结果 | 新增正式结果快照时 |
| `outputs/` | 不入库 | 运行产物目录 | 本地跑实验时自动生成 |

正式结果包规范见 `docs/RESULT_STORAGE_SPEC.zh-CN.md`，当前结果索引见 `results/README.md`。

## 4. `configs/` 职责表

| 路径 | 作用 | 注意事项 |
|---|---|---|
| `configs/app.yaml` | 全局默认配置，如 data root、output root、默认并发、默认 retry | 默认最大并发当前为 `100` |
| `configs/module_registry.yaml` | 模块注册表 | 这里只定义模块，不定义文件 |
| `configs/file_registry.yaml` | benchmark 文件注册表 | 文件是否“正式接入”以这里为准 |
| `configs/field_mappings.yaml` | 原始字段到统一 schema 的映射 | 字段变了优先改这里 |
| `configs/providers.yaml` | provider 注册与默认能力 | 不放具体模型实例 |
| `configs/models.yaml` | 被测模型和 judge 模型实例配置 | 单模型超时、限流、provider kwargs 在这里配 |
| `configs/judges.yaml` | task 到 rule / llm judge 的路由 | judge 路由唯一入口 |
| `configs/basic_education.yaml` | 基本教育桥接执行配置 | 只管执行，不替代 benchmark 数据注册 |

## 5. `src/elbench/` 代码职责表

| 路径 | 作用 | 什么时候改 |
|---|---|---|
| `src/elbench/cli.py` | CLI 参数入口 | 新增命令行参数时 |
| `src/elbench/config/loader.py` | 配置加载 | 新增配置文件或 schema 时 |
| `src/elbench/schemas/config.py` | 配置 schema | 配置结构变更时 |
| `src/elbench/schemas/evaluation.py` | Sample / Result / Judge schema | 改输出契约时，需极慎重 |
| `src/elbench/registry/file_registry.py` | 文件发现与解析 | 文件定位逻辑变化时 |
| `src/elbench/loaders/base.py` | loader 抽象基类 | loader 接口变化时 |
| `src/elbench/loaders/jsonl_loader.py` | JSONL 加载 | JSONL 读取逻辑变更时 |
| `src/elbench/loaders/xlsx_loader.py` | XLSX 加载 | Excel 读取逻辑变更时 |
| `src/elbench/loaders/basic_education_yaml_loader.py` | 基本教育 YAML 样本加载 | 基本教育数据模板解析变化时 |
| `src/elbench/loaders/normalizer.py` | 统一 sample 归一化 | Sample 构造规则变化时 |
| `src/elbench/loaders/resolvers.py` | 派生字段解析，如 `dimension` | 某些字段需要特殊派生逻辑时 |
| `src/elbench/providers/base.py` | 模型客户端统一接口 | provider 抽象变化时 |
| `src/elbench/providers/factory.py` | provider 工厂 | 新增 provider 类时 |
| `src/elbench/providers/openai_compatible.py` | OpenAI 兼容 provider | 接 OpenAI-compatible 厂商时 |
| `src/elbench/providers/mock.py` | 本地 mock provider | 本地 smoke 或 judge 回归时 |
| `src/elbench/execution/runner.py` | 主执行器 | 执行调度变更时，慎改 |
| `src/elbench/execution/basic_education.py` | 基本教育多轮桥接 | 调整内部运行时调用、结果导入、恢复逻辑时 |
| `src/elbench/execution/rate_limit.py` | 并发与限流控制 | 限流策略变化时 |
| `src/elbench/execution/retry.py` | 重试与退避 | 重试策略变化时 |
| `src/elbench/judges/router.py` | judge 路由 | task 到 judge 映射变化时 |
| `src/elbench/judges/base.py` | judge 抽象基类 | judge 接口变化时 |
| `src/elbench/judges/llm_judge.py` | `LLM-as-a-Judge` 执行器 | 主观题 judge 链路升级时 |
| `src/elbench/judges/llm_prompting.py` | judge prompt 模板 | rubric 和 judge prompt 调整时 |
| `src/elbench/judges/judge_safety.py` | 安全可信相关判分 | 安全任务判分逻辑变化时 |
| `src/elbench/judges/judge_teaching_harm.py` | 教学安全客观题判分 | SATAs 等规则判分变化时 |
| `src/elbench/judges/judge_highlevel.py` | 高阶育人判分 | `highlevel_edu` / `highlevel_omni` 变化时 |
| `src/elbench/judges/judge_mmlu.py` | MMLU-Pro / C-Eval 判分 | 通用客观题扩展时 |
| `src/elbench/judges/judge_ifeval.py` | IFEval 判分 | 指令遵循判分变化时 |
| `src/elbench/judges/judge_math.py` | Math-500 / AIME 判分 | 数学题判分变化时 |
| `src/elbench/basic_education_runtime/` | 基本教育内部多轮运行时 | 修改运行时模型编排、导出、评估逻辑时 |
| `src/elbench/persistence/writers.py` | 结果写盘 | 输出结构变化时 |
| `src/elbench/persistence/checkpoint.py` | checkpoint 持久化 | 断点续跑行为变化时 |
| `src/elbench/persistence/logging.py` | 日志初始化 | 日志格式或级别调整时 |
| `src/elbench/summary/aggregator.py` | 汇总统计 | 新增 summary 维度时 |
| `src/elbench/utils/parsing.py` | 通用解析工具 | 文本/答案解析工具扩展时 |

## 6. `tests/` 职责表

| 路径 | 作用 | 什么时候必须更新 |
|---|---|---|
| `tests/test_registry_and_loaders.py` | 校验注册表是否能解析当前 bench 文件，以及 loader 是否能产出统一 sample | 新增 benchmark 文件、调整 loader、改 field mapping 后 |
| `tests/test_judges.py` | 校验各类客观题和 `LLM-as-a-Judge` 路径的基本行为 | 改 judge、改 prompt 解析、改客观题判分逻辑后 |
| `tests/test_basic_education_bridge.py` | 校验基本教育 45 题配置、场景任务计数和内部运行时结果导入辅助逻辑 | 改基本教育配置或桥接实现后 |

## 7. `src/elbench/basic_education_runtime/` 保留范围

这个目录是 `基本教育` 的内置运行时，不是独立仓库。

当前应保留：

- `src/elbench/basic_education_runtime/` 下的源码
- `src/elbench/basic_education_runtime/assets/fonts/sarasa-mono-sc-regular.ttf`

当前不应再出现：

- 外挂式旧 third-party 运行时目录
- 与 ELBench 无关的上游示例、CI、锁文件
- 单独的外部项目说明口径

## 8. 新增评测数据的标准流程

在任何“新增/待接入/重构”场景下，都适用一条硬规则：

- 不要删除任何 benchmark 数据集文件。未接入的数据集要保留在 `data/benchmark_root/` 下，并通过文档或注册状态明确标记为“待接入”。

### 场景 A：新增文件，但仍属于现有模块

1. 把数据放到 `data/benchmark_root/` 下正确的模块目录。
2. 在 `configs/file_registry.yaml` 注册文件。
3. 在 `configs/field_mappings.yaml` 增加字段映射。
4. 在 `configs/judges.yaml` 增加或调整 judge 路由。
5. 如需派生字段逻辑，在 `src/elbench/loaders/resolvers.py` 扩展。
6. 客观题优先规则判分，主观题再走 `LLM-as-a-Judge`。
7. 补至少一条测试。
8. 跑 `inspect` 和单测。

### 场景 B：新增一个全新模块

1. 在 `configs/module_registry.yaml` 注册模块。
2. 在 `data/benchmark_root/` 下建立正式目录。
3. 为该模块每个文件补 `file_registry.yaml`。
4. 为该模块每个文件补 `field_mappings.yaml`。
5. 为该模块 task 补 `judges.yaml`。
6. 只有配置表达不了时，才新增 loader、judge 或执行逻辑。

补充：`基本教育` 不是数据组织例外。它的数据也必须放在 `data/benchmark_root/基本教育/`，只是运行时要通过 `src/elbench/execution/basic_education.py` 调用内部 `basic_education_runtime`。

## 9. 什么时候优先改配置，什么时候改代码

### 优先改配置

- 文件路径变化
- 文件归属模块或子集变化
- 原始字段名变化
- task 从 `rule` 切到 `llm`
- 模型 timeout / retry / rate limit 变化

### 必须改代码

- 数据格式不是现有 loader 能覆盖的格式
- 字段派生逻辑无法由映射表达
- 新 provider API 形态与现有 adapter 不兼容
- judge 逻辑本身是新的
- summary 要新增新的聚合方式

### 修改 `runner.py` 的前提

只有当配置、adapter、judge、loader 都不能表达需求时，才允许改 `src/elbench/execution/runner.py`。改完必须补测试并跑 smoke run。

## 10. judge 维护规范

### 客观题

客观题默认优先规则判分，适用于：

- 单选题
- 多选题
- 判断题
- 有稳定标准答案的结构化任务

优先修改位置：

- `src/elbench/judges/judge_teaching_harm.py`
- `src/elbench/judges/judge_highlevel.py`
- `src/elbench/judges/judge_mmlu.py`
- `src/elbench/judges/judge_ifeval.py`
- `src/elbench/judges/judge_math.py`
- `src/elbench/utils/parsing.py`

### 主观题

主观题优先 `LLM-as-a-Judge`，且 judge model 必须和被测模型解耦。

优先修改位置：

- `configs/judges.yaml`
- `src/elbench/judges/llm_prompting.py`
- `src/elbench/judges/llm_judge.py`

禁止：

- 让被测模型给自己打分
- 把 judge prompt 写死在 `runner.py`
- 把 rubric 散落到无关文件

## 11. 模型与 provider 接入规范

### 新增模型

1. 先确认现有 adapter 是否已支持。
2. 只改 `configs/models.yaml`。
3. 如需新的限流或 timeout，也只在该模型配置里加。

### 新增 provider

1. 在 `src/elbench/providers/` 新增 adapter。
2. 在 `factory.py` 注册。
3. 在 `configs/providers.yaml` 增加 provider。
4. 在 `configs/models.yaml` 增加具体模型实例。

要求：

- provider 差异必须封装在 adapter 内
- 不允许在 runner 或 judge 中写 provider-specific 参数名

## 12. 与大模型协作维护的要求

每次让大模型改 ELBench 时，至少要明确告诉它：

- 项目名是 `ELBench`
- 这是长期维护 benchmark 框架，不是 demo
- 正式数据根目录是 `data/benchmark_root/`
- 当前 judge 策略是“客观题 rule，主观题 llm judge”
- 先看 `README.md`、本指南和相关配置，再改代码

并要求它：

1. 优先改配置
2. 新增数据时同步改 `file_registry`、`field_mappings`、`judges`
3. 改 judge 时补测试
4. 提交前跑最小验证命令

## 13. 提交前检查清单

- [ ] 新数据是否放在 `data/benchmark_root/` 正确目录
- [ ] `configs/file_registry.yaml` 是否已同步更新
- [ ] `configs/field_mappings.yaml` 是否已同步更新
- [ ] `configs/judges.yaml` 是否已同步更新
- [ ] 是否误改输出结构
- [ ] 是否新增或更新了测试
- [ ] 是否清掉了 `outputs/`、缓存目录、系统垃圾文件
- [ ] 是否执行了：
  - `python -m unittest tests.test_basic_education_bridge tests.test_registry_and_loaders tests.test_judges`
  - `python scripts/run_benchmark.py inspect`
  - 至少一次小样本 `run`

## 14. 推荐验证命令

```bash
python -m unittest tests.test_basic_education_bridge tests.test_registry_and_loaders tests.test_judges
python scripts/run_benchmark.py inspect
python scripts/run_benchmark.py run --model-id mock.default --max-samples 3 --run-id smoke-test --no-resume
```

## 15. 常见错误

### 错误 1：只改代码，不改注册表

后果：文件不会被主流程识别，`inspect` 看不到。

### 错误 2：字段名变了，直接改 loader

正确做法：先看能不能只改 `configs/field_mappings.yaml`。

### 错误 3：新增模型时去改 runner

正确做法：优先改 `configs/models.yaml`，必要时才新增 adapter。

### 错误 4：把一次性分析稿留在 `docs/`

正确做法：只保留活文档；临时分析不要长期入库。

### 错误 5：把 `outputs/`、缓存或 `.DS_Store` 提交进仓库

正确做法：确保它们继续被 `.gitignore` 忽略，并在提交前清理工作区。
