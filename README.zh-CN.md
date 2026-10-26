# ELBench

[![English](https://img.shields.io/badge/Language-English-blue.svg)](./README.md)
[![简体中文](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-red.svg)](./README.zh-CN.md)

**ELBench** 是一个面向教育大模型的多维度评测基准。它从四个模块评估模型——通用能力、安全可信、基本教育、高阶育人——并在统一的、配置驱动的评测框架下运行。

基准共覆盖 **2939 道题**，横跨四个模块；设计上让单一总分无法掩盖模型在各模块上的真实强弱。

## 排行榜

总分采用 0–100 量纲，并给出基于条目级 bootstrap（10000 次重采样）的 95% 置信区间。头部模型相当接近：前六名分差不到两分，多个模型并列第 4。

| 排名 | 模型 | 总分 [95% CI] | 通用 | 安全 | 基本教育 | 高阶育人 |
|:----:|:-----|:------------:|:----:|:----:|:--------:|:--------:|
| 1 | DeepSeek-V4-Flash | **83.7** [82, 85] | 88.5 | 89.7 | 84.9 | 71.5 |
| 2 | Gemini-3.5-Flash | **83.4** [82, 84] | 93.4 | 70.2 | 94.6 | 75.3 |
| 3 | Doubao-Seed-2.0-Pro | **83.2** [82, 84] | 86.6 | 83.0 | 89.7 | 73.5 |
| 4 | Claude-Opus-4.8 | **83.1** [82, 84] | 91.9 | 78.5 | 92.7 | 69.5 |
| 4 | GPT-5.4 | **83.1** [82, 84] | 86.4 | 76.6 | 94.4 | 75.1 |
| 4 | DeepSeek-V4-Pro | **83.1** [82, 84] | 88.2 | 86.1 | 88.5 | 69.5 |
| 7 | GLM-5.1 | **81.7** [80, 84] | 83.2 | 89.5 | 79.2 | 75.0 |
| 8 | Safe-InnoSpark | **77.0** [76, 78] | 68.0 | 87.6 | 87.3 | 65.2 |
| 9 | InnoSpark-235B | **76.4** [75, 78] | 74.2 | 78.0 | 87.5 | 65.9 |

各模块、各任务的明细榜、置信区间以及评分可靠性审计，以 HuggingFace 数据集
[ZeroLoss-Lab/ELBench-results](https://huggingface.co/datasets/ZeroLoss-Lab/ELBench-results) 发布。

## 模块

| 模块 | 题量 | 评测内容 |
|:-----|:----:|:---------|
| 通用能力（`通用模型`） | 894 | 知识、推理、指令遵循与数学（MMLU-Pro、C-Eval、IFEval、MATH-500、AIME 采样） |
| 安全可信（`安全可信`） | 1000 | 有害请求拒答、良性请求的有用性、对抗鲁棒性 |
| 基本教育（`基本教育`） | 45 | 多轮教学：知识点讲解、情景化出题、跨学科教案、引导式讲题 |
| 高阶育人（`高阶育人`） | 1000 | 高阶育人目标的开放式培养 |

客观题由规则或参考答案匹配判分；主观和开放式任务采用 `LLM-as-a-Judge`，并配有经过验证的评分模型面板。评分一致性审计（对人工金标的二次加权 Cohen κ）见
[结果数据集](https://huggingface.co/datasets/ZeroLoss-Lab/ELBench-results)。

## 安装

```bash
pip install -e .                      # 核心框架
pip install -e .[basic-education]     # + 基本教育多轮运行时
pip install -e .[basic-education,dev] # + 贡献者工具链
```

## 快速开始

```bash
# 查看已注册的 benchmark 文件
python scripts/run_benchmark.py inspect

# 用 mock provider 跑本地冒烟测试（无需 API key）
python scripts/run_benchmark.py run \
  --model-id mock.default --max-samples 3 --run-id smoke-test --no-resume
```

editable 安装后也可直接用 `elbench` 入口：

```bash
elbench inspect
elbench run --model-id mock.default --max-samples 3 --run-id smoke-test --no-resume
```

评测真实模型时，先在 `configs/models.yaml` 里登记模型，再按模块运行：

```bash
python scripts/run_benchmark.py run \
  --model-id <你的模型ID> --module 基本教育 --run-id basic-education-run
```

`基本教育` 需要真实的 API 模型配置；`mock.*` 模型仅用于单轮流水线冒烟检查。

## HuggingFace 数据集

ELBench 以两个 HuggingFace 数据集发布:

- [**ZeroLoss-Lab/ELBench**](https://huggingface.co/datasets/ZeroLoss-Lab/ELBench) —— 评测题目集(四模块共 2939 题)。
- [**ZeroLoss-Lab/ELBench-results**](https://huggingface.co/datasets/ZeroLoss-Lab/ELBench-results) —— 完整评测结果,含逐样本明细。

本仓库只保留评测框架与代码；评测题目与结果都在上面两个数据集中。原始单次运行产物（`outputs/raw_responses/`、`judged_results/`、`logs/`）只在运行时存在，不进版本库。

## 文档

- [架构说明](docs/ARCHITECTURE.zh-CN.md)：注册表、provider 抽象、执行模型，以及如何扩展框架
- [团队维护指南](docs/TEAM_MAINTENANCE_GUIDE.zh-CN.md)：各目录的用途，以及哪些该提交、哪些不该提交

## 测试

```bash
python -m unittest \
  tests.test_basic_education_bridge tests.test_registry_and_loaders \
  tests.test_judges tests.test_response_format_retry
```

## 许可证

发布前请补充你选择的许可证。
