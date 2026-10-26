---
title: ELBench-live
emoji: "📚"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.4.0
python_version: "3.12"
app_file: dashboard/app.py
suggested_hardware: cpu-basic
pinned: false
---

# ELBench / ELBench-live

[![English](https://img.shields.io/badge/Language-English-blue.svg)](./README.md)
[![简体中文](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-red.svg)](./README.zh-CN.md)

📄 [**Paper (arXiv:2608.09548)**](https://arxiv.org/abs/2608.09548) · 💻 [**GitHub (ZeroLoss-Lab/ELBench)**](https://github.com/ZeroLoss-Lab/ELBench) · 🏆 [**AI for Education Leaderboard**](https://benchmarks.ai-for-education.org/) · 🗂️ [**Benchmark Dataset**](https://huggingface.co/datasets/ZeroLoss-Lab/ELBench) · 📊 [**Results Dataset**](https://huggingface.co/datasets/ZeroLoss-Lab/ELBench-results)

**ELBench** is a multi-dimensional benchmark for education-facing large language
models. It evaluates a model across four modules — general capability, safety
and trustworthiness, basic education, and high-level educational cultivation —
under one consistent, config-driven evaluation framework.

The benchmark covers **2,939 items** across the four modules and is built so a
single aggregate score never hides where a model actually succeeds or fails.

## Leaderboard

Overall scores are reported on a 0–100 scale with a bootstrap 95% confidence
interval over items (10,000 resamples). The top models are statistically close:
the spread among the first six is under two points, and several share rank 4.

| Rank | Model | Overall [95% CI] | General | Safety | Basic Ed. | High-Level |
|:----:|:------|:----------------:|:-------:|:------:|:---------:|:----------:|
| 1 | DeepSeek-V4-Flash | **83.7** [82, 85] | 88.5 | 89.7 | 84.9 | 71.5 |
| 2 | Gemini-3.5-Flash | **83.4** [82, 84] | 93.4 | 70.2 | 94.6 | 75.3 |
| 3 | Doubao-Seed-2.0-Pro | **83.2** [82, 84] | 86.6 | 83.0 | 89.7 | 73.5 |
| 4 | Claude-Opus-4.8 | **83.1** [82, 84] | 91.9 | 78.5 | 92.7 | 69.5 |
| 4 | GPT-5.4 | **83.1** [82, 84] | 86.4 | 76.6 | 94.4 | 75.1 |
| 4 | DeepSeek-V4-Pro | **83.1** [82, 84] | 88.2 | 86.1 | 88.5 | 69.5 |
| 7 | GLM-5.1 | **81.7** [80, 84] | 83.2 | 89.5 | 79.2 | 75.0 |
| 8 | Safe-InnoSpark | **77.0** [76, 78] | 68.0 | 87.6 | 87.3 | 65.2 |
| 9 | InnoSpark-235B | **76.4** [75, 78] | 74.2 | 78.0 | 87.5 | 65.9 |

Per-module and per-task leaderboards, confidence intervals, and the
judge-reliability audit are released as the
[ZeroLoss-Lab/ELBench-results](https://huggingface.co/datasets/ZeroLoss-Lab/ELBench-results)
dataset on HuggingFace.

## Modules

| Module | Items | What it measures |
|:-------|:-----:|:-----------------|
| General Capability (`通用模型`) | 894 | Knowledge, reasoning, instruction following, and math (MMLU-Pro, C-Eval, IFEval, MATH-500, AIME samples) |
| Safety & Trustworthiness (`安全可信`) | 1,000 | Harmful-request refusal, helpfulness on benign requests, and adversarial robustness |
| Basic Education (`基本教育`) | 45 | Multi-turn teaching: knowledge explanation, situated question design, cross-subject lesson plans, guided problem solving |
| High-Level Educational Cultivation (`高阶育人`) | 1,000 | Open-ended cultivation of higher-order educational goals |

Objective tasks are graded by rule or reference matching; subjective and
open-ended tasks use `LLM-as-a-Judge` with a validated judge panel. The
judge-agreement audit (quadratic weighted Cohen's κ against a human gold set)
is in the [results dataset](https://huggingface.co/datasets/ZeroLoss-Lab/ELBench-results).

## Installation

```bash
pip install -e .                      # core framework
pip install -e .[basic-education]     # + Basic Education multi-turn runtime
pip install -e .[basic-education,dev] # + contributor tooling
```

## Quick Start

```bash
# Inspect the registered benchmark files
python scripts/run_benchmark.py inspect

# Local smoke test against the mock provider (no API keys needed)
python scripts/run_benchmark.py run \
  --model-id mock.default --max-samples 3 --run-id smoke-test --no-resume
```

After an editable install the `elbench` entrypoint is also available:

```bash
elbench inspect
elbench run --model-id mock.default --max-samples 3 --run-id smoke-test --no-resume
```

To evaluate a real model, add it to `configs/models.yaml` and run a module:

```bash
python scripts/run_benchmark.py run \
  --model-id <your_model_id> --module 基本教育 --run-id basic-education-run
```

`基本教育` (Basic Education) requires a real API-backed model config; the
`mock.*` models are for single-turn pipeline smoke checks only.

## Hosted runner

`dashboard/app.py` is a Gradio app for a visitor who brings an OpenRouter key.
It calls `BenchmarkRunner` and `JudgeRouter`. Prompts, retries, and scores are
not reimplemented in the UI. The estimate states the cost range and the ways a
subset, a `max_samples` prefix, or an unlisted model diverges from a published
run. The YAML block at the top of this file is the Hugging Face Space config.
CPU Basic has no hourly price, but creating the Gradio Space requires a paid plan.
Details are in [dashboard/README.md](dashboard/README.md).

```bash
pip install -e ".[basic-education,dashboard]"
python dashboard/app.py
```

## Data on HuggingFace

ELBench is released as two HuggingFace datasets:

- [**ZeroLoss-Lab/ELBench**](https://huggingface.co/datasets/ZeroLoss-Lab/ELBench) — the benchmark items (2,939 across the four modules).
- [**ZeroLoss-Lab/ELBench-results**](https://huggingface.co/datasets/ZeroLoss-Lab/ELBench-results) — the full evaluation results, including per-sample detail.

This repository holds the evaluation framework and code only; the benchmark
items and results live in the two datasets above. Run artifacts
(`outputs/raw_responses/`, `judged_results/`, `logs/`) are runtime-only and not
committed.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — registry, provider abstraction,
  execution model, and how to extend the framework
- [Team maintenance guide](docs/TEAM_MAINTENANCE_GUIDE.zh-CN.md) — what each
  directory is for and what should or should not be committed

## Tests

```bash
python -m unittest \
  tests.test_basic_education_bridge tests.test_registry_and_loaders \
  tests.test_judges tests.test_response_format_retry
```

## 🔬 Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{jiang2026elbench,
  title={ELBench: A Multi-Dimensional Benchmark for Education-Facing Large Language Models},
  author={Jiang, Yilin and Zhu, Xiaorong and Tan, Fei and Zhang, Zicheng and Huang, Kaiyi and Yu, Yang and Fei, Zexuan and Luo, Yiming and Li, Keqian and Hao, Hao and others},
  journal={arXiv preprint arXiv:2608.09548},
  year={2026}
}
```

## License

Add your preferred license before publishing.
