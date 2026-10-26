# ELBench 结果保存规范

## 1. 目标

这份规范只解决一件事：让评测结果在仓库里保存得清楚、可追溯、适合提交。

- `outputs/` 只放运行时全量产物。
- `results/` 只放审核后的正式结果包。
- Git 默认只跟踪 `results/`，不跟踪 `outputs/`。

## 2. 目录边界

| 路径 | 是否入库 | 用途 | 说明 |
|---|---|---|---|
| `outputs/` | 否 | 运行时全量产物 | 大、杂、可删除，默认由 `.gitignore` 忽略 |
| `results/` | 是 | 正式结果包 | 小、稳定、适合汇报和复查 |
| `docs/RESULT_STORAGE_SPEC.zh-CN.md` | 是 | 本规范 | 结果保存规则的唯一规范入口 |

`outputs/` 当前标准子目录如下：

- `outputs/raw_responses/<run_id>/`
- `outputs/judged_results/<run_id>/`
- `outputs/summaries/<run_id>/`
- `outputs/logs/<run_id>/`

## 3. 标准流程

1. 所有评测先按真实 `run_id` 落到 `outputs/`。
2. 跑完后先检查 `summary`、`failures`、日志和样本范围，确认这是不是一次正式可引用的 run。
3. 只有正式结果才允许“升格”到 `results/`。
4. 升格时只整理最小必要文件，不复制原始响应、checkpoint、重试记录、全量日志。
5. 在 `results/` 下创建结果包目录，并补齐 `README.md`、`manifest.json`、`summary.json`。
6. 提交 Git 时只提交 `results/` 下的正式结果包和必要文档。

## 4. 命名规范

正式结果包目录名统一为：

`results/<model_id>-<scope>-<YYYYMMDD>[-revN]/`

规则如下：

- `model_id`：必须与 `configs/models.yaml` 中的模型 id 一致。
- `scope`：推荐使用 `standard`、`basic-education`、`full`、`safety`、`highlevel`、`general`、`custom-<slug>`。
- `YYYYMMDD`：结果定稿日，不强制等于任务启动日。
- 同一天同一范围如果重新定稿，追加 `-rev2`、`-rev3`。
- 真实运行时使用的 `run_id` 不要求和结果包目录名一致，但必须写进 `manifest.json`。

## 5. 结果包内容规范

### 5.1 必备文件

每个正式结果包至少包含：

- `README.md`
- `manifest.json`
- `summary.json`

### 5.2 可选文件

可按需增加：

- `notes.md`
- `artifacts/`

`artifacts/` 只允许放精选的小文件，例如手工整理后的表格、图表、失败样例摘录，不允许把运行时全量产物整体搬进来。

### 5.3 `README.md` 最少字段

每个结果包内的 `README.md` 至少写清楚：

- `run_id`
- `model_id`
- `scope`
- `finished_at`
- `source_outputs`
- `total_judged`
- `total_failures`
- 已知 caveat 或未完成项

### 5.4 `manifest.json` 必备字段

每个结果包内的 `manifest.json` 至少包含：

- `schema_version`
- `result_id`
- `run_id`
- `model_id`
- `provider_name`
- `benchmark_scope`
- `finished_at`
- `git_commit`
- `source_outputs`
- `included_files`
- `excluded_artifacts`

可选字段：

- `base_url`
- `judge_model_ids`
- `notes`

## 6. 禁止提交的内容

以下内容默认禁止进入 `results/`：

- `outputs/` 下的任何目录
- 全量 `raw_responses/`
- `*.checkpoint.json`
- `*.retries.jsonl`
- `*.failures.jsonl`
- `*.log`
- 密钥文件、环境变量导出、任何 secret
- `smoke-*`、`probe-*`、`tmp-*`、`debug-*` 之类的试跑结果

`judged_results/*.jsonl` 默认也不提交。只有在它本身就是正式交付物的一部分，而且单文件不超过 `5 MB` 时，才允许作为 `artifacts/` 的受控附件入库，并且必须在该结果包的 `README.md` 里说明原因。

## 7. 大小与稳定性要求

- 单个结果包目标大小：`<= 2 MB`
- 没有明确理由时，不得超过 `20 MB`
- `summary.json` 一经提交，就视为该次结果的 canonical snapshot
- 如果只修正文案，可以在原目录内修改
- 如果指标、范围、样本集或统计口径发生变化，必须新建 `-revN` 目录，而不是静默覆盖旧结果

## 8. 清理策略

- `outputs/` 里的 `smoke-*`、`probe-*`、`tmp-*`、`debug-*` 可随时本地删除。
- 已经升格到 `results/` 的 run，其 `outputs/` 只作为本地审计缓存，不再作为仓库正式资产。
- 需要长期保留的全量原始产物，应转存到仓库外归档位置；`manifest.json` 只记录来源路径，不把大文件带入 Git。

## 9. 标准目录模板

```text
results/
├── README.md
└── innospark-235b-standard-20260424/
    ├── README.md
    ├── manifest.json
    ├── summary.json
    └── artifacts/   # optional
```

## 10. 当前落地约定

- `outputs/` 继续保持 Git ignore。
- 当前首个标准样例是 `results/innospark-235b-standard-20260424/`。
- 后续所有正式评测结果都按本规范落入 `results/`，不再直接把运行时垃圾留给 Git 视图。
