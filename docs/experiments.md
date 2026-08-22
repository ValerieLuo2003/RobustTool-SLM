# 实验记录协议

## 1. 目标

实验记录必须做到：

- 同一配置可以复现；
- 指标可以追溯到逐任务预测和轨迹；
- Failure Distribution 可以重新计算；
- README 和报告中的数字不能与原始日志不一致；
- 不同 Base、SFT、Failure-SFT 和 GRPO 运行可以公平比较。

## 2. 目录规范

每次正式运行创建独立目录：

```text
experiments/results/<experiment_name>/
├── config.json
├── metrics.json
├── failure_stats.json
├── predictions.jsonl
├── trajectories.jsonl
├── evaluation.jsonl
└── run.log
```

实验名称应只包含字母、数字、点、下划线和连字符。例如：

```text
qwen2_5_1_5b_base_seed42
qwen2_5_1_5b_sft_v1_seed42
qwen2_5_1_5b_failure_sft_wrong_tool_seed42
qwen2_5_1_5b_grpo_dense_seed42
```

## 3. 文件含义

### `config.json`

保存解析后的完整运行配置，而不只是 Shell 命令。至少包括：

- `run_name`；
- 模型名称与精确 revision；
- 数据文件、数据版本与 SHA-256；
- seed；
- 推理或训练超参数；
- 最大轮数与最大长度；
- Python、CUDA、GPU 和关键依赖版本；
- Git commit；
- 运行开始时间。

训练实验还应保存：

- learning rate；
- batch size；
- gradient accumulation；
- epochs / max steps；
- optimizer 和 scheduler；
- LoRA rank、alpha、dropout 和 target modules；
- dtype、gradient checkpointing；
- 实际训练时长和峰值显存。

### `predictions.jsonl`

面向快速检查的逐任务预测摘要，包括 Assistant Action、Tool Calls、Final Answer 和推理期 Final State。

### `trajectories.jsonl`

完整 user / assistant / tool 交互，是重新评测和后续失败挖掘的主要输入。

### `metrics.json`

每个指标同时保存 `value`、`numerator` 和 `denominator`。禁止只保存四舍五入后的百分比。

### `failure_stats.json`

保存失败任务数、每个 Failure Label 的任务数和任务占比。

### `evaluation.jsonl`

保存逐任务环境重放结果、最终状态、指标诊断、失败标签和证据。

### `run.log`

保存标准输出、警告、异常、中断、恢复和关键时间点。后续训练应将 ms-swift 的训练日志完整保留。

## 4. 基线命令

当前无模型 Oracle：

```bash
python scripts/run_baseline.py \
  --policy oracle \
  --run-name toy_oracle

python scripts/run_eval.py --run-name toy_oracle
```

当前无模型 Random：

```bash
python scripts/run_baseline.py \
  --policy random \
  --seed 7 \
  --run-name toy_random_seed7

python scripts/run_eval.py --run-name toy_random_seed7
```

Oracle 只用于验证环境、任务目标和 Evaluator 自洽，不代表模型能力上限；Random 用于确认失败路径和 Failure Classifier 能工作，也不是正式模型基线。

当前 Qwen 基线先在 Validation 上做 5 条任务的 GPU smoke test：

```bash
python scripts/run_qwen_baseline.py \
  --config configs/models/qwen2_5_1_5b_instruct.json \
  --tasks data/eval/toy_validation.jsonl \
  --limit 5 \
  --run-name qwen2_5_1_5b_base_val_smoke

python scripts/run_eval.py \
  --run-name qwen2_5_1_5b_base_val_smoke
```

`run_qwen_baseline.py` 会把本次实际使用的任务写入实验目录下的 `tasks.jsonl`。因此 `run_eval.py` 未显式传入 `--tasks` 时，会优先使用该快照，避免 `--limit` 运行和完整数据文件发生 task ID 不匹配。

Smoke test 只验证模型下载、Chat Template、工具解析、环境执行、显存和实验产物链路。通过后才运行完整 Validation；协议冻结后再运行 Clean Test，不能根据 Test 结果修改提示词或解析规则。

## 5. 正式对比规则

Base、SFT、Failure-SFT 和 GRPO 之间必须尽量保持：

- 相同主模型与 revision；
- 相同 Test / Robust Test；
- 相同 Tool Schema 和 Prompt Protocol；
- 相同最大轮数和停止规则；
- 相同解码配置或明确记录差异；
- 相同 seed 集合；
- 相同 Evaluator 版本。

训练方法不能使用 Test 失败样本生成 Hard Cases，也不能根据 Test 结果修改目标定义。

## 6. 结果发布规则

README 和最终报告中的表格必须由机器可读产物生成。正式发布前：

1. 检查 `config.json` 是否完整；
2. 检查 Task 数量与预期一致；
3. 检查是否存在缺失或重复 task_id；
4. 重新运行 Evaluator；
5. 对 Failure Top Categories 抽样检查证据；
6. 检查数据与代码 Git commit；
7. 再生成 Markdown / CSV 表格。

禁止手工修改表格数字来匹配预期结论。

## 7. 计划中的核心实验

### Overall

比较 Base、SFT、Failure-SFT 和 GRPO 的 Tool Selection、Argument、Execution、Task Success、Recovery 和 Invalid Call。

### Robustness

比较 Clean、Distractor、Missing Tool、Tool Error、Noisy Response 和 Ambiguous Query 等扰动设置。

### Failure Distribution

对比训练前、SFT 后、Failure-SFT 后和 GRPO 后的失败类型分布。

### Ablation

优先完成：

- Random Augmentation vs Failure-aware Augmentation；
- Outcome-only Reward vs Failure-aware Dense Reward。

只有资源与时间充足时再进行模型尺度对比。
