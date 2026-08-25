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

Smoke test 只验证模型下载、Chat Template、工具解析、环境执行、显存和实验产物链路。当前已经完成正式 Validation；Clean Test 仍保持冻结，不能根据 Test 结果修改提示词、解析规则或 Hard-case 方向。

SFT 训练链路 smoke 使用配置文件启动：

```bash
python scripts/build_sft_data.py
python scripts/run_sft.py \
  --config configs/sft/qwen2_5_1_5b_lora_smoke.json
```

该配置只运行 20 step LoRA。需要确认日志中出现有限且非 NaN 的 loss、global step 持续增加、Validation loss 可以计算、LoRA checkpoint 成功保存且 CUDA 没有 OOM。通过这些检查不等于 SFT 有效果；正式对比必须使用扩展后的 Train 数据，并重新执行同一套 Environment Evaluation。

正式数据、训练和 adapter 评测命令为：

```bash
python scripts/generate_formal_data.py
python scripts/build_formal_sft_data.py
python scripts/run_sft.py --config configs/sft/qwen2_5_1_5b_lora_formal_v1.json
python scripts/select_sft_checkpoint.py experiments/results/qwen2_5_1_5b_sft_formal_v1

python scripts/run_qwen_baseline.py \
  --tasks data/processed/calendar_formal_v1/tasks/validation.jsonl \
  --adapter-path <selected_checkpoint.json 中的 selected_checkpoint> \
  --run-name qwen2_5_1_5b_sft_formal_v1_validation
python scripts/run_eval.py --run-name qwen2_5_1_5b_sft_formal_v1_validation
```

`scripts/select_sft_checkpoint.py` 按最低 Validation loss 选择实际存在且包含完整 adapter 文件的 checkpoint，并输出配置与权重 SHA-256；不允许根据 Test 结果选 checkpoint。Base 和 adapter 运行必须使用相同任务快照哈希。`scripts/compare_runs.py` 会拒绝比较不同快照；`scripts/select_failure_targets.py` 只接受 LoRA Validation 运行，并拒绝读取 Test 结果来选择 Hard-case 类别。

本次已完成的普通 SFT 正式运行使用 Qwen2.5-1.5B-Instruct、6000 条 Train、1 epoch、LoRA rank 16。Failure-SFT 使用原始 6000 条加 3000 条 Failure-aware Train，并从同一个 Base Model 重新训练，而不是继续训练普通 SFT adapter。两者都按最低 Validation loss 选择 checkpoint-750。在相同 500 条任务快照上：

| 指标 | Base | SFT | Failure-SFT |
|---|---:|---:|---:|
| Call Decision Accuracy | 92.40% | 98.20% | 100.00% |
| Tool Selection Accuracy | 91.35% | 94.94% | 98.31% |
| Argument Semantic Accuracy | 83.49% | 94.52% | 97.57% |
| Executable Call Rate | 80.16% | 96.69% | 99.79% |
| Task Success Rate | 66.00% | 92.00% | 93.80% |
| Multi-turn Task Success Rate | 18.92% | 56.76% | 62.16% |
| Invalid Tool Call Rate | 18.40% | 2.65% | 0.00% |
| Unnecessary Tool Call Rate | 6.80% | 0.00% | 0.00% |

这些数字由远程实验目录中的 `metrics.json` 和比较脚本生成，只用于 Validation 方法选择，不代替最终 Test 结果。

Failure-aware 数据阶段使用以下命令：

```bash
python scripts/select_failure_targets.py \
  experiments/results/qwen2_5_1_5b_sft_formal_v1_validation_new3090 \
  --top-k 3

python scripts/build_hard_cases.py \
  --failure-targets experiments/results/qwen2_5_1_5b_sft_formal_v1_validation_new3090/failure_targets.json
```

两条命令已经完成：冻结 `wrong_argument_value`、`ignore_tool_result` 和 `missing_argument`，并只使用这三个类别作为生成策略，产出 3000 条全新 Train 轨迹。全量 Oracle Task Success 为 100%，800 条两步任务也全部成功，新数据与原三份 split 的 ID 和规范化问题文本重叠均为 0。它没有把 Validation 失败任务直接改写成训练样本，也没有根据 Test 选择方向。

Failure-SFT 的正式配置为：

```bash
python scripts/run_sft.py \
  --config configs/sft/qwen2_5_1_5b_lora_failure_aware_v1.json
```

本次训练共 1125 step、1 epoch，运行约 1 小时 30 分钟，峰值显存约 7.4 GiB。三个候选 checkpoint 的 Validation loss 分别为：

| Step | Validation loss |
|---:|---:|
| 375 | 0.03876981 |
| 750 | **0.03750928** |
| 1125 | 0.04001660 |

最终选择 `checkpoint-750`。其 `adapter_model.safetensors` 大小为 73,911,112 bytes，SHA-256 为 `2862ddc3ceb8d7e5502a05967660d8673e60aab66a9c22036f59643c290788de`。统一评测的任务快照 SHA-256 为 `ad2202da79bdbae87a486d40ebf3ab44ee3223481c1021d29e129213ec261dee`。

Failure-SFT 相比普通 SFT 的目标 failure 变化如下：

| Failure | SFT | Failure-SFT | 变化 |
|---|---:|---:|---:|
| `wrong_argument_value` | 28 | 23 | -5 |
| `ignore_tool_result` | 15 | 1 | -14 |
| `missing_argument` | 11 | 0 | -11 |
| 失败任务总数 | 40 | 31 | -9 |

逐任务配对检查显示，Failure-SFT 修复了 11 条普通 SFT 失败任务，同时让 2 条原本成功的任务失败，净增 9 条成功。新增的 7 个 `final_answer_failure` 都发生在 `list_events → delete_event` 任务：模型已经找到目标事件，却提前返回查询结果。这里有 5 条在普通 SFT 下也失败，只是原标签为 `wrong_call_decision`；真正由成功转失败的是 2 条。因此应同时报告 failure 标签分布和逐任务成功转移，不能把“新增 7 个标签”误写成“回归 7 条任务”。

Failure-SFT 推理耗时约 4064.7 秒，普通 SFT 为 1911.9 秒；但前者只多 1.5% 的生成步数和 2.6% 的输出 token。当前证据不足以把延迟差异归因于训练方法，正式效率比较需要在相同机器状态下重复运行并记录 GPU 时钟、功耗和并发负载。

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
