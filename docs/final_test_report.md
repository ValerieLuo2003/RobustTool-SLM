# Recovery-aware v2 与 Random Augmentation v2 最终 Test 报告

## 结论摘要

本轮实验在同一个 Qwen2.5-1.5B-Instruct Base Model、同一 Prompt / Tool Schema / Evaluator 和同一任务快照上，对比了 Base、Recovery-aware v2 和等规模 Random Augmentation v2。

- Recovery-aware v2 在 Clean Test 上将 Task Success 从 Base 的 63.00% 提升到 90.10%，在 Robust Test 上从 38.40% 提升到 85.00%。
- Recovery-aware v2 的 `recovery_success_rate` 为 92.00%，Base 为 4.00%；总体 Clean→Robust gap 从 29.20 个百分点降到 9.20 个百分点。
- Random Augmentation v2 在 Clean Test 上达到 92.80%，但 Robust Test 只有 65.80%，`recovery_success_rate` 为 0.00%，总体 gap 为 29.00 个百分点。
- 因此，Random Augmentation 能提升普通任务成功率，但不能解释 Recovery-aware v2 在工具失败后的恢复能力提升；当前结果支持“定向 recovery 数据分配有效”的结论。

## 实验协议

| 项目 | 配置 |
|---|---|
| Base Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Model revision | `3c3787b7c81927cc64ad45dc32ff1c9ce2a5de34` |
| Training machine | `kuxiaohai` / RTX 3090 24 GiB |
| Recovery-aware v2 Train | formal 6000 + failure-aware v1 3000 + recovery v2 3000 = 12000 |
| Random Augmentation v2 Train | formal 6000 + failure-aware v1 3000 + random augmentation v2 3000 = 12000 |
| LoRA | rank 16, alpha 32, dropout 0.05 |
| Training | 1 epoch, batch size 1, gradient accumulation 8, learning rate `1e-4` |
| Clean Test | `calendar_formal_v1/tasks/clean_test.jsonl`，1000 条 |
| Robust Test | `calendar_robustness_test_v1/tasks.jsonl`，500 条，10 类扰动各 50 条 |
| Robust snapshot SHA-256 | `d2cd7127185d375471710e941f569ceb18bdccf9c50466a0ff7d2a5cdd6a7eb2` |
| Code commit | `a99b1d9` |

两个 adapter 都从同一个 Base Model 重新训练，并使用 Validation loss 选择 checkpoint；Clean Test 和 Robust Test 没有参与 checkpoint 选择或数据构造。历史 failure-aware v1 的冻结选择依据已按仓库记录重建，未使用 Test 失败样本动态挖掘。

Recovery-aware v2 最终选择 `checkpoint-500`，Validation loss 为 `0.03616512`；Random Augmentation v2 最终选择 `checkpoint-1000`，Validation loss 为 `0.04126363`。Random 对照第一次训练在约 step 900 遇到远程 Windows `LiveKernelEvent 141`，从完整的 checkpoint-500 恢复后使用同步 CUDA 配置完成训练；最终报告使用恢复后完整 run 的 checkpoint-1000。

## Clean Test 与 Robust Test 总体指标

百分比由对应 run 的 `metrics.json` 生成；空值表示该集合不包含相应类型的分母。

### Clean Test（1000 条）

| 指标 | Base | Recovery-aware v2 | Random Augmentation v2 |
|---|---:|---:|---:|
| Call Decision Accuracy | 91.70% | 97.50% | 98.40% |
| Tool Selection Accuracy | 90.63% | 98.63% | 96.95% |
| Argument Schema Accuracy | 84.46% | 97.47% | 98.64% |
| Argument Semantic Accuracy | 79.42% | 95.75% | 94.31% |
| Executable Call Rate | 83.23% | 93.88% | 97.59% |
| Task Success Rate | 63.00% | **90.10%** | **92.80%** |
| Multi-turn Task Success Rate | 8.00% | 90.67% | 73.33% |
| Invalid Tool Call Rate | 15.54% | 2.53% | **1.36%** |
| Unnecessary Tool Call Rate | 8.30% | **0.00%** | 0.10% |

### Robust Test（500 条）

| 指标 | Base | Recovery-aware v2 | Random Augmentation v2 |
|---|---:|---:|---:|
| Call Decision Accuracy | 76.00% | **98.40%** | 89.00% |
| Tool Selection Accuracy | 72.55% | **86.86%** | 77.84% |
| Argument Schema Accuracy | 79.37% | **99.78%** | 98.68% |
| Argument Semantic Accuracy | 61.24% | **87.42%** | 77.61% |
| Executable Call Rate | 63.26% | **88.20%** | 82.82% |
| Task Success Rate | 38.40% | **85.00%** | 65.80% |
| Final Answer Semantic Accuracy | 4.00% | **96.00%** | 0.00% |
| Multi-turn Task Success Rate | 3.68% | **54.41%** | 16.91% |
| Recovery Success Rate | 4.00% | **92.00%** | 0.00% |
| Invalid Tool Call Rate | 20.63% | **0.22%** | 1.32% |
| Unnecessary Tool Call Rate | 23.80% | **0.00%** | 10.00% |

## Clean→Robust gap

Gap 由配对报告脚本在 500 个同源 Clean/Robust 任务对上计算：同一 task ID 在 Clean 与对应扰动版本上的 Task Success 差值；越小表示对冻结扰动更稳定。下表的 Clean Task Success 是完整 1000 条 Clean Test 的总体率，因此不能直接用表中两个总体率相减复算 Gap。

| 模型 | Clean Task Success（1000 条总体） | Robust Task Success（500 条总体） | Robustness Gap（500 对配对） |
|---|---:|---:|---:|
| Base | 63.00% | 38.40% | 29.20 个百分点 |
| Recovery-aware v2 | 90.10% | **85.00%** | **9.20 个百分点** |
| Random Augmentation v2 | **92.80%** | 65.80% | 29.00 个百分点 |

### 分扰动结果

单元格为“Robust Task Success（相对同源 Clean 的 Gap）”。

| 扰动设置 | Base | Recovery-aware v2 | Random Augmentation v2 |
|---|---:|---:|---:|
| Overall | 38.40%（29.20%） | **85.00%（9.20%）** | 65.80%（29.00%） |
| Similar Tool Distractor | 62.00%（10.00%） | 86.00%（12.00%） | **88.00%（6.00%）** |
| Tool Order Shuffle | 52.00%（4.00%） | **90.00%（-2.00%）** | 88.00%（0.00%） |
| Tool Description Rewrite | 58.00%（-4.00%） | 86.00%（2.00%） | **90.00%（2.00%）** |
| Tool Name Similarity | 72.00%（-2.00%） | **100.00%（-6.00%）** | 98.00%（-8.00%） |
| Missing Tool | 0.00%（72.00%） | **96.00%（0.00%）** | 0.00%（96.00%） |
| Tool Failure | 4.00%（70.00%） | **92.00%（6.00%）** | 0.00%（98.00%） |
| Noisy Tool Response | 72.00%（0.00%） | 100.00%（0.00%） | 100.00%（0.00%） |
| Partial Tool Response | 0.00%（68.00%） | 0.00%（100.00%） | 0.00%（100.00%） |
| Ambiguous User Query | 0.00%（76.00%） | **100.00%（-16.00%）** | **100.00%（-8.00%）** |
| Irrelevant Tool Added | 64.00%（-2.00%） | **100.00%（-4.00%）** | 94.00%（4.00%） |

## 可复现产物

六个正式 run 目录包含 `config.json`、`metrics.json`、`failure_stats.json`、`predictions.jsonl`、`trajectories.jsonl` 和运行日志。对比脚本还生成了 JSON / CSV / Markdown 版本的配对 gap 与横向报告：

- `experiments/results/final_test_comparison.*`
- `experiments/results/base_robustness_gap_test.*`
- `experiments/results/recovery_v2_robustness_gap_test.*`
- `experiments/results/random_aug_v2_robustness_gap_test.*`

LoRA checkpoint 和大体积生成文件不纳入 Git；仓库只保留配置、代码、测试、数据生成协议和本报告。
