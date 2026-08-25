# 最终报告模板

> 当前仓库已完成基础框架、正式 SFT、Failure-aware 数据生成、Failure-SFT 训练、Base / SFT / Failure-SFT 的统一 Clean Validation、500 条 Robustness Validation 正式对比，以及 3000 条 Train-only Recovery v2 数据生成和 Oracle 审计。本文件同时保存已有机器可读产物支持的阶段结论和后续待完成部分；Recovery v2 尚未训练，Clean Test、Random Augmentation 消融与 GRPO 结果仍不得提前填写。

## 1. 项目摘要

用一段话说明：项目构建了什么可执行环境和 Benchmark，研究了哪些失败模式，使用了哪些后训练方法，最终在哪些任务和鲁棒性设置上获得了什么可复现实验结果。

## 2. 研究问题

最终报告必须清晰回答：

1. Base 小模型最常见的 Tool Calling Failure 是什么？
2. SFT 主要改善了哪些 Failure？
3. 哪些 Failure 在 SFT 后仍然明显？
4. Failure-aware Hard Data 是否优于随机增加数据？
5. GRPO 是否真正提升 Task Success，而不只是改善 JSON 格式？
6. GRPO 对 Recovery Failure 是否有效？
7. Outcome Reward 与 Dense Reward 有什么差异？
8. Robustness Gap 是否降低？
9. 哪些问题仍然没有解决？
10. 哪些结论可以推广到真实 Agent 系统？

## 3. 系统与 Benchmark

需要说明：

- Tool Environment 的 domain、工具数量和状态；
- Task、Trajectory 和 Goal Check 设计；
- Clean Benchmark 与 Robust Benchmark；
- 数据规模、划分和泄漏防护；
- Evaluator 指标和 Failure Taxonomy；
- 为什么最终状态比字符串 exact match 更可靠。

## 4. 训练方法

分别记录：

- Base Model 与推理配置；
- LoRA SFT 数据、格式和超参数；
- Failure Mining 过程和 Top Failure Categories；
- Targeted Hard Cases 的生成规则；
- Failure-SFT 配置；
- GRPO Rollout、Reward、组大小和超参数；
- Reward Hacking 检查。

## 5. 核心结果

### Overall

| Model | Tool Acc | Arg Acc | Exec Rate | Task Success | Recovery | Invalid Call |
|---|---:|---:|---:|---:|---:|---:|
| Base | 91.35% | 83.49% | 80.16% | 66.00% | — | 18.40% |
| SFT | 94.94% | 94.52% | 96.69% | 92.00% | — | 2.65% |
| Failure-SFT | 98.31% | 97.57% | 99.79% | 93.80% | — | 0.00% |
| GRPO |  |  |  |  |  |  |

上表是 500 条 Validation 的阶段结果，不是 Clean Test 最终结果。当前 Validation 没有 Recovery eligible 样本，因此 Recovery 显示为 `—`，不能填写为 0%。三次运行使用相同任务快照，SHA-256 为 `ad2202da79bdbae87a486d40ebf3ab44ee3223481c1021d29e129213ec261dee`。

### Robustness

| Setting | Base | SFT | Failure-SFT | GRPO |
|---|---:|---:|---:|---:|
| Paired Clean | 69.20% | 94.80% | 96.20% |  |
| Robust Overall | 40.20% | 65.40% | 67.20% |  |
| Similar Tool Distractor | 78.00% | 88.00% | 92.00% |  |
| Tool Order Shuffle | 62.00% | 88.00% | 94.00% |  |
| Tool Description Rewrite | 54.00% | 92.00% | 96.00% |  |
| Tool Name Similarity | 72.00% | 96.00% | 98.00% |  |
| Missing Tool | 0.00% | 0.00% | 0.00% |  |
| Tool Failure | 6.00% | 0.00% | 0.00% |  |
| Noisy Tool Response | 68.00% | 100.00% | 100.00% |  |
| Partial Tool Response | 0.00% | 0.00% | 0.00% |  |
| Ambiguous User Query | 0.00% | 100.00% | 100.00% |  |
| Irrelevant Tool Added | 62.00% | 90.00% | 92.00% |  |
| Robustness Gap | 29.00% | 29.40% | 29.00% |  |

表格由 `compare_robustness_runs.py` 读取正式实验 JSON 生成。Paired Clean 是按 500 条 Robust Task 的 source 分布加权后的 Clean 成功率，不等同于整份 500 条 Clean Validation 的总体分数。Robust 任务快照 SHA-256 为 `a62c76019ad935f58d915e096cad38f02fca8701cb702be1a61fe2a7f7c9f18e`。

## 6. Failure Analysis

至少展示：

- Base 的 Failure Distribution；
- SFT 后各类 Failure 的绝对数量和变化；
- Failure-SFT 针对目标类别的改善及副作用；
- GRPO 后 Recovery、Repeated Call、Ignore Tool Result 等变化；
- 每个重要类别的真实轨迹案例。

当前已经支持的阶段结论：

- Base 的 170 个失败任务中，高频标签包括 `wrong_argument_value`（97）、`ignore_tool_result`（91）和 `final_answer_failure`（40）；
- 普通 SFT 将失败任务降到 40，Task Success 从 330/500 提升到 460/500；
- SFT 后的 Top 3 是 `wrong_argument_value`（28）、`ignore_tool_result`（15）和 `missing_argument`（11）；
- Failure-SFT 将三个目标标签分别降到 23、1 和 0，失败任务进一步降到 31；
- Failure-SFT 相比 SFT 修复 11 条、回归 2 条，Task Success 净增 9 条；
- 当前最明确的副作用是 7 个 `final_answer_failure`：`list_events` 找到待删除事件后没有继续执行 `delete_event`，其中 2 条是真实回归。
- Robust Validation 上，SFT 将 Task Success 从 201/500 提升到 327/500，Failure-SFT 进一步提升到 336/500；
- SFT 把 `ambiguous_user_query` 从 0/50 提升到 50/50，并显著改善 Schema、参数语义和可执行率；
- `missing_tool` 和 `partial_tool_response` 在三个模型上均为 0/50，`tool_failure` 则是 Base 3/50、两个 SFT 模型 0/50；
- Base 的 3 条 Tool Failure 成功均执行了真实 retry，说明当前 SFT 数据产生了需要重点验证的恢复能力负迁移；
- Failure-SFT 相比 SFT 的总体 Robustness Gap 只从 29.4 降至 29.0 个百分点，不能声称第一版 hard data 已解决鲁棒性问题。

## 7. Ablation

至少比较：

1. Random Augmentation vs Failure-aware Augmentation；
2. Outcome-only Reward vs Failure-aware Dense Reward。

不仅报告总体分数，还要说明不同失败类别和 Robustness Gap 如何变化。

## 8. 局限与结论

需要主动说明：

- 本地 Sandbox 与真实互联网 API 的差异；
- 任务模板和 domain 覆盖的限制；
- 参数语义评测的覆盖边界；
- 小规模模型与单主模型的外推限制；
- Reward Hacking 和训练稳定性问题；
- 哪些 Failure 仍未解决。

最终贡献应表述为：构建可执行 Tool Agent Benchmark 与 Evaluation Framework，系统分析小模型工具调用失败模式，并基于 Failure-driven Data Flywheel 和 Execution Feedback 完成后训练；而不是简单表述为“使用 ms-swift 微调了 Qwen”。
