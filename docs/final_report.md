# 最终报告模板

> 当前仓库只完成 Week 1 基础框架。本文件定义最终报告应包含的内容，不填写尚未运行的实验数字。

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
| Base |  |  |  |  |  |  |
| SFT |  |  |  |  |  |  |
| Failure-SFT |  |  |  |  |  |  |
| GRPO |  |  |  |  |  |  |

### Robustness

| Setting | Base | SFT | Failure-SFT | GRPO |
|---|---:|---:|---:|---:|
| Clean |  |  |  |  |
| Distractor |  |  |  |  |
| Missing Tool |  |  |  |  |
| Tool Error |  |  |  |  |
| Noisy Response |  |  |  |  |
| Ambiguous Query |  |  |  |  |

表格必须由实验 JSON 自动生成，不得手工填写无法追溯的数字。

## 6. Failure Analysis

至少展示：

- Base 的 Failure Distribution；
- SFT 后各类 Failure 的绝对数量和变化；
- Failure-SFT 针对目标类别的改善及副作用；
- GRPO 后 Recovery、Repeated Call、Ignore Tool Result 等变化；
- 每个重要类别的真实轨迹案例。

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
