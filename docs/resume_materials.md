# 简历与面试材料

以下表述只使用仓库中已经完成并有产物支撑的结果。GRPO 代码已经实现，但尚未完成 GPU
训练，因此不把它写成已经取得指标的实验结论。

## 中文简历版本

**RobustTool-SLM：面向可靠工具调用小模型的失败感知后训练框架**

- 独立搭建可执行 Calendar Tool-Calling Benchmark：实现 5 个带 JSON Schema 的真实工具、
  状态环境、工具错误/噪声注入、环境重放 Evaluator，并将调用决策、工具选择、参数语义、
  执行成功、任务完成和 15 类失败标签拆开统计；仓库包含 73 项自动化测试。
- 基于 Qwen2.5-1.5B-Instruct + LoRA 完成 `6000 + 3000 + 3000` 条训练数据的
  Recovery-aware v2 训练闭环；在冻结 Clean Test 上 Task Success 从 63.0% 提升至 90.1%，
  在 500 条 Robust Test 上从 38.4% 提升至 85.0%，Recovery Success 从 4.0% 提升至 92.0%，
  配对 Robustness Gap 从 29.2 个百分点降至 9.2 个百分点。
- 设计等规模 Random Augmentation v2 对照实验控制训练样本数、LoRA、学习率和 checkpoint
  选择协议；Random 对照 Clean Test 达到 92.8%，但 Robust Test 为 65.8%、Recovery Success
  为 0%，验证收益主要来自针对恢复行为的数据分配，而非单纯增加数据量。
- 实现 execution-feedback GRPO 训练入口、Outcome Reward 与 Failure-aware Dense Reward，
  支持多轨迹采样、真实工具执行、组内 advantage、clipped policy update、轨迹审计和多 seed
  汇总；正式 GPU 训练结果待后续补跑。

## English version

- Built an executable calendar tool-calling benchmark with five JSON-Schema tools, stateful
  execution, injected tool failures/noisy responses, fresh-environment replay, and 15-label
  failure taxonomy; added 73 automated tests for reproducibility.
- Trained a Qwen2.5-1.5B-Instruct LoRA policy with a 12k-example recovery-aware pipeline;
  improved Task Success from 63.0% to 90.1% on Clean Test and from 38.4% to 85.0% on a
  500-example Robust Test, while increasing Recovery Success from 4.0% to 92.0%.
- Designed a matched Random Augmentation control with the same 12k training scale and
  optimization protocol; the control reached 92.8% Clean Task Success but only 65.8% Robust
  Task Success and 0% Recovery Success, isolating the benefit of targeted recovery data.
- Implemented an auditable execution-feedback GRPO pipeline with binary Outcome Reward,
  failure-aware dense shaping, group-relative advantages, clipped policy updates, trajectory
  logging, and multi-seed aggregation; GPU runs are pending.

## 面试时必须说明的边界

- 当前已完成并可量化的是 Recovery-v2 / Random Augmentation v2 对照；GRPO 目前是已实现、
  未完成正式训练的下一阶段。
- Robustness Gap 使用 500 个同源 Clean/Robust task pairs 计算，不是简单用 1000 条 Clean
  和 500 条 Robust 的总体成功率相减。
- Recovery Success 的分母由 benchmark 的 `tool_failure` 任务定义，避免模型通过不触发错误
  来获得虚假的恢复分数。
