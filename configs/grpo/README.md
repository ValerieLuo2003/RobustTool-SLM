# GRPO 配置说明

本目录预留给 Week 4 的执行反馈 GRPO 配置，包括：

- Rollout 模型与环境设置；
- 每个任务的最大工具调用轮数；
- Group size 与采样参数；
- Outcome Reward 或 Failure-aware Dense Reward；
- 每个 Reward 分量及权重；
- KL、优化器和训练超参数；
- seed、日志和检查点设置。

Reward 权重必须配置化并逐分量记录。至少进行 Outcome-only 与 Dense Reward 消融，并检查重复调用、规避工具和虚假完成等 Reward Hacking 行为。
