# GRPO 配置说明

这里的实现是项目内可审计的 trajectory-level GRPO：每个任务采样一组完整多步轨迹，
在全新的 CalendarEnvironment 中执行工具调用，再按组归一化 reward advantage，
用 clipped policy objective 更新模型。Reward 只读取执行后的 evaluator 结果，不把
reference call 拼进模型输入。

当前提供两种奖励：

- `outcome`：任务成功为 1，失败为 0；
- `failure_aware_dense`：由 decision、JSON/schema、tool、argument、execution、goal、
  final answer 和 recovery progress 构成，并对 invalid/repeated/unnecessary/ignored-result
  失败施加可审计惩罚。

先用小配置做 smoke test：

```bash
python scripts/run_grpo.py \
  --config configs/grpo/qwen2_5_1_5b_grpo_outcome_smoke.json \
  --adapter-path experiments/results/<recovery-adapter>
```

正式实验从 Recovery-v2 的最佳 LoRA checkpoint 初始化；不要使用 Robust Test 或 Clean Test
作为 GRPO 训练任务。`qwen2_5_1_5b_grpo_*_from_recovery_v2.json` 使用冻结的
`calendar_robustness_validation_v1` 任务，方便把执行反馈训练和最终 Test 严格隔离。
