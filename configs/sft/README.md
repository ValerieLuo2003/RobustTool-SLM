# SFT 配置说明

本目录保存 Week 2 的 ms-swift LoRA SFT 配置。每份配置至少记录：

- 主模型及 revision；
- 训练、验证数据及版本；
- seed；
- learning rate、batch size、gradient accumulation；
- epochs / max steps；
- max length；
- LoRA rank、alpha、dropout 和 target modules；
- dtype、gradient checkpointing；
- 输出目录与日志设置。

Shell 命令不能代替配置文件。每次运行必须把解析后的最终配置复制到对应实验目录。

当前 `qwen2_5_1_5b_lora_smoke.json` 只用于验证训练链路：

- 主模型为固定 revision 的 Qwen2.5-1.5B-Instruct；
- 使用 BF16 LoRA，rank 8、alpha 16、dropout 0.05；
- 数据只有 15 条 Train Oracle 轨迹和 5 条 Validation Oracle 轨迹；
- 最多运行 20 step，每 10 step 评测和保存；
- 输出写入 `experiments/results/qwen2_5_1_5b_sft_smoke_v3/`；
- 不读取 Clean Test，不代表正式 SFT 实验。

先构建数据，再启动 smoke：

```bash
python scripts/build_sft_data.py
python scripts/run_sft.py \
  --config configs/sft/qwen2_5_1_5b_lora_smoke.json
```

ms-swift 官方 Agent 数据协议要求 `tools` 是 JSON 字符串，`tool_call` 和 `tool_response` 消息的 `content` 也必须是 JSON 字符串。本项目的转换器会检查这些约束，并拒绝 task_id 交叉、不可用工具、无效 JSON 和空 Assistant target。

正式配置为 `qwen2_5_1_5b_lora_formal_v1.json`：

- 6000 条 Train、500 条 Validation，Test 不进入配置；
- 1 epoch、有效 batch size 8；
- BF16 LoRA，rank 16、alpha 32、dropout 0.05；
- learning rate `1e-4`，cosine scheduler，3% warmup；
- 每 250 optimizer step 评测和保存，保留最近两个 checkpoint；
- 输出到 `experiments/results/qwen2_5_1_5b_sft_formal_v1/`。

```bash
python scripts/generate_formal_data.py
python scripts/build_formal_sft_data.py
python scripts/run_sft.py --config configs/sft/qwen2_5_1_5b_lora_formal_v1.json
```

该配置不包含 `max_steps`，由 6000 条数据、1 epoch 和梯度累积 8 决定 750 个 optimizer step。正式运行已经完成，最低 Validation loss 对应 checkpoint-750；如需改超参数或训练 Failure-SFT，应复制为新版本，不得覆盖这份已产生正式结果的 v1 配置。

Failure-aware 阶段提供两份新配置：

- `qwen2_5_1_5b_lora_failure_aware_smoke.json`：原始 Train 与 Hard-case 各抽样 64 条，Validation 抽样 32 条，只运行 20 step；
- `qwen2_5_1_5b_lora_failure_aware_v1.json`：从同一冻结 Base 开始，使用原始 6000 + Failure-aware 3000，Validation 仍是原来的 500 条。

Failure-SFT 不直接续训原 SFT adapter。这样后续的 Failure-aware 3000 与 Random 3000 可以都从同一 Base、在相同总数据量和训练超参数下重训，避免 optimizer 历史或二阶段学习率成为混杂因素。

```bash
python scripts/run_sft.py --config configs/sft/qwen2_5_1_5b_lora_failure_aware_smoke.json
python scripts/run_sft.py --config configs/sft/qwen2_5_1_5b_lora_failure_aware_v1.json
```
