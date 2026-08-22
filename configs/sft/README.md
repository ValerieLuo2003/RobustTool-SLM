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
