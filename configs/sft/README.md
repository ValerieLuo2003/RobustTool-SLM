# SFT 配置说明

本目录预留给 Week 2 的 ms-swift LoRA SFT 配置。正式配置至少记录：

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
