# 模型配置说明

本目录用于保存 Qwen 主模型和推理配置，包括：

- ModelScope / HuggingFace 模型 ID；
- 精确 revision；
- dtype 与 device 配置；
- 最大上下文长度；
- Chat Template 与 Tool Call 解析协议；
- 解码参数；
- seed；
- 是否启用量化。

Week 1 暂未接入模型。Evaluator 冻结后，第一阶段只选择一个 1.5B～4B 级别的 Qwen 模型，不同时铺开多个模型。
