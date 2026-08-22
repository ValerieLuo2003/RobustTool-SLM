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

第一阶段主模型已固定为：

```text
Qwen/Qwen2.5-1.5B-Instruct
revision: 3c3787b7c81927cc64ad45dc32ff1c9ce2a5de34
source: ModelScope
```

对应配置是 `qwen2_5_1_5b_instruct.json`。选择 1.5B 模型是为了先在单张 RTX 3090 上快速建立可重复的 Base → SFT → Failure-SFT → GRPO 主流程。主流程稳定前不加入第二个模型，避免把模型规模变化和训练方法收益混在一起。

配置默认使用 BF16 和确定性贪心解码。模型文件由 ModelScope 下载到机器自己的缓存目录，不进入仓库；每次运行还会在 `config.json` 中记录模型 ID、revision、dtype、解码参数、依赖版本和设备信息。
