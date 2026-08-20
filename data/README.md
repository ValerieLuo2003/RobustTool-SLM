# 数据目录说明

```text
data/
├── raw/         # 数据生成器产生的原始记录
├── processed/   # 后续转换为 ms-swift 格式的训练轨迹
└── eval/        # 小规模、带版本、可确定性复现的评测任务
```

## `raw/`

保存生成或收集后、尚未清洗的源数据。默认被 Git 忽略，只保留目录占位文件。任何进入正式训练的数据都必须经过校验和版本记录。

## `processed/`

保存清洗、去重、划分并转换后的训练数据，例如未来的 ms-swift messages/trajectory 格式。默认被 Git 忽略，避免误提交大规模生成数据。

## `eval/`

当前保存 `calendar-toy-v1`：

```text
toy_tasks.jsonl       全部 25 条任务
toy_train.jsonl       15 条 Train
toy_validation.jsonl  5 条 Validation
toy_test.jsonl        5 条 Test
clean_test.jsonl      5 条 Clean Test
manifest.json         生成器版本、seed、数量和 SHA-256
```

重新生成：

```bash
python scripts/generate_data.py --seed 20260809
```

不要直接手工修改 JSONL。需要改变任务时，应修改 `robust_tool/data/generator.py`，在语义变化时更新生成器版本，然后重新生成并运行测试。

完整规范见 [数据集协议](../docs/dataset.md)。
