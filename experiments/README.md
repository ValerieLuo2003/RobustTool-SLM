# 实验输出目录说明

每次正式运行必须拥有独立目录：

```text
experiments/results/<run_name>/
├── config.json
├── metrics.json
├── failure_stats.json
├── predictions.jsonl
├── trajectories.jsonl
├── evaluation.jsonl
└── run.log
```

- `results/` 保存可复现的逐实验产物；
- `logs/` 可保存训练器产生的额外日志；
- 自动生成的运行目录默认被 Git 忽略；
- 需要发布的正式结果应在核验后显式导出；
- README 或最终报告中的数字必须能追溯到这些文件。

不要在同一目录混合不同模型、数据版本或 seed。已存在同名运行时，默认应选择新名称；只有确认需要替换该运行时才使用 `--overwrite`。

完整规范见 [实验记录协议](../docs/experiments.md)。
