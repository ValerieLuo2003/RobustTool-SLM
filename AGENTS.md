# 仓库开发规范

## 项目范围

本仓库实现可执行、失败感知的工具调用 Benchmark 与后训练流程。模块依赖方向必须保持：

```text
tools → env → data / rollout → eval / reward → scripts
```

训练框架不能成为 Environment 或 Evaluator 的依赖。

## 工程规则

- 所有环境状态转移必须是确定性的；任何随机 Policy 或生成器都必须接收并记录显式 seed。
- Benchmark 测试文件是由生成器产生并带版本的产物。修改生成器后再重新生成，不得手工编辑冻结数据。
- Task Success 必须在新环境中重放调用后计算，不得信任模型或推理代码提供的成功标志。
- Benchmark 核心优先使用 Python 标准库；模型与训练依赖放入可选依赖。
- 修改工具语义、环境状态或 Evaluator 时，必须新增或更新对应单元测试。
- 正式实验输出必须写入 `experiments/results/<run_name>/`，并采用机器可读格式。
- 不得提交模型 checkpoint、大型 cache、凭证、敏感信息或伪造实验结果。
- Train、Validation、Test 必须严格隔离，Test 不得参与数据增强或 Hard-case Mining。
- `scripts/` 只负责命令行串联，核心逻辑应放在 `robust_tool/` 对应模块中。

## 验证命令

在仓库根目录执行：

```bash
python -m unittest discover -s tests -v
```

提交前还应运行：

```bash
python -m compileall -q robust_tool scripts
```
