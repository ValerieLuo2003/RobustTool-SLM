# RobustTool-SLM 系统设计

## 1. 设计目标

在开始训练前，系统必须能够可靠回答下面这个问题：

> 给定一个任务和一段 Agent 轨迹，模型是否做出了正确的调用决策，是否选择并执行了正确工具，以及最终环境是否真正满足用户目标？

如果这个问题无法稳定回答，那么后续 SFT、Hard-case Mining 或 GRPO 即使出现数字提升，也无法证明模型的真实工具调用能力有所改善。

因此，Week 1 的核心与 Qwen、Transformers 和 ms-swift 解耦。Oracle、Random Policy 和未来的 Qwen 模型都会输出同一种 `Trajectory`；Evaluator 使用完全相同的方式重放和评测这些轨迹。

## 2. 核心原则

项目遵循以下优先级：

```text
Evaluation > Environment > Data > Training Tricks
```

具体含义是：

1. 先定义什么是成功、什么是失败；
2. 再实现能够产生客观执行结果的工具环境；
3. 然后构造可以覆盖研究问题的数据；
4. 最后才接入 SFT、Failure-SFT 和 GRPO。

训练不能反过来改变评测定义。不同模型、不同训练方法必须使用同一个环境和同一套成功判定。

## 3. 模块边界

```text
JSON Tool Schema
       │
       v
ToolRegistry ─────> ToolExecutor
                         │
Task ───────────> CalendarEnvironment <──── CalendarState
  │                      │
  v                      v
Policy / Model ─────> Trajectory ─────> Evaluator
                                           │
                                 Metrics + Failures
```

各模块职责如下：

1. `robust_tool.tools`：定义工具能力、JSON Schema、实际工具实现和注册表；不了解模型或数据集。
2. `robust_tool.env`：负责状态、参数校验、工具分发、执行结果、错误和目标检查。
3. `robust_tool.data`：负责可序列化的 Task、数据划分、确定性生成和后续数据转换。
4. `robust_tool.rollout`：负责模型输出解析、Policy/Model 适配以及完整交互轨迹记录。
5. `robust_tool.eval`：在新环境中重放轨迹，计算指标并分类失败；不信任预测中的成功标志。
6. `robust_tool.reward`：后续根据环境执行结果计算 Outcome 或 Dense Reward。
7. `scripts`：只提供薄命令行入口，不承载核心业务逻辑。

依赖方向固定为：

```text
tools → env → data / rollout → eval / reward → scripts
```

训练框架只能依赖环境和评测核心，不能让环境反向依赖 ms-swift 或 Transformers。

## 4. 工具调用接口

所有执行器接收统一的工具调用结构：

```json
{
  "name": "create_event",
  "arguments": {
    "title": "Project sync",
    "start": "2026-08-10T15:00:00",
    "end": "2026-08-10T15:30:00"
  }
}
```

模型可能使用 Chat Template、特殊 token 或嵌套字符串输出工具调用，这些模型相关差异都在 `rollout.parser` 边界处理。进入 Environment 后，调用必须已经转换成统一结构。

这样可以把以下问题分开评测：

- 原始文本是否为合法 JSON；
- 是否满足工具参数 Schema；
- 参数值在规范化后是否语义正确；
- 工具是否真的执行成功；
- 整个任务是否最终完成。

## 5. Environment 接口

每个工具环境必须实现：

```python
env.reset(task)
result = env.execute(tool_call)
state = env.get_state()
success = env.check_goal()
```

### `reset(task)`

- 深拷贝任务的 `initial_state`；
- 清空上一个任务的调用历史；
- 重新建立确定性状态；
- 保证两个任务之间不会泄漏状态。

### `execute(tool_call)`

执行过程依次进行：

1. 检查工具是否注册；
2. 检查该工具是否在当前任务的 `available_tools` 中；
3. 检查参数的必填字段、额外字段、类型和格式；
4. 调用真实工具实现；
5. 返回结构化结果并记录历史。

预期内的错误不会让 Benchmark 崩溃，而是返回类似下面的结果：

```json
{
  "tool_name": "create_event",
  "ok": false,
  "data": null,
  "error": {
    "code": "conflict",
    "message": "requested interval overlaps: evt-0001",
    "retriable": false
  },
  "validation_issues": [],
  "state_changed": false
}
```

### `get_state()`

返回可 JSON 序列化的深拷贝快照。调用方修改快照不能影响环境内部状态。

### `check_goal()`

根据当前环境状态和成功工具观测判断任务是否达到 `goal_state`。写操作通常检查最终状态；查询类工具因为不改变状态，需要检查 `required_observations`。

## 6. Task 数据结构

任务至少包含以下字段：

```json
{
  "task_id": "calendar_create_001",
  "domain": "calendar",
  "user_query": "Add a project sync tomorrow at 3 PM for 30 minutes.",
  "available_tools": ["list_events", "create_event"],
  "initial_state": {"events": []},
  "goal_state": {
    "events": {
      "contains": [
        {
          "title": "Project sync",
          "start": "2026-08-10T15:00:00"
        }
      ]
    }
  },
  "difficulty": "basic",
  "failure_tags": ["wrong_tool", "wrong_argument_value"],
  "expected_action": "call",
  "reference_calls": [
    {"name": "create_event", "arguments": {}}
  ],
  "metadata": {
    "split": "train",
    "generator_version": "calendar-toy-v2",
    "seed": 20260809
  }
}
```

几个容易混淆的字段：

- `goal_state`：任务成功的主要依据；
- `reference_calls`：用于诊断工具选择和参数，不作为最终成功的唯一依据；
- `expected_action`：第一步应该 `call`、`clarify` 还是 `respond`；
- `failure_tags`：该任务设计上重点考察的潜在失败，不代表模型已经失败；
- `metadata.split`：用于强制训练、验证和测试隔离。

## 7. Goal 约束语言

Calendar 环境支持以下目标：

```json
{
  "events": {
    "contains": [
      {"event_id": "evt-0001", "title": "Design review"}
    ],
    "absent": [
      {"event_id": "evt-0002"}
    ],
    "count": 2
  },
  "required_observations": [
    {
      "tool_name": "check_availability",
      "arguments": {
        "start": "2026-08-10T09:00:00",
        "end": "2026-08-10T10:00:00"
      },
      "result": {"available": true}
    }
  ]
}
```

语义说明：

- `contains`：最终事件列表中必须至少存在一个部分匹配的事件；
- `absent`：最终事件列表中不能出现匹配事件；
- `count`：最终事件总数必须相等；
- `required_observations`：轨迹中必须存在匹配的成功工具调用及返回结果。

对象匹配采用确定性的“部分匹配”：期望对象中的每个键都必须在实际对象中出现并相等，但实际对象可以包含额外字段。因此，任务不关心自动生成的 ID 时，可以不在目标中填写 `event_id`。

## 8. Calendar 状态语义

- 时间使用不带时区的 ISO-8601 字符串，规范化到秒；
- 时间区间采用 `[start, end)` 半开语义；
- 两个首尾相接的事件不冲突；
- Event ID 从重置后的状态确定性分配；
- `create_event` 或 `update_event` 产生冲突时返回 `conflict`，状态保持不变；
- `update_event` 是 patch 操作，未提供的字段保持不变；
- `list_events.start` 包含边界，`list_events.end` 不包含边界；
- 输出按 `(start, event_id)` 稳定排序；
- 缺失、额外、错误类型、错误格式和无效语义参数都必须在状态修改前被拒绝。

## 9. Trajectory 数据结构

一条轨迹完整保存：

```text
user
assistant tool_call
tool result
assistant tool_call
tool result
...
assistant final answer
```

每个 Assistant 消息会显式记录动作类型，工具消息会保存完整执行结果。轨迹还可以保存生成该轨迹的 Policy、模型和截断信息。

`final_state` 只作为推理阶段的调试记录。Evaluator 仍会重新执行工具调用，因此即使 `final_state` 被篡改，也不会影响真正的 Task Success。

## 10. 评测分层

第一版 Evaluator 独立报告：

1. **调用决策**：应该调用、澄清还是直接回答；
2. **工具选择**：预测工具与参考工具序列是否一致；
3. **JSON 合法性**：模型输出是否可以解析成标准调用；
4. **参数 Schema**：必填、额外、类型和格式是否正确；
5. **参数语义**：规范化后参数值是否等价；
6. **可执行性**：环境是否成功执行；
7. **任务成功**：重放后的状态和观测是否满足用户目标。

没有 eligible 样本的指标写成 `null`，而不是容易被误读的 0。每个指标同时保存分子和分母，确保结果可审计。

## 11. 失败分类

Failure Classifier 根据以下信息执行确定性规则：

- Task 的预期动作和参考调用；
- 解析后的工具调用；
- Schema 校验结果；
- 工具执行结果；
- 最终 Goal Check；
- 工具结果之后的模型行为。

同一条轨迹可以得到多个失败标签，并且每个标签都保留触发证据。详细规则见 [失败分类体系](failure_taxonomy.md)。

## 12. 数据隔离

当前生成器产生 25 条固定 Calendar 任务，并写入 Train、Validation 和 Test 文件。v1 内容主要由模板确定，但仍要求显式 seed，并将 seed 与生成器版本写入每条任务。

隔离规则：

- 只有 Train 可以进入未来的数据增强或训练；
- Validation 只用于配置选择；
- `clean_test` 和后续 `robust_test` 只能用于最终评测；
- 数据生成器升级时必须更新 `generator_version`；
- 冻结数据必须通过 `manifest.json` 的 SHA-256 校验。

## 13. 实验产物

每次运行拥有独立目录：

```text
experiments/results/<run_name>/
  config.json
  metrics.json
  failure_stats.json
  predictions.jsonl
  trajectories.jsonl
  evaluation.jsonl
  run.log
```

Evaluator 可以只使用 Task 文件和 `trajectories.jsonl` 重新计算全部指标与失败统计。任何无法追溯到这些产物的 README 数字都不应被视为正式结果。

## 14. Qwen 基线推理协议

第一阶段固定使用 `Qwen/Qwen2.5-1.5B-Instruct`，不同时比较多个模型。模型适配层位于 `robust_tool.models`，只向通用 Rollout Runner 暴露 `Policy.act(task, trajectory)`，Environment 和 Evaluator 不依赖 Transformers 或 ms-swift。

一次模型动作按下面的顺序产生：

1. 从 `available_tools` 读取本任务允许使用的工具；
2. 由 Tool Registry 生成标准 function schemas；
3. 把已有 Trajectory 转换为 Qwen Chat Template 消息；
4. 使用固定模型 revision、seed 和解码配置生成一个回合；
5. 将模型特有的 `<tool_call>` 输出解析成统一 `ToolCall`；
6. Environment 执行该调用，并把完整结果加入下一轮上下文；
7. 模型直接回答或询问澄清后结束本次 Rollout。

推理提示词要求每个回合最多生成一个工具调用。多工具任务通过“调用一个工具 → 接收结果 → 再决定下一步”完成，这样每一步都能独立执行、记录和分析。

解析器不会把格式错误的工具调用悄悄降级为普通文本。只要模型开始输出工具调用标记，即使 JSON 或闭合标签损坏，也会保留原始输出并生成 `json_valid=false` 的调用，使 Evaluator 可以统计 `invalid_json`。

模型只会看到 System Prompt、用户消息、允许使用的工具 Schema 和已经发生的工具结果。`goal_state`、`reference_calls`、测试标签和 Evaluator 诊断绝不会进入模型上下文。

每一步推理记录输入与输出 token 数、延迟、原始生成文本和 CUDA 峰值显存。实验目录还保存本次实际使用的 Task 快照，因此少量 smoke test 与完整 Validation/Test 运行都能使用相同 Evaluator 重算。

## 15. 当前有意延后的内容

当前阶段仍有意延后以下内容：

- ms-swift 训练格式的完整适配；
- 多工具和多轮长任务；
- 工具超时、不可用、恶意或部分返回等故障注入；
- 鲁棒性扰动套件；
- Hard-case Mining；
- Dense Reward、SFT 和 GRPO。

这些模块会复用已经冻结的 Task、Trajectory、Environment 和 Evaluator 接口，不能为了某个模型的结果而修改环境真值。
