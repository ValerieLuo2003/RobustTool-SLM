# 数据集协议

## 1. 数据集的作用

数据协议先保证环境与评测自洽，再为正式 SFT 提供可执行监督：

1. Calendar 工具环境是否可以确定性执行；
2. Task 与 Trajectory 是否能够完整表达工具 Agent 任务；
3. Evaluator 是否能正确区分成功、格式错误、参数错误、工具错误和任务失败。

上述三件事已通过 toy benchmark 和 LoRA smoke 验证，当前已经进入 6000 条正式 Train 轨迹阶段。

## 2. 当前规模与划分

`calendar-toy-v2` 共包含 25 条任务：

| 划分 | 数量 | 允许用途 |
|---|---:|---|
| Train | 15 | 数据转换、训练流程调试、未来数据增强 |
| Validation | 5 | 参数与配置选择 |
| Test | 5 | 冻结基础评测 |
| Clean Test | 5 | 当前与 Test 完全相同，作为后续鲁棒性对照 |

Train、Validation 和 Test 的 `task_id` 不允许重叠。未来构造 Hard Cases 时只能读取 Train 或模型在开发集上的失败，不能读取 Test 内容。

v2 在首次 Qwen Validation smoke test 后修复了任务文本歧义：凡是参考参数包含绝对日期，用户请求也必须显式给出年份；工具 Schema 同时明确时间值必须使用不带时区后缀的本地 ISO-8601 格式。该修改发生在正式 Clean Test 基线冻结前。

正式 `calendar-formal-sft-v1` 的冻结规模为：

| 划分 | 数量 | 允许用途 |
|---|---:|---|
| Train | 6000 | 正式 SFT |
| Validation | 500 | checkpoint 和训练配置评测、失败挖掘 |
| Test / Clean Test | 1000 | 冻结后的最终 Base/SFT 比较，不参与训练或失败挖掘 |

配置位于 `configs/data/calendar_formal_sft_v1.json`。生成器会检查 7500 个 task ID、7500 条规范化用户请求均唯一，并让 Oracle 在全新环境中重放全部任务；任何重复或目标失败都会终止构建。

## 3. 当前任务类型

25 条任务覆盖：

- `list_events`：按时间范围查询事件，包括空结果；
- `create_event`：创建事件，覆盖必填和可选参数；
- `update_event`：根据 Event ID 修改标题、时间、地点或参与者；
- `delete_event`：删除指定事件；
- `check_availability`：判断空闲与冲突时间；
- `clarify`：用户缺少时间、日期、时长等必要信息时要求澄清；
- `respond`：用户只询问能力时直接回答，不调用工具。

正式数据包含八类配额：查询、创建、更新、删除、空闲检查、澄清、无工具和多步任务。多步任务包括 `check_availability → create_event`、`list_events → update_event` 和 `list_events → delete_event`，后一步需要使用前一步工具结果中的状态或 event ID。工具错误恢复与扰动任务仍留到 Failure-aware 阶段。

## 4. ms-swift SFT 数据

`scripts/build_sft_data.py` 使用 Train 和 Validation Task 的 Oracle 轨迹构造官方 Agent 格式：

```json
{
  "task_id": "calendar_create_001",
  "source_split": "train",
  "tools": "[{...}]",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "tool_call", "content": "{...}"},
    {"role": "tool_response", "content": "{...}"},
    {"role": "assistant", "content": "Done."}
  ]
}
```

其中 `tools`、`tool_call.content` 和 `tool_response.content` 都是 JSON 字符串。`goal_state`、`reference_calls` 和 Failure 标签不会进入训练记录。生成前会同时加载 Test task_id 做集合交叉检查，但 Test 内容不会被转换或写入训练文件。

`calendar-sft-smoke-v1` 的 15/5 轨迹只用于训练链路验证。正式转换入口是 `scripts/build_formal_sft_data.py`，输出 6000/500 条 Agent 轨迹，覆盖划分独立的表达改写、参数组合、多步、澄清和无工具决策。Test 文件只做集合与哈希检查，不会被转换。

## 5. Task Schema

每条 Task 包含：

| 字段 | 类型 | 含义 |
|---|---|---|
| `task_id` | string | 全局唯一任务 ID |
| `domain` | string | 当前为 `calendar` |
| `user_query` | string | 用户自然语言请求 |
| `available_tools` | list[string] | 该任务允许模型使用的工具 |
| `initial_state` | object | 环境重置时的初始状态 |
| `goal_state` | object | 最终状态或必要观测约束 |
| `difficulty` | string | 任务难度或类型标记 |
| `failure_tags` | list[string] | 该任务重点考察的潜在失败 |
| `expected_action` | string | `call`、`clarify` 或 `respond` |
| `reference_calls` | list[ToolCall] | 用于工具与参数诊断的参考调用 |
| `metadata` | object | 数据划分、生成器版本、seed 等信息 |

示例：

```json
{
  "task_id": "calendar_create_001",
  "domain": "calendar",
  "user_query": "Create 'Focus time' from 2026-08-10T10:00:00 to 2026-08-10T11:00:00.",
  "available_tools": [
    "list_events",
    "create_event",
    "update_event",
    "delete_event",
    "check_availability"
  ],
  "initial_state": {"events": []},
  "goal_state": {
    "events": {
      "contains": [
        {
          "title": "Focus time",
          "start": "2026-08-10T10:00:00",
          "end": "2026-08-10T11:00:00"
        }
      ]
    }
  },
  "difficulty": "basic",
  "failure_tags": [
    "missing_argument",
    "wrong_argument_type",
    "wrong_argument_value"
  ],
  "expected_action": "call",
  "reference_calls": [
    {
      "name": "create_event",
      "arguments": {
        "title": "Focus time",
        "start": "2026-08-10T10:00:00",
        "end": "2026-08-10T11:00:00"
      }
    }
  ],
  "metadata": {
    "split": "train",
    "generator_version": "calendar-toy-v2",
    "seed": 20260809
  }
}
```

## 6. ToolCall Schema

标准调用格式为：

```json
{
  "name": "check_availability",
  "arguments": {
    "start": "2026-08-11T14:00:00",
    "end": "2026-08-11T15:00:00"
  },
  "json_valid": true
}
```

模型解析失败时仍需保留原始证据：

```json
{
  "name": "",
  "arguments": {},
  "json_valid": false,
  "raw": "{\"name\": ...",
  "parse_error": "..."
}
```

这样 Evaluator 才能把 `invalid_json` 与工具或参数错误分开统计。

## 7. Trajectory Schema

Trajectory 保存完整交互，而不是只保存最终答案：

```json
{
  "task_id": "calendar_create_001",
  "messages": [
    {"role": "user", "content": "..."},
    {
      "role": "assistant",
      "action": "call",
      "tool_call": {
        "name": "create_event",
        "arguments": {"title": "..."},
        "json_valid": true
      }
    },
    {
      "role": "tool",
      "tool_result": {
        "tool_name": "create_event",
        "ok": true,
        "data": {"event": {}},
        "error": null,
        "validation_issues": [],
        "state_changed": true
      }
    },
    {"role": "assistant", "action": "respond", "content": "Done."}
  ],
  "final_state": {"events": []},
  "metadata": {"policy": "oracle"}
}
```

SFT trajectory 保持同样的 user / assistant / tool 顺序，再由 `converter_swift.py` 转为 ms-swift 所需格式。

## 8. 确定性与版本管理

生成命令：

```bash
python scripts/generate_data.py --seed 20260809
```

输出目录为 `data/eval/`，其中：

```text
toy_tasks.jsonl       全部 25 条任务
toy_train.jsonl       Train
toy_validation.jsonl  Validation
toy_test.jsonl        Test
clean_test.jsonl      Clean Test
manifest.json         版本、数量与 SHA-256
```

修改生成数据时必须遵循：

1. 修改生成器代码，而不是手工编辑生成后的 JSONL；
2. 影响任务语义时更新 `GENERATOR_VERSION`；
3. 使用显式 seed 重新生成；
4. 检查划分数量、ID 唯一性和哈希；
5. 运行全量测试；
6. 在实验文档中记录变更原因。

## 9. 后续扩展计划

正式训练数据计划扩展到：

- SFT Train：5k～15k trajectories；
- Validation：500～1000 tasks；
- Test：约 1000 tasks；
- 单独保留 `clean_test` 与 `robust_test`；
- 新增 Calendar、Travel、Shopping、Weather 等 domain；
- 新增多工具、多轮、澄清、无工具、工具故障和恢复任务。

规模扩展不能以牺牲可执行性、数据隔离和确定性评测为代价。
