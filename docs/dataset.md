# 数据集协议

## 1. 数据集的作用

数据协议先保证环境与评测自洽，再为正式 SFT 提供可执行监督：

1. Calendar 工具环境是否可以确定性执行；
2. Task 与 Trajectory 是否能够完整表达工具 Agent 任务；
3. Evaluator 是否能正确区分成功、格式错误、参数错误、工具错误和任务失败。

上述三件事已通过 toy benchmark、LoRA smoke、6000 条正式 SFT 和 500 条统一 Validation 评测验证。当前进入 Failure-aware Train 数据阶段。

## 2. 当前规模与划分

`calendar-toy-v2` 共包含 25 条任务：

| 划分 | 数量 | 允许用途 |
|---|---:|---|
| Train | 15 | 数据转换、训练流程调试、未来数据增强 |
| Validation | 5 | 参数与配置选择 |
| Test | 5 | 冻结基础评测 |
| Clean Test | 5 | 当前与 Test 完全相同，作为后续鲁棒性对照 |

Train、Validation 和 Test 的 `task_id` 不允许重叠。构造 Hard Cases 时只允许使用模型在 Validation 上汇总出的失败类别、名次和比例，不能复制 Validation 失败样本，也不能用 Test 结果选择数据方向。

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

正式 Clean 数据包含八类配额：查询、创建、更新、删除、空闲检查、澄清、无工具和多步任务。多步任务包括 `check_availability → create_event`、`list_events → update_event` 和 `list_events → delete_event`，后一步需要使用前一步工具结果中的状态或 event ID。工具错误恢复与扰动任务由独立的 Robustness Validation 生成器构造，不混入 Clean 或 SFT Train。

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

## 5. Failure-aware Train 数据

正式 SFT 在 500 条 Validation 上达到 92% Task Success。失败统计的 Top 3 是：

| 失败标签 | 失败任务数 | 占 500 条任务 |
|---|---:|---:|
| `wrong_argument_value` | 28 | 5.6% |
| `ignore_tool_result` | 15 | 3.0% |
| `missing_argument` | 11 | 2.2% |

`configs/data/calendar_failure_aware_v1.json` 冻结 3000 条全新 Train 数据：

| 目标失败 | 数量 | 主要任务族 |
|---|---:|---|
| `wrong_argument_value` | 1200 | 半开时间边界、窄时间窗口、精确创建参数、相似事件的精确更新 |
| `ignore_tool_result` | 1000 | list 后按返回 ID 更新/删除、availability 后创建/停止、create 后按新 ID 更新 |
| `missing_argument` | 800 | 创建必填字段、更新所需 event ID、时间区间端点、缺少必填信息时澄清 |

生成命令：

```bash
python scripts/build_hard_cases.py \
  --failure-targets experiments/results/qwen2_5_1_5b_sft_formal_v1_validation_new3090/failure_targets.json
```

生成器遵循以下隔离规则：

1. Validation 只提供失败标签、排名、计数和任务快照哈希；
2. 不读取或改写具体失败任务的用户问题与参考调用；
3. Test 只用于哈希及 task ID/规范化问题文本碰撞审计，不参与生成；
4. Hard-case 只标记为 Train，不生成 Validation 或 Test 文件；
5. 新数据与原 Train、Validation、Test 的 task ID 和规范化问题文本都必须零重叠；
6. 3000 条任务必须通过同一 Calendar Environment 的全量 Oracle 重放，失败一条就拒绝写出；
7. 最终同时保存 Task、Trajectory、ms-swift Agent JSONL 和带 SHA-256 的 manifest。

## 6. Task Schema

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

## 7. ToolCall Schema

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

## 8. Trajectory Schema

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

## 9. 确定性与版本管理

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

## 10. Robustness Validation 数据

生成命令：

```bash
python scripts/generate_robustness_data.py
```

正式配置固定在 `configs/data/calendar_robustness_validation_v1.json`：

```json
{
  "dataset_name": "calendar-robustness-validation-v1",
  "generator_version": "calendar-robustness-v1",
  "seed": 20260825,
  "source_split": "validation",
  "count_per_kind": 50
}
```

输出目录为 `data/processed/calendar_robustness_validation_v1/`：

```text
tasks.jsonl                 500 条 Robust Validation Task
oracle_trajectories.jsonl   可执行 Oracle 轨迹
manifest.json               配置、来源、输出哈希与 Oracle 审计
```

每条 Robust Task 都在 `metadata.robustness` 中保存：

- `kind`：10 类扰动之一；
- `source_task_id`：对应的 Clean Validation Task；
- `source_task_sha256`：源 Task 的规范化内容哈希；
- 该扰动需要的 `faults`、`response_mutations` 或 `synthetic_tool_results`。

当前数据包含 10 类 × 50 条，使用 323 个不同的源 Validation Task；单个源 Task 最多产生 5 个不同 setting 的变体。类别选择采用确定性分层抽样，例如 ambiguous 在 create / update / delete 间近似均衡，noisy / partial 在 list / availability 间各占一半。

隔离规则：

- 只读取 Validation，不读取 Train；
- 输出用途固定为 Robust Validation，不转换为 ms-swift Train；
- 不读取 Clean Test，也不根据 Test 表现修改扰动；
- 后续 Robust Test 必须从冻结 Test 单独生成，并使用独立配置与输出目录；
- 每次生成都必须达到 Oracle Task Success 100%，否则拒绝写入正式结果。

当前任务文件 SHA-256 为 `a62c76019ad935f58d915e096cad38f02fca8701cb702be1a61fe2a7f7c9f18e`；源 Validation SHA-256 为 `ad2202da79bdbae87a486d40ebf3ab44ee3223481c1021d29e129213ec261dee`。生成代码发生语义变化后必须重新生成并以新 manifest 为准。

最终测试阶段使用独立配置 `configs/data/calendar_robustness_test_v1.json`，从冻结的 `calendar_formal_v1/tasks/test.jsonl` 单独生成 500 条 `robust_test`。它与 Robustness Validation 使用同一套 10 类扰动和 50 条/类配额，但不读取 Validation 结果，也不进入任何训练或 checkpoint 选择。

## 11. Recovery Failure-aware v2 训练数据

正式鲁棒性验证显示 `missing_tool`、`tool_failure` 和 `partial_tool_response` 是 SFT 后仍未解决的三类失败。生成命令为：

```bash
python scripts/build_recovery_cases.py \
  --config configs/data/calendar_recovery_failure_aware_v2_smoke.json \
  --output-dir data/processed/calendar_recovery_failure_aware_v2_smoke

python scripts/build_recovery_cases.py
```

正式配置固定生成 3000 条 Train-only 轨迹，三类各 1000。生成器从原始 6000 条 Train 中无放回选择 3000 个源任务，并为训练任务创建新 ID、新问题表达和独立 metadata；Validation/Test 只参与 ID 与规范化问题碰撞审计，不参与选择或变换。

三类轨迹协议为：

- `missing_tool`：移除必需工具，Oracle 直接说明具体工具不可用，不生成工具调用；
- `tool_failure`：第一次调用返回 `retriable=true` 的 timeout，Oracle 使用完全相同的参数重试一次；
- `partial_tool_response`：第一次只读结果移除关键字段，Oracle 不猜测并重复查询一次。

当前正式审计结果：Task Success 3000/3000，Multi-turn 2000/2000，Recovery 1000/1000；3000 个源 Train ID 无复用，生成任务与原 Train/Validation/Test 的 Task ID 和规范化问题文本重叠均为 0。输出 SHA-256 为：

- Task：`9e2d8ec5823ea189953b9f0a562d0a1028135aea3e93144e687fb72138e775f7`；
- Trajectory：`caca1c92120f7e8436bd2f50c0d66d4038510da416c7757f575ae3daf3acdfdb`；
- ms-swift Train：`1328078cc38fc3c604166cb0242be16570bde39fe87eb28765f027ea97466e1a`。

这些文件属于生成产物，不提交 Git；仓库提交生成器、冻结配置和测试。正式训练必须等 Random Augmentation 对照完成后，使用同样新增 3000 条、相同 Base 数据和相同优化配置。

## 12. 后续扩展计划

当前正式规模已经达到 SFT 6000 条 Train、第一版 Failure-aware 3000 条 Train，并准备好第二版 Recovery 3000 条 Train。后续扩展重点不是继续无目的增加数量，而是：

- 保持 500 条 Validation 和 1000 条冻结 Clean Test；
- 在方法冻结后从 Clean Test 单独构建 `robust_test`；
- 新增 Calendar、Travel、Shopping、Weather 等 domain；
- 新增多工具、多轮、澄清、无工具、工具故障和恢复任务。

规模扩展不能以牺牲可执行性、数据隔离和确定性评测为代价。
