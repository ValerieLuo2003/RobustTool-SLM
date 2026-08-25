# 评测协议

## 1. 评测目标

Evaluator 不只回答“模型生成的字符串像不像参考答案”，而是回答下面几个彼此独立的问题：

1. 模型是否正确判断了要不要调用工具；
2. 如果要调用，是否选对工具；
3. 输出能否被解析；
4. 参数结构是否合法；
5. 参数语义是否正确；
6. 调用是否实际执行成功；
7. 最终是否完成了整个任务。

分层评测能够区分“格式改善”和“能力改善”。例如，SFT 可能显著提高 JSON Valid Rate，但 Task Success 不一定同步提高。

## 2. 评测流程

```text
加载 Task 与 Trajectory
        ↓
检查 task_id 一一对应
        ↓
创建全新的 CalendarEnvironment
        ↓
按顺序重放 Trajectory 中的 ToolCall
        ↓
记录每次校验与真实执行结果
        ↓
检查最终状态 / 必要观测 / 最终回答
        ↓
计算逐任务诊断与 Failure Labels
        ↓
聚合 Metrics 与 Failure Distribution
```

Task 和 Trajectory 必须一一对应。缺失轨迹、额外轨迹或重复 `task_id` 都会直接报错，防止因为样本丢失而得到虚高指标。

## 3. Task Success 的判定

### 工具调用任务

同时满足以下条件才算成功：

- 第一动作是 `call`；
- 环境重放后 `check_goal()` 为真；
- 模型给出了非空最终回答。

如果 Task 定义了 `metadata.response_expectation`，最终回答还必须通过确定性短语检查。例如 `missing_tool` 任务要求回答同时提到被移除的工具，并包含“not available / unavailable / cannot”之一。未定义该字段的普通任务不受影响。

### 澄清任务

同时满足：

- 第一动作是 `clarify`；
- 没有工具调用；
- 环境状态未被破坏且目标成立；
- 澄清文本非空。

### 无工具回答任务

同时满足：

- 第一动作是 `respond`；
- 没有工具调用；
- 环境目标成立；
- 最终回答非空。

Evaluator 不读取模型提供的 `success`，也不信任 Trajectory 中保存的 `final_state`。所有状态都通过新环境重放重新得到。

## 4. 参数规范化

参数评测不完全依赖字符串 exact match。目前支持以下确定性规范化：

- `start`、`end`：解析 ISO-8601 后统一到秒；
- `title`、`location`、`description`、`event_id`：去除首尾空格、合并连续空格并忽略大小写；
- `attendees`：逐项规范化后排序，因此参与者顺序不影响结果；
- 字符串 `true` / `false`：规范化为 Boolean；
- 对象：按键递归规范化；
- 其他数值和类型：保持原始语义。

例如：

```text
2026-08-10 09:00
2026-08-10T09:00:00
```

会被视为相同时间。只有无法可靠使用确定性规则判断的参数，未来才考虑 LLM-as-Judge。

## 5. 指标定义

所有比率都保存：

```json
{
  "value": 0.8,
  "numerator": 8,
  "denominator": 10
}
```

没有 eligible 样本时：

```json
{
  "value": null,
  "numerator": 0,
  "denominator": 0
}
```

### `call_decision_accuracy`

```text
正确的 call / clarify / respond 决策任务数
──────────────────────────────────────────
总任务数
```

### `tool_selection_accuracy`

```text
与 reference_calls 同位置且名称正确的调用数
───────────────────────────────────────────
参考调用总数
```

### `json_valid_rate`

```text
json_valid = true 的预测调用数
──────────────────────────────
预测调用总数
```

### `argument_schema_accuracy`

```text
通过工具 JSON Schema 校验的调用数
────────────────────────────────
预测调用总数
```

Schema 校验包含必填字段、额外字段、类型、数组元素类型和 datetime 格式。

### `argument_semantic_accuracy`

```text
规范化后正确的参考参数个数
────────────────────────────
参考参数总数
```

如果工具名称错误或调用无法解析，该参考调用的参数计为错误。

### `executable_call_rate`

```text
Environment 返回 ok = true 的调用数
───────────────────────────────────
预测调用总数
```

Schema 正确不等于可执行成功。例如，合法的时间参数仍可能与现有事件冲突。

### `task_success_rate`

```text
通过完整重放与 Goal Check 的任务数
─────────────────────────────────
总任务数
```

这是项目最重要的 Outcome 指标。

### `final_answer_semantic_accuracy`

只在定义了 `response_expectation` 的任务上统计最终回答是否满足确定性内容条件。当前用于 `missing_tool`；没有 eligible 样本时为 `null`。它不替代环境状态检查，也不使用 LLM-as-Judge。

### `multi_turn_task_success_rate`

只在 `reference_calls` 大于 1 的任务上统计完整成功率。当前 toy benchmark 没有 eligible 样本，因此值为 `null`。

### `recovery_success_rate`

只在轨迹中实际遇到 retriable 错误的任务上统计最终成功率。Robustness v1 的 `tool_failure` 会在第一次调用时注入确定性 timeout；模型只有重试成功、达到 Goal 并给出最终回答，才计为恢复成功。

### `invalid_tool_call_rate`

```text
JSON 无效、工具未注册或参数 Schema 无效的调用数
──────────────────────────────────────────────
预测调用总数
```

### `unnecessary_tool_call_rate`

```text
expected_action 不是 call 但发生工具调用的任务数
─────────────────────────────────────────────
总任务数
```

### `average_tool_calls_per_task`

```text
预测工具调用总数
───────────────
总任务数
```

该指标用于发现重复调用、低效率轨迹和潜在 reward hacking。

## 6. 输出文件

运行：

```bash
python scripts/run_eval.py --run-name <run_name>
```

Evaluator 会在对应实验目录写入：

- `metrics.json`：汇总指标；
- `failure_stats.json`：每类失败的任务数和任务占比；
- `evaluation.jsonl`：逐任务重放状态、诊断、失败标签与证据；
- `run.log`：追加评测任务数。

查看 Failure Distribution：

```bash
python scripts/analyze_failures.py \
  experiments/results/<run_name>/failure_stats.json
```

## 7. 当前限制

- 当前语义规范化只覆盖 Calendar v1 中的确定性字段；
- Tool Selection 依赖 `reference_calls` 的位置对应；
- Final Answer 的确定性语义检查目前只覆盖显式配置的 `missing_tool`，普通自然语言回答仍主要检查非空与环境目标；
- `partial_tool_response` 是 `ok=true` 但缺失关键字段，因此不进入 retriable-error Recovery 分母，只通过 Task Success 检查是否正确重试；
- 还没有实现 pass@k 和 pass^k。

## 8. Robustness Gap

模型在 Clean 与 Robust Task 上分别运行并完成 Environment Evaluation 后，执行：

```bash
python scripts/compare_robustness.py \
  --clean-run experiments/results/<clean_run> \
  --robust-run experiments/results/<robust_run> \
  --output-prefix experiments/results/<model>_robustness_validation
```

每条 Robust Task 都通过 `source_task_id` 与 Clean Task 配对。对某个 setting：

```text
Robustness Gap
= 同源 Clean Task Success
- Perturbed Task Success
```

脚本还输出每个 setting 的分层指标、Failure Distribution，以及 `Clean fail → Perturbed success` / `Clean success → Perturbed fail` 的配对迁移。Clean 与 Robust 运行的模型 revision、adapter 权重哈希、解码配置和最大步数必须一致，否则拒绝比较。

这些限制应在解释结果时明确说明，不能把 Oracle 100% 当作模型能力结论；Oracle 只证明数据目标、工具执行和 Evaluator 自洽。
