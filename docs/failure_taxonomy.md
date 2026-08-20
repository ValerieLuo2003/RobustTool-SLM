# 失败分类体系

## 1. 为什么使用多标签

工具调用失败通常不是单一原因。例如，一条轨迹可能同时出现：

1. 选错工具；
2. 参数缺失；
3. 执行失败后仍然输出“已完成”。

如果只给一个标签，会丢失真实错误链。因此 Failure Classifier 使用确定性、多标签分类，并在 `evaluation.jsonl` 中保存每个标签的证据。

## 2. 标准失败标签

| 标签 | 中文含义 | 第一版触发条件 |
|---|---|---|
| `wrong_call_decision` | 调用决策错误 | 第一动作与 `expected_action` 不一致 |
| `wrong_tool` | 首个工具选择错误 | 第一个预测工具与第一个参考工具不同 |
| `missing_argument` | 缺少参数 | Schema 中的 required 参数未提供 |
| `extra_argument` | 多余参数 | 提供了 Schema 不允许的字段 |
| `wrong_argument_type` | 参数类型错误 | 字符串、Boolean、数组或数组元素等类型不匹配 |
| `wrong_argument_value` | 参数值错误 | 格式、枚举、长度无效，或规范化后与参考值不同 |
| `invalid_json` | JSON 无法解析 | 原始工具调用不能转换成标准 JSON 对象 |
| `hallucinated_tool` | 幻觉工具 | 调用了 Tool Registry 中不存在的工具 |
| `unnecessary_tool_call` | 不必要工具调用 | 应澄清或直接回答时调用了工具 |
| `repeated_tool_call` | 重复调用 | 相同工具名和参数的调用在同一轨迹中重复出现 |
| `ignore_tool_result` | 忽略工具结果 | 最后一次工具调用失败，但模型仍给出非空最终回答 |
| `wrong_next_tool` | 后续工具错误 | 多步任务中后续工具与参考顺序不同，或产生额外调用 |
| `tool_error_recovery_failure` | 工具错误恢复失败 | 工具出现执行错误，后续没有成功调用完成恢复 |
| `clarification_failure` | 澄清失败 | 应该澄清时没有执行 `clarify` |
| `final_answer_failure` | 最终回答失败 | 最终回答为空，或任务失败但没有更具体的已知证据 |

## 3. Failure Evidence

分类结果不仅保存标签，还保存证据。例如：

```json
{
  "failures": [
    "wrong_tool",
    "missing_argument"
  ],
  "failure_evidence": {
    "wrong_tool": [
      {
        "call_index": 0,
        "expected": "create_event",
        "predicted": "update_event"
      }
    ],
    "missing_argument": [
      {
        "call_index": 0,
        "field": "event_id",
        "message": "missing required argument: event_id"
      }
    ]
  }
}
```

这使得后续 Hard-case Mining 可以根据标签筛选样本，也可以进一步检查具体错误字段。

## 4. 标签之间的关系

一些常见组合：

- `wrong_call_decision` + `unnecessary_tool_call`：不该调用工具时进行了调用；
- `wrong_tool` + `missing_argument`：选择了错误工具，因此该工具需要的参数也不存在；
- `invalid_json` + `final_answer_failure`：调用无法解析，之后也没有正常回答；
- `tool_error_recovery_failure` + `ignore_tool_result`：工具失败后没有恢复，却继续给出完成式回答；
- `repeated_tool_call` + `wrong_next_tool`：模型重复同一调用，而不是进入下一步。

Failure Distribution 统计的是“包含该标签的任务数”，因此一条任务可以进入多个类别，各类别比例之和可以超过 100%。

## 5. 当前边界

- `wrong_argument_value` 目前只依赖确定性规范化和 Schema，不使用 LLM Judge；
- `ignore_tool_result` 当前通过“最后调用失败但仍回答”近似判断，未来需要更细的内容分析；
- `wrong_next_tool` 要在真正的多步任务中才有充分意义；
- `tool_error_recovery_failure` 需要超时、不可用、部分返回等故障注入后才能系统评测；
- `final_answer_failure` 当前未细分为遗漏关键信息、错误总结或与工具结果矛盾。

后续扩展标签时必须保持：确定性、可测试、带证据、向后兼容，并在实验报告中记录版本变化。
