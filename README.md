# RobustTool-SLM

**面向可靠工具调用小模型的失败感知后训练框架**
**Failure-Aware Post-Training for Reliable Tool-Calling Small Language Models**

## 一句话介绍

RobustTool-SLM 不是一个简单的“Qwen + LoRA”微调项目，而是一套可以实际执行工具、自动判断任务成败、统计模型失败原因，并用失败样本继续训练模型的完整实验框架。

项目最终要建立下面这条闭环：

```text
本地工具环境
    ↓
可执行 Benchmark 与自动评测
    ↓
Qwen 基础模型评测
    ↓
SFT
    ↓
失败检测与失败分类
    ↓
针对高频失败生成 Hard Cases
    ↓
Failure-aware SFT
    ↓
基于真实工具执行反馈的 GRPO
    ↓
鲁棒性评测、消融实验与最终分析
```

核心研究问题是：小参数语言模型在工具调用中究竟会怎么失败，SFT 能修复什么，失败驱动的数据增强能否优于随机扩充数据，执行反馈与 GRPO 又能否真正提高任务完成率和错误恢复能力。

## 为什么不能只比较 Tool Call 字符串

一个工具调用 Agent 的回答看起来格式正确，并不代表任务真的完成。例如：

- 工具名称正确，但时间参数填错；
- 工具成功执行，但最终环境状态不符合用户目标；
- 工具返回错误后，模型仍然声称“已经完成”；
- 用户信息不完整，模型没有澄清，而是自行猜测；
- 用户只是询问能力，模型却进行了不必要的工具调用。

因此，本项目把评测拆成多个独立层次：调用决策、工具选择、JSON 格式、参数 Schema、参数语义、可执行性和最终任务成功。最终是否成功，优先由工具执行后的环境状态决定，而不是依赖模型自己报告成功，也不是只做字符串完全匹配。

## 当前完成到哪里

目前已经完成 Week 1 基础框架、正式 SFT、Failure-aware 数据闭环、Base / SFT / Failure-SFT Validation 对比，以及 10 类 Robustness Validation 的数据与配对评测框架。Failure-SFT 使用普通 SFT 的高频失败生成 3000 条针对性 Train 轨迹，在 RTX 3090 上从同一个 Base Model 重新训练；500 条 Validation 的 Task Success 从 SFT 的 92.0% 进一步提升到 93.8%。Clean Test 仍保持冻结，下一步是运行三种模型的 Robust Validation，并实现 Random Augmentation 对照实验。

| 模块 | 当前状态 | 说明 |
|---|---|---|
| 仓库与配置骨架 | 已完成 | 代码、数据、配置、实验和文档分层组织 |
| Calendar 工具环境 | 已完成 | 5 个真实可执行工具，状态完全在本地维护 |
| Task / Trajectory Schema | 已完成 | 支持完整记录用户、工具调用、工具结果和最终回答 |
| Toy Benchmark | 已完成 | 25 条确定性 Calendar 任务，按 15/5/5 划分 |
| 正式 Calendar 数据 | 已完成 | 6000/500/1000，覆盖八类任务和三种多步状态依赖 |
| Oracle / Random Baseline | 已完成 | 不依赖模型，用来验证环境与评测器 |
| 第一版 Evaluator | 已完成 | 支持分层指标、环境重放和失败分类 |
| 单元测试 | 已完成 | 57 项测试通过，覆盖环境、工具、数据、轨迹、指标、评测、鲁棒性与训练前检查 |
| Qwen Base Inference | 已完成正式 Validation | 已固定 Qwen2.5-1.5B-Instruct，并在 RTX 3090 完成 500 条环境推理与自动评测 |
| SFT | 已完成正式训练与评测 | 6000 条 Train、1 epoch、LoRA；Validation Task Success 从 66% 提升到 92% |
| Failure-aware 数据 | 已完成 | 从 SFT Validation 选择 Top 3 failure，3000 条全新 Train 已通过 Oracle 和跨 split 泄漏审计 |
| Failure-SFT | 已完成正式训练与评测 | 原始 6000 + 针对性 3000，从同一 Base 重训；Validation Task Success 达到 93.8% |
| 鲁棒性 Benchmark | 数据与评测框架已完成 | 10 类 × 50 条 Robust Validation 已通过 Oracle；待 Base / SFT / Failure-SFT 模型实测 |
| GRPO | 未开始 | Week 4 |

## 当前系统是怎样工作的

```text
Task
 ├─ 用户请求 user_query
 ├─ 初始环境 initial_state
 ├─ 可用工具 available_tools
 ├─ 目标环境 goal_state
 └─ 参考调用 reference_calls（只用于诊断）
                │
                v
        Oracle / Random / Model
                │
                v
           Trajectory
     user → assistant tool_call
          → tool result → ...
          → assistant final answer
                │
                v
       Evaluator 在新环境中重放
         ├─ 检查调用决策
         ├─ 检查工具和参数
         ├─ 真正执行每个工具
         ├─ 检查最终状态与观测
         └─ 输出指标、失败标签和证据
```

这里最重要的设计是：`Trajectory` 中保存的 `final_state` 不能作为可信答案。Evaluator 会重新创建一个干净环境并重放所有调用，防止模型或推理代码伪造成功状态。

## 快速运行

### 1. 环境要求

- Python 3.10 或更高版本；
- 当前 Week 1 核心只使用 Python 标准库；
- 不需要 GPU，不需要下载模型，也不需要安装 ms-swift。

在仓库根目录执行：

```bash
python scripts/generate_data.py --seed 20260809
python scripts/run_baseline.py --policy oracle --run-name toy_oracle
python scripts/run_eval.py --run-name toy_oracle
python -m unittest discover -s tests -v
```

这四条命令分别完成：

1. 生成 25 条确定性 Calendar toy tasks；
2. 用 Oracle Policy 生成完整轨迹；
3. 在新环境中重放轨迹并计算指标；
4. 运行全部单元测试。

### 2. 验证失败分析是否有效

Random Policy 是一个带固定随机种子的弱基线，专门用于制造错误并验证 Evaluator：

```bash
python scripts/run_baseline.py \
  --policy random \
  --seed 7 \
  --run-name toy_random_seed7

python scripts/run_eval.py --run-name toy_random_seed7

python scripts/analyze_failures.py \
  experiments/results/toy_random_seed7/failure_stats.json
```

如果是在 PowerShell 中，也可以把多行命令写成单行执行。

### 3. 运行 Qwen GPU 基线

第一阶段只使用 `Qwen/Qwen2.5-1.5B-Instruct`，模型 revision 已固定在配置文件中。先安装可选训练依赖，再在 Validation 上运行少量任务：

```bash
python -m pip install -e ".[train]"

python scripts/run_qwen_baseline.py \
  --config configs/models/qwen2_5_1_5b_instruct.json \
  --tasks data/eval/toy_validation.jsonl \
  --limit 5 \
  --run-name qwen2_5_1_5b_base_val_smoke

python scripts/run_eval.py \
  --run-name qwen2_5_1_5b_base_val_smoke
```

模型权重由 ModelScope 下载到本机缓存，不会写入 Git 仓库。推理过程使用工具的标准 JSON Schema 和模型 Chat Template；模型看不到 `goal_state` 或 `reference_calls`。每一步的原始输出、token 数、延迟和峰值显存都会随 Trajectory 保存。

### 4. 构建并检查 SFT smoke 数据

```bash
python scripts/build_sft_data.py
python scripts/run_sft.py \
  --config configs/sft/qwen2_5_1_5b_lora_smoke.json
```

这一步只使用 15 条 Train 与 5 条 Validation Oracle 轨迹运行 20 step LoRA，用来检查 ms-swift 格式、forward、backward、loss、显存和 checkpoint。RTX 3090 smoke 已通过，但它不是正式训练，也不会读取 Clean Test。数据和训练输出都被 `.gitignore` 排除，只提交生成器和配置。

### 5. 生成正式 SFT 数据并启动正式训练

```bash
python scripts/generate_formal_data.py
python scripts/build_formal_sft_data.py
python scripts/run_sft.py \
  --config configs/sft/qwen2_5_1_5b_lora_formal_v1.json
```

正式配置生成 6000 条 Train、500 条 Validation 和 1000 条 Clean Test。只有 Train/Validation 会转换为 ms-swift Agent 轨迹；Test 只参与哈希和隔离审计。生成器要求 task ID 和用户请求全局唯一，执行全量 Oracle 重放，并拒绝任何不能达到目标状态的数据。

LoRA checkpoint 使用原来的推理入口评测：

```bash
python scripts/select_sft_checkpoint.py \
  experiments/results/qwen2_5_1_5b_sft_formal_v1

python scripts/run_qwen_baseline.py \
  --tasks data/processed/calendar_formal_v1/tasks/validation.jsonl \
  --adapter-path <selected_checkpoint.json 中的 selected_checkpoint> \
  --run-name qwen2_5_1_5b_sft_formal_v1_validation

python scripts/run_eval.py --run-name qwen2_5_1_5b_sft_formal_v1_validation
```

`select_sft_checkpoint.py` 只读取训练时的 Validation loss，在实际存在且完整的 LoRA checkpoint 中选择最低值，并将所有候选、最终路径和权重哈希写入 `selected_checkpoint.json`。它不会读取 Test。Base 与 LoRA adapter 复用同一个 `QwenTransformersPolicy`、Environment Rollout 和 Evaluator；Adapter 配置与权重哈希也会写入推理运行产物，防止误用 checkpoint。

本次正式 Validation 结果如下。三次运行使用相同的 500 条任务快照、模型 revision、最大轮数、解码协议和 Environment Evaluator。这里仍不是最终 Test 结论，但可以用于选择方法和分析失败：

| 模型 | Tool Selection | Argument Semantic | Executable | Task Success | Multi-turn Success | Invalid Call |
|---|---:|---:|---:|---:|---:|---:|
| Base | 91.35% | 83.49% | 80.16% | 66.00% | 18.92% | 18.40% |
| SFT | 94.94% | 94.52% | 96.69% | 92.00% | 56.76% | 2.65% |
| Failure-SFT | 98.31% | 97.57% | 99.79% | 93.80% | 62.16% | 0.00% |

SFT 的 40 个失败任务中，Top 3 标签是 `wrong_argument_value`（28）、`ignore_tool_result`（15）和 `missing_argument`（11）。选择过程只读取 LoRA Validation 运行：

```bash
python scripts/select_failure_targets.py \
  experiments/results/qwen2_5_1_5b_sft_formal_v1_validation_new3090 \
  --top-k 3

python scripts/build_hard_cases.py \
  --failure-targets experiments/results/qwen2_5_1_5b_sft_formal_v1_validation_new3090/failure_targets.json
```

`build_hard_cases.py` 根据失败类别生成 3000 条全新 Train 轨迹：1200 条精确参数值、1000 条工具结果依赖、800 条必填参数/澄清。它不会复制 Validation 失败样本；Test 只参与哈希和 ID/问题文本碰撞审计，不进入生成逻辑，也不会产生 Validation/Test 训练输出。每条轨迹都必须通过真实 Calendar Environment 的 Oracle 重放后才会写入 ms-swift 数据。

### 6. 运行 Failure-SFT 并复用同一评测器

Failure-SFT 不是从普通 SFT adapter 继续训练，而是从冻结的 Base Model 重新训练“原始 6000 条 + 针对性 3000 条”。这样可以把差异归因于训练数据，并为后续“随机增加 3000 条 vs 失败感知增加 3000 条”消融保留公平对照。

```bash
python scripts/run_sft.py \
  --config configs/sft/qwen2_5_1_5b_lora_failure_aware_smoke.json

python scripts/run_sft.py \
  --config configs/sft/qwen2_5_1_5b_lora_failure_aware_v1.json

python scripts/select_sft_checkpoint.py \
  experiments/results/qwen2_5_1_5b_failure_sft_formal_v1

python scripts/run_qwen_baseline.py \
  --tasks data/processed/calendar_formal_v1/tasks/validation.jsonl \
  --adapter-path <selected_checkpoint.json 中的 selected_checkpoint> \
  --run-name qwen2_5_1_5b_failure_sft_formal_v1_validation

python scripts/run_eval.py \
  --run-name qwen2_5_1_5b_failure_sft_formal_v1_validation
```

正式训练共 1125 step、1 epoch，最低 Validation loss 为 `0.03750928`，对应 `checkpoint-750`。在同一 500 条 Validation 上，Failure-SFT 相比普通 SFT 修复 11 条任务、回归 2 条，成功任务净增 9 条：

- `missing_argument`：11 → 0；
- `ignore_tool_result`：15 → 1；
- `wrong_argument_value`：28 → 23；
- 失败任务总数：40 → 31；
- Task Success：460/500 → 469/500；
- Multi-turn Success：21/37 → 23/37。

仍需注意一个明确副作用：Failure-SFT 出现 7 个 `final_answer_failure`，共同模式是在 `list_events` 找到待删除事件后提前回答，没有继续调用 `delete_event`。其中 5 条在普通 SFT 中本来就会失败，只是失败类型从错误澄清转成提前停止；另外 2 条是真实回归。这个模式将进入 Robustness Benchmark 与后续数据消融，而不会继续无目的扩充数据。

### 7. 生成 Robustness Validation 并计算 Robustness Gap

```bash
python scripts/generate_robustness_data.py
```

该命令从冻结的 500 条 Validation 中确定性选择同源任务，生成 500 条 Robust Validation；不会读取 Train，也不会写入任何 SFT 数据。当前 10 类扰动各 50 条：

| 扰动 | 检查内容 |
|---|---|
| `similar_tool_distractor` | 增加功能描述相近但目标不同的工具 |
| `tool_order_shuffle` | 改变工具 Schema 顺序 |
| `tool_description_rewrite` | 保持语义，改写工具说明 |
| `tool_name_similarity` | 增加名称与正确工具高度相似的 preview 工具 |
| `missing_tool` | 移除完成任务必需的工具，要求明确说明不可用 |
| `tool_failure` | 首次调用确定性 timeout，允许重试恢复 |
| `noisy_tool_response` | 在正确结果中加入无关 metadata |
| `partial_tool_response` | 首次返回缺少关键字段，要求识别并重试 |
| `ambiguous_user_query` | 删除必要信息，正确行为变为澄清 |
| `irrelevant_tool_added` | 增加完全无关的日历时区工具 |

所有故障和响应变换都由 Task metadata 配置，并由同一个 Calendar Environment 执行；模型、Oracle 和 Evaluator 看到的是同一行为。当前全量 Oracle Task Success 为 500/500，50 个可恢复 timeout 的 Recovery Success 为 50/50，证明目标与执行协议自洽。

模型完成 Clean 与 Robust 两次运行后，使用配对报告脚本：

```bash
python scripts/compare_robustness.py \
  --clean-run experiments/results/<clean_run> \
  --robust-run experiments/results/<robust_run> \
  --output-prefix experiments/results/<model>_robustness_validation
```

脚本按照每条 Robust Task 的 `source_task_id` 找到对应 Clean 结果，并自动输出 JSON、CSV 和 Markdown。每个 setting 的核心指标为：

```text
Robustness Gap = 同源 Clean Task Success - Perturbed Task Success
```

当前只完成了数据与评测框架，尚未填写模型 Robustness 数字。

## Calendar 工具环境

当前环境包含 5 个工具：

| 工具 | 作用 | 是否修改状态 |
|---|---|---|
| `list_events` | 按时间范围列出日历事件 | 否 |
| `create_event` | 创建一个不与现有事件冲突的新事件 | 是 |
| `update_event` | 修改现有事件的部分字段 | 是 |
| `delete_event` | 根据事件 ID 删除事件 | 是 |
| `check_availability` | 检查时间段是否空闲 | 否 |

工具定义来自 [Calendar JSON Schema](robust_tool/tools/schemas/calendar.json)，可直接转换为模型所需的 function schema。环境接口如下：

```python
from robust_tool.env import CalendarEnvironment

env = CalendarEnvironment()
env.reset(task)

result = env.execute({
    "name": "list_events",
    "arguments": {}
})

state = env.get_state()
success = env.check_goal()
```

Calendar v1 的时间规则：

- 使用不带时区的 ISO-8601 本地时间；
- 精确到秒；
- 时间段采用半开区间 `[start, end)`；
- 相邻事件不算冲突；
- Event ID 按 `evt-0001`、`evt-0002` 的顺序确定性生成；
- 参数错误或事件冲突不会修改环境状态。

## 数据与任务类型

用于开发期 smoke 的 toy benchmark 共 25 条任务：

| 划分 | 数量 | 用途 |
|---|---:|---|
| Train | 15 | 后续数据转换和训练流程调试 |
| Validation | 5 | 配置选择与开发期评测 |
| Test / Clean Test | 5 | 冻结的基础测试集 |

任务覆盖事件查询、创建、更新、删除、空闲检查、信息不完整时的澄清，以及不需要调用工具的回答。

正式 `calendar-formal-sft-v1` 进一步覆盖表达改写、可选参数组合、相似功能工具选择、工具顺序打乱，以及 `check → create`、`list → update`、`list → delete` 三种多步状态依赖。三个划分使用不同模板，全部 7500 条用户请求全局去重。

生成数据时会同步创建 `data/eval/manifest.json`，其中记录生成器版本、随机种子、各划分数量和 SHA-256。Week 1 中 `test` 与 `clean_test` 的哈希必须一致。

任务与轨迹的详细字段见 [数据集协议](docs/dataset.md)。

## 第一版评测指标

Evaluator 当前输出以下指标。每个比率都保存 `value`、`numerator` 和 `denominator`，便于追溯计算过程。

| JSON 指标名 | 中文含义 | 回答的问题 |
|---|---|---|
| `call_decision_accuracy` | 调用决策准确率 | 应调用、应澄清、应直接回答的决策是否正确 |
| `tool_selection_accuracy` | 工具选择准确率 | 是否选择了正确工具 |
| `json_valid_rate` | JSON 合法率 | 工具调用能否被解析成标准 JSON |
| `argument_schema_accuracy` | 参数 Schema 准确率 | 必填、额外、类型和格式是否正确 |
| `argument_semantic_accuracy` | 参数语义准确率 | 规范化后参数值是否满足参考语义 |
| `executable_call_rate` | 可执行调用率 | 环境是否成功执行调用 |
| `task_success_rate` | 任务成功率 | 重放后的环境状态和观测是否达到目标 |
| `final_answer_semantic_accuracy` | 最终回答语义准确率 | 显式配置内容约束的回答是否满足确定性短语条件 |
| `multi_turn_task_success_rate` | 多轮任务成功率 | 多轮任务是否整体完成 |
| `recovery_success_rate` | 错误恢复成功率 | 工具失败后是否最终恢复并完成任务 |
| `invalid_tool_call_rate` | 无效调用率 | 调用是否因为解析、工具或参数问题无效 |
| `unnecessary_tool_call_rate` | 不必要调用率 | 不该调用工具时是否调用了工具 |
| `average_tool_calls_per_task` | 平均每任务调用数 | 模型完成任务使用了多少次工具 |

当前 toy tasks 还没有真正的多步任务和可恢复故障，因此相应指标没有 eligible 样本时输出 `null`，不会用 0 制造误导。

完整计算规则见 [评测协议](docs/evaluation.md)。

## 失败分类

一次失败可以同时拥有多个标签。例如，模型选择了错误工具，传入额外参数，工具执行失败后又声称完成，那么可能同时得到：

```text
wrong_tool
extra_argument
ignore_tool_result
```

每个标签都会在 `evaluation.jsonl` 中保存证据，而不仅仅保存标签名称。全部 15 个失败类型及触发条件见 [失败分类体系](docs/failure_taxonomy.md)。

## 如何查看实验输出

每个正式运行都写入独立目录：

```text
experiments/results/<run_name>/
├── config.json          # 本次运行的解析后配置和环境信息
├── metrics.json         # 汇总指标及分子/分母
├── failure_stats.json   # 失败类型计数与任务占比
├── predictions.jsonl    # 便于阅读的预测摘要
├── trajectories.jsonl   # 完整交互轨迹
├── evaluation.jsonl     # 逐任务重放结果和失败证据
└── run.log              # 运行日志
```

推荐按下面的顺序排查结果：

1. 先看 `metrics.json` 判断哪一层出现明显下降；
2. 再看 `failure_stats.json` 找出最高频的 failure；
3. 最后打开对应任务的 `evaluation.jsonl` 和 `trajectories.jsonl` 查看证据和原始轨迹。

更完整的实验记录规范见 [实验协议](docs/experiments.md)。

## 仓库结构

```text
robust-tool-slm/
├── configs/                 # 模型、SFT、GRPO 和评测配置
├── robust_tool/
│   ├── tools/               # 工具 Schema、工具实现与注册表
│   ├── env/                 # 状态、校验、执行、错误和目标检查
│   ├── data/                # Task Schema、数据生成、扰动和格式转换
│   ├── rollout/             # 解析器、Policy Runner 和完整轨迹
│   ├── eval/                # 参数评测、环境重放、指标和失败分类
│   └── reward/              # 后续 Outcome / Dense Reward
├── scripts/                 # 只负责串联模块的命令行入口
├── data/                    # 生成的数据及冻结评测集
├── experiments/             # 每次实验的配置、轨迹、指标和日志
├── tests/                   # 不依赖 GPU 的单元与集成测试
└── docs/                    # 设计、数据、评测、实验与最终报告
```

## 可复现性约束

- 所有随机生成和随机 Policy 必须显式传入并记录 seed；
- Task Success 必须通过全新环境重放得到；
- Train、Validation、Test 必须严格分离；
- Test 不得参与 hard-case mining 或数据增强；
- 所有正式结果先写入 JSON/JSONL，再由脚本生成表格；
- README 中不得手工填写无法追溯到实验产物的数字；
- 模型 checkpoint、大型 cache、凭证和本地敏感配置不得提交。

## 一个月路线

### Week 1：环境、Benchmark 与评测

实现本地工具环境、数据结构、基础任务、无模型 baseline、Evaluator 和 Failure Taxonomy。已经完成。

### Week 2：Qwen Baseline 与 SFT

主模型固定为 Qwen2.5-1.5B-Instruct。RTX 3090 上的 20-step smoke、6000 条正式 LoRA SFT 和 500 条统一 Validation 对比已经完成。Clean Test 仍保持冻结，留给阶段性方法确定后的正式比较。

### Week 3：失败驱动优化

已完成 Top 3 failure 冻结、3000 条全新 Train Hard Cases、Failure-SFT 训练、统一 Validation 评测，以及 500 条 / 10 类 Robust Validation 的生成与配对评测框架。下一步运行三种模型的 Robust Validation，再做随机增强对照，验证当前 1.8 个百分点的 Task Success 增益是否来自失败感知数据，而不是单纯增加训练样本。

### Week 4：执行反馈 GRPO

实现多步环境 Rollout、Outcome Reward、Failure-aware Dense Reward、GRPO 和奖励消融，比较 Base、SFT、Failure-SFT 与 GRPO。

## 推荐阅读顺序

1. 当前文件：理解项目目标、现状和运行方法；
2. [系统设计](docs/design.md)：理解各模块为什么这样拆分；
3. [数据集协议](docs/dataset.md)：理解 Task、Trajectory 和数据隔离；
4. [评测协议](docs/evaluation.md)：理解每个指标如何计算；
5. [失败分类体系](docs/failure_taxonomy.md)：理解失败标签和证据；
6. [实验协议](docs/experiments.md)：理解如何保存和比较正式实验；
7. [最终报告模板](docs/final_report.md)：理解项目最终要回答哪些研究问题。
