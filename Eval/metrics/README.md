# Metrics（自定义评测脚本）

本目录包含一组**离线评测脚本**，用于从 `Eval/run.py` 生成的结果目录（`RESULT_DIR/`）中计算不同的指标。

结果目录通常包含：
- `actions/<task_id>.json`：每个任务的最终 `score`（以及 interrupt 元信息）
- `trajectories/<task_id>.json`：actions-only 轨迹（`actions` 列表，含每步 `raw_prediction`、`action_type` 等）

> 约定：本仓库里的 “成功/失败（accuracy）” 默认遵循 `Eval/score.py`：**`score >= 1.0` 才算成功**；`0 < score < 1` 不计入 overall accuracy 的成功数。

---

## `score_by_steps.py`：step-wise overall accuracy（success@N）

### 指标含义
- 输出 step \(n\)（从 1 到 `max_action`）下的 **overall accuracy** 曲线：  
  - 任务若在 step \(\le n\) 前“完成”，且最终 `score>=1.0`，则在 step \(n\) 计为成功；否则为 0。

### step 的定义
默认：**post-interrupt action index**（只从 interrupt 注入后开始计数）：
- `k = interrupt.interrupt_at_action`（interrupt 前 replay 了多少个 action）
- `stop_step_total`：整段 action 列表里首次出现 STOP（`action_type==17`）的位置（1-based）
- `stop_step_post = max(0, stop_step_total - k)`
- 所以 interrupt 后的第一个 action 是 step=1。

也可切换为全局 action index（legacy）：`--step_origin global`

### 用法
在 `Eval/` 目录下运行：

```bash
python metrics/score_by_steps.py <result_dir> --max_action 30
```

多个目录可用逗号合并（同 `score.py` 逻辑：同一任务取最高 score；分数相同时更偏好更早停止）：

```bash
python metrics/score_by_steps.py dirA,dirB --max_action 30
```

切换 step 定义：

```bash
python metrics/score_by_steps.py <result_dir> --max_action 30 --step_origin global
```

输出格式：TSV，每行 `step<TAB>overall_accuracy`（百分比）。

---

## `adapt_action_efficiency.py`：Adapt Action Efficiency（平均 action 数差值）

### 指标含义
比较两次 run 的**平均 action 数**：
- **no-interrupt**：每任务 action 数 = 轨迹中（去掉末尾 termination placeholder STOP 后）的 action 数
- **with-interrupt**：每任务 action 数 = **post-interrupt action 数** = `max(0, total_actions - k)`
- **early stop penalty**：若检测到 early stop，则该任务 action 数强制视为 `max_step`（默认 30）

最终输出：
- `avg_actions_no_interrupt`
- `avg_actions_with_interrupt`
- `diff_with_minus_without = with - without`

### 用法

```bash
python metrics/adapt_action_efficiency.py \
  --no_interrupt_dir <without_interrupt_result_dir> \
  --interrupt_dir <with_interrupt_result_dir> \
  --task_count 165 \
  --max_step 30
```

缺失轨迹的处理：
- `--missing_policy penalize`（默认）：缺失按 `max_step` 计入
- `--missing_policy ignore`：缺失任务不计入平均的分母

---

## `action_diff_by_outcome.py`：按成功/失败分桶的 action 差值分析

### 指标含义
对每个任务计算：
- `success_no = (score_no >= 1.0)`
- `success_int = (score_int >= 1.0)`
- `actions_no`（no-interrupt）与 `actions_int`（post-interrupt）
- `diff = actions_int - actions_no`

然后按以下桶汇总 diff 的统计量（mean/median/p25/p75/min/max）：
- `no_interrupt_success__interrupt_fail`
- `no_interrupt_fail__interrupt_success`
- `both_success`
- `both_fail`
- 边际：`no_interrupt_success/no_interrupt_fail/interrupt_success/interrupt_fail`

### 用法

```bash
python metrics/action_diff_by_outcome.py \
  --no_interrupt_dir <without_interrupt_result_dir> \
  --interrupt_dir <with_interrupt_result_dir> \
  --task_count 165 \
  --max_step 30 \
  --missing_policy penalize
```

可选：`--print_task_ids` 输出每个桶的 task_id 列表。

---

## `token_diff_by_outcome.py`：按成功/失败分桶的 token 差值分析（含 early-stop token 扩展惩罚）

### token 统计口径（逐任务）
- 从 `trajectories/<task_id>.json` 的 action 列表里取每步的 `raw_prediction` 文本。
- token 计数：对每个 `raw_prediction` 用 `tiktoken` 编码，token 数 = `len(encode(text))`。
  - 优先 `tiktoken.encoding_for_model(model)`
  - 如果无法映射（如 Bedrock Claude 的 model id），回退到 `cl100k_base`（近似）
- 默认 **不计 STOP（`action_type==17`）** 的 token（STOP 多为 runner 生成的 early stop/placeholder，不是模型动作本身）。
- with-interrupt 只统计 post-interrupt：若 `k=interrupt_at_action`，只取 actions[k:]（再去 STOP）。

### early stop penalty（你定义的规则）
若检测到 early stop（轨迹中存在 STOP，且其 `answer` 含 `"Early stop"`）：
- action 数强制视为 `max_step`（默认 30）
- token 也补齐到 `max_step`：
  - 令 `avg_last3` = 最后 3 个（若不足 3 则用已有的）已计入 action 的平均 token
  - 额外补齐 token = `(max_step - existing_actions) * avg_last3`
  - `tokens_total = tokens_existing + extra`

### 输出与 diff 符号
同 `action_diff_by_outcome.py` 的分桶方式，区别是：
- `diff = tokens_int - tokens_no`
- **diff > 0 表示 interrupt 更耗 token**；diff < 0 表示 interrupt 更省 token

### 用法

```bash
python metrics/token_diff_by_outcome.py \
  --no_interrupt_dir <without_interrupt_result_dir> \
  --interrupt_dir <with_interrupt_result_dir> \
  --task_count 165 \
  --max_step 30 \
  --missing_policy penalize
```

可选：`--print_task_ids` 输出每个桶的 task_id 列表。

### 常见问题：tiktoken 缓存目录权限（PermissionError）
部分系统上 `/tmp/data-gym-cache` 可能不可写，导致 tiktoken 报 `PermissionError`。

本脚本已做了自动兜底：若默认目录不可写，会自动把缓存切到 `~/.cache/` 下。

你也可以手动指定（推荐）：

```bash
export DATA_GYM_CACHE_DIR="$HOME/.cache/data-gym-cache"
export TIKTOKEN_CACHE_DIR="$HOME/.cache/tiktoken"
mkdir -p "$DATA_GYM_CACHE_DIR" "$TIKTOKEN_CACHE_DIR"
```

---

## 结果目录要求（Checklist）
- `RESULT_DIR/actions/<task_id>.json`：需要包含 `score`（用于 success/fail 分桶）
- `RESULT_DIR/trajectories/<task_id>.json`：需要包含 `actions`（用于 step/action/token 统计）
- interrupt 相关指标依赖：`interrupt.interrupt_at_action`（通常在 actions 或 trajectories 的 `interrupt` 字段里）

