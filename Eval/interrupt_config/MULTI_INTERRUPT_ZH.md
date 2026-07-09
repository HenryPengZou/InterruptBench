# 多次 Interrupt（多轮意图更新）评测工作流

这份文档说明如何在本仓库现有的 **Interrupt/Replay** 评测逻辑基础上，把“单次 interrupt + 单次 intent 更新”扩展为“多次 interrupt + 多次 intent 更新（按轮次依次用完 updates）”。

核心思想：**不在一次 run 里做多个 interrupt**，而是按你描述的策略进行 **多轮运行**：

- **Stage 0（baseline）**：只用 `transformed_initial_intent` 跑完整轨迹并保存 actions（用于后续 replay）。
- **Stage 1**：基于 Stage 0 的轨迹，在 **整段 actions 的中点** interrupt，注入 `updates[0]`，继续跑完并保存新轨迹。
- **Stage 2**：基于 Stage 1 的轨迹，在 **最近一次 interrupt 之后的 action 段中点** interrupt，注入 `updates[1]`，继续跑完并保存新轨迹。
- **Stage 3..**：依次类推，直到用完 `updates`。

本仓库新增了一个离线生成脚本：

- `interrupt_config/make_multi_interrupt_stage.py`
  - 输入：`raw/3mixed.json` + 上一轮 `result_dir/trajectories/<task_id>.json`
  - 输出：
    - 下一轮要用的 `interrupt_spec_stageK.json`
    - 下一轮要用的 config 目录（把“之前已发生的更新”累计拼进 `intent`，保证 agent 继续执行时看到的是“最新累计意图”）

> 说明：`Eval/run.py` 仍然只处理“单次 interrupt”。多次 interrupt 是靠多轮 run + replay 串起来的。

---

## 0) 准备 raw 数据

你提供的 raw 文件：

- `Eval/interrupt_config/raw/3mixed.json`

其中每个 item 至少包含：

- `task_id`
- `transformed_initial_intent`
- `updates`（字符串列表，按顺序依次使用）

---

## 1) Stage 0：生成 transformed 初始意图的 config 目录

用已有脚本把 base config 的 `intent` 替换为 `transformed_initial_intent`：

```bash
cd WebAgent-R1/Eval

python interrupt_config/make_transformed_intent_configs.py \
  --base_config_dir config_files/wa/test_webarena_lite \
  --raw_interrupt_file interrupt_config/raw/3mixed.json \
  --out_dir config_files/wa/test_webarena_transformed_3mixed
```

> `--base_config_dir` 需要与你想跑的 benchmark 对齐（例如 `test_webarena_lite`）。raw 里只会覆盖出现的 task_id。

---

## 2) Stage 0：跑 baseline（保存 trajectories）

你可以用 `run.py` 或 `scripts/parallel_by_sites.py`。这里给一个示例（参数按你自己的 provider/model/observation 配置替换）：

```bash
cd WebAgent-R1/Eval
source setup_vars.sh

python scripts/parallel_by_sites.py \
  --test_config_base_dir config_files/wa/test_webarena_transformed_3mixed \
  --result_dir test_result_3mixed_stage0_baseline \
  --test_start_idx 0 --test_end_idx 165 \
  --max_parallel 8 \
  --reset_before_run \
  --reset_server_url "http://<host>:7565" \
  -- \
  --instruction_path agent/prompts/jsons/p_webrl_chat_think.json \
  --provider bedrock --mode chat \
  --model <your_model> \
  --region <your_region> \
  --action_set_tag webrl_id \
  --observation_type webrl \
  --max_obs_length 0 \
  --max_tokens 2048 \
  --save_trajectory
```

Stage 0 的关键产物是：

- `test_result_3mixed_stage0_baseline/trajectories/<task_id>.json`

---

## 3) Stage 1：生成（config + interrupt_spec）

Stage 1 的 interrupt 点是 **baseline actions 的中点**。

```bash
cd WebAgent-R1/Eval

python interrupt_config/make_multi_interrupt_stage.py \
  --raw_interrupt_file interrupt_config/raw/3mixed.json \
  --base_config_dir config_files/wa/test_webarena_transformed_3mixed \
  --result_dirs test_result_3mixed_stage0_baseline \
  --stage 1 \
  --out_config_dir config_files/wa/test_webarena_3mixed_stage1 \
  --out_interrupt_spec interrupt_config/process/interrupt_spec_3mixed_stage1.json
```

---

## 4) Stage 1：跑 replay + interrupt（保存 trajectories）

```bash
cd WebAgent-R1/Eval
source setup_vars.sh

python scripts/parallel_by_sites.py \
  --test_config_base_dir config_files/wa/test_webarena_3mixed_stage1 \
  --result_dir test_result_3mixed_stage1 \
  --test_start_idx 0 --test_end_idx 165 \
  --max_parallel 8 \
  --reset_before_run \
  --reset_server_url "http://<host>:7565" \
  -- \
  --instruction_path agent/prompts/jsons/p_webrl_chat_think.json \
  --provider bedrock --mode chat \
  --model <your_model> \
  --region <your_region> \
  --action_set_tag webrl_id \
  --observation_type webrl \
  --max_obs_length 0 \
  --max_tokens 2048 \
  --save_trajectory \
  --replay_trajectory_dir test_result_3mixed_stage0_baseline/trajectories \
  --interrupt_spec interrupt_config/process/interrupt_spec_3mixed_stage1.json
```

---

## 5) Stage 2 / Stage 3：重复（直到用完 updates）

Stage 2 的 interrupt 点是：

- \(k_2 = k_1 + \lfloor (N_1 - k_1) / 2 \rfloor\)

也就是 **Stage 1 中“最近一次 interrupt 之后”的 actions 段中点**。

生成 Stage 2（推荐只传上一轮 result_dir 即可；脚本会从上一轮 trajectory 的 `interrupt.multi_interrupt` 里恢复历史）：

```bash
cd WebAgent-R1/Eval

python interrupt_config/make_multi_interrupt_stage.py \
  --raw_interrupt_file interrupt_config/raw/3mixed.json \
  --base_config_dir config_files/wa/test_webarena_transformed_3mixed \
  --result_dirs test_result_3mixed_stage1 \
  --stage 2 \
  --out_config_dir config_files/wa/test_webarena_3mixed_stage2 \
  --out_interrupt_spec interrupt_config/process/interrupt_spec_3mixed_stage2.json
```

然后跑 Stage 2（replay 来自 Stage 1）：

```bash
python scripts/parallel_by_sites.py \
  --test_config_base_dir config_files/wa/test_webarena_3mixed_stage2 \
  --result_dir test_result_3mixed_stage2 \
  ... \
  -- \
  ... \
  --save_trajectory \
  --replay_trajectory_dir test_result_3mixed_stage1/trajectories \
  --interrupt_spec interrupt_config/process/interrupt_spec_3mixed_stage2.json
```

Stage 3 同理：

- 生成：`--result_dirs test_result_3mixed_stage2 --stage 3`
- 运行：`--replay_trajectory_dir test_result_3mixed_stage2/trajectories`

---

## 6) 产物说明（便于调试）

每一轮 `result_dir/trajectories/<task_id>.json` 都会包含：

- `num_actions`
- `actions`（可 replay）
- `interrupt`（本轮注入的 update）
  - `interrupt_at_action`
  - `update_intent`
  - `multi_interrupt`（如果本轮 spec/config 带了 prior_updates，会被带进来）

每一轮生成的 config 目录下还会额外写一个：

- `_multi_interrupt_manifest_stageK.json`：记录该 stage 写了多少 task、跳过了哪些 task 以及原因。

