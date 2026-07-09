# Multi-Interrupt (Multi-Round Intent Update) Evaluation Workflow

This document explains how, on top of the existing **Interrupt/Replay** evaluation logic in this repo, to extend "single interrupt + single intent update" into "multiple interrupts + multiple intent updates (consuming the updates round by round in order)".

Core idea: **do not perform multiple interrupts within a single run**, but instead run **multiple rounds** following the strategy you described:

- **Stage 0 (baseline)**: run the full trajectory using only `transformed_initial_intent` and save the actions (used for later replay).
- **Stage 1**: based on the Stage 0 trajectory, interrupt at the **midpoint of the entire action sequence**, inject `updates[0]`, finish the run and save a new trajectory.
- **Stage 2**: based on the Stage 1 trajectory, interrupt at the **midpoint of the action segment after the most recent interrupt**, inject `updates[1]`, finish the run and save a new trajectory.
- **Stage 3..**: and so on, until the `updates` are exhausted.

This repo adds an offline generation script:

- `interrupt_config/make_multi_interrupt_stage.py`
  - Input: `raw/3mixed.json` + the previous round's `result_dir/trajectories/<task_id>.json`
  - Output:
    - The `interrupt_spec_stageK.json` to use for the next round
    - The config directory to use for the next round (accumulating "the updates that already happened" into `intent`, so that when the agent continues, what it sees is the "latest accumulated intent")

> Note: `Eval/run.py` still only handles a "single interrupt". Multiple interrupts are chained together via multiple rounds of run + replay.

---

## 0) Prepare the raw data

The raw file you provide:

- `Eval/interrupt_config/raw/3mixed.json`

Each item contains at least:

- `task_id`
- `transformed_initial_intent`
- `updates` (list of strings, used in order)

---

## 1) Stage 0: generate the config directory for the transformed initial intent

Use the existing script to replace the base config's `intent` with `transformed_initial_intent`:

```bash
cd WebAgent-R1/Eval

python interrupt_config/make_transformed_intent_configs.py \
  --base_config_dir config_files/wa/test_webarena_lite \
  --raw_interrupt_file interrupt_config/raw/3mixed.json \
  --out_dir config_files/wa/test_webarena_transformed_3mixed
```

> `--base_config_dir` needs to align with the benchmark you want to run (e.g. `test_webarena_lite`). raw will only override the task_ids that appear.

---

## 2) Stage 0: run the baseline (save trajectories)

You can use `run.py` or `scripts/parallel_by_sites.py`. Here is an example (replace the parameters with your own provider/model/observation config):

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

The key output of Stage 0 is:

- `test_result_3mixed_stage0_baseline/trajectories/<task_id>.json`

---

## 3) Stage 1: generate (config + interrupt_spec)

Stage 1's interrupt point is the **midpoint of the baseline actions**.

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

## 4) Stage 1: run replay + interrupt (save trajectories)

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

## 5) Stage 2 / Stage 3: repeat (until updates are exhausted)

Stage 2's interrupt point is:

- \(k_2 = k_1 + \lfloor (N_1 - k_1) / 2 \rfloor\)

That is, the **midpoint of the action segment after the "most recent interrupt" in Stage 1**.

Generate Stage 2 (recommended to only pass the previous round's result_dir; the script will recover the history from the previous round's trajectory `interrupt.multi_interrupt`):

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

Then run Stage 2 (replay comes from Stage 1):

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

Stage 3 is analogous:

- Generate: `--result_dirs test_result_3mixed_stage2 --stage 3`
- Run: `--replay_trajectory_dir test_result_3mixed_stage2/trajectories`

---

## 6) Output description (for easier debugging)

Each round's `result_dir/trajectories/<task_id>.json` contains:

- `num_actions`
- `actions` (replayable)
- `interrupt` (the update injected this round)
  - `interrupt_at_action`
  - `update_intent`
  - `multi_interrupt` (if this round's spec/config carried prior_updates, it will be brought in)

Each round's generated config directory additionally writes:

- `_multi_interrupt_manifest_stageK.json`: records how many tasks this stage wrote, which tasks were skipped, and why.
