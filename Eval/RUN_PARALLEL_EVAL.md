# Parallel Evaluation From Zero

This document targets two scenarios: "**setting up the environment from scratch for the first time**" and "**repeatedly resetting and running evaluations in parallel afterwards**". It covers:
- WebArena environment (docker) setup and day-to-day reset
- Starting and using the reset server
- Installing evaluation (Python side) dependencies: **use conda instead of venv**
- Environment variables (website URLs / Bedrock / OpenAI)
- **Complete parameter reference** for the parallel evaluation script `scripts/parallel_by_sites.py`

> Convention: below we assume you operate from the repo root `WebAgent-R1/`; the evaluation code lives in `WebAgent-R1/Eval/`.

---

## 1) Setting up the WebArena environment (docker, first-time full setup / subsequent reset)

### 1.1 Prepare images and resource files (only once)

Go into `WebAgent-R1/WebArena-Env-Setup/` and, per its `README.md`, download and place:
- Several docker image tars provided officially by WebArena (shopping / reddit / gitlab, etc.)
- The Wikipedia `.zim`
- The 3 OpenStreetMap tar.gz files (Zenodo)

And modify `WebAgent-R1/WebArena-Env-Setup/00_vars.sh` as needed:
- `PUBLIC_HOSTNAME`: the IP/domain of this machine that is reachable from the evaluation machine
- The port of each website (`SHOPPING_PORT/REDDIT_PORT/...`) and `RESET_PORT`
- `ARCHIVES_LOCATION` (default `./images`): the path where you store the downloaded tar/zim files

First-time image loading (takes a while, only once):

```bash
cd WebArena-Env-Setup
sudo bash 01_docker_load_images.sh
```

### 1.2 Start the website environment (run 02-06 on every reset)

Whenever you want to return to a "clean environment", run the following in order (recommended to run inside tmux/screen):

```bash
cd WebArena-Env-Setup
sudo bash 02_docker_remove_containers.sh
sudo bash 03_docker_create_containers.sh
sudo bash 04_docker_start_containers.sh
sudo bash 05_docker_patch_containers.sh
sudo bash 06_serve_homepage.sh
```

- `06_serve_homepage.sh` starts the "landing homepage service" and **needs to keep occupying a terminal**.

#### Tip: use nohup to run in the background (avoid tying up the terminal long-term)

If you don't want `06_serve_homepage.sh` to keep occupying the current terminal, use:

```bash
cd WebArena-Env-Setup
nohup sudo bash 06_serve_homepage.sh > homepage.log 2>&1 &
```

---

## 2) Setting up the reset server (optional, but strongly recommended for automated reset)

The reset server is used to trigger an "all-instances reset" **without rebuilding containers** (very convenient for interrupt/replay or multi-round testing).

Start the reset server (also needs to keep running):

```bash
cd WebArena-Env-Setup
sudo bash 07_serve_reset.sh
```

It provides two key endpoints on the `RESET_PORT` from `00_vars.sh`:
- `GET /reset`: trigger a reset (if a reset is already running, it may return 418)
- `GET /status`: check status (only `Ready` means the reset is complete and you can continue running)

For example (replace with the host/port you configured in `00_vars.sh`):

```bash
curl "http://<PUBLIC_HOSTNAME>:7565/status"
curl "http://<PUBLIC_HOSTNAME>:7565/reset"
```

#### Tip: nohup is also recommended for the reset server

```bash
cd WebArena-Env-Setup
nohup sudo bash 07_serve_reset.sh > reset_server.log 2>&1 &
```

---

## 3) Evaluation environment configuration (use conda instead of venv)

Go into `WebAgent-R1/Eval/` and create an isolated environment with conda (Python 3.10 recommended; 3.11 also works; 3.12 not recommended):

```bash
cd Eval
conda create -n webagent-eval python=3.10 -y
conda activate webagent-eval

pip install -r requirements.txt
playwright install
pip install -e .
```

---

## 4) Environment variables (website URLs / Bedrock / OpenAI)

### 4.1 Website URLs and ports (recommended to manage uniformly via `setup_vars.sh`)

This repo provides `WebAgent-R1/Eval/setup_vars.sh`, used to export in one go:
- `DATASET=webarena`
- The URL of each website (`HOMEPAGE/SHOPPING/REDDIT/...`)
- OpenAI-related variables (used for fuzzy match during evaluation)

You need to do two things:
- **Change `PUBLIC_HOSTNAME` to your WebArena environment machine's IP/domain**
- **Make sure the ports inside match `WebArena-Env-Setup/00_vars.sh`**

Then before every eval run:

```bash
cd Eval
source setup_vars.sh
```

> Note: `setup_vars.sh` currently exports full URLs (e.g. `http://10.49.48.242:8082`); do not manually append the protocol/port again, to avoid duplication.

### 4.2 OpenAI (used for fuzzy-match scoring, very cheap)

Even if your agent uses Bedrock, some tasks in evaluation use `llm_fuzzy_match / llm_ua_match` for semantic scoring (only triggered on evaluators that need fuzzy match, **very low call volume**).

You need to set:
- `OPENAI_API_KEY`
- (Optional) `OPENAI_API_URL` (default `https://api.openai.com/v1`)

Corresponding code location:
- `llm_fuzzy_match()` / `llm_ua_match()` in `Eval/evaluation_harness/helper_functions.py`

### 4.3 AWS Bedrock (used for agent inference)

When you use `--provider bedrock`, inference goes through `boto3`'s credential chain (usually it just works if `aws sts get-caller-identity` succeeds).

- **Directly export the environment variable** `AWS_BEARER_TOKEN_BEDROC`

Region resolution priority:
- Command line `--region us-east-2` (recommended, most explicit)
- Or environment variables: `BEDROCK_REGION` / `AWS_BEDROCK_REGION` / `AWS_REGION` / `AWS_DEFAULT_REGION`

---

## 5) Running parallel evaluation (recommended: `scripts/parallel_by_sites.py`)

### 5.1 Example command

The following command will:
- Run task indices \([0, 165)\) in `config_files/wa/test_webarena_lite/` (**half-open interval**)
- Launch at most 8 `run.py` subprocesses in parallel
- Skip if `test_result_webr1_newprompt_think/render_<idx>.html` already exists
- Perform a single global reset via the reset server before starting
- Forward the arguments after `--` verbatim to `run.py`

```bash
cd Eval
source setup_vars.sh

python scripts/parallel_by_sites.py \
  --test_config_base_dir config_files/wa/test_webarena_lite \
  --result_dir test_result_webr1_newprompt_think \
  --test_start_idx 0 --test_end_idx 165 \
  --max_parallel 8 \
  --skip_finished \
  --reset_server_url "http://local_host:7565" \
  --reset_before_run \
  -- \
  --instruction_path agent/prompts/jsons/p_webrl_chat_think.json \
  --provider bedrock --mode chat \
  --model us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
  --region us-east-2 \
  --action_set_tag webrl_id \
  --observation_type webrl \
  --planner_ip '' \
  --max_obs_length 0 \
  --max_tokens 2048 \
  --viewport_width 1280 \
  --viewport_height 720 \
  --save_trajectory
```

### 5.2 `parallel_by_sites.py` (scheduler) parameter reference

These parameters are parsed by `Eval/scripts/parallel_by_sites.py`.

- **`--test_config_base_dir` (required)**
  - **Purpose**: single-task config directory, which should contain `${idx}.json` files
  - **Example**: `config_files/wa/test_webarena_lite`

- **`--result_dir` (required)**
  - **Purpose**: result output directory (`render_*.html`, `actions/`, `traces/`, `trajectories/`, etc. are written here)
  - **Example**: `test_result_webr1_newprompt_think`

- **`--test_start_idx` / `--test_end_idx`**
  - **Purpose**: specify the task index range, using Python's `range(start, end)`, i.e. **includes start, excludes end**
  - **Default**: `start=0`, `end=910`
  - **Options**: any integer; recommended to set end to "last task index + 1"

- **`--test_indices`**
  - **Purpose**: a comma-separated list of indices, overrides `--test_start_idx/--test_end_idx`
  - **Example**: `--test_indices 0,5,7`

- **`--max_parallel`**
  - **Purpose**: upper limit on the number of concurrently running `run.py` subprocesses
  - **Default**: 8
  - **Options**: positive integer (recommended to adjust based on machine CPU/RAM and website load capacity)

- **`--chunk_size`**
  - **Purpose**: how many tasks each `run.py` process runs serially (implemented via `run.py --test_indices a,b,c`), to reduce startup overhead
  - **Default**: 0 (auto-balance: roughly split into `--max_parallel` chunks)
  - **Options**:
    - `0`: auto
    - `>=1`: fixed chunk size (larger means each process runs longer and failure-rollback cost is higher)

- **`--skip_finished`**
  - **Purpose**: skip a task if `result_dir/render_<idx>.html` already exists
  - **Options**: flag (no value)

- **`--python`**
  - **Purpose**: the Python interpreter path used to launch `run.py`
  - **Default**: the current interpreter (`sys.executable`)
  - **Common**: usually no need to specify explicitly in a conda environment; to pin it, use `--python "$(which python)"`

- **`--run_script`**
  - **Purpose**: the script to be scheduled (path relative to `Eval/`)
  - **Default**: `run.py`

- **`--reset_server_url`**
  - **Purpose**: reset server base URL (e.g. `http://<host>:7565`)
  - **Options**: if not passed, no automatic reset is performed

- **`--reset_before_run`**
  - **Purpose**: before starting any parallel tasks, trigger **one** global reset (calls `GET <reset_server_url>/reset` and polls `GET .../status` until `Ready`)
  - **Options**: flag (no value)

- **`--reset_timeout_s` / `--reset_poll_interval_s` / `--reset_request_timeout_s` / `--reset_max_retries`**
  - **Purpose**: control the reset wait timeout, poll interval, HTTP timeout, and reset-trigger retry count
  - **Default**: `600 / 2 / 10 / 3`

- **`--dry_run`**
  - **Purpose**: only print the `run.py` commands that would be executed, without actually running them

> Important reminder: this scheduler **does not do site isolation**; when running multiple tasks concurrently, do not enable "per-task reset" inside each `run.py` (it would globally reset and disrupt other parallel tasks).

### 5.3 Common parameters after `--` (forwarded to `run.py`)

These parameters are parsed by `Eval/run.py` (`parallel_by_sites.py` forwards them verbatim).

- **`--instruction_path`**
  - **Purpose**: path to the prompt/instruction template JSON
  - **Examples**:
    - `agent/prompts/jsons/p_webrl_chat.json`
    - `agent/prompts/jsons/p_webrl_chat_think.json`

- **`--provider`**
  - **Purpose**: LLM invocation backend
  - **Options** (common values consistent with the code):
    - `bedrock` (AWS Bedrock)
    - `openai` (OpenAI / OpenAI-compatible gateway)
    - `google` / `api` / `finetune` (see code and README for details)

- **`--mode`**
  - **Purpose**: invocation mode (may have different meanings across providers)
  - **Common**: `chat` (chat-style)

- **`--model`**
  - **Purpose**: model name/ID
  - **Example (Bedrock)**: `us.anthropic.claude-sonnet-4-5-20250929-v1:0`

- **`--region`**
  - **Purpose**: Bedrock region (effective when `--provider bedrock`)
  - **Example**: `us-east-2`
  - **Options**: any AWS region; if not passed, it will try to infer from environment variables (see 4.3)

- **`--action_set_tag`**
  - **Purpose**: action space / parsing method (needs to pair with observation_type)
  - **Common options**:
    - `webrl_id` (WebRL text setting, uses raw string actions)
    - `id_accessibility_tree` (ID actions based on the accessibility tree)
    - `som` (SoM visual setting)
    - `id_accessibility_tree_with_captioner` (variant with captioner)

- **`--observation_type`**
  - **Purpose**: form of environment observation
  - **Options** (explicit choices in the code):
    - `accessibility_tree`
    - `accessibility_tree_with_captioner`
    - `html`
    - `image`
    - `image_som`
    - `webrl`

- **`--planner_ip`**
  - **Purpose**: the model service address in some self-host / finetune modes (related to OpenAI-compatible endpoint)
  - **Common**: if not needed, pass an empty string `''` or omit

- **`--max_obs_length`**
  - **Purpose**: observation text truncation length; `0` means no truncation
  - **Default**: 3840

- **`--max_tokens`**
  - **Purpose**: upper limit on model output tokens
  - **Default**: 384

- **`--viewport_width` / `--viewport_height`**
  - **Purpose**: browser viewport size (affects screenshots / visible area / some site layouts)
  - **Default**: `1280 / 2048`

- **`--save_trajectory`**
  - **Purpose**: save each task's action sequence (used for interrupt/replay)
  - **Output**: `<result_dir>/trajectories/<task_id>.json`

#### Compatibility constraints (very important)

`run.py` checks whether certain combinations are valid, e.g.:
- When `--action_set_tag id_accessibility_tree`, `--observation_type` can only be:
  - `accessibility_tree`
  - `accessibility_tree_with_captioner`
  - `image_som`

And the WebRL example you gave is:
- `--action_set_tag webrl_id`
- `--observation_type webrl`

This is a valid match.

---

## 6) Interrupt dataset workflow (raw → transformed_initial_intent → baseline → replay interrupt)

This section explains:
- How to generate the interrupt config (`interrupt_spec`) usable by `run.py` from raw data
- How to generate "task config files based on transformed_initial_intent" from raw data (a new `--test_config_base_dir`)
- How to first run the transformed_initial_intent baseline evaluation (saving trajectories), then use those trajectories for replay interrupt evaluation

### 6.1 Format and location of raw data

Raw interrupt data is placed by default in:
- `Eval/interrupt_config/raw/*.json`

The file format is a JSON array, where each element contains at least:
- `task_id` (int)
- `transformed_initial_intent` (str)
- `updates` (list[str], at least one)

For example (excerpt from `interrupt_config/raw/1update_opus.json`):

```json
[
  {
    "task_id": 0,
    "intent": "What are the top-3 best-selling product in Jan 2023",
    "transformed_initial_intent": "What are the top-3 best-selling products?",
    "updates": ["I meant for Jan 2023 specifically"]
  }
]
```

> The `intent` field is optional for the generation script; what the two-stage evaluation actually uses is `transformed_initial_intent` + `updates`.

### 6.2 raw → interrupt_spec (generate the config needed for replay/interrupt)

Script: `Eval/interrupt_config/convert_raw_to_interrupt_spec.py`

It concatenates the `updates` in raw into `update_intent`, generating what `run.py` needs:

```json
{
  "tasks": {
    "0": {
      "interrupt_at_action": 5,
      "update_mode": "append",
      "update_intent": "...",
      "extra_steps": 0
    }
  }
}
```

Recommended usage (put the output in `interrupt_config/process/` to avoid mixing with raw):

```bash
cd Eval
python interrupt_config/convert_raw_to_interrupt_spec.py \
  --inputs interrupt_config/raw/1update_opus.json \
  --out_dir interrupt_config/process \
  --interrupt_at_action 5 \
  --update_mode append \
  --extra_steps 0
```

Output example:
- `Eval/interrupt_config/process/interrupt_spec_1update_opus.json`

#### Optional: specify the interrupt point by percentage (`interrupt_at_pct`)

`run.py` supports specifying the interrupt point by "percentage of the number of saved actions":

```json
{
  "tasks": {
    "0": {
      "interrupt_at_pct": "20%",
      "update_mode": "append",
      "update_intent": "...",
      "extra_steps": 0
    }
  }
}
```

Where \(N\) is the number of actions in the replay file, the actual replay step count is:
\[
K=\lfloor 0.2\times N\rfloor
\]

### 6.3 raw → transformed_initial_intent task config files (generate a new `test_config_base_dir`)

Script: `Eval/interrupt_config/make_transformed_intent_configs.py`

It reads the base config dir (e.g. `0.json..164.json` under `config_files/wa/test_webarena_lite/`), and for `task_id`s that appear in raw:
- Replaces `cfg["intent"]` with `transformed_initial_intent`
- Also backs up the original intent to `cfg["true_intent"]` (if it wasn't there before)

There is already a sample directory in this repo:
- `Eval/config_files/wa/test_webarena_lite_transformed_1update/`

You can also customize the output directory per raw file (recommended to make the output directory name correspond to the raw file name):

```bash
cd Eval
python interrupt_config/make_transformed_intent_configs.py \
  --base_config_dir config_files/wa/test_webarena_lite \
  --raw_interrupt_file interrupt_config/raw/1update_opus.json \
  --out_dir config_files/wa/test_webarena_lite_transformed_1update_opus
```

> Note: if you want to force every base task to have a transformed intent, add `--strict`; otherwise it will skip task_ids not present in raw.

#### (Optional) raw → transformed dataset JSON (single-file list[dict])

If you also need to generate a "single-file dataset" like `config_files/wa/test_webarena_lite.json`, use:
- `Eval/interrupt_config/make_transformed_intent_dataset_json.py`

Example:

```bash
cd Eval
python interrupt_config/make_transformed_intent_dataset_json.py \
  --base_dataset_json config_files/wa/test_webarena_lite.json \
  --raw_interrupt_file interrupt_config/raw/1update_opus.json \
  --out_dataset_json config_files/wa/test_webarena_lite_transformed_1update_opus.json
```

> Note: what `run.py`/`parallel_by_sites.py` actually use is the "config directory split by task_id" (`--test_config_base_dir`); the single file above is mainly for data processing/statistics purposes.

### 6.4 Stage A: first run the transformed_initial_intent baseline evaluation (save trajectories)

Goal: have the agent run once under **transformed_initial_intent**, producing a replayable action sequence:
- `<baseline_result_dir>/trajectories/<task_id>.json`

Parallel run example (point `--test_config_base_dir` to the transformed directory, and add `--save_trajectory`):

```bash
cd Eval
source setup_vars.sh

python scripts/parallel_by_sites.py \
  --test_config_base_dir config_files/wa/test_webarena_lite_transformed_1update \
  --result_dir test_result_transformed_1update_baseline \
  --test_start_idx 0 --test_end_idx 165 \
  --max_parallel 8 \
  --skip_finished \
  --reset_server_url "http://<PUBLIC_HOSTNAME>:7565" \
  --reset_before_run \
  -- \
  --instruction_path agent/prompts/jsons/p_webrl_chat_think.json \
  --provider bedrock --mode chat \
  --model us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
  --region us-east-2 \
  --action_set_tag webrl_id \
  --observation_type webrl \
  --max_obs_length 0 \
  --max_tokens 2048 \
  --save_trajectory
```

### 6.5 Stage B: use Stage A's trajectories for replay + interrupt evaluation

Goal: on the same **transformed_initial_intent** starting point:
1) replay the first \(K\) steps (from the baseline trajectories)
2) inject `updates` (append/replace)
3) continue completing the task and score it

Key inputs:
- `--test_config_base_dir`: **same as Stage A (transformed configs)**
- `--replay_trajectory_dir`: Stage A output directory (e.g. `test_result_transformed_1update_baseline/trajectories`)
- `--interrupt_spec`: the interrupt spec generated in 6.2 (e.g. `interrupt_config/process/interrupt_spec_1update_opus.json`)

Parallel run example:

```bash
cd Eval
source setup_vars.sh

python scripts/parallel_by_sites.py \
  --test_config_base_dir config_files/wa/test_webarena_lite_transformed_1update \
  --result_dir test_result_transformed_1update_interrupt \
  --test_start_idx 0 --test_end_idx 165 \
  --max_parallel 8 \
  --skip_finished \
  --reset_server_url "http://<PUBLIC_HOSTNAME>:7565" \
  --reset_before_run \
  -- \
  --instruction_path agent/prompts/jsons/p_webrl_chat_think.json \
  --provider bedrock --mode chat \
  --model us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
  --region us-east-2 \
  --action_set_tag webrl_id \
  --observation_type webrl \
  --interrupt_spec interrupt_config/process/interrupt_spec_1update_opus.json \
  --replay_trajectory_dir test_result_transformed_1update_baseline/trajectories \
  --max_obs_length 0 \
  --max_tokens 2048 \
  --save_trajectory
```

> Reminder: in parallel mode it is not recommended to enable `run.py`'s `--reset_before_each_task` (it would globally reset and disrupt other parallel processes). If you need "per-task-level isolation", run `run.py` serially (or lower `--max_parallel` to 1).
