# 从零到并行 Evaluation

这份文档面向“**第一次从零搭环境**”以及“**之后反复 reset 并并行跑 evaluation**”两类场景，覆盖：

- WebArena 环境（docker）搭建与日常重置
- reset server 启动与使用
- evaluation（Python 侧）依赖安装：**用 conda 替代 venv**
- 环境变量（网站 URL / Bedrock / OpenAI）
- 并行 evaluation 脚本 `scripts/parallel_by_sites.py` 的**完整参数说明**

> 约定：下文默认你在仓库根目录 `WebAgent-R1/` 下操作；其中 evaluation 代码在 `WebAgent-R1/Eval/`。

---

## 1) Setup WebArena 环境（docker，首次全量 / 后续 reset）

### 1.1 准备镜像与资源文件（只做一次）

进入 `WebAgent-R1/WebArena-Env-Setup/`，按其 `README.md` 要求下载并放置：

- WebArena 官方提供的若干 docker image tar（shopping / reddit / gitlab 等）
- Wikipedia 的 `.zim`
- OpenStreetMap 的 3 个 tar.gz（Zenodo）

并按需修改 `WebAgent-R1/WebArena-Env-Setup/00_vars.sh`：

- `PUBLIC_HOSTNAME`: 这台机器对评测机可访问的 IP/域名
- 各网站端口（`SHOPPING_PORT/REDDIT_PORT/...`）与 `RESET_PORT`
- `ARCHIVES_LOCATION`（默认 `./images`）：你下载好的 tar/zim 存放路径

首次加载镜像（耗时较长，仅一次）：

```bash
cd WebArena-Env-Setup
sudo bash 01_docker_load_images.sh
```

### 1.2 启动网站环境（每次 reset 都跑 02-06）

每次希望回到“干净环境”时，按顺序执行（建议在 tmux/screen 里跑）：

```bash
cd WebArena-Env-Setup
sudo bash 02_docker_remove_containers.sh
sudo bash 03_docker_create_containers.sh
sudo bash 04_docker_start_containers.sh
sudo bash 05_docker_patch_containers.sh
sudo bash 06_serve_homepage.sh
```

- `06_serve_homepage.sh` 会启动“入口首页服务”，**需要持续占用一个终端**。

#### 小 tip：用 nohup 挂后台（避免长期占用 terminal）

如果你不想让 `06_serve_homepage.sh` 持续占用当前 terminal，可用：

```bash
cd WebArena-Env-Setup
nohup sudo bash 06_serve_homepage.sh > homepage.log 2>&1 &
```

---

## 2) Setup reset server（可选，但强烈推荐用于自动化 reset）

reset server 用于在**不重建容器**的情况下触发“全实例 reset”（对 interrupt/replay 或多轮跑测非常方便）。

启动 reset server（同样需要持续运行）：

```bash
cd WebArena-Env-Setup
sudo bash 07_serve_reset.sh
```

它会在 `00_vars.sh` 的 `RESET_PORT` 上提供两个关键接口：

- `GET /reset`：触发 reset（如果已有 reset 正在执行，可能返回 418）
- `GET /status`：查看状态（Ready 才表示 reset 完成并可继续跑）

例如（请替换为你在 `00_vars.sh` 里配置的 host/port）：

```bash
curl "http://<PUBLIC_HOSTNAME>:7565/status"
curl "http://<PUBLIC_HOSTNAME>:7565/reset"
```

#### 小 tip：reset server 也建议 nohup

```bash
cd WebArena-Env-Setup
nohup sudo bash 07_serve_reset.sh > reset_server.log 2>&1 &
```

---

## 3) Evaluation 环境配置（用 conda 代替 venv）

进入 `WebAgent-R1/Eval/`，用 conda 创建独立环境（推荐 Python 3.10；3.11 也可；不建议 3.12）：

```bash
cd Eval
conda create -n webagent-eval python=3.10 -y
conda activate webagent-eval

pip install -r requirements.txt
playwright install
pip install -e .
```

---

## 4) 环境变量设置（网站 URL / Bedrock / OpenAI）

### 4.1 网站 URL 与端口（建议统一用 `setup_vars.sh` 管理）

本仓库提供了 `WebAgent-R1/Eval/setup_vars.sh`，用于一次性导出：

- `DATASET=webarena`
- 各网站 URL（`HOMEPAGE/SHOPPING/REDDIT/...`）
- OpenAI 相关变量（用于 evaluation 的 fuzzy match）

你需要做两件事：

- **把 `PUBLIC_HOSTNAME` 改成你的 WebArena 环境机器 IP/域名**
- **确保里面的端口与 `WebArena-Env-Setup/00_vars.sh` 一致**

然后在每次跑 eval 前：

```bash
cd Eval
source setup_vars.sh
```

> 注意：`setup_vars.sh` 当前导出的是完整 URL（例如 `http://10.49.48.242:8082`），不要再额外手动拼协议/端口，避免重复。

### 4.2 OpenAI（用于 fuzzy match 判分，开销很小）

即使你的 agent 走的是 Bedrock，evaluation 里某些任务会用 `llm_fuzzy_match / llm_ua_match` 做语义判分（只在需要 fuzzy match 的 evaluator 上触发，**调用量很小**）。

需要设置：

- `OPENAI_API_KEY`
- （可选）`OPENAI_API_URL`（默认 `https://api.openai.com/v1`）

对应代码位置：

- `Eval/evaluation_harness/helper_functions.py` 中的 `llm_fuzzy_match()` / `llm_ua_match()`

### 4.3 AWS Bedrock（用于 agent 推理）

当你用 `--provider bedrock` 时，推理走 `boto3` 的 credential chain（通常只要 `aws sts get-caller-identity` 可用就行）。

- **直接导出环境变量** `AWS_BEARER_TOKEN_BEDROC`

region 的来源优先级：

- 命令行 `--region us-east-2`（推荐，最明确）
- 或环境变量：`BEDROCK_REGION` / `AWS_BEDROCK_REGION` / `AWS_REGION` / `AWS_DEFAULT_REGION`

---

## 5) 运行并行 evaluation（推荐：`scripts/parallel_by_sites.py`）

### 5.1 示例命令

下面这条命令会：

- 在 `config_files/wa/test_webarena_lite/` 里跑任务索引 [0, 165)（**右开区间**）
- 并行启动最多 8 个 `run.py` 子进程
- 若 `test_result_webr1_newprompt_think/render_<idx>.html` 已存在则跳过
- 开跑前通过 reset server 做一次全局 reset
- 把 `--` 之后的参数原样转发给 `run.py`

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

### 5.2 `parallel_by_sites.py`（调度器）参数说明

这些参数由 `Eval/scripts/parallel_by_sites.py` 解析。

- `**--test_config_base_dir`（必填）**
  - **作用**：单任务配置目录，内部应存在 `${idx}.json`
  - **示例**：`config_files/wa/test_webarena_lite`
- `**--result_dir`（必填）**
  - **作用**：结果输出目录（`render_*.html`、`actions/`、`traces/`、`trajectories/` 等会写在这里）
  - **示例**：`test_result_webr1_newprompt_think`
- `**--test_start_idx` / `--test_end_idx`**
  - **作用**：指定任务索引范围，使用 Python 的 `range(start, end)`，即 **包含 start，不包含 end**
  - **默认**：`start=0`，`end=910`
  - **可选**：任意整数；建议 end 写成“最后一个任务索引 + 1”
- `**--test_indices`**
  - **作用**：用逗号分隔的索引列表，覆盖 `--test_start_idx/--test_end_idx`
  - **示例**：`--test_indices 0,5,7`
- `**--max_parallel`**
  - **作用**：同时运行的 `run.py` 子进程数量上限
  - **默认**：8
  - **可选**：正整数（建议根据机器 CPU/RAM 与网站承载能力调整）
- `**--chunk_size`**
  - **作用**：每个 `run.py` 进程里串行跑多少个 task（用 `run.py --test_indices a,b,c` 实现），用于减少启动开销
  - **默认**：0（自动均衡：大约分成 `--max_parallel` 份）
  - **可选**：
    - `0`：自动
    - `>=1`：固定 chunk 大小（越大单进程越久，失败回滚成本更高）
- `**--skip_finished`**
  - **作用**：若 `result_dir/render_<idx>.html` 已存在则跳过该任务
  - **可选**：开关参数（不带值）
- `**--python`**
  - **作用**：用于启动 `run.py` 的 Python 解释器路径
  - **默认**：当前解释器（`sys.executable`）
  - **常用**：在 conda 环境下通常无需显式指定；如需固定可写 `--python "$(which python)"`
- `**--run_script`**
  - **作用**：被调度执行的脚本（相对 `Eval/` 路径）
  - **默认**：`run.py`
- `**--reset_server_url`**
  - **作用**：reset server base URL（例如 `http://<host>:7565`）
  - **可选**：不传则不做任何自动 reset
- `**--reset_before_run`**
  - **作用**：在启动任何并行任务前，触发 **一次** 全局 reset（调用 `GET <reset_server_url>/reset` 并轮询 `GET .../status` 直到 Ready）
  - **可选**：开关参数（不带值）
- `**--reset_timeout_s` / `--reset_poll_interval_s` / `--reset_request_timeout_s` / `--reset_max_retries`**
  - **作用**：控制 reset 的等待超时、轮询间隔、HTTP 超时、触发 reset 的重试次数
  - **默认**：`600 / 2 / 10 / 3`
- `**--dry_run`**
  - **作用**：只打印将要执行的 `run.py` 命令，不实际运行

> 重要提醒：这个调度器**不做站点隔离**；并发跑多任务时，不要在每个 `run.py` 内启用“每任务 reset”（会全局 reset 干扰其他并行任务）。

### 5.3 `--` 之后（转发给 `run.py`）的常用参数说明

这些参数由 `Eval/run.py` 解析（`parallel_by_sites.py` 会把它们原样转发）。

- `**--instruction_path`**
  - **作用**：prompt/指令模板 JSON 路径
  - **示例**：
    - `agent/prompts/jsons/p_webrl_chat.json`
    - `agent/prompts/jsons/p_webrl_chat_think.json`
- `**--provider`**
  - **作用**：LLM 调用后端
  - **可选**（与代码一致的常用值）：
    - `bedrock`（AWS Bedrock）
    - `openai`（OpenAI / 兼容 OpenAI 的网关）
    - `google` / `api` / `finetune`（详见代码与 README）
- `**--mode`**
  - **作用**：调用模式（不同 provider 可能有不同含义）
  - **常用**：`chat`（聊天式）
- `**--model`**
  - **作用**：模型名/ID
  - **示例（Bedrock）**：`us.anthropic.claude-sonnet-4-5-20250929-v1:0`
- `**--region`**
  - **作用**：Bedrock region（当 `--provider bedrock` 时生效）
  - **示例**：`us-east-2`
  - **可选**：任意 AWS region；不传则会尝试从环境变量推断（见 4.3）
- `**--action_set_tag`**
  - **作用**：动作空间/解析方式（需要与 observation_type 搭配）
  - **常用可选**：
    - `webrl_id`（WebRL text setting，用原始字符串动作）
    - `id_accessibility_tree`（基于可访问性树的 ID 动作）
    - `som`（SoM 视觉 setting）
    - `id_accessibility_tree_with_captioner`（带 captioner 的变体）
- `**--observation_type`**
  - **作用**：环境观测形式
  - **可选**（代码里有显式 choices）：
    - `accessibility_tree`
    - `accessibility_tree_with_captioner`
    - `html`
    - `image`
    - `image_som`
    - `webrl`
- `**--planner_ip`**
  - **作用**：某些 self-host / finetune 模式下的模型服务地址（OpenAI-compatible endpoint 相关）
  - **常用**：不需要则传空字符串 `''` 或不传
- `**--max_obs_length`**
  - **作用**：观测文本截断长度；`0` 表示不截断
  - **默认**：3840
- `**--max_tokens`**
  - **作用**：模型输出 token 上限
  - **默认**：384
- `**--viewport_width` / `--viewport_height`**
  - **作用**：浏览器 viewport 尺寸（影响截图/可视区域/某些站点布局）
  - **默认**：`1280 / 2048`
- `**--save_trajectory`**
  - **作用**：保存每个任务的 action 序列（用于 interrupt/replay）
  - **输出**：`<result_dir>/trajectories/<task_id>.json`

#### 兼容性约束（很重要）

`run.py` 会检查某些组合是否合法，例如：

- 当 `--action_set_tag id_accessibility_tree` 时，`--observation_type` 只能是：
  - `accessibility_tree`
  - `accessibility_tree_with_captioner`
  - `image_som`

而你给出的 WebRL 示例是：

- `--action_set_tag webrl_id`
- `--observation_type webrl`

这是匹配的。

---

## 6) Interrupt 数据集工作流（raw → transformed_initial_intent → baseline → replay interrupt）

这一节说明：

- 如何从 raw data 生成 `run.py` 可用的 interrupt 配置（`interrupt_spec`）
- 如何从 raw data 生成“基于 transformed_initial_intent 的 task config files”（新的 `--test_config_base_dir`）
- 如何先跑 transformed_initial_intent 的 baseline evaluation（保存 trajectories），再用这些 trajectories 做 replay 的 interrupt evaluation

### 6.1 raw data 的格式与位置

raw interrupt 数据默认放在：

- `Eval/interrupt_config/raw/*.json`

文件格式是一个 JSON 数组，每个元素至少包含：

- `task_id`（int）
- `transformed_initial_intent`（str）
- `updates`（list[str]，至少一个）

例如（节选自 `interrupt_config/raw/1update_opus.json`）：

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

> `intent` 字段对生成脚本来说是可选的；两阶段评测真正用的是 `transformed_initial_intent` + `updates`。

### 6.2 raw → interrupt_spec（生成 replay/interrupt 所需配置）

脚本：`Eval/interrupt_config/convert_raw_to_interrupt_spec.py`

它会把 raw 里的 `updates` 拼接成 `update_intent`，生成 `run.py` 需要的：

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

推荐用法（把产物放到 `interrupt_config/process/`，避免和 raw 混在一起）：

```bash
cd Eval
python interrupt_config/convert_raw_to_interrupt_spec.py \
  --inputs interrupt_config/raw/1update_opus.json \
  --out_dir interrupt_config/process \
  --interrupt_at_action 5 \
  --update_mode append \
  --extra_steps 0
```

输出示例：

- `Eval/interrupt_config/process/interrupt_spec_1update_opus.json`

#### 可选：用百分比指定中断点（`interrupt_at_pct`）

`run.py` 支持按“保存的 action 数量百分比”指定中断点：

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

其中 N 为 replay 文件里的 action 数量，实际 replay 的步数是：

K=\lfloor 0.2\times N\rfloor


### 6.3 raw → transformed_initial_intent 的 task config files（生成新的 `test_config_base_dir`）

脚本：`Eval/interrupt_config/make_transformed_intent_configs.py`

它会读取 base config dir（例如 `config_files/wa/test_webarena_lite/` 下的 `0.json..164.json`），并对 raw 中出现的 `task_id`：

- 把 `cfg["intent"]` 替换为 `transformed_initial_intent`
- 同时把原始 intent 备份到 `cfg["true_intent"]`（如果原来没有）

本仓库里已有一份示例目录：

- `Eval/config_files/wa/test_webarena_lite_transformed_1update/`

你也可以按 raw 文件自定义生成目录（推荐把输出目录名和 raw 文件名对应起来）：

```bash
cd Eval
python interrupt_config/make_transformed_intent_configs.py \
  --base_config_dir config_files/wa/test_webarena_lite \
  --raw_interrupt_file interrupt_config/raw/1update_opus.json \
  --out_dir config_files/wa/test_webarena_lite_transformed_1update_opus
```

> 说明：如需强制每个 base 任务都必须有 transformed intent，可加 `--strict`；否则会跳过 raw 里没有的 task_id。

####（可选）raw → transformed dataset JSON（单文件 list[dict]）

如果你还需要生成类似 `config_files/wa/test_webarena_lite.json` 这种“单文件数据集”，用：

- `Eval/interrupt_config/make_transformed_intent_dataset_json.py`

示例：

```bash
cd Eval
python interrupt_config/make_transformed_intent_dataset_json.py \
  --base_dataset_json config_files/wa/test_webarena_lite.json \
  --raw_interrupt_file interrupt_config/raw/1update_opus.json \
  --out_dataset_json config_files/wa/test_webarena_lite_transformed_1update_opus.json
```

> 注意：`run.py`/`parallel_by_sites.py` 真正用的是“按 task_id 拆开的 config 目录”（`--test_config_base_dir`），上面这个单文件主要用于数据处理/统计用途。

### 6.4 Stage A：先跑 transformed_initial_intent 的 baseline evaluation（保存 trajectories）

目标：让 agent 在 **transformed_initial_intent** 下跑一遍，产出可 replay 的动作序列：

- `<baseline_result_dir>/trajectories/<task_id>.json`

并行运行示例（把 `--test_config_base_dir` 指向 transformed 目录，并加上 `--save_trajectory`）：

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

### 6.5 Stage B：用 Stage A 的 trajectories 做 replay + interrupt evaluation

目标：在同一个 **transformed_initial_intent** 起点上：

1. replay 前 \(K\) 步（来自 baseline trajectories）
2. 注入 `updates`（append/replace）
3. 继续完成任务并评分

关键输入：

- `--test_config_base_dir`: **同 Stage A（transformed 版 configs）**
- `--replay_trajectory_dir`: Stage A 产物目录（例如 `test_result_transformed_1update_baseline/trajectories`）
- `--interrupt_spec`: 6.2 生成的 interrupt spec（例如 `interrupt_config/process/interrupt_spec_1update_opus.json`）

并行运行示例：

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

> 提醒：并行模式下不建议启用 `run.py` 的 `--reset_before_each_task`（会全局 reset 干扰其他并行进程）。如需“每任务级别隔离”，建议串行跑 `run.py`（或把 `--max_parallel` 降到 1）。

