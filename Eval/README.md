# Setup for VAB-WebArena-Lite

## Table of Contents

- [中文：从零到并行 Evaluation 指南](RUN_PARALLEL_EVAL_ZH.md)
- [Brief Introduction](#brief-introduction)
- [Install](#install)
- [Setup WebArena-Lite Environments](#setup-webarena-lite-environments)
- [🚨 Important: Refresh all websites before re-run another round of testing](#-important-refresh-all-websites-before-re-run-another-round-of-testing)
- [🛑 Interrupt / Replay Evaluation](#-interrupt--replay-evaluation)
- [🖼️ Evaluating in VAB Standard Setting with SoM (Set-of-Marks) Visual Agents](#️-evaluating-in-vab-standard-setting-with-som-set-of-marks-visual-agents)
  - [👎 Run Single Agent For Evalution](#-run-single-agent-for-evalution-slow-but-please-read-to-understand-meaning-of-arguments)
  - [👍 Run Parallel Agent For Evaluation](#-run-parallel-agent-for-evaluation-recommended)
- [🚀 Evaluating in WebRL Setting (Text Modal)](#-evaluating-in-webrl-setting-text-modal)
  - [Evaluation of Finetuned Models](#evaluation-of-finetuned-models)
  - [Evaluation of Proprietary Models](#evaluation-of-proprietary-models)
- [Run Visualized Demostration](#run-visualized-demostration)
- [Acknowledgements](#acknowledgements)

## Brief Introduction

VAB-WebArena-Lite is a 165-task refined subset from <a href="https://webarena.dev/" target="_blank">WebArena</a>.
The purpose of building this subset is to manually ensure task correctness & feasibility, and speed up testing (original 812-task WebArena usually takes more than 6h to run through, while VAB-WebArena-Lite takes around 40m in practice). 
The modified version of the test cases can be found in `config_files/wa/test_webarena_lite.raw.json`.


## Install

First, you should clone the official repository of <a href="https://github.com/web-arena-x/visualwebarena">VisualWebArena</a> to this directory

```bash
# Assume you have cloned VAB and is now in the `VAB-WebArena-Lite` directory
git clone https://github.com/web-arena-x/visualwebarena.git visualwebarena
cd visualwebarena
git reset --hard ad57aae4dad71531504726900b80db02e0526158
cd ..
```

Then, you should substitute the file with the commands below:

```bash
bash replace.sh
```

After that, you should install the dependencies for VAB-WebArena-Lite (recommend using an independent conda environment to VAB):

```bash
# Python 3.10 (or 3.11, but not 3.12 cause 3.12 deprecated distutils needed here)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install
pip install -e .
```

You can also run the unit tests to ensure that WebArena-Lite is installed correctly:

```bash
pytest -x
```

## Setup WebArena-Lite Environments

1. Setup the standalone environments.
Please check out [this page](https://github.com/web-arena-x/webarena/tree/main/environment_docker) for details.

2. Configurate the urls for each website.
First, export the `DATASET` to be `webarena`:

```bash
export DATASET=webarena
```

Then, set the URL for the websites
(🚨 Notice: check if default ports of websites below correspond to those you setup in the first step)

```bash
# Actually, the CLASSIFIEDS environment is not included in the WebArena-Lite evaluation, we keep the environment variables here just for consistency.
export CLASSIFIEDS="<your_classifieds_domain>:9980"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"

# Below are the variables you should set for the evaluation.
export SHOPPING="<your_shopping_site_domain>:7770"
export REDDIT="<your_reddit_domain>:9999"
export SHOPPING_ADMIN="<your_e_commerce_cms_domain>:7780/admin"
export GITLAB="<your_gitlab_domain>:8023"
export MAP="<your_map_domain>:3000"
export WIKIPEDIA="<your_wikipedia_domain>:8888"
export HOMEPAGE="<your_homepage_domain>:4399"
```

3. Generate config files for each test example:

```bash
python scripts/generate_test_data.py
```

You will see `*.json` files generated in the [config_files](./config_files) folder. Each file contains the configuration for one test example.

4. Obtain and save the auto-login cookies for all websites:

```bash
bash prepare.sh
```

5. Set up API keys.

```bash
export OPENAI_API_KEY=your_key

# Optional: if you use a different OpenAI model source
export OPENAI_API_URL=your_url 

# Optional: you can set the following variables to evaluate the preset model in llms/providers/api_utils.py
export GEMENI_API_KEY=your_key
export QWEN_API_KEY=your_key
export CLAUDE_API_KEY=your_key

# Optional: if you have trained your model, we recommend deploying it as an API service, where you can set a FINETUNED_URL to evaluate it.
export FINETUNED_URL=your_url

```

If using Gemini, first install the [gcloud CLI](https://cloud.google.com/sdk/docs/install). Configure the API key by authenticating with Google Cloud:

```bash
gcloud auth login
gcloud config set project <your_project_name>
```

## 🚨 Important: Refresh all websites before re-run another round of testing!
Since tasks in WebArena may involve changing status and database of websites (e.g., posting comments on Reddit), if websites are not all refreshed before another round of evaluation, the results would be problematic.

Please remember to always run following command (assume you are hosting WebArena websites on your own) to restart and refresh all website dockers to avoid potential contamination.
The process usually takes 3-5 minites.

```bash
# Make sure the script is executed on the machine that you run those website dockers
bash refresh_website_docker.sh
```

You may need to change some contents in the script (e.g. configured ports of websites, names of dockers, etc.).

## 🛑 Interrupt / Replay Evaluation

This repo supports an "interrupt" evaluation mode to simulate **a user interrupting the agent mid-run** and **updating the task instruction**.

The recommended workflow is **two-stage**:

1) **Baseline run**: run the full task normally, while saving the per-task action sequence (for replay).

2) **Interrupt run**: (optionally) reset the website dockers to a clean state, replay the saved actions up to an interruption point, inject an updated instruction, then continue the task and evaluate.

### Stage 1: Baseline evaluation (save trajectories for replay)

Add `--save_trajectory` to your normal evaluation command. This will write:
`<result_dir>/trajectories/<task_id>.json`

Example:

```bash
# Make sure env vars like DATASET/SHOPPING/REDDIT/... are set
source setup_vars.sh

python run.py \
  --instruction_path agent/prompts/jsons/p_cot_id_actree_3s.json \
  --test_config_base_dir config_files/wa/test_webarena_lite \
  --test_start_idx 0 --test_end_idx 1 \
  --result_dir test_result_deepseek_full \
  --provider bedrock --mode chat \
  --model deepseek.v3-v1:0 \
  --region us-east-2 \
  --action_set_tag id_accessibility_tree \
  --observation_type accessibility_tree \
  --save_trajectory
```

### Stage 2: Interrupt evaluation (reset once, replay, inject update, continue)

#### (Optional) Start the reset server

If you host WebArena websites yourself, you can use the reset server to reset all dockers.
From `WebAgent-R1/WebArena-Env-Setup/`:

```bash
sudo bash 07_serve_reset.sh
```

Then in another terminal you can check status:

```bash
curl http://127.0.0.1:7565/status
```

#### Prepare `interrupt_spec.json`

Create a JSON file to define where to interrupt and what instruction update to inject.
An example is provided at `interrupt_config/interrupt_spec.json`.

Minimal format:

```json
{
  "tasks": {
    "0": {
      "interrupt_at_action": 5,
      "update_mode": "append",
      "update_intent": "User update after interruption (placeholder text).",
      "extra_steps": 0
    }
  }
}
```

You can also specify the interruption point as a **percentage of the saved action sequence**:

```json
{
  "tasks": {
    "0": {
      "interrupt_at_pct": "20%",
      "update_mode": "append",
      "update_intent": "User update after interruption (placeholder text).",
      "extra_steps": 0
    }
  }
}
```

In this case, if the saved trajectory has \(N\) actions, the runner will replay
\(\lfloor 0.2 \times N \rfloor\) actions before injecting the update.

- `interrupt_at_action`: replay the first K actions (0-based count) before injecting the update.
- `update_mode`:
  - `append`: append update to the original intent (default).
  - `replace`: replace the original intent entirely.
- `update_intent`: the new instruction text to inject (currently injected into the `intent` string used by prompt).
- `extra_steps`: optionally increase the max step budget after interruption (per-task).

#### Run interrupt evaluation

Provide:
- `--interrupt_spec <path>`
- `--replay_trajectory_dir <baseline_result_dir>/trajectories`

If you want to avoid cross-run contamination, also provide:
- `--reset_server_url http://127.0.0.1:7565`
- `--reset_before_run` (reset ONCE before evaluating the task range)

Example:

```bash
source setup_vars.sh

python run.py \
  --instruction_path agent/prompts/jsons/p_cot_id_actree_3s.json \
  --test_config_base_dir config_files/wa/test_webarena_lite \
  --test_start_idx 0 --test_end_idx 1 \
  --result_dir test_result_deepseek_interrupt \
  --provider bedrock --mode chat \
  --model deepseek.v3-v1:0 \
  --region us-east-2 \
  --action_set_tag id_accessibility_tree \
  --observation_type accessibility_tree \
  --interrupt_spec interrupt_config/interrupt_spec.json \
  --replay_trajectory_dir test_result_deepseek_full/trajectories \
  --reset_server_url "http://127.0.0.1:7565" \
  --reset_before_run \
  --save_trajectory
```

Notes:
- `--reset_server_url` must point to a real host/IP (do NOT use `http://HOST:7565` placeholder).
- Replay re-executes recorded actions in a clean `env.reset()` state; this assumes the reset returns websites to a consistent initial state.

## 🖼️ Evaluating in VAB Standard Setting with SoM (Set-of-Marks) Visual Agents

### 👎 Run Single Agent For Evalution (Slow, but please read to understand meaning of arguments)

To run your own model with SoM visual agent,  you can run evaluation with the following flags:

```bash
python run.py
--instruction_path agent/prompts/jsons/p_cot_id_actree_3s.json
--test_config_base_dir config_files/wa/test_webarena_lite
--test_start_idx 0
--test_end_idx 1
--result_dir test_result_deepseek
--provider bedrock
--mode chat
--model deepseek.v3-v1:0
--region us-east-2
--action_set_tag id_accessibility_tree
--observation_type accessibility_tree
```

Besides the original model providers (OpenAI, Google), you can also add your models in `llms/providers/api_utils.py`. Remember to set `--provider` to:

- `api`: Keep the same input style as WebArena, suitable for regular API calls
- `finetune`: This is required for models trained with the data we provide.
- `bedrock`: Use AWS Bedrock `converse` API (set `BEDROCK_REGION`/`AWS_REGION` and `AWS_BEARER_TOKEN_BEDROCK` if needed).

For the `--model` variable, we use the format `<source>_<model-name>` .

- If there is no more optional models under source, you can set it to just `source`.
- Remember that the source name here should be added in the init function of `APIModel` in `llms/providers/api_utils.py`.
- For example, if you want to use the openai model "gpt-4o", you can set the flag like this: `--model openai_gpt-4o`.
- For Bedrock, you can set `--provider bedrock` and use the native model id, e.g. `--model us.anthropic.claude-sonnet-4-5-20250929-v1:0`.
- Bedrock region can be provided via `--region us-east-2` (recommended) or env vars `BEDROCK_REGION` / `AWS_REGION` / `AWS_DEFAULT_REGION`.

Finally, run `score.py` to get the pass rate
```bash 
python score.py <your_result_dir>
```

### 👍 Run Parallel Agent For Evaluation (Recommended)

To run the tests in parallel, you can first configure `wa_parallel_run.sh`, then run it. We default split the test set to 8 parallel-group for evaluation in VAB.

```bash
# Remember to first launch a tmux session
tmux
bash wa_parallel_run.sh
```

The script is enabled with auto-resuming if it is interrupted or met unexpected error. Please feel free to rerun the above command until all tasks finish.

After all parallel groupes finish, run `score.py` to get the pass rate
```bash 
python score.py <your_result_dir>
```

#### Parallel scheduler (no site isolation)

If you want to run many evaluation tasks in parallel, this repo provides a scheduler script. It **does not** isolate tasks by site; it assumes tasks are independent and can run concurrently (bounded by `--max_parallel`).

```bash
python scripts/parallel_by_sites.py \
  --test_config_base_dir config_files/wa/test_webarena_lite \
  --result_dir <your_result_dir> \
  --test_start_idx 0 --test_end_idx 165 \
  --max_parallel 8 \
  --skip_finished \
  --chunk_size 0 \
  -- \
  --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json \
  --provider openai --mode chat --model openai_gpt-4o
```

Notes:
- `--chunk_size` controls how many tasks are executed sequentially inside one `run.py` process (reduces startup overhead).
  - Use `--chunk_size 0` to auto-balance tasks across `--max_parallel`.
- Do **NOT** enable `--reset_server_url/--reset_before_each_task` in `run.py` when running multiple jobs concurrently, since a reset is global and will break other running jobs.
- You can also run arbitrary tasks by passing `--test_indices` to `run.py` (e.g. `--test_indices 0,5,7`).
 - If you want to reset the environment once before starting parallel jobs, use:
   `python scripts/parallel_by_sites.py ... --reset_server_url http://127.0.0.1:7565 --reset_before_run ...`

## 🚀 Evaluating in WebRL Setting (Text Modal)

[WebRL](https://github.com/THUDM/WebRL) is one of the top-performing models on WebArena-Lite. It uses a plain text modality as input. Additionally, we provide evaluation scripts that support this plain text modality.

**Before running, remember to read and follow the above environmental setup procedures!**

### Evaluation of Finetuned Models

To run the finetuned model in WebRL setting,  you can run evaluation with the following flags:

```bash
python run.py \
  --instruction_path agent/prompts/jsons/p_webrl.json \
  --test_start_idx 0 \
  --test_end_idx 1 \
  --result_dir <your_result_dir> \
  --test_config_base_dir config_files/wa/test_webarena_lite \
  --provider openai \
  --mode completion \
  --model <your_deployed_model_name> \
  --planner_ip <your_deployed_model_ip> \
  --stop_token "<|eot_id|>" \
  --max_obs_length 0 \
  --max_tokens 2048 \
  --viewport_width 1280 \
  --viewport_height 720 \
  --action_set_tag webrl_id  --observation_type webrl
```

You need to first use tools like vllm to deploy the finetuned model locally. Once deployed, the model can be accessed through the OpenAI API call method. 

Ensure that the `--model` and `--planner_ip` fields are completed with the appropriate model name and the IP address of the deployed model instance.

We also provide the parallel script.

```bash
# Remember to first launch a tmux session
tmux
bash wa_parallel_run_webrl.sh
```

### Evaluation of Proprietary Models

To run the proprietary model in WebRL setting,  you can run evaluation with the following flags:

```bash
python run.py \
  --instruction_path agent/prompts/jsons/p_webrl_chat.json \
  --test_start_idx 0 \
  --test_end_idx 1 \
  --result_dir <your_result_dir> \
  --test_config_base_dir config_files/wa/test_webarena_lite \
  --provider openai \
  --model GPT-4o \
  --mode chat \
  --planner_ip '' \
  --max_obs_length 0 \
  --max_tokens 2048 \
  --viewport_width 1280 \
  --viewport_height 720 \
  --action_set_tag webrl_id  --observation_type webrl
```

You can switch the evaluation model by modifying `--model`. We also provide the parallel script.

```bash
# Remember to first launch a tmux session
tmux
bash wa_parallel_run_webrl_chat.sh
```


## Run Visualized Demostration
Original WebArena have also prepared a demo for you to run the agents on your own task on an arbitrary webpage. An example is shown above where the agent is tasked to find the best Thai restaurant in Pittsburgh.

After following the setup instructions above and setting the OpenAI API key (the other environment variables for website URLs aren't really used, so you should be able to set them to some dummy variable), you can run the GPT-4V + SoM agent with the following command:

```bash
python run_demo.py \
  --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json \
  --start_url "https://www.amazon.com" \
  --image "https://media.npr.org/assets/img/2023/01/14/this-is-fine_wide-0077dc0607062e15b476fb7f3bd99c5f340af356-s1400-c100.jpg" \
  --intent "Help me navigate to a shirt that has this on it." \
  --result_dir demo_test_amazon \
  --model gpt-4-vision-preview \
  --action_set_tag som  --observation_type image_som \
  --render
```

This tasks the agent to find a shirt that looks like the provided image (the "This is fine" dog) from Amazon. Have fun!

## Acknowledgements

Please cite our paper if you find VAB-WebArena-Lite useful for your work:

```bibtex
@article{liu2024visualagentbench,
  title={VisualAgentBench: Towards Large Multimodal Models as Visual Foundation Agents},
  author={Liu, Xiao and Zhang, Tianjie and Gu, Yu and Iong, Iat Long and Xu, Yifan and Song, Xixuan and Zhang, Shudan and Lai, Hanyu and Liu, Xinyi and Zhao, Hanlin and others},
  journal={arXiv preprint arXiv:2408.06327},
  year={2024}
}
```

Our code is heavily based off the <a href="https://github.com/web-arena-x/webarena">WebArena codebase</a> and <a href="https://github.com/web-arena-x/visualwebarena">VisualWebArena codebase</a>.
If you find this environment useful, please also consider citing <a href="https://jykoh.com/vwa" target="_blank">VisualWebArena</a> as well as <a href="https://webarena.dev/" target="_blank">WebArena</a>:

```bibtex
@article{koh2024visualwebarena,
  title={VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks},
  author={Koh, Jing Yu and Lo, Robert and Jang, Lawrence and Duvvur, Vikram and Lim, Ming Chong and Huang, Po-Yu and Neubig, Graham and Zhou, Shuyan and Salakhutdinov, Ruslan and Fried, Daniel},
  journal={arXiv preprint arXiv:2401.13649},
  year={2024}
}

@article{zhou2024webarena,
  title={WebArena: A Realistic Web Environment for Building Autonomous Agents},
  author={Zhou, Shuyan and Xu, Frank F and Zhu, Hao and Zhou, Xuhui and Lo, Robert and Sridhar, Abishek and Cheng, Xianyi and Bisk, Yonatan and Fried, Daniel and Alon, Uri and others},
  journal={ICLR},
  year={2024}
}
```

