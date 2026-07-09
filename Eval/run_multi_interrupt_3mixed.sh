#!/bin/bash

# Multi-Interrupt Evaluation Script for 3mixed
# Based on MULTI_INTERRUPT_ZH.md

cd /home/ubuntu/Learn/WebAgent-R1/WebAgent-R1/Eval

# ========================================
# Stage 0.1: Generate transformed intent configs
# ========================================
echo "Stage 0.1: Generating transformed intent configs..."

python interrupt_config/make_transformed_intent_configs.py \
  --base_config_dir config_files/wa/test_webarena_lite \
  --raw_interrupt_file interrupt_config/raw/3mixed.json \
  --out_dir config_files/wa/test_webarena_transformed_3mixed

# ========================================
# Stage 0.2: Run baseline (no interrupt)
# ========================================
echo "Stage 0.2: Running baseline..."

source setup_vars.sh

python scripts/parallel_by_sites.py \
  --test_config_base_dir config_files/wa/test_webarena_transformed_3mixed \
  --result_dir test_result_3mixed_stage0_baseline \
  --test_start_idx 0 --test_end_idx 165 \
  --max_parallel 8 \
  --skip_finished \
  --reset_before_run \
  --reset_server_url "http://localhost:7565" \
  -- \
  --instruction_path agent/prompts/jsons/p_webrl_chat_think.json \
  --provider bedrock --mode chat \
  --model us.anthropic.claude-opus-4-5-20251101-v1:0 \
  --region us-east-2 \
  --action_set_tag webrl_id \
  --observation_type webrl \
  --max_obs_length 0 \
  --max_tokens 2048 \
  --save_trajectory

# ========================================
# Stage 1.1: Generate interrupt spec
# ========================================
echo "Stage 1.1: Generating interrupt spec for stage 1..."

python interrupt_config/make_multi_interrupt_stage.py \
  --raw_interrupt_file interrupt_config/raw/3mixed.json \
  --base_config_dir config_files/wa/test_webarena_transformed_3mixed \
  --result_dirs test_result_3mixed_stage0_baseline \
  --stage 1 \
  --out_config_dir config_files/wa/test_webarena_3mixed_stage1 \
  --out_interrupt_spec interrupt_config/process/interrupt_spec_3mixed_stage1.json

# ========================================
# Stage 1.2: Run replay + interrupt
# ========================================
echo "Stage 1.2: Running replay + interrupt for stage 1..."

source setup_vars.sh

python scripts/parallel_by_sites.py \
  --test_config_base_dir config_files/wa/test_webarena_3mixed_stage1 \
  --result_dir test_result_3mixed_stage1 \
  --test_start_idx 0 --test_end_idx 165 \
  --max_parallel 8 \
  --skip_finished \
  --reset_before_run \
  --reset_server_url "http://localhost:7565" \
  -- \
  --instruction_path agent/prompts/jsons/p_webrl_chat_think.json \
  --provider bedrock --mode chat \
  --model us.anthropic.claude-opus-4-5-20251101-v1:0 \
  --region us-east-2 \
  --action_set_tag webrl_id \
  --observation_type webrl \
  --max_obs_length 0 \
  --max_tokens 2048 \
  --save_trajectory \
  --replay_trajectory_dir test_result_3mixed_stage0_baseline/trajectories \
  --interrupt_spec interrupt_config/process/interrupt_spec_3mixed_stage1.json

# ========================================
# Stage 2.1: Generate interrupt spec
# ========================================
echo "Stage 2.1: Generating interrupt spec for stage 2..."

python interrupt_config/make_multi_interrupt_stage.py \
  --raw_interrupt_file interrupt_config/raw/3mixed.json \
  --base_config_dir config_files/wa/test_webarena_transformed_3mixed \
  --result_dirs test_result_3mixed_stage1 \
  --stage 2 \
  --out_config_dir config_files/wa/test_webarena_3mixed_stage2 \
  --out_interrupt_spec interrupt_config/process/interrupt_spec_3mixed_stage2.json

# ========================================
# Stage 2.2: Run replay + interrupt
# ========================================
echo "Stage 2.2: Running replay + interrupt for stage 2..."

source setup_vars.sh

python scripts/parallel_by_sites.py \
  --test_config_base_dir config_files/wa/test_webarena_3mixed_stage2 \
  --result_dir test_result_3mixed_stage2 \
  --test_start_idx 0 --test_end_idx 165 \
  --max_parallel 8 \
  --skip_finished \
  --reset_before_run \
  --reset_server_url "http://localhost:7565" \
  -- \
  --instruction_path agent/prompts/jsons/p_webrl_chat_think.json \
  --provider bedrock --mode chat \
  --model us.anthropic.claude-opus-4-5-20251101-v1:0 \
  --region us-east-2 \
  --action_set_tag webrl_id \
  --observation_type webrl \
  --max_obs_length 0 \
  --max_tokens 2048 \
  --save_trajectory \
  --replay_trajectory_dir test_result_3mixed_stage1/trajectories \
  --interrupt_spec interrupt_config/process/interrupt_spec_3mixed_stage2.json

# ========================================
# Stage 3.1: Generate interrupt spec
# ========================================
echo "Stage 3.1: Generating interrupt spec for stage 3..."

python interrupt_config/make_multi_interrupt_stage.py \
  --raw_interrupt_file interrupt_config/raw/3mixed.json \
  --base_config_dir config_files/wa/test_webarena_transformed_3mixed \
  --result_dirs test_result_3mixed_stage2 \
  --stage 3 \
  --out_config_dir config_files/wa/test_webarena_3mixed_stage3 \
  --out_interrupt_spec interrupt_config/process/interrupt_spec_3mixed_stage3.json

# ========================================
# Stage 3.2: Run replay + interrupt
# ========================================
echo "Stage 3.2: Running replay + interrupt for stage 3..."

source setup_vars.sh

python scripts/parallel_by_sites.py \
  --test_config_base_dir config_files/wa/test_webarena_3mixed_stage3 \
  --result_dir test_result_3mixed_stage3 \
  --test_start_idx 0 --test_end_idx 165 \
  --max_parallel 8 \
  --skip_finished \
  --reset_before_run \
  --reset_server_url "http://localhost:7565" \
  -- \
  --instruction_path agent/prompts/jsons/p_webrl_chat_think.json \
  --provider bedrock --mode chat \
  --model us.anthropic.claude-opus-4-5-20251101-v1:0 \
  --region us-east-2 \
  --action_set_tag webrl_id \
  --observation_type webrl \
  --max_obs_length 0 \
  --max_tokens 2048 \
  --save_trajectory \
  --replay_trajectory_dir test_result_3mixed_stage2/trajectories \
  --interrupt_spec interrupt_config/process/interrupt_spec_3mixed_stage3.json

echo "All stages completed!"
