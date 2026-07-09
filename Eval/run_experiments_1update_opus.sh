#!/bin/bash


# Run first experiment: baseline
echo "=========================================="
echo "Starting baseline experiment..."
echo "=========================================="


source setup_vars.sh


python scripts/parallel_by_sites.py \
 --test_config_base_dir config_files/wa/test_webarena_lite_transformed_1update \
 --result_dir test_result_transformed_1update_opus_1 \
 --test_start_idx 0 --test_end_idx 165 \
 --max_parallel 8 \
 --skip_finished \
 --reset_server_url "http://localhost:7565" \
 --reset_before_run \
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


# Check if first experiment succeeded
if [ $? -ne 0 ]; then
   echo "=========================================="
   echo "First experiment failed! Exiting..."
   echo "=========================================="
   exit 1
fi


echo "=========================================="
echo "First experiment completed successfully!"
echo "=========================================="


# # Run second experiment: with interrupts
# echo "=========================================="
# echo "Starting interrupt experiment..."
# echo "=========================================="


# source setup_vars.sh


# python scripts/parallel_by_sites.py \
#  --test_config_base_dir config_files/wa/test_webarena_lite_transformed_1update \
#  --result_dir test_result_transformed_1update_interrupt_opus_04 \
#  --test_start_idx 0 --test_end_idx 165 \
#  --max_parallel 8 \
#  --skip_finished \
#  --reset_server_url "http://localhost:7565" \
#  --reset_before_run \
#  -- \
#  --instruction_path agent/prompts/jsons/p_webrl_chat_think.json \
#  --provider bedrock --mode chat \
#  --model us.anthropic.claude-opus-4-5-20251101-v1:0 \
#  --region us-east-2 \
#  --action_set_tag webrl_id \
#  --observation_type webrl \
#  --interrupt_spec interrupt_config/process/interrupt_spec_1update_opus.json \
#  --replay_trajectory_dir test_result_transformed_1update_opus_1/trajectories \
#  --max_obs_length 0 \
#  --max_tokens 2048 \
#  --save_trajectory


# # Check if second experiment succeeded
# if [ $? -ne 0 ]; then
#    echo "=========================================="
#    echo "Second experiment failed!"
#    echo "=========================================="
#    exit 1
# fi


# echo "=========================================="
# echo "Second experiments completed successfully!"
# echo "=========================================="


########################################################
# Run third experiment: with interrupts at 20%
echo "=========================================="
echo "Starting interrupt experiment at 20%..."
echo "=========================================="


source setup_vars.sh


python scripts/parallel_by_sites.py \
 --test_config_base_dir config_files/wa/test_webarena_lite_transformed_1update \
 --result_dir test_result_transformed_1update_interrupt_opus_02 \
 --test_start_idx 0 --test_end_idx 165 \
 --max_parallel 8 \
 --skip_finished \
 --reset_server_url "http://localhost:7565" \
 --reset_before_run \
 -- \
 --instruction_path agent/prompts/jsons/p_webrl_chat_think.json \
 --provider bedrock --mode chat \
 --model us.anthropic.claude-opus-4-5-20251101-v1:0 \
 --region us-east-2 \
 --action_set_tag webrl_id \
 --observation_type webrl \
 --interrupt_spec interrupt_config/process/interrupt_spec_1update_opus_02.json \
 --replay_trajectory_dir test_result_transformed_1update_opus_1/trajectories \
 --max_obs_length 0 \
 --max_tokens 2048 \
 --save_trajectory


# Check if third experiment succeeded
if [ $? -ne 0 ]; then
   echo "=========================================="
   echo "Third experiment failed!" 
   echo "=========================================="
   exit 1
fi


echo "=========================================="
echo "Third experiments completed successfully!"
echo "=========================================="


########################################################
# Run Fourth experiment: with interrupts at 40%
echo "=========================================="
echo "Starting interrupt experiment at 40%..."
echo "=========================================="


source setup_vars.sh


python scripts/parallel_by_sites.py \
 --test_config_base_dir config_files/wa/test_webarena_lite_transformed_1update \
 --result_dir test_result_transformed_1update_interrupt_opus_04 \
 --test_start_idx 0 --test_end_idx 165 \
 --max_parallel 8 \
 --skip_finished \
 --reset_server_url "http://localhost:7565" \
 --reset_before_run \
 -- \
 --instruction_path agent/prompts/jsons/p_webrl_chat_think.json \
 --provider bedrock --mode chat \
 --model us.anthropic.claude-opus-4-5-20251101-v1:0 \
 --region us-east-2 \
 --action_set_tag webrl_id \
 --observation_type webrl \
 --interrupt_spec interrupt_config/process/interrupt_spec_1update_opus_04.json \
 --replay_trajectory_dir test_result_transformed_1update_opus_1/trajectories \
 --max_obs_length 0 \
 --max_tokens 2048 \
 --save_trajectory


# Check if fourth experiment succeeded
if [ $? -ne 0 ]; then
   echo "=========================================="
   echo "Fourth experiment failed!" 
   echo "=========================================="
   exit 1
fi


echo "=========================================="
echo "Fourth experiments completed successfully!"
echo "=========================================="


########################################################
# Run Fifth experiment: with interrupts at 60%
echo "=========================================="
echo "Starting interrupt experiment at 60%..."
echo "=========================================="


source setup_vars.sh


python scripts/parallel_by_sites.py \
 --test_config_base_dir config_files/wa/test_webarena_lite_transformed_1update \
 --result_dir test_result_transformed_1update_interrupt_opus_06 \
 --test_start_idx 0 --test_end_idx 165 \
 --max_parallel 8 \
 --skip_finished \
 --reset_server_url "http://localhost:7565" \
 --reset_before_run \
 -- \
 --instruction_path agent/prompts/jsons/p_webrl_chat_think.json \
 --provider bedrock --mode chat \
 --model us.anthropic.claude-opus-4-5-20251101-v1:0 \
 --region us-east-2 \
 --action_set_tag webrl_id \
 --observation_type webrl \
 --interrupt_spec interrupt_config/process/interrupt_spec_1update_opus_06.json \
 --replay_trajectory_dir test_result_transformed_1update_opus_1/trajectories \
 --max_obs_length 0 \
 --max_tokens 2048 \
 --save_trajectory


# Check if fifth experiment succeeded
if [ $? -ne 0 ]; then
   echo "=========================================="
   echo "Fifth experiment failed!" 
   echo "=========================================="
   exit 1
fi


echo "=========================================="
echo "Fifth experiments completed successfully!"
echo "=========================================="


########################################################
# Run Sixth experiment: with interrupts at 80%
echo "=========================================="
echo "Starting interrupt experiment at 80%..."
echo "=========================================="


source setup_vars.sh


python scripts/parallel_by_sites.py \
 --test_config_base_dir config_files/wa/test_webarena_lite_transformed_1update \
 --result_dir test_result_transformed_1update_interrupt_opus_08 \
 --test_start_idx 0 --test_end_idx 165 \
 --max_parallel 8 \
 --skip_finished \
 --reset_server_url "http://localhost:7565" \
 --reset_before_run \
 -- \
 --instruction_path agent/prompts/jsons/p_webrl_chat_think.json \
 --provider bedrock --mode chat \
 --model us.anthropic.claude-opus-4-5-20251101-v1:0 \
 --region us-east-2 \
 --action_set_tag webrl_id \
 --observation_type webrl \
 --interrupt_spec interrupt_config/process/interrupt_spec_1update_opus_08.json \
 --replay_trajectory_dir test_result_transformed_1update_opus_1/trajectories \
 --max_obs_length 0 \
 --max_tokens 2048 \
 --save_trajectory


# Check if sixth experiment succeeded
if [ $? -ne 0 ]; then
   echo "=========================================="
   echo "Sixth experiment failed!" 
   echo "=========================================="
   exit 1
fi


echo "=========================================="
echo "Sixth experiments completed successfully!"
echo "=========================================="


########################################################
# Run Seventh experiment: No Interruption%
echo "=========================================="
echo "Starting No Interruption experiment..."   
echo "=========================================="


source setup_vars.sh

python scripts/parallel_by_sites.py \
 --test_config_base_dir config_files/wa/test_webarena_lite \
 --result_dir test_result_opus_0 \
 --test_start_idx 0 --test_end_idx 165 \
 --max_parallel 8 \
 --skip_finished \
 --reset_server_url "http://localhost:7565" \
 --reset_before_run \
 -- \
 --instruction_path agent/prompts/jsons/p_webrl_chat_think.json \
 --provider bedrock --mode chat \
 --model us.anthropic.claude-opus-4-5-20251101-v1:0 \
 --region us-east-2 \
 --action_set_tag webrl_id \
 --observation_type webrl \
 --planner_ip '' \
 --max_obs_length 0 \
 --max_tokens 2048 \
 --viewport_width 1280 \
 --viewport_height 720 \
 --save_trajectory

# Check if sixth experiment succeeded
if [ $? -ne 0 ]; then
   echo "=========================================="
   echo "Sixth experiment failed!" 
   echo "=========================================="
   exit 1
fi


echo "=========================================="
echo "Sixth experiments completed successfully!"
echo "=========================================="