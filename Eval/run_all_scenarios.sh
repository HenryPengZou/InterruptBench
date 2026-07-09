#!/bin/bash


# Run  experiment: baseline for Modification Scenario
echo "=========================================="
echo "Starting baseline experiment for Modification Scenario..." 
echo "=========================================="


source setup_vars.sh


python scripts/parallel_by_sites.py \
 --test_config_base_dir config_files/wa/test_webarena_lite_transformed_2modification \
 --result_dir test_result_transformed_2modification_haiku_1 \
 --test_start_idx 0 --test_end_idx 165 \
 --max_parallel 8 \
 --reset_server_url "http://localhost:7565" \
 --reset_before_run \
 -- \
 --instruction_path agent/prompts/jsons/p_webrl_chat_think.json \
 --provider bedrock --mode chat \
 --model us.anthropic.claude-haiku-4-5-20251001-v1:0 \
 --region us-east-2 \
 --action_set_tag webrl_id \
 --observation_type webrl \
 --max_obs_length 0 \
 --max_tokens 2048 \
 --save_trajectory


# Check if experiment succeeded
if [ $? -ne 0 ]; then
   echo "=========================================="
   echo "Baseline experiment for Modification Scenario failed! Exiting..."
   echo "=========================================="
   exit 1
fi


echo "=========================================="
echo "Baseline experiment for Modification Scenario completed successfully!"
echo "=========================================="



########################################################
# Run experiment: with interrupts at 60% for Modification Scenario
echo "=========================================="
echo "Starting interrupt experiment at 60% for Modification Scenario..."
echo "=========================================="


source setup_vars.sh


python scripts/parallel_by_sites.py \
 --test_config_base_dir config_files/wa/test_webarena_lite_transformed_2modification \
 --result_dir test_result_transformed_2modification_interrupt_haiku_06 \
 --test_start_idx 0 --test_end_idx 165 \
 --max_parallel 8 \
 --reset_server_url "http://localhost:7565" \
 --reset_before_run \
 -- \
 --instruction_path agent/prompts/jsons/p_webrl_chat_think.json \
 --provider bedrock --mode chat \
 --model us.anthropic.claude-haiku-4-5-20251001-v1:0 \
 --region us-east-2 \
 --action_set_tag webrl_id \
 --observation_type webrl \
 --interrupt_spec interrupt_config/process/interrupt_spec_2modification_06.json \
 --replay_trajectory_dir test_result_transformed_2modification_haiku_1/trajectories \
 --max_obs_length 0 \
 --max_tokens 2048 \
 --save_trajectory


# Check if experiment succeeded
if [ $? -ne 0 ]; then
   echo "=========================================="
   echo "Experiment for Modification Scenario 60% Interrupts failed!" 
   echo "=========================================="
   exit 1
fi


echo "=========================================="
echo "Experiment for Modification Scenario 60% Interrupts completed successfully!"
echo "=========================================="



########################################################
########################################################


# Run  experiment: baseline for Retraction Scenario
echo "=========================================="
echo "Starting baseline experiment for Retraction Scenario..." 
echo "=========================================="


source setup_vars.sh


python scripts/parallel_by_sites.py \
 --test_config_base_dir config_files/wa/test_webarena_lite_transformed_2retraction \
 --result_dir test_result_transformed_2retraction_haiku_1 \
 --test_start_idx 0 --test_end_idx 165 \
 --max_parallel 8 \
 --reset_server_url "http://localhost:7565" \
 --reset_before_run \
 -- \
 --instruction_path agent/prompts/jsons/p_webrl_chat_think.json \
 --provider bedrock --mode chat \
 --model us.anthropic.claude-haiku-4-5-20251001-v1:0 \
 --region us-east-2 \
 --action_set_tag webrl_id \
 --observation_type webrl \
 --max_obs_length 0 \
 --max_tokens 2048 \
 --save_trajectory


# Check if experiment succeeded
if [ $? -ne 0 ]; then
   echo "=========================================="
   echo "Baseline experiment for Retraction Scenario failed! Exiting..."
   echo "=========================================="
   exit 1
fi


echo "=========================================="
echo "Baseline experiment for Retraction Scenario completed successfully!"
echo "=========================================="



########################################################
# Run experiment: with interrupts at 60% for Retraction Scenario
echo "=========================================="
echo "Starting interrupt experiment at 60% for Retraction Scenario..."
echo "=========================================="


source setup_vars.sh


python scripts/parallel_by_sites.py \
 --test_config_base_dir config_files/wa/test_webarena_lite_transformed_2retraction \
 --result_dir test_result_transformed_2retraction_interrupt_haiku_06 \
 --test_start_idx 0 --test_end_idx 165 \
 --max_parallel 8 \
 --reset_server_url "http://localhost:7565" \
 --reset_before_run \
 -- \
 --instruction_path agent/prompts/jsons/p_webrl_chat_think.json \
 --provider bedrock --mode chat \
 --model us.anthropic.claude-haiku-4-5-20251001-v1:0 \
 --region us-east-2 \
 --action_set_tag webrl_id \
 --observation_type webrl \
 --interrupt_spec interrupt_config/process/interrupt_spec_2retraction_06.json \
 --replay_trajectory_dir test_result_transformed_2retraction_haiku_1/trajectories \
 --max_obs_length 0 \
 --max_tokens 2048 \
 --save_trajectory


# Check if experiment succeeded
if [ $? -ne 0 ]; then
   echo "=========================================="
   echo "Experiment for Retraction Scenario 60% Interrupts failed!" 
   echo "=========================================="
   exit 1
fi


echo "=========================================="
echo "Experiment for Retraction Scenario 60% Interrupts completed successfully!"
echo "=========================================="




########################################################
########################################################


########################################################
# Run experiment: No Interruption%
echo "=========================================="
echo "Starting No Interruption experiment..."   
echo "=========================================="


source setup_vars.sh

python scripts/parallel_by_sites.py \
 --test_config_base_dir config_files/wa/test_webarena_lite \
 --result_dir test_result_haiku_0 \
 --test_start_idx 0 --test_end_idx 165 \
 --max_parallel 8 \
 --reset_server_url "http://localhost:7565" \
 --reset_before_run \
 -- \
 --instruction_path agent/prompts/jsons/p_webrl_chat_think.json \
 --provider bedrock --mode chat \
 --model us.anthropic.claude-haiku-4-5-20251001-v1:0 \
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




########################################################
########################################################

# Run  experiment: baseline for Update Scenario
echo "=========================================="
echo "Starting baseline experiment for Update Scenario..." 
echo "=========================================="


source setup_vars.sh


python scripts/parallel_by_sites.py \
 --test_config_base_dir config_files/wa/test_webarena_lite_transformed_1update \
 --result_dir test_result_transformed_1update_haiku_1 \
 --test_start_idx 0 --test_end_idx 165 \
 --max_parallel 8 \
 --reset_server_url "http://localhost:7565" \
 --reset_before_run \
 -- \
 --instruction_path agent/prompts/jsons/p_webrl_chat_think.json \
 --provider bedrock --mode chat \
 --model us.anthropic.claude-haiku-4-5-20251001-v1:0 \
 --region us-east-2 \
 --action_set_tag webrl_id \
 --observation_type webrl \
 --max_obs_length 0 \
 --max_tokens 2048 \
 --save_trajectory


# Check if experiment succeeded
if [ $? -ne 0 ]; then
   echo "=========================================="
   echo "Baseline experiment for Update Scenario failed! Exiting..."
   echo "=========================================="
   exit 1
fi


echo "=========================================="
echo "Baseline experiment for Update Scenario completed successfully!"
echo "=========================================="



########################################################
########################################################



########################################################
# Run experiment: with interrupts at 60% for Update Scenario
echo "=========================================="
echo "Starting interrupt experiment at 60% for Update Scenario..."
echo "=========================================="


source setup_vars.sh


python scripts/parallel_by_sites.py \
 --test_config_base_dir config_files/wa/test_webarena_lite_transformed_1update \
 --result_dir test_result_transformed_1update_interrupt_haiku_06 \
 --test_start_idx 0 --test_end_idx 165 \
 --max_parallel 8 \
 --reset_server_url "http://localhost:7565" \
 --reset_before_run \
 -- \
 --instruction_path agent/prompts/jsons/p_webrl_chat_think.json \
 --provider bedrock --mode chat \
 --model us.anthropic.claude-haiku-4-5-20251001-v1:0 \
 --region us-east-2 \
 --action_set_tag webrl_id \
 --observation_type webrl \
 --interrupt_spec interrupt_config/process/interrupt_spec_1update_06.json \
 --replay_trajectory_dir test_result_transformed_1update_haiku_1/trajectories \
 --max_obs_length 0 \
 --max_tokens 2048 \
 --save_trajectory


# Check if experiment succeeded
if [ $? -ne 0 ]; then
   echo "=========================================="
   echo "Experiment for Update Scenario 60% Interrupts failed!" 
   echo "=========================================="
   exit 1
fi


echo "=========================================="
echo "Experiment for Update Scenario 60% Interrupts completed successfully!"
echo "=========================================="















