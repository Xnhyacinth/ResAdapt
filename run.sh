#!/bin/bash
# ==============================================================================
# ResAdapt Execution Flow Example
# ==============================================================================

# ------------------------------------------------------------------------------
# STEP 1: Save the Allocator Model
# ------------------------------------------------------------------------------
# Before training, initialize and save the allocator model weights.
# You can configure the allocator architecture inside resadapt/allocator/smol_config.py
echo "1. Initializing and saving the allocator..."
# Uncomment and run the following commands to save the allocator:
# cd resadapt/allocator
# python3 modeling_allocator_smol_v3.py
# cd ../../

# ------------------------------------------------------------------------------
# STEP 2: Configure the Allocator Path
# ------------------------------------------------------------------------------
# After saving, ensure the path to the saved allocator is correctly set in the
# training script: resadapt/scripts/main.sh
# Example: ALLOCATOR_PATH=YOUR_WORKSPACE_PATH/models/allocator_smolv3

# ------------------------------------------------------------------------------
# STEP 3: Run Training
# ------------------------------------------------------------------------------
echo "2. Launching training examples..."

# Example 1: Qwen2.5-VL-7B training with 4 nodes, FSDP2
# NNODES=4 nohup bash resadapt/scripts/main.sh 7B mix_pt_asym_scale_smolv4_sim_ccen_filter_frozen_newtie_acc_cost_enc0.25_cen0.4_nframes128_video_mrope 8 8 fsdp2 1.8 1 16 2e-5 0 > logs_run/mix_ray4_pt_asym_scale_smolv4_sim_ccen_filter_frozen_newtie_acc_cost_enc0.25_cen0.4_nframes128_video_mrope_fsdp2_7b_s16_n1_min0.2_max_1.8_mini64_bsz128_len8_resp8_2e-5.log 2>&1 &

# Example 2: Qwen3-VL-4B training with 2 nodes, FSDP2
# NNODES=2 MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct nohup bash resadapt/scripts/main.sh 4B mix_base_nframes32_video 4 8 fsdp2 1.5 16 1 1e-5 0 > logs_run/mix_ray2_base_nframes32_video_4b_n16_mini64_bsz128_len4_resp8.log 2>&1 &

# Example 3: Qwen2.5-VL-7B baseline training with 2 nodes
# NNODES=2 nohup bash resadapt/scripts/main.sh 7B mix_base_nframes64_video 8 8 fsdp2 1.5 16 1 1e-5 0 > logs_run/mix_ray2_base_nframes64_video_7b_n16_mini64_bsz128_len8_resp8.log 2>&1 &


# ------------------------------------------------------------------------------
# STEP 4: Evaluation
# ------------------------------------------------------------------------------
echo "3. Launching evaluation examples..."

# Example 1: Evaluate baseline Qwen2.5-VL-7B on video tasks
# nohup bash resadapt/scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate video_all 0 0 0 0 0 logs_eval fix0.5 > logs/fix0.5.log 2>&1 &

# Example 2: Evaluate baseline Qwen2.5-VL-7B on image tasks
# nohup bash resadapt/scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate images 0 0 0 0 0 logs_eval fix1.5 > logs/fix1.5.log 2>&1 &

# Example 3: Evaluate trained model on video tasks
# bash resadapt/scripts/eval.sh YOUR_WORKSPACE_PATH/resadapt/ckpts/VideoQA_Qwen_Verify vllm_generate r4-mix_pt_asym_sc_smolv3_sim_ccen_filt_fr_new_enc0.25_cen0.4_nf32_vidm-dp2-s16-7B-bsz128-mini64-n1-max2.0-l4-r8_5e-5 250 290 video_add 1 all
