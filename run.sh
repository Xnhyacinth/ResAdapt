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

# Example 1: Qwen2.5-VL-7B training with 1 node, FSDP2
# NNODES=1 \
# NFRAMES=8 \
# ALLOCATOR_PATH=YOUR_WORKSPACE_PATH/models/allocator_smol_init \
# TRAIN_FILE=YOUR_WORKSPACE_PATH/data/train.parquet \
# TEST_FILE=YOUR_WORKSPACE_PATH/data/test.parquet \
# nohup bash resadapt/scripts/main.sh Qwen/Qwen2.5-VL-7B-Instruct scale > logs_run/train_7b.log 2>&1 &

# Example 2: Qwen3-VL-4B training with 2 nodes, FSDP2
# NNODES=2 \
# NFRAMES=8 \
# ALLOCATOR_PATH=YOUR_WORKSPACE_PATH/models/allocator_smol_init \
# TRAIN_FILE=YOUR_WORKSPACE_PATH/data/train.parquet \
# TEST_FILE=YOUR_WORKSPACE_PATH/data/test.parquet \
# nohup bash resadapt/scripts/main.sh Qwen/Qwen3-VL-4B-Instruct scale > logs_run/train_4b.log 2>&1 &

# Example 3: Qwen2.5-VL-7B baseline training with 2 nodes
# NNODES=2 \
# NFRAMES=8 \
# ALLOCATOR_PATH=YOUR_WORKSPACE_PATH/models/allocator_smol_init \
# TRAIN_FILE=YOUR_WORKSPACE_PATH/data/train.parquet \
# TEST_FILE=YOUR_WORKSPACE_PATH/data/test.parquet \
# nohup bash resadapt/scripts/main.sh Qwen/Qwen2.5-VL-7B-Instruct base > logs_run/train_7b_base.log 2>&1 &


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
