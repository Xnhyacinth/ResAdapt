# ResAdapt: Resource-Aware Adaptation for Large Vision-Language Models

This is the official repository for **ResAdapt**, a novel framework designed to adaptively allocate resources (such as frames or tokens) for Vision-Language Models (VLMs) and Video-LMs. By employing a lightweight **allocator**, ResAdapt dynamically determines the optimal resource allocation strategy to balance performance and computational efficiency.

## Quick Start

The training process involves three main steps: 
1. Saving the initial allocator weights.
2. Configuring the training script.
3. Launching the training.

### 1. Save the Allocator
Before starting the RL training, you need to initialize and save the allocator weights. Run the modeling script corresponding to your allocator version. For example, using the `smol_v3` allocator:

```bash
cd resadapt/allocator
python3 modeling_allocator_smol_v3.py
```
This will save the initialized allocator model to your configured path (e.g., `YOUR_WORKSPACE_PATH/models/allocator_smolv3`). 
You can adjust the configuration in `smol_config.py` if necessary.

### 2. Configure the Main Script
Once the allocator is saved, update the allocator path in the main training script `resadapt/scripts/main.sh`. 

Ensure that `ALLOCATOR_PATH` matches the directory where you saved the model in Step 1.
For example:
```bash
ALLOCATOR_PATH=YOUR_WORKSPACE_PATH/models/allocator_smolv3
```

### 3. Run Training
Use `run.sh` to configure the training parameters and launch the job. The `run.sh` script provides examples of how to run the pipeline across multiple nodes with FSDP2.

Example execution from the repository root:
```bash
# Example: 7B model with smol_v4 allocator, FSDP2, 4 nodes
NNODES=4 nohup bash resadapt/scripts/main.sh 7B mix_pt_asym_scale_smolv4_sim_ccen_filter_frozen_newtie_acc_cost_enc0.25_cen0.4_nframes128_video_mrope 8 8 fsdp2 1.8 1 16 2e-5 0 > logs_run/train_7b.log 2>&1 &
```

### Evaluation
You can evaluate your trained models using `resadapt/scripts/eval.sh` as shown in `run.sh`:
```bash
nohup bash resadapt/scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate video_all 0 0 0 0 0 logs_eval fix0.5 > logs/eval.log 2>&1 &
```

## Repository Structure

- `resadapt/allocator/`: Contains the definitions, configurations, and initialization scripts for the lightweight resource allocator.
- `resadapt/reward_fn/`: Includes reward functions and advantage computations used during RL training.
- `resadapt/scripts/`: Main bash scripts for launching training (`main.sh`) and evaluation (`eval.sh`).
- `resadapt/eval/`: Offline evaluation scripts and utilities.
- `resadapt/verl_patches/`: Custom patches for the `verl` framework, including data parallel actors and FSDP workers.
- `run.sh`: High-level entry point showing example commands for training and evaluation.

## Acknowledgments
This repository is built upon [verl](https://github.com/volcengine/verl) and [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). We thank the authors for their great open-source contributions.
