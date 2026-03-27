# ResAdapt: Efficient Multimodal Reasoning via Adaptive Resolution

**ResAdapt** is an **Input-side adaptation** framework designed to adaptively allocate visual budgets (such as spatial resolution or tokens) for Vision-Language Models (VLMs) and Video-LMs *before* encoding.

By coupling a lightweight **Allocator** with an unchanged MLLM backbone, ResAdapt dynamically determines the optimal resource allocation strategy to balance performance and computational efficiency. It converts sparse rollout feedback into a stable learning signal using **Cost-Aware Policy Optimization (CAPO)**.

## Highlights

- **Input-side Adaptation:** Reallocates visual budget before encoding, preserving the backbone's native token interface and compatibility with optimized inference engines (e.g., FlashAttention, vLLM).
- **Cost-Aware Policy Optimization (CAPO):** Formulates visual allocation as a contextual bandit, training the Allocator to optimize the efficiency-accuracy Pareto frontier.
- **Extreme Efficiency:** Matches or surpasses uncompressed baselines while compressing over **90%** of visual tokens.
- **Long-context Video Reasoning:** Supports up to **16× more frames** at the same visual budget, delivering over 15% performance gain on reasoning-heavy benchmarks.
- **Active Perception:** The learned policy exhibits open-loop active perception, concentrating visual budget on information-dense content without explicit saliency supervision.

## Installation

To set up the environment, run the following commands:

```bash
conda create -n resadapt python=3.12
conda activate resadapt
pip install -e ".[vllm]"
pip install "transformers<5.0"
pip install flash-attn --no-build-isolation
pip install rouge_score sympy num2words numpy==1.26.4
# For CUDA-enabled TorchCodec (replace cu128 with your CUDA version, e.g., cu118, cu121)
conda install "ffmpeg" -c conda-forge
pip install torchcodec --index-url=https://download.pytorch.org/whl/cu128
```

> **Note**: For training with Qwen3-VL models, it is recommended to use PyTorch 2.10 and `vllm==0.18.0` for optimal compatibility.

## Quick Start

The training process involves three main steps:

1. Saving the initial allocator weights.
2. Configuring the training script.
3. Launching the training.

### 1. Save the Allocator

Before starting the RL training, you need to initialize and save the allocator weights. We recommend using the `smol_v2` allocator which features an optimized configuration.

```bash
cd resadapt/allocator
python3 modeling_allocator_smol_v2.py
```

This will save the initialized allocator model to your configured path (e.g., `YOUR_WORKSPACE_PATH/models/allocator_smol_init`).
You can adjust the configuration in `smol_config.py` if necessary.

### 2. Configure the Main Script

Once the allocator is saved, update the allocator path in the main training script `resadapt/scripts/main.sh`.

Ensure that `ALLOCATOR_PATH` matches the directory where you saved the model in Step 1.
For example:

```bash
ALLOCATOR_PATH=YOUR_WORKSPACE_PATH/models/allocator_smol_init
```

### 3. Run Training

Use `run.sh` to configure the training parameters and launch the job. The training script supports configurable environment variables for adjusting paths and parameters.

Example execution from the repository root:

```bash
# Example: 7B model with smol allocator, FSDP2, 1 node
NNODES=1 \
ALLOCATOR_PATH=YOUR_WORKSPACE_PATH/models/allocator_smol_init \
TRAIN_FILE=YOUR_WORKSPACE_PATH/data/train.parquet \
TEST_FILE=YOUR_WORKSPACE_PATH/data/test.parquet \
nohup bash resadapt/scripts/main.sh Qwen/Qwen2.5-VL-7B-Instruct scale > logs_run/train_7b.log 2>&1 &
```

**Key Environment Variables & Parameters:**
- `NNODES`: Number of nodes to use for distributed training (default: 1).
- `ALLOCATOR_PATH`: Path to your initialized allocator weights.
- `TRAIN_FILE` / `TEST_FILE`: Paths to your training and validation parquet files.
- `Qwen/Qwen2.5-VL-7B-Instruct`: The backbone model path (first positional argument).
- `scale`: The configuration tag for multimodal data scaling (second positional argument).

Alternatively, you can use named arguments:

```bash
NNODES=4 \
ALLOCATOR_PATH=YOUR_WORKSPACE_PATH/models/allocator_smol_init \
TRAIN_FILE=YOUR_WORKSPACE_PATH/data/train.parquet \
TEST_FILE=YOUR_WORKSPACE_PATH/data/test.parquet \
nohup bash resadapt/scripts/main.sh --model_path Qwen/Qwen2.5-VL-7B-Instruct --scale_data scale > logs_run/train_7b.log 2>&1 &
```

### Evaluation

You can evaluate your trained models using `resadapt/scripts/eval.sh` as shown in `run.sh`:

```bash
nohup bash resadapt/scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct vllm_generate video_all 0 0 0 0 0 logs_eval fix0.5 > logs/eval.log 2>&1 &
```

## Repository Structure

- `examples/data_preprocess/`: Example scripts and documentation for preprocessing raw video datasets into the standardized Parquet format required for training.
- `resadapt/allocator/`: Contains the definitions, configurations, and initialization scripts for the lightweight resource allocator.
- `resadapt/reward_fn/`: Includes reward functions and advantage computations used during RL training (e.g., CAPO and temporal similarity regularizers).
- `resadapt/scripts/`: Main bash scripts for launching training (`main.sh`) and evaluation (`eval.sh`).
- `resadapt/eval/`: Offline evaluation scripts and utilities.
- `resadapt/verl_patches/`: Custom patches for the `verl` framework, including data parallel actors and FSDP workers.
- `run.sh`: High-level entry point showing example commands for training and evaluation.

## Acknowledgments

This repository is built upon [verl](https://github.com/volcengine/verl) and [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). We thank the authors for their great open-source contributions.
