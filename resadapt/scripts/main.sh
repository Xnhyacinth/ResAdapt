#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# 0. Project Paths & Environment Variables
# ==============================================================================
# Resolve project root directory automatically based on script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." &> /dev/null && pwd)"

# Setup Python path to include project roots
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/resadapt:${PYTHONPATH:-}"
export PYTORCH_KERNEL_CACHE_PATH="${PROJECT_ROOT}/.cache/torch/kernels"

# Hugging Face offline mode configuration (default: disabled)
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-"1"}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-"1"}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-"1"}
# export HF_ENDPOINT=https://hf-mirror.com/ # Uncomment if using HF mirror

# Ray and Backend Configuration
export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.98}

# Clear shared memory if requested (useful for avoiding OOM in distributed training)
: "${CLEAR_SHM:=1}"
if [[ "${CLEAR_SHM}" == "1" ]]; then
    if [[ -d /dev/shm ]]; then
        rm -rf /dev/shm/* || true
    fi
fi

# ==============================================================================
# 1. CLI Arguments Parsing
# ==============================================================================
# Define default values for script arguments
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"  # Path or name of the model
SCALE_MULTI_MODAL_DATA="scale"          # Configuration tag for multimodal data scaling (e.g., "scale", "base")
PROMPT_LEN=8                            # Maximum prompt length in thousands (K) of tokens
RESP_LEN=8                              # Maximum response length in thousands (K) of tokens
STRATEGY="fsdp2"                        # Distributed training strategy (e.g., "fsdp2", "fsdp")
MAX_SCALE=2.0                           # Maximum scaling factor for resolution/frames
N_RESP_PER_PROMPT=2                     # Number of responses to generate per prompt
SCALE_N=8                               # Number of scales to use per prompt
LR="1e-5"                               # Learning rate for the allocator model
DEBUG="0"                               # Debug mode flag (0 for off, 1 for on)

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model_path PATH         Path or name of the model (default: ${MODEL_PATH})"
    echo "  --scale_data DATA         Configuration tag for multimodal data scaling (e.g., \"scale\", \"base\") (default: ${SCALE_MULTI_MODAL_DATA})"
    echo "  --prompt_len LEN          Maximum prompt length in thousands (K) of tokens (default: ${PROMPT_LEN})"
    echo "  --resp_len LEN            Maximum response length in thousands (K) of tokens (default: ${RESP_LEN})"
    echo "  --strategy STRATEGY       Distributed training strategy (e.g., \"fsdp2\", \"fsdp\") (default: ${STRATEGY})"
    echo "  --max_scale SCALE         Maximum scaling factor for resolution/frames (default: ${MAX_SCALE})"
    echo "  --n_resp N                Number of responses to generate per prompt (default: ${N_RESP_PER_PROMPT})"
    echo "  --scale_n N               Number of scales to use per prompt (default: ${SCALE_N})"
    echo "  --lr LR                   Learning rate for the allocator model (default: ${LR})"
    echo "  --debug 0/1               Debug mode flag (0 for off, 1 for on) (default: ${DEBUG})"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Environment (cost / advantage):"
    echo "  SCALE_ENABLE_COST=0|1     (default: ${SCALE_ENABLE_COST})"
    echo "  SCALE_USE_COST            Short tag, e.g. capo, piecewise_v2, saliency_share_v1 (default: ${SCALE_USE_COST})"
    echo "  SCALE_COST_ENCOURAGEMENT_COEF  (default: ${SCALE_COST_ENCOURAGEMENT_COEF})"
    echo "  SCALE_COST_PENALTY_COEF        (default: ${SCALE_COST_PENALTY_COEF})"
    echo "  SCALE_CENTERED_TERM       (default: ${SCALE_CENTERED_TERM})"
    echo ""
    echo "Note: Positional arguments are also supported for backward compatibility:"
    echo "  $0 [model_path] [scale_data] [prompt_len] [resp_len] [strategy] [max_scale] [n_resp] [scale_n] [lr] [debug]"
}

# Parse named arguments or fallback to positional for backward compatibility
if [[ $# -gt 0 && "$1" == -* ]]; then
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model_path) MODEL_PATH="$2"; shift 2 ;;
            --scale_data) SCALE_MULTI_MODAL_DATA="$2"; shift 2 ;;
            --prompt_len) PROMPT_LEN="$2"; shift 2 ;;
            --resp_len) RESP_LEN="$2"; shift 2 ;;
            --strategy) STRATEGY="$2"; shift 2 ;;
            --max_scale) MAX_SCALE="$2"; shift 2 ;;
            --n_resp) N_RESP_PER_PROMPT="$2"; shift 2 ;;
            --scale_n) SCALE_N="$2"; shift 2 ;;
            --lr) LR="$2"; shift 2 ;;
            --debug) DEBUG="$2"; shift 2 ;;
            -h|--help) usage; exit 0 ;;
            *) echo "Unknown parameter: $1"; usage; exit 1 ;;
        esac
    done
else
    # Fallback to positional arguments
    MODEL_PATH=${1:-$MODEL_PATH}
    SCALE_MULTI_MODAL_DATA=${2:-$SCALE_MULTI_MODAL_DATA}
    PROMPT_LEN=${3:-$PROMPT_LEN}
    RESP_LEN=${4:-$RESP_LEN}
    STRATEGY=${5:-$STRATEGY}
    MAX_SCALE=${6:-$MAX_SCALE}
    N_RESP_PER_PROMPT=${7:-$N_RESP_PER_PROMPT}
    SCALE_N=${8:-$SCALE_N}
    LR=${9:-$LR}
    DEBUG=${10:-$DEBUG}
fi

# ==============================================================================
# 2. Ray Cluster Configuration
# ==============================================================================
# Determine IP address for Ray cluster
IP=""
if command -v hostname >/dev/null 2>&1 && hostname -I >/dev/null 2>&1; then
    IP=$(hostname -I | awk '{print $1}')
fi
if [[ -z "${IP}" ]] && command -v python3 >/dev/null 2>&1; then
    IP=$(python3 -c 'import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80)); print(s.getsockname()[0]); s.close()' 2>/dev/null || echo "")
fi
IP=${IP:-127.0.0.1}

export RAY_ADDRESS="http://${IP}:8888"
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}

# Configure Ray environment variables arguments
ray_env_args=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE='${HF_HUB_OFFLINE}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.HF_DATASETS_OFFLINE='${HF_DATASETS_OFFLINE}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_OFFLINE='${TRANSFORMERS_OFFLINE}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='${PYTHONPATH}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_KERNEL_CACHE_PATH='${PYTORCH_KERNEL_CACHE_PATH}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.RAY_memory_usage_threshold='${RAY_memory_usage_threshold}'"
)

# ==============================================================================
# 3. Model & Data Paths
# ==============================================================================
RAY_DATA_HOME=${RAY_DATA_HOME:-"${WORKING_DIR}"}

# NOTE: Ensure max_position_embeddings in config.json is set to 32768 after downloading from HF
ALLOCATOR_PATH=${ALLOCATOR_PATH:-"${PROJECT_ROOT}/models/allocator"}
TRAIN_FILE=${TRAIN_FILE:-"${PROJECT_ROOT}/data/train.parquet"}
TEST_FILE=${TEST_FILE:-"${PROJECT_ROOT}/data/test.parquet"}

# ==============================================================================
# 4. Scale Configuration Tags
# ==============================================================================
# Assembles SCALE_MULTI_MODAL_DATA (underscore-separated pieces). Passed to Hydra as
# algorithm.scale_multi_modal_data, allocator.scale_multi_modal_data, and
# actor_rollout_ref.scale_multi_modal_data when SCALE_ENABLE_SCALE=1 (section 6).
#
# Substring reference (full table): resadapt/utils/scale_multi_modal_tags.py
# Not in scale_parts below but used in code when appended manually: aw (frame-aware),
# ispred (predictor log-prob alignment with use_filter_sid), hadw (GRPO cost path).
#
# Base scale configurations
SCALE_BASE=${SCALE_BASE:-"${SCALE_MULTI_MODAL_DATA}"} # Base identifier for the scale config
SCALE_ENABLE_SEP=${SCALE_ENABLE_SEP:-"0"}             # Enable separate allocation node
SCALE_ENABLE_SCALE=${SCALE_ENABLE_SCALE:-"1"}         # Enable scaling mechanism
SCALE_ENABLE_FILTER=${SCALE_ENABLE_FILTER:-"1"}       # Enable filtering of repeat prompts
SCALE_ENABLE_CCEN=${SCALE_ENABLE_CCEN:-"1"}           # Enable concentration loss/coefficient
SCALE_ENABLE_SIM=${SCALE_ENABLE_SIM:-"1"}             # Enable similarity/contrastive loss
SCALE_ENABLE_NOTEST=${SCALE_ENABLE_NOTEST:-"0"}       # Disable validation/testing if 1

# Frozen model configurations (mutually exclusive)
SCALE_ENABLE_ACTOR_FROZEN=${SCALE_ENABLE_ACTOR_FROZEN:-"0"}       # Freeze entire actor model
SCALE_ENABLE_ALLOCATOR_FROZEN=${SCALE_ENABLE_ALLOCATOR_FROZEN:-"1"} # Freeze actor except allocator

# Cost / allocator advantage (algorithm.use_cost; script appends "_acc")
# capo (default): cost-aware mix — group z-score on acc, optional HADW, correct/wrong vs
#   normalized mean scale + gas tax; optional frame terms if frame_metrics exist (see advantage.py).
# Other SCALE_USE_COST: gdpo | mygdpo | piecewise_v1 | piecewise_v2 | saliency_share_v1 (…).
# piecewise_* defaults live in piecewise_adaptive_cost.py + advantage.py; optional overrides e.g. gas(0.03), noframeaux.
SCALE_ENABLE_COST=${SCALE_ENABLE_COST:-"1"}
SCALE_USE_COST=${SCALE_USE_COST:-"capo"}
SCALE_COST_ENCOURAGEMENT_COEF=${SCALE_COST_ENCOURAGEMENT_COEF:-"0.25"}
SCALE_COST_PENALTY_COEF=${SCALE_COST_PENALTY_COEF:-"0.05"}
SCALE_CENTERED_TERM=${SCALE_CENTERED_TERM:-"0.4"}

# Build scale configuration name
scale_parts=("${SCALE_BASE}")
[[ "${SCALE_ENABLE_SEP}" == "1" ]] && scale_parts+=("sep")
[[ "${SCALE_ENABLE_FILTER}" == "1" ]] && scale_parts+=("filter")
[[ "${SCALE_ENABLE_CCEN}" == "1" ]] && scale_parts+=("ccen")
[[ "${SCALE_ENABLE_SIM}" == "1" ]] && scale_parts+=("sim")
[[ "${SCALE_ENABLE_COST}" == "1" ]] && scale_parts+=("cost${SCALE_USE_COST}")
[[ "${SCALE_ENABLE_NOTEST}" == "1" ]] && scale_parts+=("notest")
[[ "${SCALE_ENABLE_ACTOR_FROZEN}" == "1" ]] && scale_parts+=("actor_frozen")
[[ "${SCALE_ENABLE_ALLOCATOR_FROZEN}" == "1" ]] && scale_parts+=("allocator_frozen")

scale_core_norm="$(IFS=_; echo "${scale_parts[*]}")"

if [[ "$NNODES" != 1 ]]; then
    SCALE_MULTI_MODAL_DATA="ray${NNODES}-${scale_core_norm}"
else
    SCALE_MULTI_MODAL_DATA="${scale_core_norm}"
fi

project_name='GeneralQA_Qwen_Verify'

# ==============================================================================
# 5. Training Hyperparameters
# ==============================================================================
extra_args=()

# Basic Training Settings
total_training_steps=500                    # Total number of PPO training steps
fsdp_dtype="bfloat16"                       # Data type for FSDP (bfloat16 recommended for Ampere+)
attn_implementation="flash_attention_2"     # Attention backend: flash_attention_2 or sdpa

# Data Subsampling (-1 means use all data)
train_max_samples=-1                        # Max training samples (-1 for all)
val_max_samples=-1                          # Max validation samples (-1 for all)

# Distributed Training Node Configuration
train_nodes=${NNODES}                       # Number of nodes for training
if [[ "${SCALE_ENABLE_SEP}" == "1" ]]; then
    extra_args+=( "allocator.enable_resource_pool=True" )
    extra_args+=( "allocator.nnodes=2" )
    train_nodes=$((NNODES - 2))
    if [[ $train_nodes -le 0 ]]; then
        train_nodes=1
    fi
fi

# Dataloader & Worker Settings
dataloader_num_workers=$((8 * NNODES))      # Workers for standard dataloader
filter_num_workers=$((16 * NNODES))         # Workers for data filtering
rollout_workers=$((8 * train_nodes))        # vLLM rollout workers per node

# Batch Size Configuration
train_prompt_bsz=$((128 * train_nodes))     # Global prompt batch size
train_prompt_mini_bsz=$((32 * train_nodes)) # Mini-batch size for PPO
val_batch_size=512                          # Validation batch size
gpu_memory_utilization=0.55                 # vLLM GPU memory utilization
max_factor=36                               # Factor for max token length calculation

# RLHF / PPO Algorithm Settings
adv_estimator="grpo"                        # Advantage estimator (e.g., grpo, gae)
use_kl_in_reward=False                      # Whether to add KL penalty directly to reward
kl_coef=0.0                                 # Coefficient for KL penalty in reward
use_kl_loss=False                           # Whether to use KL divergence loss
kl_loss_coef=0.0                            # Coefficient for KL loss

# PPO Clipping Parameters
clip_ratio_low=0.2                          # Lower bound for PPO clip
clip_ratio_high=0.28                        # Upper bound for PPO clip

# Sequence Length Settings
max_prompt_length=$((1024 * PROMPT_LEN))    # Maximum length for input prompts
max_response_length=$((1024 * RESP_LEN))    # Maximum length for generated responses
enable_overlong_buffer=True                 # Enable buffer for overlong responses
overlong_buffer_len=$((1024 * 4))           # Length of the overlong buffer
overlong_penalty_factor=1.0                 # Penalty factor for exceeding length

loss_agg_mode="token-mean"                  # Loss aggregation mode (token-mean or batch-mean)
filter_prompt_length=${max_prompt_length}   # Prompt length threshold for filtering

# Scaled Length Calculation
scaled_length=$((( $(echo "$max_prompt_length * $MAX_SCALE * $MAX_SCALE" | bc -l | xargs printf "%.0f") + 1023 ) / 1024 * 1024))

# Generation Parameters (vLLM)
temperature=1.0                             # Sampling temperature
top_p=1.0                                   # Top-p (nucleus) sampling
top_k=-1                                    # Top-k sampling (-1 for vLLM rollout, 0 for HF)
val_top_p=0.7                               # Validation top-p

# Model Parallelism & FSDP
sp_size=1                                   # Ulysses sequence parallel size
use_dynamic_bsz=False                       # Dynamic batch size for PPO actor
actor_ppo_max_token_len=$(( (scaled_length + max_response_length) > (1024 * max_factor) ? (1024 * max_factor) : (scaled_length + max_response_length) ))
infer_ppo_max_token_len=${actor_ppo_max_token_len} # Max token length for inference
offload=False                               # CPU offload for parameters/optimizer
gen_tp=1                                    # Tensor parallelism size for generation
fsdp_size=-1                                # FSDP group size (-1 for full node)

# Evaluation Settings
test_freq=10                                # Frequency of evaluation (in steps)
val_before_train=True                       # Run validation before training starts

# Multimedia & Batch Settings
NFRAMES=${NFRAMES:-"8"}                     # Number of frames for video inputs
ppo_micro_batch_size_per_gpu=1              # Micro batch size for PPO per GPU
min_scale=0.2                               # Minimum scale factor

# WandB Logging Configuration
WANDB_RUN_ID=${WANDB_RUN_ID:-"0"}
if [[ "$WANDB_RUN_ID" != "0" ]]; then
    extra_args+=( "+trainer.wandb_run_id=${WANDB_RUN_ID}" )
    extra_args+=( "+trainer.wandb_resume=allow" )
fi

extra_args+=( "data.max_frames=${NFRAMES}" )
extra_args+=( "+actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=16" )
extra_args+=( "+actor_rollout_ref.rollout.max_model_len=32000" )

export VLLM_MROPE_PATCH=True
ray_env_args+=( "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_MROPE_PATCH='${VLLM_MROPE_PATCH}'" )

# ==============================================================================
# 6. Scale Features Assembly
# ==============================================================================
# When SCALE_ENABLE_SCALE=1, extra_args push allocator + actor_rollout_ref overrides and
# sync scale_multi_modal_data across algorithm/allocator/rollout (see scale_multi_modal_tags.py).
MODEL_NAME=$(basename "${MODEL_PATH}")
# Extract critical part from model name (e.g., Qwen2.5-VL-7B-Instruct -> Qwen2.5-VL-7B, Llama-3-8B -> Llama-3-8B)
SHORT_MODEL_NAME=$(echo "$MODEL_NAME" | sed -E 's/-Instruct|-Chat|-Base//g')

exp_name="${SCALE_MULTI_MODAL_DATA}-${STRATEGY}-${SHORT_MODEL_NAME}-bsz${train_prompt_bsz}-mini${train_prompt_mini_bsz}-n${N_RESP_PER_PROMPT}-nf${NFRAMES}-l${PROMPT_LEN}-r${RESP_LEN}"

if [[ "${SCALE_ENABLE_SCALE}" == "1" ]]; then
    max_prompt_length=${scaled_length}
    use_text=True
    use_discrete_action=False
    exp_name="${SCALE_MULTI_MODAL_DATA}-${STRATEGY}-s${SCALE_N}-${SHORT_MODEL_NAME}-bsz${train_prompt_bsz}-mini${train_prompt_mini_bsz}-n${N_RESP_PER_PROMPT}-max${MAX_SCALE}-nf${NFRAMES}-l${PROMPT_LEN}-r${RESP_LEN}"
    
    extra_args+=( "+allocator.model.override_config.max_frames=${NFRAMES}" )
    extra_args+=( "algorithm.max_scale=${MAX_SCALE}" )
    extra_args+=( "algorithm.min_scale=${min_scale}" )
    extra_args+=( "algorithm.use_discrete_action=${use_discrete_action}" )
    # Keep algorithm.scale_multi_modal_data in sync with allocator/rollout so GRPO hadw/switch logic in ray_trainer sees the tag.
    extra_args+=( "algorithm.scale_multi_modal_data=${SCALE_MULTI_MODAL_DATA}" )

    extra_args+=( "allocator.enable=True" )
    extra_args+=( "allocator.scale_multi_modal_data=${SCALE_MULTI_MODAL_DATA}" )
    extra_args+=( "allocator.model.path=${ALLOCATOR_PATH}" )
    extra_args+=( "allocator.scale_n=${SCALE_N}" )
    extra_args+=( "+allocator.model.override_config.attn_implementation=flash_attention_2" )
    extra_args+=( "+allocator.model.override_config.min_scale=${min_scale}" )
    extra_args+=( "+allocator.model.override_config.max_scale=${MAX_SCALE}" )
    extra_args+=( "+allocator.model.override_config.use_text=${use_text}" )
    extra_args+=( "allocator.actor.loss_agg_mode=token-mean" )
    extra_args+=( "allocator.actor.optim.lr=${LR}" )
    extra_args+=( "allocator.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu}" )
    extra_args+=( "allocator.actor.use_dynamic_bsz=False" )
    extra_args+=( "allocator.return_mm_data=False" )
    extra_args+=( "allocator.scale.num_workers=$((8 * NNODES))" )

    extra_args+=( "actor_rollout_ref.scale_n=${SCALE_N}" )
    extra_args+=( "actor_rollout_ref.scale_multi_modal_data=${SCALE_MULTI_MODAL_DATA}" )

    if [[ "$LR" != "1e-5" ]]; then
        exp_name="${exp_name}_${LR}"
    fi
    
    if [[ "${SCALE_ENABLE_FILTER}" == "1" ]]; then
        extra_args+=( "algorithm.use_filter_sid=True" )
    fi

    if [[ "${SCALE_ENABLE_CCEN}" == "1" ]]; then
        extra_args+=( "allocator.scale.concentration_coef=5e-4" )
    fi
    extra_args+=( "+allocator.model.override_config.beta_param_scale=0.5" )

    if [[ "${SCALE_ENABLE_SIM}" == "1" ]]; then
        extra_args+=( "allocator.scale.contrastive_coef=1e-3" )
        extra_args+=( "allocator.scale.sim_scale_coef=1e-3" )
        extra_args+=( "+allocator.model.override_config.sim_scale_weight=0.2" )
        extra_args+=( "+allocator.model.override_config.contrastive_weight=0.2" )
    else
        extra_args+=( "+allocator.model.override_config.sim_scale_weight=0.0" )
        extra_args+=( "+allocator.model.override_config.contrastive_weight=0.0" )
    fi
    
    if [[ "${SCALE_ENABLE_COST}" == "1" ]]; then
        extra_args+=( "algorithm.use_cost=${SCALE_USE_COST}_acc" )
        if [[ -n "${SCALE_COST_ENCOURAGEMENT_COEF}" ]]; then extra_args+=( "algorithm.cost_encouragement_coef=${SCALE_COST_ENCOURAGEMENT_COEF}" ); fi
        if [[ -n "${SCALE_COST_PENALTY_COEF}" ]]; then extra_args+=( "algorithm.cost_penalty_coef=${SCALE_COST_PENALTY_COEF}" ); fi
        if [[ -n "${SCALE_CENTERED_TERM}" ]]; then extra_args+=( "algorithm.centered_term=${SCALE_CENTERED_TERM}" ); fi
    else
        extra_args+=( "algorithm.use_cost=null" )
    fi
fi

if [[ "${SCALE_ENABLE_NOTEST}" == "1" ]]; then
    test_freq=-1
    val_before_train=False
fi

# Apply string replacements to shorten the experiment name (similar to predictor/main.sh)
exp_name=$(echo "$exp_name" | sed \
    -e 's/tie_acc_cost_enc/tie_enc/g' \
    -e 's/ispred/is/g' \
    -e 's/flash/fla/g' \
    -e 's/gate/gt/g' \
    -e 's/filter/filt/g' \
    -e 's/nframes8_offline/n8_off/g' \
    -e 's/sta/s/g' \
    -e 's/newtie/new/g' \
    -e 's/scale/sc/g' \
    -e 's/frozen/fr/g' \
    -e 's/nframes/nf/g' \
    -e 's/video_mrope/vidm/g' \
    -e 's/fsdp2/dp2/g' \
    -e 's/ray/r/g' \
    -e 's/allocator/allo/g' \
    -e 's/cost//g' \
    -e 's/Qwen2.5-VL-/Q25-/g' \
    -e 's/Qwen2-VL-/Q2-/g' \
    -e 's/Qwen3-VL-/Q3-/g')

# ==============================================================================
# 7. Execution Command Construction
# ==============================================================================
CONFIG_PATH="${PROJECT_ROOT}/resadapt/config"
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}

# Build the main command array
# We do not use outer quotes for keys, but we quote the variable values to prevent word splitting
cmd=(
    python3
    -m
    resadapt.main_ppo
    --config-path="${CONFIG_PATH}"
    --config-name="pred_config.yaml"
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.prompt_key="prompt"
    data.max_prompt_length="${filter_prompt_length}"
    data.max_response_length="${max_response_length}"
    data.train_batch_size="${train_prompt_bsz}"
    data.return_raw_chat=True
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers="${filter_num_workers}"
    data.truncation="error"
    data.image_key="images"
    data.video_key="videos"
    data.val_batch_size="${val_batch_size}"
    data.dataloader_num_workers="${dataloader_num_workers}"
    data.train_max_samples="${train_max_samples}"
    data.val_max_samples="${val_max_samples}"
    algorithm.adv_estimator="${adv_estimator}"
    algorithm.use_kl_in_reward="${use_kl_in_reward}"
    algorithm.kl_ctrl.kl_coef="${kl_coef}"
    actor_rollout_ref.nccl_timeout=7200
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
    +actor_rollout_ref.model.override_config.attn_implementation="${attn_implementation}"
    actor_rollout_ref.actor.use_kl_loss="${use_kl_loss}"
    actor_rollout_ref.actor.kl_loss_coef="${kl_loss_coef}"
    actor_rollout_ref.actor.clip_ratio_low="${clip_ratio_low}"
    actor_rollout_ref.actor.clip_ratio_high="${clip_ratio_high}"
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.use_dynamic_bsz="${use_dynamic_bsz}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${actor_ppo_max_token_len}"
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=30
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.ppo_mini_batch_size="${train_prompt_mini_bsz}"
    actor_rollout_ref.actor.fsdp_config.param_offload="${offload}"
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${offload}"
    actor_rollout_ref.actor.fsdp_config.fsdp_size="${fsdp_size}"
    actor_rollout_ref.actor.fsdp_config.dtype="${fsdp_dtype}"
    actor_rollout_ref.actor.fsdp_config.strategy="${STRATEGY}"
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.grad_clip=1.0
    actor_rollout_ref.actor.loss_agg_mode="${loss_agg_mode}"
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${sp_size}"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${use_dynamic_bsz}"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${infer_ppo_max_token_len}"
    actor_rollout_ref.rollout.n="${N_RESP_PER_PROMPT}"
    actor_rollout_ref.rollout.name="vllm"
    actor_rollout_ref.rollout.mode="async"
    actor_rollout_ref.rollout.prompt_length="${max_prompt_length}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${gpu_memory_utilization}"
    actor_rollout_ref.rollout.tensor_model_parallel_size="${gen_tp}"
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length))
    actor_rollout_ref.rollout.temperature="${temperature}"
    actor_rollout_ref.rollout.top_p="${top_p}"
    actor_rollout_ref.rollout.top_k="${top_k}"
    actor_rollout_ref.rollout.val_kwargs.temperature="${temperature}"
    actor_rollout_ref.rollout.val_kwargs.top_p="${val_top_p}"
    actor_rollout_ref.rollout.val_kwargs.top_k="${top_k}"
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.agent.num_workers="${rollout_workers}"
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=8192
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${use_dynamic_bsz}"
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${infer_ppo_max_token_len}"
    actor_rollout_ref.ref.fsdp_config.param_offload="${offload}"
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="${sp_size}"
    reward_model.reward_manager="dapo"
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable="${enable_overlong_buffer}"
    +reward_model.reward_kwargs.overlong_buffer_cfg.len="${overlong_buffer_len}"
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor="${overlong_penalty_factor}"
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=True
    +reward_model.reward_kwargs.max_resp_len="${max_response_length}"
    custom_reward_function.path="${PROJECT_ROOT}/resadapt/reward_fn/reward_r1.py"
    custom_reward_function.name="compute_score"
    trainer.logger="['console','wandb']"
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    trainer.n_gpus_per_node=8
    trainer.nnodes="${train_nodes}"
    trainer.val_before_train="${val_before_train}"
    trainer.test_freq="${test_freq}"
    trainer.save_freq="${test_freq}"
    trainer.total_epochs=10
    trainer.total_training_steps="${total_training_steps}"
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.resume_mode="auto"
    trainer.log_val_generations="${test_freq}"
)

: "${DRY_RUN:=0}"
if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${cmd[@]}" "${extra_args[@]}"
    echo
    exit 0
fi

if [[ "$DEBUG" == "0" ]]; then
    log_dir="logs_vid"
    mkdir -p "${log_dir}"
    log_file="${log_dir}/${exp_name}.log"
    echo "Starting training. Logs will be saved to ${log_file}"
    "${cmd[@]}" "${extra_args[@]}" "${ray_env_args[@]}" > "${log_file}" 2>&1
else
    echo "Starting training in debug mode..."
    "${cmd[@]}" "${extra_args[@]}" "${ray_env_args[@]}"
fi
