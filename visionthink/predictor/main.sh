#!/usr/bin/env bash
set -xeuo pipefail

: "${CLEAR_SHM:=1}"
if [[ "${CLEAR_SHM}" == "1" ]]; then
    if [[ -d /dev/shm ]]; then
        rm -rf /dev/shm/*
    fi
fi
# rm -rf /tmp/torchinductor_*

unset VLLM_MROPE_PATCH
# export BYTED_RAY_POD_IP=127.0.0.1
export HF_HUB_OFFLINE=True HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com/
export PYTHONPATH="/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink:/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/visionthink:${PYTHONPATH:-}"
export PYTORCH_KERNEL_CACHE_PATH="/mnt/bn/jiangzhongtao/users/liaohuanxuan/.cache/torch/kernels"
export RAY_memory_usage_threshold=0.98
export TK_HOST=https://ml.byteintl.net

ray_env_args=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE='${HF_HUB_OFFLINE}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.HF_DATASETS_OFFLINE='${HF_DATASETS_OFFLINE}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_OFFLINE='${TRANSFORMERS_OFFLINE}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.HF_ENDPOINT='${HF_ENDPOINT}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='${PYTHONPATH}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_KERNEL_CACHE_PATH='${PYTORCH_KERNEL_CACHE_PATH}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.RAY_memory_usage_threshold='${RAY_memory_usage_threshold}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.TK_HOST='${TK_HOST}'"
)

# export CUDA_HOME=$CONDA_PREFIX
# export PATH=$CONDA_PREFIX/bin:$PATH
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
echo "${CONDA_PREFIX:-}"

model_size=${1:-"3B"}
scale_multi_modal_data=${2:-"scale_flash"}
prompt_len_arg=${3:-4}
resp_len_arg=${4:-4}
strategy=${5:-"fsdp2"}
max_scale=${6:-3.0}
n_resp_per_prompt=${7:-4}
scale_n=${8:-4}
lr=${9:-"1e-5"}
# nframes=${10:-"2"}
# load_type=${11:-"0"}
debug=${10:-"0"}

# Ray
IP=""
if hostname -I >/dev/null 2>&1; then
    IP=$(hostname -I | awk '{print $1}')
fi
if [[ -z "${IP}" ]] && command -v python3 >/dev/null 2>&1; then
    IP=$(python3 - <<'PY'
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    sock.connect(("8.8.8.8", 80))
    print(sock.getsockname()[0])
finally:
    sock.close()
PY
)
fi
IP=${IP:-127.0.0.1}
RAY_ADDRESS="http://${IP}:8888"
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}

if [[ "$NNODES" != 1 ]]; then
    scale_multi_modal_data="ray${NNODES}-${scale_multi_modal_data}"
fi

project_name='GeneralQA_Qwen_Verify'

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${WORKING_DIR}"}
# very important! please modify the max_position_embeddings in config.json to 32768 after downloading from huggingface
# MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-VL-${model_size}-Instruct"}
MODEL_PATH=${MODEL_PATH:-"/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/qwen2.5_vl-${model_size}"}
TRAIN_FILE=${TRAIN_FILE:-"/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/VisionThink-General-Train/train.parquet"}
TEST_FILE=${TEST_FILE:-"/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/VisionThink-General-Val/train.parquet"}

extra_args=()
total_training_steps=500
fsdp_dtype=bfloat16
attn_implementation=flash_attention_2  # flash_attention_2 sdpa

train_max_samples=-1
val_max_samples=-1

train_nodes=${NNODES}
if [[ "$scale_multi_modal_data" == *"sep"* ]]; then
    # extra_args+=( "predictor.dedicated_8gpu=True" )
    extra_args+=( "predictor.enable_resource_pool=True" )
    extra_args+=( "predictor.nnodes=2" )
    train_nodes=$((NNODES - 2))
    if [[ $train_nodes -le 0 ]]; then
        train_nodes=1
    fi
fi

dataloader_num_workers=$((8 * ${NNODES}))
filter_num_workers=$((16 * ${NNODES}))
rollout_workers=$((8 * ${train_nodes}))
train_prompt_bsz=$((128 * ${train_nodes}))
train_prompt_mini_bsz=$((32 * ${train_nodes}))
gpu_memory_utilization=0.55
val_batch_size=512

max_factor=36
# train_prompt_bsz=8
# n_resp_per_prompt=4
# train_prompt_mini_bsz=2

adv_estimator=grpo

# use_kl_in_reward=False
# kl_coef=0.001
# use_kl_loss=True
# kl_loss_coef=0.001
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * ${prompt_len_arg}))
max_response_length=$((1024 * ${resp_len_arg}))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

filter_prompt_length=${max_prompt_length}

# scaled_length=$((( $(echo "$max_prompt_length * $max_scale * $max_scale + 0.999999" | bc -l | xargs printf "%.0f") + 1023 ) / 1024 * 1024))
scaled_length=$((( $(echo "$max_prompt_length * $max_scale * $max_scale" | bc -l | xargs printf "%.0f") + 1023 ) / 1024 * 1024))

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$(( (scaled_length + max_response_length) > $((1024 * ${max_factor})) ? $((1024 * ${max_factor})) : (scaled_length + max_response_length) ))
infer_ppo_max_token_len=$(( (scaled_length + max_response_length) > $((1024 * ${max_factor})) ? $((1024 * ${max_factor})) : (scaled_length + max_response_length) ))
offload=False
gen_tp=1
fsdp_size=-1

test_freq=10
val_before_train=True

nframes=2
ppo_micro_batch_size_per_gpu=16
min_scale=0.2

WANDB_RUN_ID=${WANDB_RUN_ID:-"0"}
if [[ "$WANDB_RUN_ID" != "0" ]]; then
    extra_args+=( "+trainer.wandb_run_id=${WANDB_RUN_ID}" )
    extra_args+=( "+trainer.wandb_resume=allow" )
fi

if [[ "$scale_multi_modal_data" == *"nframes"* ]]; then
    val_before_train=False
    # min_scale=0.25
    project_name='VideoQA_Qwen_Verify'
    
    if [[ "$scale_multi_modal_data" =~ nframes([0-9]*\.?[0-9]+) ]]; then
        nframes="${BASH_REMATCH[1]}"
    fi

    if [ "$NNODES" -ge 8 ]; then
        train_prompt_bsz=$((${train_prompt_bsz} * 2))
        train_prompt_mini_bsz=$((${train_prompt_mini_bsz} * 2))

    elif [ "$NNODES" -ge 4 ]; then
        train_prompt_bsz=$((${train_prompt_bsz} * 4))
        train_prompt_mini_bsz=$((${train_prompt_mini_bsz} * 4))

    elif [ "$NNODES" -ge 2 ]; then
        train_prompt_bsz=$((${train_prompt_bsz} * 8))
        train_prompt_mini_bsz=$((${train_prompt_mini_bsz} * 8))
    fi

    extra_args+=( "data.max_frames=${nframes}" )

    if [ "$nframes" -ge 128 ]; then
        # 12845056 6422528 5880000 3211264 1605632 802816
        # extra_args+=( "data.max_pixels=6422528" )

        train_prompt_bsz=$((${train_prompt_bsz} / 16))
        train_prompt_mini_bsz=$((${train_prompt_mini_bsz} / 8))
        ppo_micro_batch_size_per_gpu=$((${ppo_micro_batch_size_per_gpu} / 16))

        # rollout_workers=$((rollout_workers / 4))
        # dataloader_num_workers=$((dataloader_num_workers / 2))
        # filter_num_workers=$((filter_num_workers / 2))

        use_dynamic_bsz=False
        extra_args+=( "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1" )
        extra_args+=( "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1" )
        rollout_workers=$((rollout_workers / 2))

        # rollout_workers=$((rollout_workers / 2))
        # dataloader_num_workers=$((dataloader_num_workers / 2))
        # filter_num_workers=$((filter_num_workers / 2))

    elif [ "$nframes" -ge 64 ]; then
        # extra_args+=( "data.max_pixels=3211264" )

        train_prompt_bsz=$((${train_prompt_bsz} / 16))
        train_prompt_mini_bsz=$((${train_prompt_mini_bsz} / 8))
        ppo_micro_batch_size_per_gpu=$((${ppo_micro_batch_size_per_gpu} / 16))

        use_dynamic_bsz=False
        extra_args+=( "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1" )
        extra_args+=( "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1" )
        rollout_workers=$((rollout_workers / 2))

    elif [ "$nframes" -ge 32 ]; then
        # extra_args+=( "data.max_pixels=1605632" )

        # train_prompt_bsz=$((${train_prompt_bsz} / 16))
        # train_prompt_mini_bsz=$((${train_prompt_mini_bsz} / 8))
        # ppo_micro_batch_size_per_gpu=$((${ppo_micro_batch_size_per_gpu} / 4))

        if [[ "$scale_multi_modal_data" == *"sep"* ]]; then
            train_prompt_bsz=$((${train_prompt_bsz} / 16))
            train_prompt_mini_bsz=$((${train_prompt_mini_bsz} / 16))
            ppo_micro_batch_size_per_gpu=$((${ppo_micro_batch_size_per_gpu} / 16))
        elif [[ "$scale_multi_modal_data" == *"v2"* ]]; then
            train_prompt_bsz=$((${train_prompt_bsz} / 16))
            train_prompt_mini_bsz=$((${train_prompt_mini_bsz} / 8))
            ppo_micro_batch_size_per_gpu=$((${ppo_micro_batch_size_per_gpu} / 16))
            # rollout_workers=$((rollout_workers / 2))
        else
            train_prompt_bsz=$((${train_prompt_bsz} / 16))
            train_prompt_mini_bsz=$((${train_prompt_mini_bsz} / 8))
            ppo_micro_batch_size_per_gpu=$((${ppo_micro_batch_size_per_gpu} / 16))
        fi
        # sp_size=2
        use_dynamic_bsz=False
        extra_args+=( "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1" )
        extra_args+=( "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1" )
        rollout_workers=$((rollout_workers / 2))
        
        # rollout_workers=$((rollout_workers / 4))
        # dataloader_num_workers=$((dataloader_num_workers / 2))
    fi

    if [[ "$scale_multi_modal_data" == *"sft"* ]]; then
        MODEL_PATH=/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/videor1
        # ppo_micro_batch_size_per_gpu=$((${ppo_micro_batch_size_per_gpu} / 2))
    fi

    if [[ "$scale_multi_modal_data" == *"pt"* ]]; then
        MODEL_PATH=/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify/r4-mix_base_nf64_video-dp2-7B-bsz128-mini64-n16-l8-r8/global_step_40
        # ppo_micro_batch_size_per_gpu=$((${ppo_micro_batch_size_per_gpu} / 2))
    fi

    if [[ "$scale_multi_modal_data" == *"offline"* ]]; then
        TRAIN_FILE=/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/TSPO-${nframes}frames/train.parquet
        TEST_FILE=/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/TSPO-${nframes}frames/val.parquet
    elif [[ "$scale_multi_modal_data" == *"mix"* ]]; then
        TRAIN_FILE=/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/videoautor1_mixed/train_lens.parquet
        TEST_FILE=/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/TSPO-10K/val.parquet
        total_training_steps=600
        if [ "$train_prompt_bsz" -le 32 ]; then
            total_training_steps=2500
        elif [ "$train_prompt_bsz" -le 64 ]; then
            total_training_steps=1500
        elif [ "$train_prompt_bsz" -le 128 ]; then
            total_training_steps=800
        fi
    elif [[ "$scale_multi_modal_data" == *"auto"* ]]; then
        TRAIN_FILE=/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/videoautor1/train.parquet
        TEST_FILE=/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/TSPO-10K/val.parquet
        total_training_steps=600
    elif [[ "$scale_multi_modal_data" == *"r1"* ]]; then
        TRAIN_FILE=/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/videor1/train.parquet
        TEST_FILE=/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/TSPO-10K/val.parquet
        total_training_steps=800
    else
        TRAIN_FILE=/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/TSPO-10K/train.parquet
        TEST_FILE=/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/TSPO-10K/val.parquet
    fi

    extra_args+=( "+actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=16" )
    extra_args+=( "+actor_rollout_ref.rollout.max_model_len=32000" )
    
    if [[ "$scale_multi_modal_data" == *"vid_list"* ]]; then
        extra_args+=( "data.video2list=True" )
        extra_args+=( "predictor.video2list=True" )

        ppo_micro_batch_size_per_gpu=$((${ppo_micro_batch_size_per_gpu} / 2))
        train_prompt_bsz=$((${train_prompt_bsz} / 2))
        train_prompt_mini_bsz=$((${train_prompt_mini_bsz} / 2))

    elif [[ "$scale_multi_modal_data" != *"video"* ]]; then
        extra_args+=( "data.video2image=True" )
        extra_args+=( "predictor.video2image=True" )

        ppo_micro_batch_size_per_gpu=$((${ppo_micro_batch_size_per_gpu} / 2))
        train_prompt_bsz=$((${train_prompt_bsz} / 2))
        train_prompt_mini_bsz=$((${train_prompt_mini_bsz} / 2))
    fi

    if [[ "$scale_multi_modal_data" == *"mrope"* ]]; then
        # extra_args+=" +actor_rollout_ref.model.override_config.architectures=MyQwen2_5_VLForConditionalGeneration"
        export VLLM_MROPE_PATCH=True
        ray_env_args+=( "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_MROPE_PATCH='${VLLM_MROPE_PATCH}'" )
    fi
    # gpu_memory_utilization=0.75
    # extra_args+=" data.custom_cls.path=/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/visionthink/predictor/rl_dataset.py"
    # extra_args+=" data.custom_cls.name=CustomRLHFDataset"
    # gen_tp=2
    # ppo_micro_batch_size_per_gpu=8
    # val_batch_size=256
    # train_prompt_bsz=$((${train_prompt_bsz} / 2))
    # train_prompt_mini_bsz=$((${train_prompt_mini_bsz} / 2))
fi

if [[ "$NNODES" == 1 ]]; then
    train_max_samples=20000
    total_training_steps=600
fi

rollout_workers=$(( (rollout_workers > 2 ? rollout_workers : 2) < 64 ? (rollout_workers > 2 ? rollout_workers : 2) : 64 ))
dataloader_num_workers=$(( (dataloader_num_workers > 8 ? dataloader_num_workers : 8) < 64 ? (dataloader_num_workers > 8 ? dataloader_num_workers : 8) : 64 ))
filter_num_workers=$(( (filter_num_workers > 16 ? filter_num_workers : 16) < 128 ? (filter_num_workers > 16 ? filter_num_workers : 16) : 128 ))

train_prompt_bsz=$(( (train_prompt_bsz > 8 ? train_prompt_bsz : 8) < 512 ? (train_prompt_bsz > 8 ? train_prompt_bsz : 8) : 512 ))
train_prompt_mini_bsz=$(( (train_prompt_mini_bsz > 4 ? train_prompt_mini_bsz : 4) < 128 ? (train_prompt_mini_bsz > 4 ? train_prompt_mini_bsz : 4) : 128 ))

if [[ "$debug" != "0" ]]; then
    echo "use debug"
    export ARNOLD_BYTEDRAY_start_param_all_ray_debugger_external=true
    export RAY_DEBUG=legacy
    # val_before_train=True

    train_max_samples=500
    # total_training_steps=20
    # train_max_samples=2048
    # val_max_samples=100

    train_prompt_bsz=8
    # n_resp_per_prompt=4
    train_prompt_mini_bsz=2
    # ppo_micro_batch_size_per_gpu=4

    project_name='debug'
    ray_env_args+=( "+ray_kwargs.ray_init.runtime_env.env_vars.DEBUG_MODE=debug" )
fi

# scale
exp_name="${scale_multi_modal_data}-${strategy}-${model_size}-bsz${train_prompt_bsz}-mini${train_prompt_mini_bsz}-n${n_resp_per_prompt}-len${prompt_len_arg}-resp${resp_len_arg}"

if [[ "$scale_multi_modal_data" == *"scale"* ]]; then
    max_prompt_length=${scaled_length}
    # scale_multi_modal_data=scale_flash
    # self_depth=2
    # cross_depth=2
    use_text=True
    use_discrete_action=False
    max_frames=${nframes}
    exp_name="${scale_multi_modal_data}-${strategy}-s${scale_n}-${model_size}-bsz${train_prompt_bsz}-mini${train_prompt_mini_bsz}-n${n_resp_per_prompt}-max${max_scale}-len${prompt_len_arg}-resp${resp_len_arg}"
    
    if [[ "$scale_multi_modal_data" == *"ent"* ]]; then
        extra_args+=( "predictor.actor.entropy_coeff=5e-3" )
    fi
    
    if [[ "$scale_multi_modal_data" == *"smol"* ]]; then
        if [[ "$scale_multi_modal_data" == *"head"* ]]; then
            PREDICTOR_PATH=/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictor_head
        elif [[ "$scale_multi_modal_data" == *"embed"* ]]; then
            PREDICTOR_PATH=/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictor_embed
        elif [[ "$scale_multi_modal_data" == *"v2"* ]]; then
            PREDICTOR_PATH=/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictor_smolv2_zero
        elif [[ "$scale_multi_modal_data" == *"v1"* ]]; then
            PREDICTOR_PATH=/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictor_smolv1
        elif [[ "$scale_multi_modal_data" == *"v3"* ]]; then
            PREDICTOR_PATH=/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictor_smolv3
        else
            PREDICTOR_PATH=/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictor_smol
        fi
        extra_args+=( "+predictor.model.override_config.max_frames=${max_frames}" )
    else
        if [[ "$scale_multi_modal_data" == *"v2bd"* ]]; then
            PREDICTOR_PATH=/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictorv2_beta_deep
        elif [[ "$scale_multi_modal_data" == *"v2b"* ]]; then
            PREDICTOR_PATH=/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictorv2_beta
        elif [[ "$scale_multi_modal_data" == *"v1b"* ]]; then
            PREDICTOR_PATH=/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictorv1_beta
        elif [[ "$scale_multi_modal_data" == *"v3bd0"* ]]; then
            PREDICTOR_PATH=/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictorv3_beta_deep_zero
        elif [[ "$scale_multi_modal_data" == *"v3b"* ]]; then
            PREDICTOR_PATH=/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictorv3
        elif [[ "$scale_multi_modal_data" == *"v2"* ]]; then
            PREDICTOR_PATH=/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictorv2
        elif [[ "$scale_multi_modal_data" == *"v1"* ]]; then
            PREDICTOR_PATH=/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictorv1
        elif [[ "$scale_multi_modal_data" == *"flash"* ]]; then
            PREDICTOR_PATH=/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictor_flash_new
        elif [[ "$scale_multi_modal_data" == *"gate"* ]]; then
            PREDICTOR_PATH=/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictor_gate
        elif [[ "$scale_multi_modal_data" == *"disc"* ]]; then
            PREDICTOR_PATH=/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictor_disc_new
            min_scale=0.25
            use_discrete_action=True
        else
            PREDICTOR_PATH=/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictor
        fi
        extra_args+=( "+predictor.model.override_config.max_frames=$((${max_frames} / 2))" )
    fi

    # if [[ "$model_size" == "7B" && $scaled_length -gt $((1024*24)) ]]; then
    #     gpu_memory_utilization=0.75
    #     # max_factor=36
    # fi

    if [[ "$scale_multi_modal_data" =~ sta[_\=\ ]*([0-9]+) ]]; then
        num=${BASH_REMATCH[1]}
        extra_args+=( "predictor.scale.scale_start_step=${num}" )

        PREDICTOR_PATH=/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/VideoQA_Qwen_Verify/ray2-scale_fla_is_filt_tie_enc0.15_cen0.5_nframes8_video_mrope-fsdp2-s16-7B-bsz128-mini32-n1-max2.0-len4-resp8_5e-6/global_step_110/pred
    fi

    extra_args+=( "algorithm.max_scale=${max_scale}" )
    extra_args+=( "algorithm.min_scale=${min_scale}" )
    extra_args+=( "algorithm.use_discrete_action=${use_discrete_action}" )

    extra_args+=( "predictor.enable=True" )
    extra_args+=( "predictor.scale_multi_modal_data=${scale_multi_modal_data}" )
    extra_args+=( "predictor.model.path=${PREDICTOR_PATH}" )
    extra_args+=( "predictor.scale_n=${scale_n}" )
    extra_args+=( "+predictor.model.override_config.attn_implementation=flash_attention_2" )
    extra_args+=( "+predictor.model.override_config.min_scale=${min_scale}" )
    extra_args+=( "+predictor.model.override_config.max_scale=${max_scale}" )
    # extra_args+=( "+predictor.model.override_config.self_depth=${self_depth}" )
    # extra_args+=( "+predictor.model.override_config.cross_depth=${cross_depth}" )
    extra_args+=( "+predictor.model.override_config.use_text=${use_text}" )
    extra_args+=( "predictor.actor.loss_agg_mode=token-mean" )
    extra_args+=( "predictor.actor.optim.lr=${lr}" )
    extra_args+=( "predictor.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu}" )
    extra_args+=( "predictor.actor.use_dynamic_bsz=False" )
    extra_args+=( "predictor.return_mm_data=False" )
    extra_args+=( "predictor.scale.num_workers=$((8 * ${NNODES}))" )

    extra_args+=( "actor_rollout_ref.scale_n=${scale_n}" )
    extra_args+=( "actor_rollout_ref.scale_multi_modal_data=${scale_multi_modal_data}" )
    # extra_args+=" actor_rollout_ref.model.scale_multi_modal_data=${scale_multi_modal_data}"

    if [[ "$lr" != "1e-5" ]]; then
        exp_name="${exp_name}_${lr}"
    fi
    
    if [[ "$scale_multi_modal_data" == *"filter"* ]]; then
        extra_args+=( "algorithm.use_filter_sid=True" )
    fi

    if [[ "$scale_multi_modal_data" == *"bn"* ]]; then
        extra_args+=( "algorithm.batch_norm_adv=True" )
    fi

    if [[ "$scale_multi_modal_data" == *"ccen"* ]]; then
        extra_args+=( "predictor.scale.concentration_coef=5e-4" )
    fi
    extra_args+=( "+predictor.model.override_config.beta_param_scale=0.5" )

    if [[ "$scale_multi_modal_data" == *"sim"* ]]; then
        extra_args+=( "predictor.scale.contrastive_coef=1e-3" )
        extra_args+=( "predictor.scale.sim_scale_coef=1e-3" )

        extra_args+=( "+predictor.model.override_config.sim_scale_weight=0.2" )
        extra_args+=( "+predictor.model.override_config.contrastive_weight=0.2" )
        
    else
        extra_args+=( "+predictor.model.override_config.sim_scale_weight=0.0" )
        extra_args+=( "+predictor.model.override_config.contrastive_weight=0.0" )
    fi
    
    if [[ "$scale_multi_modal_data" == *"cost"* ]]; then
        use_cost="penalty"
        if [[ "$scale_multi_modal_data" == *"mul_cost"* ]]; then
            use_cost="multiply"
        elif [[ "$scale_multi_modal_data" == *"abs_cost"* ]]; then
            use_cost="absolute"
        elif [[ "$scale_multi_modal_data" == *"awid"* ]]; then
            use_cost="frame_ideal_wu1.0_wr1.0_wrel1.0_winf0.5_lam0.45_eta0.08_smin0.25_smax1.0_delta0.12_correct0.35_norm"
        elif [[ "$scale_multi_modal_data" == *"awrk"* ]]; then
            # use_cost="frame_rank_wu1.2_wr0.9_wrel1.0_winf0.4_lam0.25_rho0.08_eta0.02_smin0.25_smax1.0_delta0.08_margin0.12_krank6_correct0.35"
            use_cost="frame_rank_wu1.0_wr1.0_wrel1.0_winf0.5_lam0.5_rho0.3_eta0.1_smin0.25_smax1.0_delta0.1_margin0.1_krank4_correct0.35_wrongeta0.3_wrongexp0.2"
        elif [[ "$scale_multi_modal_data" == *"awid"* ]]; then
            use_cost="frame_ideal_wu1.1_wr0.8_wrel1.2_winf0.5_lam0.55_eta0.03_smin0.15_smax0.95_delta0.06_correct0.35"
        elif [[ "$scale_multi_modal_data" == *"awne"* ]]; then
            use_cost="frame_new_alpha0.5_beta0.3_gamma0.2_gas0.1_budget0.8"
        elif [[ "$scale_multi_modal_data" == *"awa"* ]]; then
            use_cost="frameaware_alpha1.0_beta0.5_gamma0.5_gas0.05"
        elif [[ "$scale_multi_modal_data" == *"newtie"* ]]; then
            use_cost="newtie"
            # - newtie_pull0.2_noclamp 
            # - newtie_pull0.3_noclamp
            # - newtie_pull0.25_noclamp_gas0.1
            # - newtie_acc_tau0.15_alpha0.5_accpow0.8_accfloor0.02_pull0.2_gas0.08_scalecap0.5_noclamp
            # - newtie_acc_tau0.12_alpha0.4_accpow1.0_accfloor0.01_pull0.3_gas0.1_scalecap0.4_noclamp
            if [[ "$scale_multi_modal_data" == *"asym_m"* ]]; then
                use_cost+="_asym_tau0.15_alpha0.4_rightscale1.0_wrongscale0.6_wrongpow1.1_accpow0.6_accfloor0.05_pull0.2_gas0.1_scalecap0.6_noclamp"

            elif [[ "$scale_multi_modal_data" == *"asym_h"* ]]; then
                use_cost+="_asym_tau0.15_alpha0.1_rightscale1.1_wrongscale0.8_wrongpow1.5_accpow0.7_accfloor0.07_pull0.25_gas0.15_scalecap0.6_noclamp"

            elif [[ "$scale_multi_modal_data" == *"asym_s"* ]]; then
                use_cost+="_asym_tau0.2_alpha0.2_rightscale0.9_wrongscale1.0_wrongpow1.3_accpow1.0_accfloor0.10_pull0.25_gas0.18_scalecap0.6_noclamp"

            elif [[ "$scale_multi_modal_data" == *"asym"* ]]; then
                use_cost+="_asym_alpha0.5_tau0.15_gas0.05_accpow0.5_accfloor0.05_wrongscale1.2_rightscale0.9_scalecap0.25_noclamp"

                # use_cost+="_asym_rightscale0.5_wrongscale1.0_wrongpow1.2_scalecap0.5"
                # newtie_asym_rightscale0.6_wrongscale1.0_wrongpow1.2_pull0.2_gas0.06_scalecap0.6_noclamp
                # newtie_asym_rightscale0.5_wrongscale1.0_wrongpow1.3_pull0.25_gas0.08_scalecap0.5_noclamp
                # newtie_asym_tau0.2_alpha0.6_rightscale0.6_wrongscale1.0_wrongpow1.1_accpow0.6_accfloor0.05_pull0.1_gas0.05_scalecap0.7_noclamp
            fi
        elif [[ "$scale_multi_modal_data" == *"tie"* ]]; then
            use_cost="tie"
        elif [[ "$scale_multi_modal_data" == *"mygdpo"* ]]; then
            use_cost="mygdpo"
        elif [[ "$scale_multi_modal_data" == *"gdpo"* ]]; then
            use_cost="gdpo"
        fi

        if [[ "$scale_multi_modal_data" == *"swi"* ]]; then
            use_cost+="_switch"
        fi

        if [[ "$scale_multi_modal_data" == *"hadw"* ]]; then
            # use_cost+="_hadw"
            # extra_args+=( "algorithm.use_hadw_in_grpo=True" )
            extra_args+=( "algorithm.scale_multi_modal_data=${scale_multi_modal_data}" )
        fi

        if [[ "$scale_multi_modal_data" == *"acc"* ]]; then
            use_cost+="_acc"
        fi

        if [[ "$scale_multi_modal_data" == *"norm"* ]]; then
            use_cost+="_norm"
        fi

        extra_args+=( "algorithm.use_cost=${use_cost}" )
        
        if [[ "$scale_multi_modal_data" =~ cost([0-9]*\.?[0-9]+) ]]; then
            num="${BASH_REMATCH[1]}"
            extra_args+=( "algorithm.cost_penalty_coef=${num}" )
        fi

        if [[ "$scale_multi_modal_data" =~ enc([0-9]*\.?[0-9]+) ]]; then
            num="${BASH_REMATCH[1]}"
            extra_args+=( "algorithm.cost_encouragement_coef=${num}" )
        fi

        if [[ "$scale_multi_modal_data" =~ cen([0-9]*\.?[0-9]+) ]]; then
            num="${BASH_REMATCH[1]}"
            extra_args+=( "algorithm.centered_term=${num}" )
        fi
    fi
fi

if [[ "$scale_multi_modal_data" == *"notest"* ]]; then
    test_freq=-1
    val_before_train=False
fi

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
    -e 's/len/l/g' \
    -e 's/resp/r/g' \
    -e 's/fsdp2/dp2/g' \
    -e 's/ray/r/g')

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/visionthink/config"
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}

# ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
#     --address "${RAY_ADDRESS}" \
#     --working-dir "${WORKING_DIR}" \
    # -- python3 -m verl.visionthink.main_ppo \
        # +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
cmd=(
    python3 -m visionthink.main_ppo
    --config-path "${CONFIG_PATH}"
    --config-name pred_config.yaml
    "data.train_files=${TRAIN_FILE}"
    "data.val_files=${TEST_FILE}"
    data.prompt_key=prompt
    "data.max_prompt_length=${filter_prompt_length}"
    "data.max_response_length=${max_response_length}"
    "data.train_batch_size=${train_prompt_bsz}"
    data.return_raw_chat=True
    data.filter_overlong_prompts=True
    "data.filter_overlong_prompts_workers=${filter_num_workers}"
    data.truncation=error
    data.image_key=images
    data.video_key=videos
    "data.val_batch_size=${val_batch_size}"
    "data.dataloader_num_workers=${dataloader_num_workers}"
    "data.train_max_samples=${train_max_samples}"
    "data.val_max_samples=${val_max_samples}"
    "algorithm.adv_estimator=${adv_estimator}"
    "algorithm.use_kl_in_reward=${use_kl_in_reward}"
    "algorithm.kl_ctrl.kl_coef=${kl_coef}"
    actor_rollout_ref.nccl_timeout=7200
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
    "+actor_rollout_ref.model.override_config.attn_implementation=${attn_implementation}"
    "actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}"
    "actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}"
    "actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}"
    "actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}"
    actor_rollout_ref.actor.clip_ratio_c=10.0
    "actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}"
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=30
    actor_rollout_ref.actor.optim.weight_decay=0.1
    "actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${offload}"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload}"
    "actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size}"
    "actor_rollout_ref.actor.fsdp_config.dtype=${fsdp_dtype}"
    "actor_rollout_ref.actor.fsdp_config.strategy=${strategy}"
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.grad_clip=1.0
    "actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}"
    "actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size}"
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}"
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}"
    "actor_rollout_ref.rollout.n=${n_resp_per_prompt}"
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    "actor_rollout_ref.rollout.prompt_length=${max_prompt_length}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}"
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    "actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length))"
    "actor_rollout_ref.rollout.temperature=${temperature}"
    "actor_rollout_ref.rollout.top_p=${top_p}"
    "actor_rollout_ref.rollout.top_k=${top_k}"
    "actor_rollout_ref.rollout.val_kwargs.temperature=${temperature}"
    "actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}"
    "actor_rollout_ref.rollout.val_kwargs.top_k=${top_k}"
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    "actor_rollout_ref.rollout.agent.num_workers=${rollout_workers}"
    "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=8192"
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}"
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}"
    "actor_rollout_ref.ref.fsdp_config.param_offload=${offload}"
    "actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size}"
    reward_model.reward_manager=dapo
    "+reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer}"
    "+reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len}"
    "+reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor}"
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=True
    "+reward_model.reward_kwargs.max_resp_len=${max_response_length}"
    custom_reward_function.path=/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/visionthink/reward_fn/reward_r1.py
    custom_reward_function.name=compute_score
    'trainer.logger=["console","wandb"]'
    "trainer.project_name=${project_name}"
    "trainer.experiment_name=${exp_name}"
    trainer.n_gpus_per_node=8
    "trainer.nnodes=${train_nodes}"
    "trainer.val_before_train=${val_before_train}"
    "trainer.test_freq=${test_freq}"
    "trainer.save_freq=${test_freq}"
    trainer.total_epochs=10
    "trainer.total_training_steps=${total_training_steps}"
    "trainer.default_local_dir=${CKPTS_DIR}"
    trainer.resume_mode=auto
    "trainer.log_val_generations=${test_freq}"
)

: "${DRY_RUN:=0}"
if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${cmd[@]}" "${extra_args[@]}"
    echo
    exit 0
fi

if [[ "$debug" == "0" ]]; then
    log_dir="logs_vid"
    mkdir -p "${log_dir}"
    log_file="${log_dir}/${exp_name}.log"
    "${cmd[@]}" "${extra_args[@]}" "${ray_env_args[@]}" > "${log_file}" 2>&1
else
    "${cmd[@]}" "${extra_args[@]}" "${ray_env_args[@]}"
fi
