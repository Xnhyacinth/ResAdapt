#!/usr/bin/env bash
set -xeuo pipefail

rm -rf /dev/shm/*
# rm -rf /tmp/torchinductor_*

unset NFRAMES VLLM_MROPE_PATCH
# export BYTED_RAY_POD_IP=127.0.0.1
export HF_HUB_OFFLINE=True HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com/
export PYTHONPATH="/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink:$PYTHONPATH"
export PYTHONPATH="/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/visionthink:$PYTHONPATH"
export PYTORCH_KERNEL_CACHE_PATH="/mnt/bn/jiangzhongtao/users/liaohuanxuan/.cache/torch/kernels"
export RAY_memory_usage_threshold=0.98

# unset WANDB_RESUME
WANDB_RUN_ID=${WANDB_RUN_ID:-"0"}
if [[ "$WANDB_RUN_ID" != "0" ]]; then
    export WANDB_RUN_ID=${WANDB_RUN_ID}
    # export WANDB_RESUME=must
fi

# export CUDA_HOME=$CONDA_PREFIX
# export PATH=$CONDA_PREFIX/bin:$PATH
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
echo $CONDA_PREFIX

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
IP=$(hostname -I | awk '{print $1}')
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
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-VL-${model_size}-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/VisionThink-General-Train/train.parquet"}
TEST_FILE=${TEST_FILE:-"/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/VisionThink-General-Val/train.parquet"}

extra_args=""
total_training_steps=400
fsdp_dtype=bfloat16
attn_implementation=flash_attention_2  # flash_attention_2 sdpa

train_max_samples=-1
val_max_samples=-1

train_prompt_bsz=$((128 * ${NNODES}))
# n_resp_per_prompt=4
train_prompt_mini_bsz=$((32 * ${NNODES}))
gpu_memory_utilization=0.75
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

nframes=2
ppo_micro_batch_size_per_gpu=16
min_scale=0.2
if [[ "$scale_multi_modal_data" == *"nframes"* ]]; then
    if [[ "$NNODES" != 1 ]]; then
        train_prompt_bsz=$((${train_prompt_bsz} / 2))
        train_prompt_mini_bsz=$((${train_prompt_mini_bsz} / 2))
    fi

    min_scale=0.25
    project_name='VideoQA_Qwen_Verify'
    
    if [[ "$scale_multi_modal_data" =~ nframes([0-9]*\.?[0-9]+) ]]; then
        nframes="${BASH_REMATCH[1]}"
    fi
    export NFRAMES=${nframes}
    ppo_micro_batch_size_per_gpu=$((${ppo_micro_batch_size_per_gpu} / 2))
    train_prompt_bsz=$((${train_prompt_bsz} / 2))
    train_prompt_mini_bsz=$((${train_prompt_mini_bsz} / 2))

    if [[ "$scale_multi_modal_data" == *"offline"* ]]; then
        TRAIN_FILE=/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/TSPO-${nframes}frames/train.parquet
        TEST_FILE=/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/TSPO-${nframes}frames/val.parquet
    else
        TRAIN_FILE=/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/TSPO-10K/train.parquet
        TEST_FILE=/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/TSPO-10K/val.parquet
    fi

    extra_args+=" +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=16"
    if [[ "$scale_multi_modal_data" == *"vid_list"* ]]; then
        extra_args+=" data.video2list=True"
        extra_args+=" predictor.video2list=True"

        ppo_micro_batch_size_per_gpu=$((${ppo_micro_batch_size_per_gpu} / 2))
        train_prompt_bsz=$((${train_prompt_bsz} / 2))
        train_prompt_mini_bsz=$((${train_prompt_mini_bsz} / 2))

    elif [[ "$scale_multi_modal_data" != *"video"* ]]; then
        extra_args+=" data.video2image=True"
        extra_args+=" predictor.video2image=True"

        ppo_micro_batch_size_per_gpu=$((${ppo_micro_batch_size_per_gpu} / 2))
        train_prompt_bsz=$((${train_prompt_bsz} / 2))
        train_prompt_mini_bsz=$((${train_prompt_mini_bsz} / 2))
    fi

    if [[ "$scale_multi_modal_data" == *"mrope"* ]]; then
        # extra_args+=" +actor_rollout_ref.model.override_config.architectures=MyQwen2_5_VLForConditionalGeneration"
        export VLLM_MROPE_PATCH=True
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

if [[ "$debug" != "0" ]]; then
    echo "use debug"
    export ARNOLD_BYTEDRAY_start_param_all_ray_debugger_external=true
    export RAY_DEBUG=legacy

    total_training_steps=20
    train_max_samples=2048
    val_max_samples=10

    train_prompt_bsz=32
    # n_resp_per_prompt=4
    train_prompt_mini_bsz=8
    ppo_micro_batch_size_per_gpu=4

    project_name='debug'
fi

# scale
exp_name="${scale_multi_modal_data}-${strategy}-${model_size}-bsz${train_prompt_bsz}-mini${train_prompt_mini_bsz}-n${n_resp_per_prompt}-len${prompt_len_arg}-resp${resp_len_arg}"

if [[ "$scale_multi_modal_data" == *"scale"* ]]; then
    max_prompt_length=${scaled_length}
    # scale_multi_modal_data=scale_flash
    self_depth=2
    cross_depth=2
    use_text=True
    use_discrete_action=False
    max_frames=${nframes}
    exp_name="${scale_multi_modal_data}-${strategy}-s${scale_n}-${model_size}-bsz${train_prompt_bsz}-mini${train_prompt_mini_bsz}-n${n_resp_per_prompt}-max${max_scale}-len${prompt_len_arg}-resp${resp_len_arg}"
    if [[ "$scale_multi_modal_data" == *"flash"* ]]; then
        PREDICTOR_PATH=/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictor_flash_new
    elif [[ "$scale_multi_modal_data" == *"disc"* ]]; then
        PREDICTOR_PATH=/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictor_disc_new
        min_scale=0.25
        use_discrete_action=True
    else
        PREDICTOR_PATH=/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictor
    fi

    # if [[ "$model_size" == "7B" && $scaled_length -gt $((1024*24)) ]]; then
    #     gpu_memory_utilization=0.75
    #     # max_factor=36
    # fi

    extra_args+=" algorithm.max_scale=${max_scale}"
    extra_args+=" algorithm.min_scale=${min_scale}"
    extra_args+=" algorithm.use_discrete_action=${use_discrete_action}"

    extra_args+=" predictor.enable=True"
    extra_args+=" predictor.scale_multi_modal_data=${scale_multi_modal_data}"
    extra_args+=" predictor.model.path=${PREDICTOR_PATH}"
    extra_args+=" predictor.scale_n=${scale_n}"
    extra_args+=" +predictor.model.override_config.attn_implementation=flash_attention_2"
    extra_args+=" +predictor.model.override_config.min_scale=${min_scale}"
    extra_args+=" +predictor.model.override_config.max_scale=${max_scale}"
    extra_args+=" +predictor.model.override_config.self_depth=${self_depth}"
    extra_args+=" +predictor.model.override_config.cross_depth=${cross_depth}"
    extra_args+=" +predictor.model.override_config.use_text=${use_text}"
    extra_args+=" +predictor.model.override_config.max_frames=${max_frames}"
    extra_args+=" predictor.actor.loss_agg_mode=token-mean"
    extra_args+=" predictor.actor.optim.lr=${lr}"
    extra_args+=" predictor.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu}"
    extra_args+=" predictor.actor.use_dynamic_bsz=False"
    extra_args+=" predictor.return_mm_data=False"

    extra_args+=" actor_rollout_ref.scale_n=${scale_n}"
    extra_args+=" actor_rollout_ref.scale_multi_modal_data=${scale_multi_modal_data}"
    # extra_args+=" actor_rollout_ref.model.scale_multi_modal_data=${scale_multi_modal_data}"

    if [[ "$lr" != "1e-5" ]]; then
        exp_name="${exp_name}_${lr}"
    fi
    
    if [[ "$scale_multi_modal_data" == *"filter"* ]]; then
        extra_args+=" algorithm.use_filter_sid=True" 
    fi
    
    if [[ "$scale_multi_modal_data" == *"cost"* ]]; then
        if [[ "$scale_multi_modal_data" == *"mul_cost"* ]]; then
            extra_args+=" algorithm.use_cost=multiply"
        elif [[ "$scale_multi_modal_data" == *"abs_cost"* ]]; then
            extra_args+=" algorithm.use_cost=absolute"
        elif [[ "$scale_multi_modal_data" == *"tie"* ]]; then
            extra_args+=" algorithm.use_cost=tie"
        else
            extra_args+=" algorithm.use_cost=penalty"
        fi
        
        if [[ "$scale_multi_modal_data" =~ cost([0-9]*\.?[0-9]+) ]]; then
            num="${BASH_REMATCH[1]}"
            extra_args+=" algorithm.cost_penalty_coef=${num}"
        fi

        if [[ "$scale_multi_modal_data" =~ enc([0-9]*\.?[0-9]+) ]]; then
            num="${BASH_REMATCH[1]}"
            extra_args+=" algorithm.cost_encouragement_coef=${num}"
        fi

        if [[ "$scale_multi_modal_data" =~ cen([0-9]*\.?[0-9]+) ]]; then
            num="${BASH_REMATCH[1]}"
            extra_args+=" algorithm.centered_term=${num}"
        fi
    fi
fi

test_freq=10
val_before_train=True
if [[ "$scale_multi_modal_data" == *"notest"* ]]; then
    test_freq=-1
    val_before_train=False
fi

exp_name=${exp_name//scale_flash_ispred/scale_fla_is}
exp_name=${exp_name//filter_tie_acc_cost/filt_tie}
exp_name=${exp_name//nframes8_offline/n8_off}
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/visionthink/config"
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}

# ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
#     --address "${RAY_ADDRESS}" \
#     --working-dir "${WORKING_DIR}" \
    # -- python3 -m verl.visionthink.main_ppo \
        # +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
python3 -m visionthink.main_ppo \
    --config-path $CONFIG_PATH \
    --config-name pred_config.yaml \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.max_prompt_length=${filter_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=16 \
    data.truncation='error' \
    data.image_key=images \
    data.val_batch_size=${val_batch_size} \
    data.dataloader_num_workers=8 \
    data.train_max_samples=${train_max_samples} \
    data.val_max_samples=${val_max_samples} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.nccl_timeout=9600 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.attn_implementation=${attn_implementation} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.fsdp_config.dtype=${fsdp_dtype} \
    actor_rollout_ref.actor.fsdp_config.strategy=${strategy} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.agent.num_workers=$((8 * ${NNODES})) \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=True \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    custom_reward_function.path=visionthink/reward_fn/reward.py \
    custom_reward_function.name=compute_score \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=10 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 ${extra_args}

