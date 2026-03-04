export HF_HUB_OFFLINE=True
export PYTHONPATH=/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink:$PYTHONPATH
export OPENAI_API_URL="https://gpt-i18n.byteintl.net/gpt/openapi/online/v2/crawl/openai/deployments/gpt_openapi"
export OPENAI_API_KEY="dxMlgIJpXgkdou8z77OKt5rg4BQjwgJZ_GPT_AK"
export MODEL_VERSION="gpt-5-2025-08-07"

# export PYTORCH_ALLOC_CONF="max_split_size_mb:64,garbage_collection_threshold:0.8"

unset PREDICTOR_PATH ENABLE_BASELINE_SCALE BASELINE_SCALE_FACTOR USE_DEBUG MICRO_BATCH WORKERS max_inflight_per_gpu
unset CONVERT2IMAGES REMOVEPAD VLLM_MROPE_PATCH VIDEO2LIST VIDEO2IMAGE ADD_SYS
cp -r /mnt/bn/jiangzhongtao/users/liaohuanxuan/longvu/lmms-eval/lmms_eval /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/lmms-eval

model=${1:-"liuhaotian/llava-v1.5-7b"}
conv_template=${2:-"0"}
# model_type=${3:-"llava"}
task_type=${3:-"images"}
max_num_frames=${4:-"32"}
sa_pattern=${5:-"0"}
sa_ratio=${6:-"0.3"}
subtitles=${7:-"0"}
idx=${8:-"0"}
log=${9:-"logs_eval"}
method=${10:-"0"}
debug=${11:-"0"}

image_min_tokens=128
image_max_tokens=16384
video_min_tokens=16
video_max_tokens=768
video_total_tokens=16384

image_min_pixels=$((image_min_tokens * 28 * 28))
image_max_pixels=$((image_max_tokens * 28 * 28))
video_min_pixels=$((video_min_tokens * 28 * 28))
video_max_pixels=$((video_max_tokens * 28 * 28))
video_total_pixels=$((video_total_tokens * 28 * 28))

extra_name="base"
NUM_GPUS=8
BATCH_SIZE=1
model_type=llava
model_args="pretrained=${model},device_map=auto,attn_implementation=flash_attention_2"

output_path="${log}/eval_${conv_template}"
log_file="${log}/eval_${conv_template}"

step=$(echo "$model" | grep -o 'global_step_[0-9]\+')
# Validate the extraction result
if [ -n "$step" ]; then
    echo "Extracted step: $step"  # Output: Extracted step: global_step_310
    # output_path="${output_path}/${step}"
    extra_name="${step}_base"
# else
#     echo "No 'global_step_<number>' in the specified format found"
fi

to_boxed_tasks() {
    local input="$1"
    local IFS=','
    read -ra items <<< "$input"
    local out=()
    for item in "${items[@]}"; do
        case "$item" in
            activitynet_tvg) out+=("activitynettvg_boxed") ;;
            *) out+=("${item}_boxed") ;;
        esac
    done
    IFS=','
    echo "${out[*]}"
}

if [[ "$task_type" == *"video_inc"* ]]; then
    tasks="videomme_inc,longvideobench_val_v_inc"
elif [[ "$task_type" == *"video"* ]]; then
    tasks="videomme,longvideobench_val_v,video_mmmu,lvbench,mmvu_val"
    # tasks="mmvu_val"
    if [[ "$task_type" == *"all"* ]]; then
        # tasks="videomme,longvideobench_val_v,mlvu_dev,mlvu_test,egoschema_subset,lvbench,video_mmmu,mmvu_val,activitynettvg,charades,nextgqa" #mvbench
        tasks="videomme,longvideobench_val_v,video_mmmu,lvbench,mmvu_val,mlvu_dev,activitynettvg,charades,nextgqa,youcook2_val" #mvbench
    elif [[ "$task_type" == *"add"* ]]; then
        # tasks="mlvu_dev,mlvu_test,egoschema_subset,activitynettvg,charades,nextgqa" #mvbench
        tasks="mlvu_dev,activitynettvg,charades,nextgqa,youcook2_val" #mvbench
    fi
else
    # tasks="mmmu_val,chartqa,docvqa_val,ai2d,gqa,realworldqa,textvqa_val,mathvista_testmini,mathvision_testmini,ocrbench"
    tasks="mmmu_val,chartqa,mathvista_testmini,mathvision_testmini,ocrbench,textvqa_val,ai2d"
    # tasks="mmmu_val,chartqa,mmbench_en_dev,pope,mme" # mmvet mathverse_testmini
fi

if [[ "$method" == *"sys"* ]]; then
    export ADD_SYS=True
    if [[ "$task_type" == *"video"* && "$task_type" != *"video_inc"* ]]; then
        tasks=$(to_boxed_tasks "$tasks")
    elif [[ "$task_type" == *"image"* ]]; then
        tasks=$(to_boxed_tasks "$tasks")
    fi
fi

if [[ "$task_type" == *"video"* ]]; then
    model_args="${model_args},max_frames=${max_num_frames}"
    extra_name="${extra_name}_frames${max_num_frames}"
fi

TENSOR_PARALLEL_SIZE=1  # Number of GPUs for tensor parallelism
DATA_PARALLEL_SIZE=8     # Number of GPUs for data parallelism
extra_name="${extra_name}_${task_type}"

if [[ "$method" != "0" ]]; then
    extra_name="${extra_name}_${method}"
fi

if [[ "$model" == *"scale"* ]]; then
    TENSOR_PARALLEL_SIZE=1
    DATA_PARALLEL_SIZE=8 
    # tasks="mmmu_val_scale"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
if [ "$debug" == "debug" ]; then
    export USE_DEBUG=1
    echo "Using debug mode"
    NUM_GPUS=1
    export CUDA_VISIBLE_DEVICES=0 
    tasks="${tasks%%,*}"

    TENSOR_PARALLEL_SIZE=1  
    DATA_PARALLEL_SIZE=1    
fi

if [ "$conv_template" == "vicuna_v1" ]; then
    model_name="llava_llama"
elif [ "$conv_template" == "qwen_2" ]; then
    model_name="llava_qwen2"
elif [ "$conv_template" == "qwen_2_5" ]; then
    model_name="llava_qwen2_5"
elif [ "$conv_template" == "qwen_3" ]; then
    model_name="llava_qwen3"
elif [ "$conv_template" == "qwen2_5_vl" ]; then
    model_type="qwen2_5_vl"
elif [ "$conv_template" == "qwen2_5_vl_autothink" ]; then
    model_type="qwen2_5_vl_autothink"
    model_args="${model_args},inference_mode=auto,early_exit_thresh=0.97,video_min_pixels=$video_min_pixels,video_max_pixels=$video_max_pixels,video_total_pixels=$video_total_pixels,max_frames=$max_num_frames,image_min_pixels=$image_min_pixels,image_max_pixels=$image_max_pixels"
elif [ "$conv_template" == "qwen2_vl" ]; then
    model_type="qwen2_vl"
elif [ "$conv_template" == "qwen3_vl" ]; then
    model_type="qwen3_vl"
elif [[ "$conv_template" == *"vllm"* ]]; then
    export TORCH_NCCL_BLOCKING_WAIT=1
    export NCCL_TIMEOUT=18000000
    model_type=${conv_template}
    BATCH_SIZE=32
    GPU_MEMORY_UTILIZATION=0.85
    base_model="${model}"

    model_type="vllm_generate_custom"

    if [[ "$model" == *"3B"* ]]; then
        BATCH_SIZE=64
    fi

    if [[ "$model" == *"sc"* ]] && [[ "$method" != *"nopred"* ]]; then
        export PREDICTOR_PATH="${model}pred"
        export PREDICTOR_NUM_GPUS=${NUM_GPUS}
        echo "Using predictor path: $PREDICTOR_PATH"
    fi

    if [[ "$model" == *"sc"* ]] && [[ "$method" == *"base"* ]]; then
        if [[ "$method" == *"qwen3vl"* ]]; then
            num=$(echo "$method" | grep -oP '(?<=-)\d+(?=B-)')
            if [ -n "$num" ]; then
                echo "base size $num"
            else
                num=8
            fi
            model="Qwen/Qwen3-VL-${num}B-Instruct"
        else
            num=$(echo "$model" | grep -oP '(?<=-)\d+(?=B-)')
            if [ -n "$num" ]; then
                echo "base size $num"
            else
                num=7
            fi
            model="Qwen/Qwen2.5-VL-${num}B-Instruct"
        fi
    fi

    if [[ "$method" == *"fix"* ]]; then
        num=$(echo "$method" | sed -E 's/.*fix([0-9]+\.?[0-9]*).*/\1/; t; s/.*/0.0/')
        if [ -n "$num" ]; then
            echo "fix $num"
        else
            num=1.5
        fi

        export ENABLE_BASELINE_SCALE=1
        export BASELINE_SCALE_FACTOR=${num}
    fi

    if [[ "$model" == *"Video-R1-7B"* ]]; then
        model=/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/videor1_7B
    fi

    model_args="model=${model},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},data_parallel_size=${DATA_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    
    if [[ "$method" == *"len"* ]]; then
        echo "Using visual length"
        export LOG_VISUAL_LEN=True
        model_args="${model_args},log_visual_len=true"
    fi

    if [ ! -n "$step" ]; then
        suffix="${base_model#*/}"
        # output_path="${output_path}/${suffix}"
        # log_file="${log_file}/${suffix}"
        extra_name="${suffix}_${extra_name}"
    fi

    if [[ "$method" == *"img"* ]]; then
        export CONVERT2IMAGES=1
        # model_args="${model_args},mm_processor_cache_gb=32"

    elif [[ "$method" == *"repad"* ]]; then    
        export REMOVEPAD=1

    elif [[ "$method" == *"vid_list"* ]]; then
        export VIDEO2LIST=True
    fi

    if [[ "$task_type" == *"video"* ]]; then
        if [[ ( "$model" =~ mrope|vidm && "$method" != *"nopred"* ) || "$method" =~ mrope ]]; then
            echo "Using mrope patch"
            export VLLM_MROPE_PATCH=True
        fi

        if [ "$max_num_frames" -ge 128 ]; then
            # frames=128
            export max_inflight_per_gpu=8
            export MICRO_BATCH=8
            export MAX_QUEUE_PER_GPU=8
            export WORKERS=8
            export SCALE_PREPROCESS_RETRIES=8
            BATCH_SIZE=4
        elif [ "$max_num_frames" -ge 64 ]; then
            # frames=64 
            export max_inflight_per_gpu=16
            export MICRO_BATCH=16
            export MAX_QUEUE_PER_GPU=16
            export WORKERS=32
            export SCALE_PREPROCESS_RETRIES=8
            BATCH_SIZE=32
        elif [ "$max_num_frames" -ge 32 ]; then
            # frames=32
            export max_inflight_per_gpu=32
            export MICRO_BATCH=32
            export MAX_QUEUE_PER_GPU=32
            export WORKERS=32
            export SCALE_PREPROCESS_RETRIES=8
            BATCH_SIZE=32
        fi
        if [[ "$model" == *"smol"* ]]; then
            export max_inflight_per_gpu=$((max_inflight_per_gpu * 8))
            export MICRO_BATCH=$((MICRO_BATCH * 8))
            export MAX_QUEUE_PER_GPU=$((MAX_QUEUE_PER_GPU * 8))
            export MICRO_BATCH_MS=100
        fi
        model_args="${model_args},max_frame_num=${max_num_frames}"
    fi
    export LMMS_EVAL_VIS_WORKERS=${LMMS_EVAL_VIS_WORKERS:-32}
    export LMMS_EVAL_VIS_USE_PROCESS=${LMMS_EVAL_VIS_USE_PROCESS:-1}

    model_args="${model_args},mm_processor_cache_gb=16,max_pixels=${video_max_pixels}" # ,enable_prefix_caching=False ,max_pixels=12845056
else
    echo "Unsupported conv_template: $conv_template"
    # exit 1
fi

# if [[ "$model" == *"rope"* ]]; then
#     echo "Using rope"
#     extra_name="${extra_name}_rope"
# fi

# if [[ "$model" == *"x1"* ]]; then
#     echo "Using xformers1"
#     extra_name="${extra_name}_x1"
# fi

# if [[ "$model" == *"dynamic_var"* ]]; then
#     echo "Using dynamic resolution with var loss"
#     extra_name="${extra_name}_dynamic_var"
# elif [[ "$model" == *"dynamic"* ]]; then
#     echo "Using dynamic resolution"
#     extra_name="${extra_name}_dynamic"
# elif [[ "$model" == *"custom"* ]]; then
#     echo "Using custom"
#     extra_name="${extra_name}_custom"
# fi

# if [[ "$model" =~ _max([0-9]+\.?[0-9]*)_ ]] || [[ "$model" =~ _max([0-9]+\.?[0-9]*)$ ]]; then
#     max_value="${BASH_REMATCH[1]}"
#     echo "Using max resolution with value: $max_value"
#     extra_name="${extra_name}_max${max_value}"
# elif [[ "$model" == *"_max_"* ]] || [[ "$model" == *"_max" ]]; then
#     echo "Using max resolution (no value specified)"
#     extra_name="${extra_name}_max"
# fi

# if [[ "$model" =~ _skip([0-9]+)_v([0-9]+) ]]; then
#     skip_number="${BASH_REMATCH[1]}"
#     skip_version="${BASH_REMATCH[2]}"
#     echo "Using skip with number: $skip_number and version: v$skip_version"
#     extra_name="${extra_name}_skip${skip_number}_v${skip_version}"
# elif [[ "$model" =~ _skip([0-9]+) ]]; then
#     skip_number="${BASH_REMATCH[1]}"
#     echo "Using skip with number: $skip_number"
#     extra_name="${extra_name}_skip${skip_number}"
# elif [[ "$model" =~ _skip_v([0-9]+) ]]; then
#     skip_version="${BASH_REMATCH[1]}"
#     echo "Using skip with version: v$skip_version"
#     extra_name="${extra_name}_skip_v${skip_version}"
# elif [[ "$model" == *"skip"* ]]; then
#     echo "Using skip (generic)"
#     extra_name="${extra_name}_skip"
# fi

# if [[ "$model" == *"mask"* ]]; then
#     echo "Using mask"
#     extra_name="${extra_name}_mask"
# fi

# if [[ "$model" =~ _lr([^_]+) ]]; then
#     lr_value="${BASH_REMATCH[1]}"
#     echo "Using learning rate: $lr_value"
#     extra_name="${extra_name}_lr${lr_value}"
# elif [[ "$model" == *"lr"* ]]; then
#     echo "Using lr (generic)"
#     extra_name="${extra_name}_lr"
# fi

# if [[ "$model" == *"dyrouter"* ]]; then
#     echo "Using dyrouter"
#     # extra_name="${extra_name}_dyrouter"
#     dyrouter_suffix="${model#*dyrouter}"
#     dyrouter_suffix="${dyrouter_suffix%%/*}"
#     extra_name="${extra_name}_dyrouter${dyrouter_suffix}"
# fi

# if [[ "$model" == *"patchmerger"* ]]; then
#     echo "Using patchmerger"
#     extra_name="${extra_name}_patchmerger"
# fi

# if [[ "$model" == *"full"* ]]; then
#     echo "Using full layers"
#     extra_name="${extra_name}_full"
# fi

if [[ "$model_type" == *"qwen2_5_vl"* ]]; then
    model_args="${model_args},interleave_visuals=False" # ,max_pixels=${video_max_pixels} ,max_pixels=12845056
elif [[ "$model_type" == *"qwen2_vl"* ]]; then
    model_args="${model_args},max_pixels=${video_max_pixels}" # ,max_pixels=2359296
elif [[ "$model_type" == *"qwen3_vl"* ]]; then
    model_args="${model_args},interleave_visuals=False,max_pixels=${video_max_pixels}"
elif [[ "$model_name" == *"llava_"* ]]; then
    model_args="${model_args},model_name=${model_name},conv_template=${conv_template}"
fi

if [ "$sa_pattern" != "0" ]; then
    model_type="${model_type}_custom"
    extra_name="${extra_name}_${sa_pattern}_${sa_ratio}"
    sa_prune_ratio=$(bc -l <<< "1 - $sa_ratio")
    model_args="${model_args},sa_prune_ratio=${sa_prune_ratio},sa_pattern=${sa_pattern},sa_start_layer_idx=${idx}"
elif [ "$conv_template" == "qwen2_vl" ]; then
    model_type="${model_type}_custom"
fi

if [[ "$sa_pattern" == *"flashvid"* && "$conv_template" != *"vllm"* ]]; then
    DO_SEGMENT=True
    MIN_SEGMENT_NUM=4
    COMPLEMENTARY_SEGMENT=True
    TOKEN_SELECTION_METHOD=attn_div
    ALPHA=0.70
    TEMPORAL_THRESHOLD=0.8
    EXPANSION=1.25
    PRUNING_LAYER=20
    LLM_RETENTION_RATIO=0.3

    if [[ "$model" == *"Qwen3"* || "$conv_template" == "qwen_3" || "$conv_template" == "qwen3_vl" ]]; then
        PRUNING_LAYER=28
        LLM_RETENTION_RATIO=0.1
    fi

    retention_ratio=${sa_ratio}
    if [ "$retention_ratio" == "0" ]; then
        if [ -n "$FLASHVID_RATIO" ]; then
            retention_ratio=${FLASHVID_RATIO}
        else
            retention_ratio=0.25
        fi
    fi
    model_args="${model_args},enable_flashvid=True,retention_ratio=${retention_ratio},do_segment=${DO_SEGMENT},min_segment_num=${MIN_SEGMENT_NUM},complementary_segment=${COMPLEMENTARY_SEGMENT},token_selection_method=${TOKEN_SELECTION_METHOD},alpha=${ALPHA},temporal_threshold=${TEMPORAL_THRESHOLD},expansion=${EXPANSION},pruning_layer=${PRUNING_LAYER},llm_retention_ratio=${LLM_RETENTION_RATIO}"

elif [[ "$sa_pattern" == *"quadtree"* ]]; then
    if [ "$sa_ratio" == "0.05" ]; then
        threshold=0.65
        sa_tree_temporal_thresh=0.49
        if [ "$idx" == "0" ]; then
            threshold=0.58
            sa_tree_temporal_thresh=0.42
        fi
        sa_var_thresh=0.4
        sa_tem_diff_thresh=30
    elif [ "$sa_ratio" == "0.1" ]; then
        threshold=0.68
        sa_tree_temporal_thresh=0.5

        if [ "$idx" == "0" ]; then
            threshold=0.62
            sa_tree_temporal_thresh=0.45
        fi
        sa_var_thresh=0.4
        sa_tem_diff_thresh=30
    elif [ "$sa_ratio" == "0.2" ]; then
        threshold=0.7
        sa_tree_temporal_thresh=0.6
        sa_var_thresh=0.3
        sa_tem_diff_thresh=25
    elif [ "$sa_ratio" == "0.3" ]; then
        threshold=0.78
        sa_tree_temporal_thresh=0.6
        sa_var_thresh=0.28
        sa_tem_diff_thresh=23
    elif [ "$sa_ratio" == "0.4" ]; then
        threshold=0.8
        sa_tree_temporal_thresh=0.6
        sa_var_thresh=0.25
        sa_tem_diff_thresh=20
    elif [ "$sa_ratio" == "0.5" ]; then
        threshold=0.85
        sa_tree_temporal_thresh=0.6
    fi
    if [[ "$sa_pattern" == *"new"* ]] && [[ "$sa_pattern" != *"sim"* ]]; then
        threshold=-1
    fi
    if [[ "$sa_pattern" == *"hash"* ]]; then
        threshold=0.9  
        sa_tree_temporal_thresh=0.75
        sa_var_thresh=0.25
        sa_tem_diff_thresh=25
        sttm_slow_ver=True
    fi
    model_args="${model_args},sa_tree_root_level=1,threshold=${threshold},sa_var_thresh=${sa_var_thresh},sa_tem_diff_thresh=${sa_tem_diff_thresh},sa_tree_temporal_thresh=${sa_tree_temporal_thresh},sttm_slow_ver=${sttm_slow_ver}"

elif [[ "$sa_pattern" == *"tome"* ]]; then
    model_args="${model_args},sa_tome_ver=video"

elif [[ "$sa_pattern" == *"dycoke-stage1"* ]]; then
    if [ "$sa_ratio" == "0.3" ]; then
        threshold=0.925
    elif [ "$sa_ratio" == "0.5" ]; then
        threshold=0.7
    elif [ "$sa_ratio" == "0.1" ]; then
        threshold=0.975
    fi
    model_args="${model_args},threshold=${threshold}"

elif [[ "$sa_pattern" == *"dpmm"* ]]; then
    if [ "$sa_ratio" == "0.1" ]; then
        sa_alpha=1e2
        chunk_size=4
        local_max_k=128
        likelihood_var=30
        prior_var=5
    elif [ "$sa_ratio" == "0.05" ]; then
        sa_alpha=1e-2
        likelihood_var=20
        prior_var=5
        chunk_size=4
        local_max_k=128
    elif [ "$sa_ratio" == "0.3" ]; then
        sa_alpha=1e1
        likelihood_var=20
        prior_var=5
        chunk_size=4
        local_max_k=128
    elif [ "$sa_ratio" == "0.5" ]; then
        sa_alpha=10.0
        sa_tau=0.1
        sa_local_sim_thresh=0.8
    fi
    model_args="${model_args},sa_alpha=${sa_alpha},chunk_size=${chunk_size},local_max_k=${local_max_k}"

elif [[ "$sa_pattern" == *"dpmm_infer"* ]]; then
    if [ "$sa_ratio" == "0.3" ]; then
        sa_alpha=30.0
        sa_tau=0.08
        sa_local_sim_thresh=0.9
    elif [ "$sa_ratio" == "0.5" ]; then
        sa_alpha=10.0
        sa_tau=0.1
        sa_local_sim_thresh=0.8
    fi
    model_args="${model_args},sa_alpha=${sa_alpha},sa_tau=${sa_tau},sa_local_sim_thresh=${sa_local_sim_thresh}"

elif [[ "$sa_pattern" == "router" ]]; then
    if [ "$sa_ratio" == "0.2" ]; then
        threshold=0.25
    elif [ "$sa_ratio" == "0.3" ]; then
        threshold=0.3
    elif [ "$sa_ratio" == "0.5" ]; then
        threshold=0.35
    fi
    model_args="${model_args},threshold=${threshold}"

elif [[ "$sa_pattern" == "router_seq" ]]; then
    if [ "$sa_ratio" == "0.3" ]; then
        threshold=0.37
    elif [ "$sa_ratio" == "0.1" ]; then
        threshold=0.35
    elif [ "$sa_ratio" == "0.5" ]; then
        threshold=0.35
    fi
    model_args="${model_args},threshold=${threshold}"

elif [ "$sa_pattern" == "win" ]; then
    if [ "$sa_ratio" == "0.3" ]; then
        threshold=0.25
        sa_tree_temporal_thresh=0.6
    elif [ "$sa_ratio" == "0.4" ]; then
        threshold=0.35 
        sa_tree_temporal_thresh=0.55
    elif [ "$sa_ratio" == "0.5" ]; then
        threshold=0.6 
        sa_tree_temporal_thresh=0.8
    fi
    model_args="${model_args},threshold=${threshold},sa_tree_temporal_thresh=${sa_tree_temporal_thresh},chunk_size=2"

elif [[ "$sa_pattern" == *"hnsw"* ]]; then
    if [ "$sa_ratio" == "0.33" ]; then
        threshold=2.5
        sa_tree_temporal_thresh=1.8
        chunk_size=2
        greedy_layers=0
    elif [ "$sa_ratio" == "0.15" ]; then
        threshold=3.2
        sa_tree_temporal_thresh=1.8
        greedy_layers=1
        chunk_size=4
    elif [ "$sa_ratio" == "0.4" ]; then
        threshold=2.5 
        sa_tree_temporal_thresh=1.8
        chunk_size=2
        greedy_layers=0
        opt_name=fast
    elif [ "$sa_ratio" == "0.41" ]; then
        threshold=2.0
        sa_tree_temporal_thresh=1.8
        chunk_size=4 
        greedy_layers=0
        opt_name=fast
    elif [ "$sa_ratio" == "0.3" ]; then
        threshold=0.7
        sa_tree_temporal_thresh=1.8
        chunk_size=8
        greedy_layers=0
        opt_name=fast
    elif [ "$sa_ratio" == "0.1" ]; then
        threshold=100.0
        sa_tree_temporal_thresh=1.8
        chunk_size=16
        greedy_layers=0
        opt_name=fast
    elif [ "$sa_ratio" == "0.43" ]; then
        a_tree_thresh=0.7
        sa_tree_temporal_thresh=1.8
        chunk_size=16
        greedy_layers=0
        opt_name=fast
    elif [ "$sa_ratio" == "0.5" ]; then
        threshold=2.5
        sa_tree_temporal_thresh=1.8
        chunk_size=2
        greedy_layers=0
    fi
    extra_name="${extra_name}_greedy${greedy_layers}_chunk${chunk_size}"
    model_args="${model_args},threshold=${threshold},sa_tree_temporal_thresh=${sa_tree_temporal_thresh},chunk_size=${chunk_size},greedy_layers=${greedy_layers}"
    if [ "$opt_name" ]; then
        extra_name="${extra_name}_${opt_name}"
        model_args="${model_args},opt_name=${opt_name}"
    fi

elif [[ "$sa_pattern" == *"framefusion"* ]]; then
    :

elif [[ "$sa_pattern" == *"fastv"* ]]; then
    sa_fastv_evict_ratio=$(bc -l <<< "1 - $sa_ratio")
    model_args="${model_args},sa_fastv_evict_ratio=${sa_fastv_evict_ratio}"

elif [[ "$sa_pattern" == *"visionzip"* ]]; then
    :

elif [[ "$sa_pattern" == *"saliency"* ]]; then
    :

elif [[ "$sa_pattern" == *"streaming"* ]]; then
    :

elif [[ "$sa_pattern" == *"random"* ]]; then
    :
fi

if [[ "$sa_pattern" == *"vit"* ]]; then
    model_args="${model_args//sa_start_layer_idx/vit_start_layer_idx}"
fi

if [ "$idx" != "0" ]; then
    extra_name="${extra_name}_${idx}"
fi

if [[ "$model" == *"ft_779k"* ]]; then
    b="${model#*ft_779k}" 
    b="ft_779k$b" 
    b_clean="${b%/*}"
    extra_name="${extra_name}_${b_clean}"
fi


if [ "$subtitles" == "sub" ]; then
    tasks="videomme_w_subtitle"
    extra_name="${extra_name}_sub"
fi


MAX_RETRIES=5
RETRY_DELAY=10


output_path="${output_path}/res/${extra_name}"
log_file="${log_file}/${extra_name}.log"

if [ -d "${output_path}/submissions" ]; then
    echo "Skip: submissions exists at ${output_path}/submissions"
    exit 0
fi

if [ ! -d "${output_path}" ]; then
    mkdir -p "${output_path}"
fi

echo "model: ${model}"
echo "model_type: ${model_type}"
echo "extra_kwargs: ${extra_kwargs}"
echo "log_file: ${log_file}"
echo "output_path: ${output_path}"
echo "model_args: ${model_args}"
echo "tasks: ${tasks}"

CMD="accelerate launch --num_processes=${NUM_GPUS} --main_process_port=10392 -m lmms_eval \
    --model ${model_type} \
    --model_args ${model_args} \
    --tasks ${tasks} \
    --batch_size ${BATCH_SIZE} \
    --log_samples \
    --log_samples_suffix all_${task_type} \
    --output_path ${output_path} ${extra_kwargs}"

echo $CMD


if [ "$debug" != "0" ]; then
    eval "$CMD"
else
    eval "$CMD" > ${log_file} 2>&1
fi
status=$?


ray stop --force
pkill -9 -f raylet
pkill -9 -f gcs_server
pkill -9 -f dashboard
ps -ef |grep spawn|grep -v grep |cut -c 9-16|xargs kill -9
# ps -ef |grep model|grep -v grep |cut -c 9-16|xargs kill -9

sleep 10s

# eval "accelerate launch --num_processes=${NUM_GPUS} --main_process_port=10392 -m lmms_eval \
#     --model ${model_type} \
#     --model_args pretrained="${model_args}" \
#     --tasks ${tasks} \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix all_${task_type} \
#     --output_path ./logs_eval/${conv_template}/base${extra_name} ${extra_kwargs}"
