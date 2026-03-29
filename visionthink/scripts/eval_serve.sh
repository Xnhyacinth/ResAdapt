export HF_HUB_OFFLINE=True
export PYTHONPATH=/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink:$PYTHONPATH
export OPENAI_API_URL="https://gpt-i18n.byteintl.net/gpt/openapi/online/v2/crawl/openai/deployments/gpt_openapi"
export OPENAI_API_KEY="dxMlgIJpXgkdou8z77OKt5rg4BQjwgJZ_GPT_AK"
export MODEL_VERSION="gpt-5-2025-08-07"

unset PREDICTOR_PATH ENABLE_BASELINE_SCALE BASELINE_SCALE_FACTOR USE_DEBUG CONVERT2IMAGES REMOVEPAD
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

if [[ "$task_type" == *"video_inc"* ]]; then
    tasks="videomme_inc,longvideobench_val_v_inc"
elif [[ "$task_type" == *"video"* ]]; then
    tasks="videomme"
    if [[ "$task_type" == *"all"* ]]; then
        tasks="videomme,longvideobench_val_v,mlvu_dev,mlvu_test,egoschema_subset,lvbench" #mvbench
    fi
else
    tasks="mmmu_val,chartqa,mmbench_en_dev,pope,docvqa_val,ai2d,gqa,mme,realworldqa,textvqa_val,mathvista,ocrbench"
    # tasks="gqa" mmvet
fi

if [[ "$task_type" == *"video"* ]]; then
    model_args="${model_args},max_num_frames=${max_num_frames}"
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
elif [ "$conv_template" == "qwen2_vl" ]; then
    model_type="qwen2_vl"
elif [[ "$conv_template" == *"vllm"* ]]; then
    export TORCH_NCCL_BLOCKING_WAIT=1
    export NCCL_TIMEOUT=18000000
    model_type=${conv_template}
    BATCH_SIZE=32
    GPU_MEMORY_UTILIZATION=0.75

    if [[ "$model" == *"3B"* ]]; then
        BATCH_SIZE=64
    fi

    if [[ "$model" == *"scale"* ]] && [[ "$method" != *"nopred"* ]]; then
        export PREDICTOR_PATH="${model}pred"
        echo "Using predictor path: $PREDICTOR_PATH"

        model_type="vllm_generate_remote"
        BATCH_SIZE=8
        GPU_MEMORY_UTILIZATION=0.6 # 0.62
        # tasks="mmmu_val_scale"

        if [[ "$model" == *"3B"* ]]; then
            BATCH_SIZE=32
            # GPU_MEMORY_UTILIZATION=0.60
        fi
    fi

    if [[ "$model" == *"scale"* ]] && [[ "$method" == *"base"* ]]; then
        num=$(echo "$model" | grep -oP '(?<=-)\d+(?=B-)')
        if [ -n "$num" ]; then
            echo "base size $num"
            model="Qwen/Qwen2.5-VL-${num}B-Instruct"
        else
            model="Qwen/Qwen2.5-VL-7B-Instruct"
        fi
    fi

    if [[ "$method" == *"fix"* ]]; then
        model_type="vllm_generate_custom"
        num=$(echo "$method" | sed -E 's/.*fix([0-9]+\.?[0-9]*).*/\1/; t; s/.*/0.0/')
        if [ -n "$num" ]; then
            echo "fix $num"
        else
            num=1.5
        fi

        export ENABLE_BASELINE_SCALE=1
        export BASELINE_SCALE_FACTOR=${num}
    fi

    model_args="model=${model},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},data_parallel_size=${DATA_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    
    if [ ! -n "$step" ]; then
        suffix="${model#*/}"
        # output_path="${output_path}/${suffix}"
        # log_file="${log_file}/${suffix}"
        extra_name="${suffix}_${extra_name}"
    fi

    if [[ "$method" == *"img"* ]]; then
        model_type="vllm_generate_custom"
        export CONVERT2IMAGES=1
        # model_args="${model_args},mm_processor_cache_gb=32"
    fi

    if [[ "$method" == *"repad"* ]]; then        
        export REMOVEPAD=1
    fi

    if [[ "$task_type" == *"video"* ]]; then
        model_args="${model_args},nframes=${max_num_frames}"
    fi

    model_args="${model_args},mm_processor_cache_gb=0"
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
    model_args="${model_args},max_pixels=12845056,interleave_visuals=False"
elif [[ "$model_type" == *"qwen2_vl"* ]]; then
    model_args="${model_args},max_pixels=2359296"
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

if [[ "$sa_pattern" == *"quadtree"* ]]; then
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

if [[ "$model" == *"scale"* ]] && [[ "$method" != *"nopred"* ]]; then
    PRED="python -u visionthink/predictor/pred_serve.py"
    echo "PRED: ${PRED}"
    eval "$PRED" > ${log_file%.log}_pred.log 2>&1 &

    echo "⏳ Waiting for Ray Serve port (8000) to open..."
    MAX_RETRIES=50
    COUNTER=0

    while true; do
        if curl -s -o /dev/null "http://localhost:8000/-/routes"; then
            echo "✅ Ray Serve is up!"
            break
        fi
        
        if [ $COUNTER -ge $MAX_RETRIES ]; then
            echo "❌ Timeout waiting for port 8000."
            exit 1
        fi
        
        sleep 5
        COUNTER=$((COUNTER+1))
    done

    echo "🔌 Initializing Model Service..."
    API_INIT_URL="http://localhost:8000/init"

    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --noproxy "*" \
        -X POST "${API_INIT_URL}" \
        -H "Content-Type: application/json" \
        -d "{\"predictor_path\": \"${PREDICTOR_PATH}\", \"num_replicas\": ${NUM_GPUS}}")

    if [ "$HTTP_CODE" -eq 200 ] || [ "$HTTP_CODE" -eq 202 ]; then
        echo "✅ Init successful."
    else
        echo "❌ Init failed with HTTP $HTTP_CODE"
        exit 1
    fi

    echo "⏳ Waiting for Model to Load (Health Check)..."

    CHECK_URL="http://localhost:8000/predict"
    MAX_WAIT_SECONDS=600
    START_TIME=$(date +%s)

    while true; do
        RESPONSE=$(curl -s --noproxy "*" \
            -X POST "${CHECK_URL}" \
            -H "Content-Type: application/json" \
            -d '{"check_health": true}')

        CURRENT_TIME=$(date +%s)
        ELAPSED=$((CURRENT_TIME - START_TIME))

        if echo "$RESPONSE" | tr -d '\n' | grep -q '"status":"ready"'; then
            echo "🎉 Model is Fully Loaded and Ready! (Time taken: ${ELAPSED}s)"
            break
        else
            echo "Waiting... Server replied: $RESPONSE"
        fi

        if [ $ELAPSED -ge $MAX_WAIT_SECONDS ]; then
            echo "❌ Timeout!"
            exit 1
        fi
        
        sleep 10
    done

    echo "🚀 Proceeding to Main Evaluation..."
fi

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

ray stop
bash pre.sh

sleep 20s

# eval "accelerate launch --num_processes=${NUM_GPUS} --main_process_port=10392 -m lmms_eval \
#     --model ${model_type} \
#     --model_args pretrained="${model_args}" \
#     --tasks ${tasks} \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix all_${task_type} \
#     --output_path ./logs_eval/${conv_template}/base${extra_name} ${extra_kwargs}"

