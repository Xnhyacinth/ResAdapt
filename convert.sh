#!/bin/bash

# Check if the correct number of arguments is provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <grandparent_directory> <conv_template>"
    echo "Example: $0 /path/to/all_experiments qwen2_5_vl"
    exit 1
fi

CONDA_SH_PATH="/mnt/bn/jiangzhongtao/users/liaohuanxuan/conda/etc/profile.d/conda.sh"  

# Check if conda.sh exists
if [ ! -f "$CONDA_SH_PATH" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Error: conda.sh not found at $CONDA_SH_PATH."
    exit 1
fi

# Load Conda environment variables
source "$CONDA_SH_PATH"

grandparent_dir="$1"
conv_template=${2:-"qwen2_5_vl"}


while [ ! -d "$grandparent_dir" ]; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for directory '$grandparent_dir' to be created..."
    sleep 60
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Directory detected: $grandparent_dir"

# ==========================================
# Helper Functions
# ==========================================

ensure_merged() {
    local ckpt_path="$1"
    local exp_dir="$2"
    local ckpt_name=$(basename "$ckpt_path")
    
    # Remove trailing slash
    ckpt_path=${ckpt_path%/}

    local actor_src="${ckpt_path}/actor"
    local actor_dst="${ckpt_path}"
    
    local pred_src="${ckpt_path}/predictor"
    local pred_dst="${ckpt_path}/pred"
    
    # -----------------------
    # Process ACTOR
    # -----------------------
    if [ -d "$actor_src" ]; then
        if [ ! -f "${actor_dst}/config.json" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ACTOR] Merging $ckpt_name in $(basename "$exp_dir")..."
            python /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/verl/scripts/legacy_model_merger.py merge \
                --backend fsdp \
                --local_dir "$actor_src" \
                --target_dir "$actor_dst"
            
            if [ $? -ne 0 ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ACTOR] Error: Merge failed for $ckpt_name"
            fi
        fi
    fi

    # -----------------------
    # Process PREDICTOR
    # -----------------------
    if [ -d "$pred_src" ]; then
        if [ ! -d "$pred_dst" ] || [ -z "$(ls -A "$pred_dst" 2>/dev/null)" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] [PREDICTOR] Merging $ckpt_name in $(basename "$exp_dir")..."
            mkdir -p "$pred_dst"
            python /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/verl/scripts/legacy_model_merger.py merge \
                --backend fsdp \
                --local_dir "$pred_src" \
                --target_dir "$pred_dst"

            if [ $? -ne 0 ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] [PREDICTOR] Error: Merge failed for $ckpt_name"
            fi
        fi
    fi
}

cleanup_old_ckpt() {
    local ckpt_path="$1"
    local ckpt_name=$(basename "$ckpt_path")
    
    # Remove trailing slash
    ckpt_path=${ckpt_path%/}

    local actor_src="${ckpt_path}/actor"
    local actor_dst="${ckpt_path}"
    
    local pred_src="${ckpt_path}/predictor"
    local pred_dst="${ckpt_path}/pred"

    # 1. Clean ACTOR source
    if [ -d "$actor_src" ]; then
        # Safety Check: Delete only if destination config.json exists
        if [ -f "${actor_dst}/config.json" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] [CLEANUP] Deleting OLD source 'actor' in $ckpt_name"
            rm -rf "$actor_src"
        fi
    fi

    # 2. Clean PREDICTOR source
    if [ -d "$pred_src" ]; then
        # Safety Check: Delete only if dest dir exists and not empty
        if [ -d "$pred_dst" ] && [ "$(ls -A "$pred_dst" 2>/dev/null)" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] [CLEANUP] Deleting OLD source 'predictor' in $ckpt_name"
            rm -rf "$pred_src"
        fi
    fi
}


process_experiment_dir() {
    local exp_dir="$1"
    

    mapfile -t ckpt_dirs < <(find "$exp_dir" -mindepth 1 -maxdepth 1 -type d | sort -V)
    
    local num_ckpts=${#ckpt_dirs[@]}
    
    if [ "$num_ckpts" -eq 0 ]; then
        return
    fi
    
    local last_idx=$((num_ckpts - 1))
    local latest_ckpt="${ckpt_dirs[$last_idx]}"
    
    for i in "${!ckpt_dirs[@]}"; do
        local curr_ckpt="${ckpt_dirs[$i]}"
        
        ensure_merged "$curr_ckpt" "$exp_dir"
        
        if [ "$i" -lt "$last_idx" ]; then
            cleanup_old_ckpt "$curr_ckpt"
        else
            if [ -d "${curr_ckpt}/actor" ] || [ -d "${curr_ckpt}/predictor" ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] [KEEP] Keeping source files for LATEST: $(basename "$curr_ckpt")"
            fi
        fi
    done
}

# ==================================================================
# Main Monitoring Loop
# ==================================================================
scan_interval=300  # 5 minutes

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting continuous monitoring on: $grandparent_dir"
echo "Policy: Merge all. Delete sources of OLD checkpoints. Keep source of LATEST checkpoint."

while true; do
    # Resilience check
    if [ ! -d "$grandparent_dir" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: Root directory '$grandparent_dir' lost. Waiting..."
        sleep 60
        continue
    fi

    # Iterate over Experiment Folders
    for exp_dir in "$grandparent_dir"/*/; do
        if [ -d "$exp_dir" ]; then
            process_experiment_dir "$exp_dir"
        fi
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Scan cycle complete. Sleeping for ${scan_interval}s..."
    sleep $scan_interval
done