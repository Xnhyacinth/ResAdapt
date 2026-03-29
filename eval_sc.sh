#!/bin/bash
export PYTHONPATH=/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink:$PYTHONPATH

# ================= Configuration =================
# Check arguments
# Modified to accept optional lower and upper bounds (at least 3 args required)
if [ $# -lt 3 ]; then
    echo "Usage: $0 <grandparent_directory> <conv_template> <exp_filter> [lower_bound] [upper_bound]"
    echo "Example 1 (No bounds): $0 /path/to/exps qwen2_5_vl 'run_2'"
    echo "Example 2 (With bounds): $0 /path/to/exps qwen2_5_vl 'run_2' 1000 5000  # Only process steps [1000, 5000]"
    echo "Example 3 (Lower only):  $0 /path/to/exps qwen2_5_vl 'run_2' 2000       # Only process steps >= 2000"
    exit 1
fi

GRANDPARENT_DIR="$1"
CONV_TEMPLATE="$2"
EXP_FILTER="$3"
LOWER_BOUND="${4:-}" # Optional: Default to empty
UPPER_BOUND="${5:-}" # Optional: Default to empty
TYPE="${6:-images}"
max_num_frames=${7:-"0"}
METHOD=${8:-"0"}

# Paths and Constants
CONDA_SH_PATH="/mnt/bn/jiangzhongtao/users/liaohuanxuan/conda/etc/profile.d/conda.sh"
EVAL_SCRIPT="bash visionthink/scripts/eval.sh"
EVAL_MODEL="Qwen/Qwen2-VL-7B-Instruct" 
EVAL_TAG=("0" "0" "0" "0")
SCAN_INTERVAL=300 

# ================= Setup =================
# Check Conda
if [ ! -f "$CONDA_SH_PATH" ]; then
    echo "Error: conda.sh not found at $CONDA_SH_PATH"
    exit 1
fi
source "$CONDA_SH_PATH"

# Check Eval Script
if [ ! -f "visionthink/scripts/eval.sh" ]; then
    echo "Error: visionthink/scripts/eval.sh not found in current directory"
    exit 1
fi

# ================= Main Logic =================

process_experiment() {
    local exp_dir="$1"
    # Remove trailing slash
    exp_dir=${exp_dir%/}
    
    local eval_root_dir="${exp_dir}/eval_${CONV_TEMPLATE}"
    local search_base="$eval_root_dir/res"

    if [[ "$max_num_frames" == "1" ]]; then
        frames_list=(32)
    else
        frames_list=("$max_num_frames")
    fi

    if [[ "$TYPE" == "all" ]]; then
        types_to_run=("video_all")
    else
        types_to_run=("$TYPE")
    fi

    for eval_type in "${types_to_run[@]}"; do
        for frames in "${frames_list[@]}"; do
            while true; do
                newest_ckpt_path=""
                newest_ckpt_name=""
                # Build a sorted candidate list every time to pick the latest
                candidates=$(for ckpt_path in "$exp_dir"/*/; do
                    if [ -d "$ckpt_path" ]; then
                        ckpt_name=$(basename "$ckpt_path")
                        if [[ "$ckpt_name" == "eval_${CONV_TEMPLATE}" ]]; then
                            continue
                        fi
                        if [ ! -f "${ckpt_path}/config.json" ]; then
                            continue
                        fi
                        ckpt_num=$(echo "$ckpt_name" | sed -n 's/.*global_step_\([0-9][0-9]*\).*/\1/p')
                        if [[ -z "$ckpt_num" ]]; then
                            ckpt_num=$(echo "$ckpt_name" | grep -oE '[0-9]+' | tail -n1)
                        fi
                        if [[ -z "$ckpt_num" ]]; then
                            ckpt_num=-1
                        fi
                        # Bounds filtering
                        if [[ -n "$LOWER_BOUND" ]] && (( ckpt_num < LOWER_BOUND )); then
                            continue
                        fi
                        if [[ -n "$UPPER_BOUND" ]] && (( ckpt_num > UPPER_BOUND )); then
                            continue
                        fi
                        printf "%s\t%s\n" "$ckpt_num" "$ckpt_path"
                    fi
                done | sort -nr -k1,1)

                found_missing=""
                while IFS=$'\t' read -r ckpt_num ckpt_path; do
                    [ -z "$ckpt_path" ] && continue
                    ckpt_name=$(basename "$ckpt_path")
                    expected_dir="${ckpt_name}_base"
                    if [[ -n "$frames" && "$frames" != "0" ]]; then
                        expected_dir="${expected_dir}_frames${frames}"
                    fi
                    expected_dir="${expected_dir}_${eval_type}"
                    if [[ -n "$METHOD" && "$METHOD" != "0" ]]; then
                        expected_dir="${expected_dir}_${METHOD}"
                    fi
                    expected_path="${search_base}/${expected_dir}"
                    results_found=$(find "$expected_path" -mindepth 2 -maxdepth 2 -type f -name "*results.json" -print -quit 2>/dev/null)
                    if [ -z "$results_found" ]; then
                        newest_ckpt_path="$ckpt_path"
                        newest_ckpt_name="$ckpt_name"
                        found_missing="1"
                        break
                    fi
                done <<< "$candidates"

                if [[ -z "$found_missing" ]]; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [EVAL] No missing results for type=${eval_type}, frames=${frames} in $(basename "$exp_dir")"
                    break
                fi

                echo "[$(date '+%Y-%m-%d %H:%M:%S')] [EVAL] Missing results for: ${newest_ckpt_name} (in $(basename "$exp_dir"))"
                echo "Expected base: ${search_base}/${newest_ckpt_name}_base ..."
                echo "--> Starting evaluation (type=${eval_type}, frames=${frames})..."

                mkdir -p "$eval_root_dir"
                $EVAL_SCRIPT "$newest_ckpt_path" "${CONV_TEMPLATE}" "${eval_type}" "${frames}" "${EVAL_TAG[@]}" "$exp_dir" "${METHOD}"
                status=$?
                if [ $status -eq 0 ]; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [EVAL] Success: ${newest_ckpt_name}"
                else
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [EVAL] Failed: ${newest_ckpt_name}"
                fi
                ps -ef |grep lmms|grep -v grep |awk '{print $2}' |xargs -r kill -9
                # Loop will re-pick the latest candidate next iteration
            done
        done
    done
}

# ================= Execution Loop =================

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Auto-Eval Monitor on: $GRANDPARENT_DIR"
if [[ -n "$LOWER_BOUND" ]]; then echo "   Lower Bound: $LOWER_BOUND"; fi
if [[ -n "$UPPER_BOUND" ]]; then echo "   Upper Bound: $UPPER_BOUND"; fi



# Loop through Experiment Directories (Parent Dirs)
for exp_dir in "$GRANDPARENT_DIR"/*/; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        # Check if the experiment name contains the filter
        if [[ "$exp_name" == *"$EXP_FILTER"* ]]; then
            process_experiment "$exp_dir"
        fi
    fi
done

# while true; do
#     # Check root dir resilience
#     if [ ! -d "$GRANDPARENT_DIR" ]; then
#         echo "Waiting for root directory..."
#         sleep 60
#         continue
#     fi

#     # Loop through Experiment Directories (Parent Dirs)
#     for exp_dir in "$GRANDPARENT_DIR"/*/; do
#         if [ -d "$exp_dir" ]; then
#             exp_name=$(basename "$exp_dir")
#             # Check if the experiment name contains the filter
#             if [[ "$exp_name" == *"$EXP_FILTER"* ]]; then
#                 process_experiment "$exp_dir"
#             fi
#         fi
#     done

#     echo "[$(date '+%Y-%m-%d %H:%M:%S')] Scan complete. Sleeping for ${SCAN_INTERVAL}s..."
#     sleep $SCAN_INTERVAL
# done
