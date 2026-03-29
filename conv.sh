#!/bin/bash

# Check if the correct number of arguments is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <parent_directory>"
    echo "Example: $0 /path/to/parent_dir  # Monitor and process folders in this directory"
    exit 1
fi

CONDA_SH_PATH="/mnt/bn/jiangzhongtao/users/liaohuanxuan/conda/etc/profile.d/conda.sh"  

# Load Conda environment variables
source "$CONDA_SH_PATH"

parent_dir="$1"
conv_template=${2:-"qwen2_5_vl"}

# Validate the parent directory
if [ ! -d "$parent_dir" ]; then
    echo "Error: Directory '$parent_dir' does not exist"
    exit 1
fi

# Function to copy modeling file with flexibility
copy_modeling_file() {
    local src_dir="$1"
    local dest_dir="$2"
    local filename="$3"

    # Try finding in src/huggingface/filename first
    if [ -f "${src_dir}/huggingface/${filename}" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Found ${filename} in ${src_dir}/huggingface. Copying..."
        cp "${src_dir}/huggingface/${filename}" "${dest_dir}/"
    # Fallback to src/filename
    elif [ -f "${src_dir}/${filename}" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Found ${filename} in ${src_dir}. Copying..."
        cp "${src_dir}/${filename}" "${dest_dir}/"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Info: ${filename} not found in source. Skipping copy."
    fi
}

# Function to process a single folder
process_folder() {
    local folder_path="$1"
    local path_name=$(basename "$folder_path")
    
    local actor_dir="${folder_path}/actor"
    local predictor_dir="${folder_path}/predictor"
    
    local actor_status=0
    local predictor_status=0

    if [ -d "$actor_dir" ]; then
        local target_dir="${folder_path}"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ACTOR] Running merge for $path_name..."
        
        python /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/verl/scripts/legacy_model_merger.py merge \
            --backend fsdp \
            --local_dir "$actor_dir" \
            --target_dir "$target_dir"

        if [ $? -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ACTOR] Merge successful."
            copy_modeling_file "$actor_dir" "$target_dir" "configuration_predictor.py"
            copy_modeling_file "$actor_dir" "$target_dir" "modeling_predictor.py"
            
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ACTOR] Deleting $actor_dir..."
            rm -rf "$actor_dir"
            actor_status=0
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ACTOR] Error: Merge failed."
            actor_status=1
        fi
    fi

    if [ -d "$predictor_dir" ]; then

        local target_dir="${folder_path}/predictor_merged"

        mkdir -p "$target_dir"
        
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [PREDICTOR] Running merge for $path_name..."
        
        python /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/verl/scripts/legacy_model_merger.py merge \
            --backend fsdp \
            --local_dir "$predictor_dir" \
            --target_dir "$target_dir"

        if [ $? -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] [PREDICTOR] Merge successful."
            
            copy_modeling_file "$predictor_dir" "$target_dir" "modeling_predictor.py"
            
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] [PREDICTOR] Deleting $predictor_dir..."
            rm -rf "$predictor_dir"
            predictor_status=0
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] [PREDICTOR] Error: Merge failed."
            predictor_status=1
        fi
    fi


    if [ $actor_status -eq 0 ] && [ $predictor_status -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Folder $path_name fully processed."
        mark_processed "$folder_path"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: Some components in $path_name failed to merge."
    fi
}

# File to track processed folders
processed_file="${parent_dir}/.processed_folders.txt"
touch "$processed_file"

is_processed() {
    local folder="$1"
    grep -qxF "$folder" "$processed_file"
}

mark_processed() {
    local folder="$1"
    echo "$folder" >> "$processed_file"
}

# Scan loop
for folder in "$parent_dir"/*/; do
    folder=${folder%/} 

    if [ -d "$folder" ] && ! is_processed "$folder"; then
        if [ -d "${folder}/actor" ] || [ -d "${folder}/predictor" ]; then
            process_folder "$folder"
        fi
    fi
done











# #!/bin/bash

# # Check if the correct number of arguments is provided
# if [ $# -ne 1 ]; then
#     echo "Usage: $0 <parent_directory>"
#     echo "Example: $0 /path/to/parent_dir  # Monitor and process folders in this directory"
#     exit 1
# fi

# CONDA_SH_PATH="/mnt/bn/jiangzhongtao/users/liaohuanxuan/conda/etc/profile.d/conda.sh"  

# # Load Conda environment variables (required for 'conda' command in script)
# source "$CONDA_SH_PATH"

# parent_dir="$1"
# conv_template=${2:-"qwen2_5_vl"}
# eval_script="bash scripts/eval.sh"  # Path to the evaluation script
# eval_model="Qwen/Qwen2-VL-7B-Instruct"  # Fixed model for evaluation
# eval_tag="images 0 0 0 0 0"  # Fixed tag for evaluation
# eval_dir="${parent_dir}/eval_${conv_template}"

# # Validate the parent directory exists and is a directory
# if [ ! -d "$parent_dir" ]; then
#     echo "Error: Directory '$parent_dir' does not exist or is not a valid directory"
#     exit 1
# fi


# # Function to process a single folder (merge + delete actor + run evaluation)
# process_folder() {
#     local folder_path="$1"
#     local path_name=$(basename "$folder_path")  # Extract folder name from full path
    
#     # [FIX] Added '/' to ensure path is correct: path/to/folder/actor
#     local local_dir="${folder_path}/predictor" 
#     local target_dir="${folder_path}"

#     # Flag: whether processing was done via the original 'actor' directory logic
#     local processed_via_actor=0

#     # Original logic: Process if 'actor' directory exists
#     if [ -d "$local_dir" ]; then
#         echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running merge for $path_name..."
        
#         # 1. 执行合并脚本
#         python /mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/verl/scripts/legacy_model_merger.py merge \
#             --backend fsdp \
#             --local_dir "$local_dir" \
#             --target_dir "$target_dir"

#         # Check if merge succeeded
#         if [ $? -eq 0 ]; then
#             echo "[$(date '+%Y-%m-%d %H:%M:%S')] Merge successful."

#             # ================= [NEW] Copy modeling_qwenvl.py logic =================
#             # Define the source path for the modeling file
#             # Assuming the structure is local_dir/huggingface/modeling_qwenvl.py
#             # If 'huggingface' is inside 'actor', then:
#             local modeling_file_src="${local_dir}/huggingface/modeling_qwenvl.py"
            
#             if [ -f "$modeling_file_src" ]; then
#                 echo "[$(date '+%Y-%m-%d %H:%M:%S')] Found modeling_qwenvl.py in ${local_dir}/huggingface. Copying to target..."
#                 cp "$modeling_file_src" "$target_dir/"
#                 if [ $? -eq 0 ]; then
#                     echo "[$(date '+%Y-%m-%d %H:%M:%S')] Copy successful."
#                 else
#                     echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: Failed to copy modeling_qwenvl.py."
#                 fi
#             else
#                 # Optional: Check directly under actor just in case
#                 if [ -f "${local_dir}/modeling_qwenvl.py" ]; then
#                      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Found modeling_qwenvl.py in ${local_dir}. Copying to target..."
#                      cp "${local_dir}/modeling_qwenvl.py" "$target_dir/"
#                 else
#                      echo "[$(date '+%Y-%m-%d %H:%M:%S')] modeling_qwenvl.py not found. Skipping copy."
#                 fi
#             fi
#             # =======================================================================

#             echo "[$(date '+%Y-%m-%d %H:%M:%S')] Deleting $local_dir..."
#             rm -rf "$local_dir"  # Delete 'actor' directory after successful merge
            
#             # Verify if 'actor' directory was actually deleted
#             if [ -d "$local_dir" ]; then
#                 echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: Failed to delete $local_dir. Please clean up manually."
#                 return 1
#             fi
            
#             # Mark as processed only if success
#             mark_processed "$folder_path"
#             return 0
#         else
#             echo "[$(date '+%Y-%m-%d %H:%M:%S')] Error: Merge failed for $path_name. Preserving $local_dir for debugging."
#             return 1
#         fi
#     fi
# }


# # File to track processed folders (avoid reprocessing)
# processed_file="${parent_dir}/.processed_folders.txt"
# touch "$processed_file"  # Create if not exists

# # Function to check if a folder has been processed
# is_processed() {
#     local folder="$1"
#     grep -qxF "$folder" "$processed_file"
# }

# # Function to mark a folder as processed
# mark_processed() {
#     local folder="$1"
#     echo "$folder" >> "$processed_file"
# }

# # Initial scan: process existing folders that haven't been processed yet
# # echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting initial scan of existing folders in $parent_dir..."
# for folder in "$parent_dir"/*/; do
#     # Remove trailing slash for consistency in tracking file
#     folder=${folder%/} 
#     if [ -d "$folder" ] && ! is_processed "$folder"; then
#         process_folder "$folder"
#     fi
# done