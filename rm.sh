#!/bin/bash

if [ $# -lt 4 ] || [ $# -gt 5 ]; then
    echo "Usage: $0 <lower_bound> <upper_bound> <root_directory> <search_string> [suffix]"
    echo "Example (delete): $0 100 200 /path/to/experiments exp_v1"
    echo "Example (rename): $0 100 200 /path/to/experiments exp_v1 _bak"
    exit 1
fi

lower_bound=$1
upper_bound=$2
root_dir=$3
search_string=$4
suffix=$5

# ---------- Validation ----------

if ! [[ $lower_bound =~ ^[0-9]+$ ]] || ! [[ $upper_bound =~ ^[0-9]+$ ]]; then
    echo "Error: bounds must be integers"
    exit 1
fi

if [ "$lower_bound" -gt "$upper_bound" ]; then
    echo "Error: lower_bound > upper_bound"
    exit 1
fi

if [ ! -d "$root_dir" ]; then
    echo "Error: root directory does not exist: $root_dir"
    exit 1
fi

echo "--------------------------------------------------------"
echo "Root directory : $root_dir"
echo "Path contains  : $search_string"
echo "Step range     : [$lower_bound, $upper_bound]"
if [ -n "$suffix" ]; then
    echo "Rename suffix  : $suffix"
fi
echo "--------------------------------------------------------"

delete_list=()

# ---------- Core logic ----------

add_suffix() {
    local path="$1"
    local suffix="$2"
    local base_name
    base_name=$(basename "$path")
    if [ -d "$path" ]; then
        echo "${base_name}${suffix}"
    else
        if [[ "$base_name" == *.* && "$base_name" != .* ]]; then
            local stem="${base_name%.*}"
            local ext="${base_name##*.}"
            echo "${stem}${suffix}.${ext}"
        else
            echo "${base_name}${suffix}"
        fi
    fi
}

while IFS= read -r dir; do
    dir_name=$(basename "$dir")

    if [[ $dir_name =~ ^global_step_([0-9]+)_.*$ ]]; then
        step_num=${BASH_REMATCH[1]}

        if [ "$step_num" -ge "$lower_bound" ] && [ "$step_num" -le "$upper_bound" ]; then
            delete_list+=("$dir")
        fi
    fi
done < <(
    find "$root_dir" \( -type d -o -type f \) -path "*$search_string*" -name "global_step_*"
)

# ---------- Result ----------

count=${#delete_list[@]}

if [ $count -eq 0 ]; then
    echo "No matching global_step directories found."
    exit 0
fi

echo ""
echo "Found $count directories to delete:"
for dir in "${delete_list[@]}"; do
    echo "  $dir"
done
echo ""

if [ -n "$suffix" ]; then
    read -p "Confirm rename? [y/N] " confirm
else
    read -p "Confirm deletion? [y/N] " confirm
fi
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Cancelled."
    exit 0
fi

for dir in "${delete_list[@]}"; do
    if [ -n "$suffix" ]; then
        parent_dir=$(dirname "$dir")
        new_name=$(add_suffix "$dir" "$suffix")
        new_path="${parent_dir}/${new_name}"
        if [ -e "$new_path" ]; then
            echo "Skip (exists): $new_path"
            continue
        fi
        echo "Renaming: $dir -> $new_path"
        mv "$dir" "$new_path"
    else
        echo "Deleting: $dir"
        rm -rf "$dir"
    fi
done

echo "Done."
