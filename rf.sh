#!/bin/bash

# Check if the number of arguments is correct
if [ $# -ne 2 ]; then
    echo "Usage: $0 <root_directory> <search_string>"
    echo "Example: $0 /path/to/experiments \"test_run\""
    exit 1
fi

root_dir=$1
search_string=$2

# --- Safety Checks ---

# 1. Verify that the root directory exists
if [ ! -d "$root_dir" ]; then
    echo "Error: Directory '$root_dir' does not exist."
    exit 1
fi

# 2. Ensure the search string is not empty
if [ -z "$search_string" ]; then
    echo "Error: Search string cannot be empty."
    exit 1
fi

# --- Search Logic ---

echo "Searching for items (files & dirs) containing '$search_string' inside '$root_dir'..."

# Array to store found items
# CHANGED: Removed '-type d' so it finds files AND directories
# NOTE: '-maxdepth 1' means it only looks in the current folder, not subfolders recursively.
target_items=()
while IFS= read -r item; do
    target_items+=("$item")
done < <(find "$root_dir" -mindepth 1 -maxdepth 1 -name "*$search_string*")

# --- Result Handling ---

count=${#target_items[@]}

# If no items are found, exit
if [ $count -eq 0 ]; then
    echo "No matching items found."
    exit 0
fi

# Display the list of items to be deleted
echo "========================================"
echo "Found $count item(s) matching the criteria:"
echo "========================================"
for item in "${target_items[@]}"; do
    # Add a visual indicator if it's a directory or a file
    if [ -d "$item" ]; then
        echo " [DIR]  $item"
    else
        echo " [FILE] $item"
    fi
done
echo "========================================"

# --- User Confirmation ---

echo "WARNING: These items will be permanently deleted!"
read -p "Are you sure you want to delete them? [y/N] " confirm

# Proceed only if user types 'y' or 'Y'
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Operation cancelled."
    exit 0
fi

# --- Execution ---

echo "Starting deletion..."
for item in "${target_items[@]}"; do
    echo "Deleting: $item"
    # rm -rf works for both files and non-empty directories
    rm -rf "$item"
done

echo "Deletion completed. Total deleted: $count."