#!/bin/bash

# Function to confirm the deletion with user input
confirm_delete() {
    read -p "Do you want to delete the listed files? (y/n): " choice
    if [ "$choice" == "y" ]; then
        return 0
    else
        return 1
    fi
}

# Check if base directory is provided as the first argument, if not, prompt the user
if [ -z "$1" ]; then
    read -p "Enter the base directory: " base_dir
else
    base_dir="$1"
fi

# Check if prefix is provided as the second argument, if not, prompt the user
if [ -z "$2" ]; then
    read -p "Enter the prefix: " prefix
else
    prefix="$2"
fi

# Show file lists before deletion
echo "Files to be deleted:"
echo "-----------------------"
find "$base_dir" -type f -name "${prefix}*" -print
find "$base_dir/log" -type f -name "${prefix}*" -print
find "$base_dir/by-product" -type f -name "${prefix}*" -print
find "$base_dir/by-product" -type d -name "${prefix}*" -print
echo "-----------------------"

# Confirm with the user before deleting
confirm_delete || exit 0

# Perform deletion
find "$base_dir" -type f -name "${prefix}*" -delete
find "$base_dir/log" -type f -name "${prefix}*" -delete
find "$base_dir/by-product" -type f -name "${prefix}*" -delete
find "$base_dir/by-product" -type d -name "${prefix}*" -exec rm -r {} +

echo "Deletion complete."
