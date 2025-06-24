#!/bin/bash

# Script to rename all .jpg files in the test directory to .jpeg

# Set the directory path
TEST_DIR="/Users/yexw/PycharmProjects/nova-fine-tunning/data/images/test"

# Check if the directory exists
if [ ! -d "$TEST_DIR" ]; then
  echo "Error: Directory $TEST_DIR does not exist."
  exit 1
fi

# Count the number of .jpg files
jpg_count=$(find "$TEST_DIR" -name "*.jpg" | wc -l)
echo "Found $jpg_count .jpg files in $TEST_DIR"

# Rename all .jpg files to .jpeg
for file in "$TEST_DIR"/*.jpg; do
  if [ -f "$file" ]; then
    # Get the base name without extension
    base_name=$(basename "$file" .jpg)
    # Create the new file name with .jpeg extension
    new_file="$TEST_DIR/$base_name.jpeg"
    # Rename the file
    mv "$file" "$new_file"
    echo "Renamed: $file -> $new_file"
  fi
done

# Verify the renaming
jpeg_count=$(find "$TEST_DIR" -name "*.jpeg" | wc -l)
jpg_count=$(find "$TEST_DIR" -name "*.jpg" | wc -l)
echo "After renaming: $jpeg_count .jpeg files, $jpg_count .jpg files"

echo "Renaming complete!"
