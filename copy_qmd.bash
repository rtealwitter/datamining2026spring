#!/bin/bash

# Target the parent directory
PARENT_DIR=".."
TXT="txt"
mkdir "$TXT"

# Find all .qmd files recursively in the current directory and subdirectories
find . -type f -name "*.qmd" | while read -r file; do
    # Get the filename without the path
    filename=$(basename "$file")
    
    # Define the new filename (change extension to .txt)
    new_filename="${filename%.qmd}.txt"
    
    # Copy to the parent directory
    cp "$file" "$TXT/$new_filename"
    
    echo "Copied $filename to $PARENT_DIR/$new_filename"
done