#!/bin/bash

# Check if a directory argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_search>"
    echo "Example: $0 /cluster/work/fuge/AISCO/temp_"
    exit 1
fi

SEARCH_DIR="$1"

# Check if directory exists
if [ ! -d "$SEARCH_DIR" ]; then
    echo "Error: Directory '$SEARCH_DIR' does not exist."
    exit 1
fi

echo "Searching for *trace.json files recursively in: $SEARCH_DIR"
echo "--------------------------------------------------------"

# Find trace.json files and print their human-readable size (-h) and path
find "$SEARCH_DIR" -type f -name "*trace.json" -exec du -h {} +

echo "--------------------------------------------------------"
echo "Search complete."
