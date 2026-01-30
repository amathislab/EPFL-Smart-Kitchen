#!/bin/bash

# Script to download all files from the motion_generation dataset on Hugging Face
# Repository: https://huggingface.co/datasets/amathislab/ESK_motion_generation

set -e

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli is not installed."
    echo "Please install it with: pip install huggingface_hub[cli]"
    exit 1
fi

# Set the dataset repository
REPO="amathislab/ESK_motion_generation"

# Set the output directory (default: ./data)
OUTPUT_DIR="${1:-./data}"

echo "Downloading ESK_motion_generation dataset from Hugging Face..."
echo "Repository: $REPO"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Download the entire dataset using huggingface-cli
huggingface-cli download "$REPO" \
    --repo-type dataset \
    --local-dir "$OUTPUT_DIR" \
    --local-dir-use-symlinks False

echo ""
echo "Download complete! Files saved to: $OUTPUT_DIR"
