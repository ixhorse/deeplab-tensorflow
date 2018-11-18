#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="./tt100k"
mkdir -p "${WORK_DIR}"

# Root path for TT100K dataset.
TT100K_ROOT="/home/mcc/data/TT100K/TT100K_voc/"

# Remove the colormap in the ground truth annotations.
SEMANTIC_SEG_FOLDER="${TT100K_ROOT}/SegmentationClass"

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

IMAGE_FOLDER="${TT100K_ROOT}/JPEGImages"
LIST_FOLDER="${TT100K_ROOT}/ImageSets"

echo "Converting TT100K dataset..."
python ./build_tt100k_data.py \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${SEMANTIC_SEG_FOLDER}" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="jpg" \
  --output_dir="${OUTPUT_DIR}"
