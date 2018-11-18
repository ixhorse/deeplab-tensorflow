#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

FLAG=$1

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Go to datasets folder and download PASCAL VOC 2012 segmentation dataset.
DATASET_DIR="datasets"

# Set up the working directories.
TT100K_FOLDER="tt100k"
EXP_FOLDER="exp"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${TT100K_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${TT100K_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${TT100K_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${TT100K_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${TT100K_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

TT100K_DATASET="${WORK_DIR}/${DATASET_DIR}/${TT100K_FOLDER}/tfrecord"
NUM_ITERATIONS=10000

echo $FLAG
if [ 1 == $FLAG ] 
then
    echo "====train===="
    python "${WORK_DIR}"/train.py \
        --logtostderr \
        --train_split="train" \
        --model_variant="xception_65" \
        --atrous_rates=6 \
        --atrous_rates=12 \
        --atrous_rates=18 \
        --output_stride=16 \
        --decoder_output_stride=4 \
        --train_crop_size=513 \
        --train_crop_size=513 \
        --min_resize_value=513 \
        --max_resize_value=513 \
        --train_batch_size=4 \
        --dataset="tt100k" \
        --training_number_of_steps="${NUM_ITERATIONS}" \
        --fine_tune_batch_norm=false \
        --initialize_last_layer=false \
        --train_logdir="${TRAIN_LOGDIR}" \
        --dataset_dir="${TT100K_DATASET}" \
        --tf_initial_checkpoint="${INIT_FOLDER}/deeplabv3_pascal_trainval/model.ckpt"
elif [ $FLAG == 2 ] 
then 
    # Run evaluation. This performs eval over the full val split (1449 images) and
    # will take a while.
    # Using the provided checkpoint, one should expect mIOU=82.20%.
    echo "====val===="
    python "${WORK_DIR}"/eval.py \
        --logtostderr \
        --eval_split="val" \
        --model_variant="xception_65" \
        --atrous_rates=6 \
        --atrous_rates=12 \
        --atrous_rates=18 \
        --output_stride=16 \
        --decoder_output_stride=4 \
        --min_resize_value=513 \
        --max_resize_value=513 \
        --eval_crop_size=513 \
        --eval_crop_size=513 \
        --dataset="tt100k" \
        --checkpoint_dir="${TRAIN_LOGDIR}" \
        --eval_logdir="${EVAL_LOGDIR}" \
        --dataset_dir="${TT100K_DATASET}" \
        --max_number_of_evaluations=1
elif [ $FLAG == 3 ] 
then
    # Visualize the results.
    echo "====vis===="
    python "${WORK_DIR}"/vis.py \
        --logtostderr \
        --vis_split="val" \
        --model_variant="xception_65" \
        --atrous_rates=6 \
        --atrous_rates=12 \
        --atrous_rates=18 \
        --output_stride=16 \
        --decoder_output_stride=4 \
        --vis_crop_size=513 \
        --vis_crop_size=513 \
        --min_resize_value=513 \
        --max_resize_value=513 \
        --dataset="tt100k" \
        --checkpoint_dir="${TRAIN_LOGDIR}" \
        --vis_logdir="${VIS_LOGDIR}" \
        --dataset_dir="${TT100K_DATASET}" \
        --max_number_of_iterations=1 \
        --save_raw_predictions=true
elif [ $FLAG == 4 ] 
then
    # Export the trained checkpoint.
    CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
    EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

    python "${WORK_DIR}"/export_model.py \
        --logtostderr \
        --checkpoint_path="${CKPT_PATH}" \
        --export_path="${EXPORT_PATH}" \
        --model_variant="xception_65" \
        --atrous_rates=6 \
        --atrous_rates=12 \
        --atrous_rates=18 \
        --output_stride=16 \
        --decoder_output_stride=4 \
        --num_classes=21 \
        --crop_size=513 \
        --crop_size=513 \
        --inference_scales=1.0
else
    echo "error"
fi

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.