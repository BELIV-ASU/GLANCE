#!/usr/bin/env bash
set -euo pipefail

module load ffmpeg-6.0-gcc-12.1.0
module load cuda-12.8.1-gcc-12.1.0
module load cmake/3.30.2
module load mamba/latest
source activate new_beginning

cd /scratch/jnolas77/SafetyVLM/reasoning_distillation

python3 prepare_reasoning_data.py \
  --california_jsonl /scratch/jnolas77/SafetyVLM/DrivingManual/california_driving_sft/sft_output/california_driving_sft.jsonl \
  --wa_jsonl /scratch/jnolas77/SafetyVLM/DrivingManual/wa_driver_sft_data/final_package/sft_training_data.jsonl \
  --image_manifest_json /scratch/jnolas77/SafetyVLM/DrivingManual/wa_driver_sft_data/final_package/image_manifest.json \
  --out_dir /scratch/jnolas77/SafetyVLM/reasoning_distillation/data
