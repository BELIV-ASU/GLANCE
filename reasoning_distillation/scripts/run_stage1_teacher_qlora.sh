#!/usr/bin/env bash
set -euo pipefail

module load ffmpeg-6.0-gcc-12.1.0
module load cuda-12.8.1-gcc-12.1.0
module load cmake/3.30.2
module load mamba/latest
source activate new_beginning

cd /scratch/jnolas77/SafetyVLM/reasoning_distillation

python3 train_teacher_qwq32b_qlora.py \
  --model_name Qwen/QwQ-32B \
  --train_jsonl /scratch/jnolas77/SafetyVLM/reasoning_distillation/data/teacher_sft_qwq32b.jsonl \
  --out_dir /scratch/jnolas77/SafetyVLM/reasoning_distillation/checkpoints/teacher_qwq32b_qlora \
  --max_length 2048 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --logging_steps 5 \
  --save_steps 200
