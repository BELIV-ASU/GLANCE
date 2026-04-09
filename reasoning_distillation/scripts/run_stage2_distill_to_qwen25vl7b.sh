#!/usr/bin/env bash
set -euo pipefail

module load ffmpeg-6.0-gcc-12.1.0
module load cuda-12.8.1-gcc-12.1.0
module load cmake/3.30.2
module load mamba/latest
source activate new_beginning

cd /scratch/jnolas77/SafetyVLM/reasoning_distillation

python3 train_qwen7b_reasoning_distill.py \
  --teacher_base Qwen/QwQ-32B \
  --teacher_adapter /scratch/jnolas77/SafetyVLM/reasoning_distillation/checkpoints/teacher_qwq32b_qlora/teacher_lora_adapter \
  --student Qwen/Qwen2.5-VL-7B-Instruct \
  --train_jsonl /scratch/jnolas77/SafetyVLM/reasoning_distillation/data/distill_train_qwen25vl7b.jsonl \
  --out_dir /scratch/jnolas77/SafetyVLM/reasoning_distillation/checkpoints/qwen25vl7b_distilled \
  --epochs 2 \
  --batch_size 1 \
  --grad_accum 8 \
  --max_length 2048 \
  --lr 1e-5 \
  --alpha_ce 0.2 \
  --beta_kl 0.8 \
  --temperature 2.0 \
  --log_every 10 \
  --save_every 200
