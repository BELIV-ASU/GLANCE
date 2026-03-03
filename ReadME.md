# GLANCE - SafetyVLM

## Setup (Local Workstation)

```bash
# Create and activate virtual environment (already done)
cd /home/sim273/GLANCE
source .venv/bin/activate
```

## Inference
```bash
chmod +x infer.sh
./infer.sh
```

## Training
```bash
chmod +x train_teacher.sh
./train_teacher.sh
```

## Distillation
```bash
source .venv/bin/activate
python3 distill_4b.py \
    --teacher Qwen/Qwen2.5-VL-7B-Instruct \
    --student Qwen/Qwen2.5-VL-3B-Instruct \
    --train_jsonl data/train.jsonl \
    --out_dir checkpoints/distill_output
```

## Model Info
- **Base model**: `Qwen/Qwen2.5-VL-7B-Instruct`
- **LoRA adapter**: `checkpoints/` (r=16, alpha=32)
- **GPU**: NVIDIA RTX 5000 Ada (30GB VRAM)
- **Quantization**: 4-bit NF4 via bitsandbytes

## Datasets and Checkpoints
[Dataset here](https://www.dropbox.com/scl/fo/d6wpk16zpd7vxbl4c8ncu/APyLVdMgb0RX2qCf9uLWrTI?rlkey=p7xjrap2w1j3bja5js4bkxv1u&st=vu9cwbqj&dl=0)
