# GLANCE + Waymo + Qwen VLM Training Setup

## Quick Start (3 Steps)

### Step 1: Source Environment
```bash
source /scratch/rbaskar5/set.bash
cd /scratch/rbaskar5/GLANCE
```

### Step 2: Prepare Front Camera Dataset
```bash
# Extract front camera images from downloaded Waymo TFRecords
# and generate training/validation annotations
python scripts/prepare_waymo_front_camera.py
```

This will:
- Extract **FRONT camera only** from 1540 Waymo TFRecord files
- Save JPEGs to `/scratch/rbaskar5/Dataset/waymo_front/`
- Generate annotations JSON for training
- Update config to point to prepared data
- Duration: ~15-30 minutes (depending on hardware)

### Step 3: Launch Training
```bash
# Train Qwen2.5-VL-7B on 2× A100-80GB with LoRA
./scripts/train_waymo.sh qwen2.5-vl-7b

# Or use the automated setup script:
chmod +x setup_glance_waymo_qwen.sh
./setup_glance_waymo_qwen.sh both
```

---

## Detailed Configuration

### Data Setup

**Input**: Waymo Open Dataset (3.2TB)
- Location: `/scratch/rbaskar5/.hf_cache/datasets--AnnaZhang--waymo_open_dataset_v_1_4_3/blobs/`
- Total files: 1540 TFRecord + TAR archives
- Data format: Multi-sensor (5 cameras, LiDAR, 3D labels, trajectories)

**Processing**: `prepare_waymo_front_camera.py`
- Extracts FRONT camera images only
- Applies 80/20 split → train/validation
- Generates `train.json` and `val.json` with annotations

**Output**: `/scratch/rbaskar5/Dataset/waymo_front/`
```
waymo_front/
├── training/camera_FRONT/
│   ├── frame_000000_file_0000.jpg
│   ├── frame_000001_file_0000.jpg
│   └── ...
├── validation/camera_FRONT/
│   ├── frame_000000_file_0400.jpg
│   └── ...
└── annotations/
    ├── train.json         # ~800 samples
    └── val.json          # ~200 samples
```

### Model: Qwen2.5-VL-7B

**Base Model**: `Qwen/Qwen2.5-VL-7B-Instruct`
- 7B parameters
- Multi-modal (text + images)
- Supports vision instructions

**LoRA Adapter** (Parameter-Efficient Fine-Tuning):
```yaml
lora:
  r: 64
  alpha: 128
  dropout: 0.05
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
```

**Output**: LoRA adapter saved to:
```
/scratch/rbaskar5/GLANCE/checkpoints/waymo_finetune/qwen2.5-vl-7b/
├── adapter_config.json
├── adapter_model.bin      # LoRA weights (~30MB)
├── training_args.bin
└── training_log.txt
```

### Training Config

**File**: `configs/waymo_finetune_config.yaml`

```yaml
data:
  waymo_root: "/scratch/rbaskar5/Dataset/waymo_front"
  annotations_json: "/scratch/rbaskar5/Dataset/waymo_front/annotations/train.json"
  val_annotations_json: "/scratch/rbaskar5/Dataset/waymo_front/annotations/val.json"
  image_size: [448, 448]
  max_seq_length: 2048

training:
  num_epochs: 3
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 2
  gradient_accumulation_steps: 8      # Effective batch = 2×2×8 = 32
  learning_rate: 2.0e-5
  gradient_checkpointing: true        # Memory optimization
  bf16: true                          # bfloat16 precision
  
  output_dir: "/scratch/rbaskar5/GLANCE/checkpoints/waymo_finetune"
  save_steps: 500
  eval_steps: 500
```

### Hardware Requirements

- **GPUs**: 2× A100-80GB (with NVLink)
- **Memory**: ~160GB total (80GB × 2)
- **Each model** (7B + activations + gradients + optimizer states in bf16):
  - Model weights: ~14GB
  - Gradients: ~14GB
  - Optimizer states: ~28GB
  - Activations: ~24GB
  - **Total per GPU**: ~80GB
- **BF16 vs FP32**: 50% memory savings
- **LoRA**: Adds only ~30MB (trainable parameters)

### Environment Variables

Set by `/scratch/rbaskar5/set.bash`:
```bash
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_BLAS_PREFER_CUBLASLT=1       # Avoid cuBLAS bugs
export HF_HOME="/scratch/rbaskar5/.hf_cache"
export TRANSFORMERS_CACHE="/scratch/rbaskar5/.hf_cache"
export TRITON_CACHE_DIR="/scratch/rbaskar5/.triton_cache"
```

---

## Workflow

### 1️⃣ **Preparation** (run once)

```bash
cd /scratch/rbaskar5/GLANCE
python scripts/prepare_waymo_front_camera.py
```

Expected output:
```
✓ Extraction complete!
  Train frames: ~850
  Val frames:   ~215
✓ Updated config file
```

### 2️⃣ **Training** (3-5 hours)

```bash
./scripts/train_waymo.sh qwen2.5-vl-7b
```

Monitoring:
```bash
# In another terminal
tail -f /scratch/rbaskar5/GLANCE/checkpoints/waymo/qwen2.5-vl-7b_train.log

# Or check GPU usage
nvidia-smi dmon
```

### 3️⃣ **Inference** (Optional)

With trained LoRA adapter:
```python
from glance_src.vlm_factory import get_vlm_and_processor
from peft import PeftModel

model, processor = get_vlm_and_processor("qwen2.5-vl-7b", cfg)
model = PeftModel.from_pretrained(
    model,
    "/scratch/rbaskar5/GLANCE/checkpoints/waymo_finetune/qwen2.5-vl-7b/adapter_model.bin"
)

# Now model is fine-tuned on Waymo front camera data!
```

---

## Troubleshooting

### Issue: "Cannot find Waymo dataset"
**Solution**: Verify download:
```bash
ls -la /scratch/rbaskar5/.hf_cache/datasets--AnnaZhang--waymo_open_dataset_v_1_4_3/blobs/ | head
# Should show ~1540 files
```

### Issue: "TensorFlow not installed"
**Solution**:
```bash
source /scratch/rbaskar5/set.bash
pip install tensorflow --no-cache-dir
# Or use prepared extraction without TF (manually)
```

### Issue: "CUDA out of memory"
**Solution**: Verify A100s are being used:
```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
# Should show: A100-PCIE-80GB

# Reduce batch size in config if needed:
per_device_train_batch_size: 1   # from 2
gradient_accumulation_steps: 16    # from 8
```

### Issue: "Training seems slow"
**Solution**: Check GPU utilization:
```bash
watch nvidia-smi dmon
# Should see >80% GPU utilization (per GPU)
```

---

## File Locations

```
/scratch/rbaskar5/GLANCE/
├── configs/
│   ├── waymo_finetune_config.yaml      ← Main config
│   └── accelerate_2gpu.yaml            ← Multi-GPU setup
├── scripts/
│   ├── prepare_waymo_front_camera.py   ← Data extraction
│   ├── train_waymo.sh                  ← Training launcher
│   └── setup_glance_waymo_qwen.sh      ← Full pipeline
├── src/
│   ├── train_waymo.py                  ← Training loop
│   ├── waymo_dataset.py                ← Data loader
│   ├── vlm_factory.py                  ← Model loader (Qwen, etc.)
│   └── infer_teacher.py                ← Inference script
└── checkpoints/
    └── waymo_finetune/
        └── qwen2.5-vl-7b/              ← Trained LoRA weights

/scratch/rbaskar5/Dataset/
└── waymo_front/                        ← Extracted front camera data
    ├── training/camera_FRONT/          ← Train images
    ├── validation/camera_FRONT/        ← Val images
    └── annotations/
        ├── train.json
        └── val.json
```

---

## Advanced Usage

### Train on Subset (for testing)
```bash
# Modify config:
sed -i 's/max_samples: 0/max_samples: 100/' configs/waymo_finetune_config.yaml

# Or via CLI:
python src/train_waymo.py \
  --config configs/waymo_finetune_config.yaml \
  --model qwen2.5-vl-7b \
  --dry_run  # Load model, run 1 step, exit
```

### Use Different Model
```bash
# Switch to Cosmos R1-7B:
./scripts/train_waymo.sh cosmos-r1-7b

# Or MiMo-VL-7B:
./scripts/train_waymo.sh mimo-vl-7b
```

### DeepSpeed ZeRO-2 (for larger models)
```bash
# Uncomment in config:
# distributed:
#   deepspeed_config: "/scratch/rbaskar5/GLANCE/configs/ds_z2_config.json"

# Then train:
./scripts/train_waymo.sh qwen2.5-vl-7b
```

---

## Key Commands Summary

```bash
# Full pipeline (one command)
cd /scratch/rbaskar5/GLANCE
chmod +x setup_glance_waymo_qwen.sh
./setup_glance_waymo_qwen.sh both

# Or step-by-step
source /scratch/rbaskar5/set.bash
python scripts/prepare_waymo_front_camera.py
./scripts/train_waymo.sh qwen2.5-vl-7b

# Monitor
tail -f checkpoints/waymo/qwen2.5-vl-7b_train.log
nvidia-smi dmon

# Check results
ls -lh checkpoints/waymo_finetune/qwen2.5-vl-7b/
```

---

## Performance Notes

- **Data preparation**: 15-30 min (depends on I/O, network)
- **Model loading**: ~2-3 min per GPU
- **Training per epoch**: ~30-60 min (3 epochs total)
- **Total time**: ~3-5 hours for full training

---

## Support & Debugging

For issues, check:
1. **Logs**: `checkpoints/waymo_finetune/qwen2.5-vl-7b/training_log.txt`
2. **GPU status**: `nvidia-smi`
3. **Disk space**: `df -h /scratch/rbaskar5`
4. **Data**: `ls -la Dataset/waymo_front/annotations/`

Good luck with your Waymo + Qwen training! 🚀
