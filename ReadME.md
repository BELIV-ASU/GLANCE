# Setup on SOL
```
module load ffmpeg-6.0-gcc-12.1.0
module load cuda-12.9.0-gcc-12.1.0
module load cmake/3.30.2
module load mamba/latest
```

# Project Structure
```
GLANCE/
├── configs/                  # accelerate & deepspeed configs
│   ├── accelerate_2gpu.yaml
│   ├── accelerate_z2.yaml
│   └── ds_z2_config.json
├── scripts/                  # shell launchers
│   ├── run_distill.sh
│   ├── train_teacher.sh
│   └── infer.sh
├── src/                      # all Python source
│   ├── distill_4b.py
│   ├── train_teacher.py
│   ├── data.py
│   ├── safety_data.py
│   ├── compute_depth.py
│   ├── infer_checkpoint200.py
│   ├── infer_distilled_student.py
│   ├── infer_teacher.py
│   └── visualize_results.py
├── results/                  # all inference JSONs + viz dirs
│   ├── infer_result_200.json
│   ├── infer_result_400.json
│   ├── infer_result_600.json
│   ├── inference_results_ckpt200.json
│   ├── viz_ckpt200/
│   ├── viz_step200/ ... viz_step600/
│   └── viz_distilled/
├── logs/                     # log files + plots
│   ├── depth_compute.log
│   ├── infer_step400.log
│   └── *.png
├── data_drivelm/             # train/val JSONL
├── checkpoints/              # teacher LoRA checkpoints
├── distilled_student/        # student LoRA checkpoints
├── ReadME.md
└── progress.md
```

# Training cmd
```
chmod +x scripts/train_teacher.sh
./scripts/train_teacher.sh
```

# Distill cmd
```
chmod +x scripts/run_distill.sh
./scripts/run_distill.sh
```

# Inference cmd
```
chmod +x scripts/infer.sh
./scripts/infer.sh
```

# Datasets and checkpoints
[Dataset here](https://www.dropbox.com/scl/fo/d6wpk16zpd7vxbl4c8ncu/APyLVdMgb0RX2qCf9uLWrTI?rlkey=p7xjrap2w1j3bja5js4bkxv1u&st=vu9cwbqj&dl=0)
