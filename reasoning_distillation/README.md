# Reasoning Distillation Pipeline

This folder contains a 2-stage training flow you requested:

1. QLoRA-train `Qwen/QwQ-32B` teacher on DrivingManual data.
2. Distill that trained teacher into `Qwen/Qwen2.5-VL-7B-Instruct`.

## Files

- `prepare_reasoning_data.py`: merges your 3 source files into training JSONL.
- `train_teacher_qwq32b_qlora.py`: stage-1 QLoRA teacher training.
- `train_qwen7b_reasoning_distill.py`: stage-2 teacher->student distillation.
- `eval_qwen7b_reasoning_student.py`: inference/eval for distilled student.
- `scripts/run_prepare_data.sh`: data prep with your module loads.
- `scripts/run_stage1_teacher_qlora.sh`: stage-1 run script.
- `scripts/run_stage2_distill_to_qwen25vl7b.sh`: stage-2 run script.
- `scripts/run_all_reasoning_distillation.sh`: full pipeline runner.

## Source Data Used

- `/scratch/jnolas77/SafetyVLM/DrivingManual/california_driving_sft/sft_output/california_driving_sft.jsonl`
- `/scratch/jnolas77/SafetyVLM/DrivingManual/wa_driver_sft_data/final_package/sft_training_data.jsonl`
- `/scratch/jnolas77/SafetyVLM/DrivingManual/wa_driver_sft_data/final_package/image_manifest.json`

## Run Order

```bash
cd /scratch/jnolas77/SafetyVLM/reasoning_distillation
bash scripts/run_prepare_data.sh
bash scripts/run_stage1_teacher_qlora.sh
bash scripts/run_stage2_distill_to_qwen25vl7b.sh
```

Or one command:

```bash
cd /scratch/jnolas77/SafetyVLM/reasoning_distillation
bash scripts/run_all_reasoning_distillation.sh
```

## Notes

- Stage 1 saves adapter at:
  `/scratch/jnolas77/SafetyVLM/reasoning_distillation/checkpoints/teacher_qwq32b_qlora/teacher_lora_adapter`
- Stage 2 reads that adapter via `--teacher_adapter`.
- Distillation is text-mode using reasoning traces; student is still `Qwen2.5-VL-7B-Instruct`.
- You can increase epochs and reduce LR once the dry run is stable.
