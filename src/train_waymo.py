#!/usr/bin/env python3
"""
train_waymo.py  –  Fine-tune 7B VLMs on Waymo images for traffic understanding
═══════════════════════════════════════════════════════════════════════════════

Supports any model in the VLM factory registry (Qwen2.5-VL-7B, Cosmos R1-7B,
MiMo-VL-7B, OpenVLA-7B, VILA-7B) using LoRA adapters on 2× A100-80GB.

Key features:
  • Config-driven (YAML)   – all hyper-params in waymo_finetune_config.yaml
  • LoRA fine-tuning        – parameter-efficient, bf16 full-precision base
  • Multi-GPU               – HF Accelerate / DeepSpeed ZeRO-2
  • Gradient checkpointing  – fits 7B models with images in 80GB
  • CUBLAS workarounds       – safe on CUDA 12.8 + PyTorch 2.10

Usage:
  # Single-GPU
  python src/train_waymo.py --config configs/waymo_finetune_config.yaml

  # 2× A100 via Accelerate
  accelerate launch --config_file configs/accelerate_2gpu.yaml \\
      src/train_waymo.py --config configs/waymo_finetune_config.yaml

  # Override model on the fly
  accelerate launch ... src/train_waymo.py \\
      --config configs/waymo_finetune_config.yaml \\
      --model mimo-vl-7b
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# ── CUBLAS workaround (before any model import) ─────────────────────────
torch.backends.cuda.preferred_blas_library("cublaslt")
os.environ.setdefault("TORCH_BLAS_PREFER_CUBLASLT", "1")

import yaml
from torch.utils.data import DataLoader, DistributedSampler

from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint

# Local imports
from vlm_factory import get_vlm_and_processor, apply_lora
from waymo_dataset import WaymoImageDataset, WaymoCollator

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════════════════

def load_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load YAML config and apply CLI overrides."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply overrides (flat key=value from CLI)
    if overrides:
        for dotted_key, value in overrides.items():
            keys = dotted_key.split(".")
            d = cfg
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value

    return cfg


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune 7B VLMs on Waymo images"
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/waymo_finetune_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override model.name from config (e.g. 'mimo-vl-7b')",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Override training.output_dir from config",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=None,
        help="Override training.num_train_epochs",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from checkpoint path (or 'auto' to find latest)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Load model & data, run 1 step, then exit (for testing)",
    )

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
#  Model & data setup
# ═══════════════════════════════════════════════════════════════════════════

def setup_model(cfg: Dict[str, Any]):
    """Load model + processor + LoRA."""
    model_name = cfg["model"]["name"]
    log.info("Loading model: %s", model_name)

    model, processor = get_vlm_and_processor(model_name, cfg)

    # Disable cache for training
    model.config.use_cache = False

    # Enable gradient checkpointing
    training_cfg = cfg.get("training", {})
    if training_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Apply LoRA
    lora_cfg = cfg.get("lora", {})
    if lora_cfg.get("enabled", True):
        model = apply_lora(model, lora_cfg, model_name=model_name)

    return model, processor


def setup_data(cfg: Dict[str, Any], processor: Any):
    """Build train/val datasets and collator."""
    data_cfg = cfg["data"]
    model_name = cfg["model"]["name"].lower()

    # Datasets
    train_ds = WaymoImageDataset(data_cfg, split="train")
    val_ds = WaymoImageDataset(data_cfg, split="val")

    log.info("Train: %d samples  |  Val: %d samples", len(train_ds), len(val_ds))

    # Collator (processor-aware)
    collator = WaymoCollator(
        processor=processor,
        max_length=data_cfg.get("max_seq_length", 2048),
        model_name=model_name,
    )

    return train_ds, val_ds, collator


# ═══════════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════════

def build_training_args(cfg: Dict[str, Any], output_dir: str) -> TrainingArguments:
    """
    Build HuggingFace TrainingArguments from config.

    Maps config keys to HF TrainingArguments fields.
    """
    t = cfg.get("training", {})

    return TrainingArguments(
        # Output
        output_dir=output_dir,

        # Epochs / steps
        num_train_epochs=t.get("num_train_epochs", 10),
        max_steps=t.get("max_steps", -1),

        # Batch size
        per_device_train_batch_size=t.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 8),

        # Optimiser
        learning_rate=t.get("learning_rate", 2e-5),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        warmup_ratio=t.get("warmup_ratio", 0.05),
        weight_decay=t.get("weight_decay", 0.01),
        max_grad_norm=t.get("max_grad_norm", 1.0),
        optim=t.get("optim", "adamw_torch"),

        # Precision
        bf16=t.get("bf16", True),
        tf32=t.get("tf32", True),

        # Gradient checkpointing
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Logging
        logging_steps=t.get("logging_steps", 5),
        logging_first_step=True,
        report_to=t.get("report_to", "none"),

        # Saving
        save_steps=t.get("save_steps", 200),
        save_total_limit=t.get("save_total_limit", 3),

        # Evaluation
        # load_best_model_at_end=True causes OOM on large quantized models (32B 4-bit)
        # because the Trainer tries to reload the checkpoint while the model is in GPU
        # memory.  Keep it False; use the saved LoRA adapter directly instead.
        eval_strategy=t.get("eval_strategy", "steps"),
        eval_steps=t.get("eval_steps", 200),
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Data
        dataloader_num_workers=t.get("dataloader_num_workers", 2),
        dataloader_pin_memory=True,
        remove_unused_columns=False,

        # Misc
        seed=t.get("seed", 42),
        data_seed=t.get("seed", 42),
    )


def train(
    model,
    processor,
    train_ds,
    val_ds,
    collator,
    cfg: Dict[str, Any],
    resume_from: Optional[str] = None,
    dry_run: bool = False,
):
    """Run the HF Trainer loop."""
    training_cfg = cfg.get("training", {})
    output_dir = training_cfg.get(
        "output_dir", "/scratch/rbaskar5/GLANCE/checkpoints/waymo"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save config alongside checkpoints for reproducibility
    cfg_save_path = os.path.join(output_dir, "config_snapshot.yaml")
    with open(cfg_save_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    log.info("Config snapshot saved to %s", cfg_save_path)

    # Training arguments
    training_args = build_training_args(cfg, output_dir)

    # Dry-run: override to 1 step
    if dry_run:
        training_args.max_steps = 1
        training_args.num_train_epochs = 1
        training_args.eval_strategy = "no"
        training_args.save_strategy = "no"
        training_args.logging_steps = 1
        log.info("DRY RUN: will train 1 step and exit")

    # Callbacks
    callbacks = []
    if training_cfg.get("early_stopping_patience", 0) > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=training_cfg["early_stopping_patience"]
            )
        )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if len(val_ds) > 0 else None,
        data_collator=collator,
        callbacks=callbacks,
    )

    # Auto-detect checkpoint
    if resume_from == "auto":
        last_ckpt = get_last_checkpoint(output_dir)
        if last_ckpt:
            log.info("Auto-resuming from: %s", last_ckpt)
            resume_from = last_ckpt
        else:
            log.info("No checkpoint found for auto-resume; training from scratch")
            resume_from = None

    # ── Train ────────────────────────────────────────────────────────────
    log.info("=" * 64)
    log.info("  Waymo VLM Fine-tuning  –  %s", cfg["model"]["name"])
    log.info("  GPUs: %d  |  Effective batch: %d",
             max(1, torch.cuda.device_count()),
             training_args.per_device_train_batch_size
             * training_args.gradient_accumulation_steps
             * max(1, torch.cuda.device_count()))
    log.info("=" * 64)

    train_result = trainer.train(resume_from_checkpoint=resume_from)

    # ── Metrics ──────────────────────────────────────────────────────────
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_ds)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    if not dry_run:
        # ── Save final model ─────────────────────────────────────────────
        final_dir = os.path.join(output_dir, "final_lora")
        log.info("Saving final LoRA adapter to %s", final_dir)
        trainer.save_model(final_dir)
        processor.save_pretrained(final_dir)

        # NOTE: merge_and_unload() on a 4-bit quantized 32B model dequantizes all
        # weights to bf16 (~64 GB), causing OOM on 2×A100-80GB.  Skip the in-process
        # merge; use the offline merge script or load the adapter at inference time:
        #
        #   from peft import PeftModel
        #   base = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_id, ...)
        #   model = PeftModel.from_pretrained(base, final_lora_dir)
        #   merged = model.merge_and_unload()   # run on a CPU-offload node
        log.info("Skipping in-process LoRA merge to avoid OOM on large model.")
        log.info("Use the LoRA adapter at %s with the base model for inference.", final_dir)

    log.info("Training complete!")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Load config ──────────────────────────────────────────────────────
    cfg = load_config(args.config)

    # CLI overrides
    if args.model:
        cfg["model"]["name"] = args.model
    if args.output_dir:
        cfg["training"]["output_dir"] = args.output_dir
    if args.num_epochs:
        cfg["training"]["num_train_epochs"] = args.num_epochs

    # ── Environment info ─────────────────────────────────────────────────
    log.info("Python:       %s", sys.version.split()[0])
    log.info("PyTorch:      %s", torch.__version__)
    log.info("CUDA:         %s", torch.version.cuda or "N/A")
    log.info("GPU count:    %d", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        log.info("  GPU %d: %s  (%.1f GB)",
             i, torch.cuda.get_device_name(i),
             torch.cuda.get_device_properties(i).total_memory / 1e9)

    # ── Distributed rank-to-GPU binding ──────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        log.info(f"Set CUDA device to local_rank={local_rank}")

    # ── Setup ────────────────────────────────────────────────────────────
    model, processor = setup_model(cfg)
    train_ds, val_ds, collator = setup_data(cfg, processor)

    # ── Train ────────────────────────────────────────────────────────────
    train(
        model=model,
        processor=processor,
        train_ds=train_ds,
        val_ds=val_ds,
        collator=collator,
        cfg=cfg,
        resume_from=args.resume,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
