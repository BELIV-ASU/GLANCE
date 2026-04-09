#!/usr/bin/env python3
"""
Stage 1: QLoRA SFT for Qwen/QwQ-32B teacher on driving-manual reasoning data.

Input JSONL rows must contain:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

# Favor cublasLt kernels and avoid bf16 gemm instability on this cluster stack.
torch.backends.cuda.preferred_blas_library("cublaslt")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {idx} in {path}") from exc
    return rows


def as_text_chat(tokenizer, messages: List[Dict[str, Any]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


class PackedSFTCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        texts = [f["text"] for f in features]
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return enc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA teacher training for Qwen/QwQ-32B")
    p.add_argument("--model_name", default="Qwen/QwQ-32B")
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--max_length", type=int, default=2048)

    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=3)

    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb,
    )
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    raw = load_jsonl(args.train_jsonl)
    if args.max_samples > 0:
        raw = raw[: args.max_samples]

    train_rows = []
    for row in raw:
        messages = row.get("messages")
        if not messages:
            continue
        train_rows.append({"text": as_text_chat(tokenizer, messages)})

    dataset = Dataset.from_list(train_rows)
    collator = PackedSFTCollator(tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=False,
        fp16=True,
        tf32=True,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()

    adapter_dir = out_dir / "teacher_lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print(f"Saved teacher LoRA adapter to: {adapter_dir}")


if __name__ == "__main__":
    main()
