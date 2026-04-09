#!/usr/bin/env python3
"""
Stage 2: Distill QLoRA-trained QwQ-32B teacher into Qwen2.5-VL-7B student.

Teacher:
- Base: Qwen/QwQ-32B (text model)
- Optional adapter: output of train_teacher_qwq32b_qlora.py

Student:
- Qwen/Qwen2.5-VL-7B-Instruct (vision-language model)
- Distilled here in text-only mode from teacher reasoning traces

Input JSONL rows must contain a `messages` field.
"""

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.optimization import get_cosine_schedule_with_warmup

try:
    from transformers import AutoModelForVision2Seq as VisionModelClass
except ImportError:
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VisionModelClass
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration as VisionModelClass


class JsonlDataset(Dataset):
    def __init__(self, path: str):
        self.items: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    self.items.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at line {idx} in {path}") from exc

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def messages_to_prompt(tokenizer, messages: List[Dict[str, Any]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


@dataclass
class DistillBatch:
    teacher_input_ids: torch.Tensor
    teacher_attention_mask: torch.Tensor
    student_input_ids: torch.Tensor
    student_attention_mask: torch.Tensor
    labels: torch.Tensor


class ReasoningDistillCollator:
    def __init__(self, teacher_tokenizer, student_processor, max_length: int):
        self.teacher_tokenizer = teacher_tokenizer
        self.student_processor = student_processor
        self.student_tokenizer = student_processor.tokenizer
        if self.student_tokenizer.pad_token is None:
            self.student_tokenizer.pad_token = self.student_tokenizer.eos_token
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> DistillBatch:
        prompts: List[str] = []
        for row in batch:
            messages = row.get("messages")
            if not messages:
                raise KeyError("Each record must contain `messages`.")
            prompts.append(messages_to_prompt(self.teacher_tokenizer, messages))

        t_enc = self.teacher_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        s_enc = self.student_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        labels = s_enc["input_ids"].clone()
        labels[s_enc["attention_mask"] == 0] = -100

        return DistillBatch(
            teacher_input_ids=t_enc["input_ids"],
            teacher_attention_mask=t_enc["attention_mask"],
            student_input_ids=s_enc["input_ids"],
            student_attention_mask=s_enc["attention_mask"],
            labels=labels,
        )


def shift_for_causal(logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()
    return shift_logits, shift_labels, shift_mask


def ce_loss(shift_logits: torch.Tensor, shift_labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def align_vocab_for_kl(
    student_shift_logits: torch.Tensor,
    teacher_shift_logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    s_vocab = student_shift_logits.size(-1)
    t_vocab = teacher_shift_logits.size(-1)
    v = min(s_vocab, t_vocab)
    return student_shift_logits[..., :v], teacher_shift_logits[..., :v]


def kl_loss(
    student_shift_logits: torch.Tensor,
    teacher_shift_logits: torch.Tensor,
    shift_labels: torch.Tensor,
    shift_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    s_logits, t_logits = align_vocab_for_kl(student_shift_logits, teacher_shift_logits)
    t = temperature
    s_logits = s_logits / t
    t_logits = t_logits / t

    log_p_s = F.log_softmax(s_logits, dim=-1)
    p_t = F.softmax(t_logits, dim=-1)
    token_kl = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1)

    valid = (shift_mask > 0) & (shift_labels != -100)
    valid = valid.to(token_kl.dtype)
    denom = valid.sum().clamp_min(1.0)
    return ((token_kl * valid).sum() / denom) * (t * t)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distill QwQ-32B teacher into Qwen2.5-VL-7B")
    p.add_argument("--teacher_base", default="Qwen/QwQ-32B")
    p.add_argument("--teacher_adapter", default="", help="Optional LoRA adapter path")
    p.add_argument("--student", default="Qwen/Qwen2.5-VL-7B-Instruct")

    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--max_length", type=int, default=2048)

    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)

    p.add_argument("--alpha_ce", type=float, default=0.2)
    p.add_argument("--beta_kl", type=float, default=0.8)
    p.add_argument("--temperature", type=float, default=2.0)

    p.add_argument("--teacher_4bit", action="store_true")
    p.add_argument("--student_4bit", action="store_true")

    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_teacher_model(args: argparse.Namespace):
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_base, trust_remote_code=True)
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    quant = None
    if args.teacher_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_base,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quant,
    )

    if args.teacher_adapter:
        teacher = PeftModel.from_pretrained(teacher, args.teacher_adapter)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    return teacher, teacher_tokenizer


def build_student_model(args: argparse.Namespace):
    quant = None
    if args.student_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    processor = AutoProcessor.from_pretrained(args.student, trust_remote_code=True)
    student = VisionModelClass.from_pretrained(
        args.student,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quant,
    )
    student.train()
    return student, processor


def save_metrics(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    teacher, teacher_tokenizer = build_teacher_model(args)
    student, student_processor = build_student_model(args)

    dataset = JsonlDataset(args.train_jsonl)
    collator = ReasoningDistillCollator(teacher_tokenizer, student_processor, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    total_updates = math.ceil(len(loader) / args.grad_accum) * args.epochs
    if args.max_steps > 0:
        total_updates = min(total_updates, args.max_steps)
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_updates)

    teacher_device = next(iter(teacher.parameters())).device
    student_device = next(iter(student.parameters())).device

    global_step = 0
    accum = 0
    running = 0.0

    print(f"Dataset size: {len(dataset)}")
    print(f"Total optimizer updates: {total_updates}")

    for epoch in range(args.epochs):
        for batch in loader:
            t_kwargs = {
                "input_ids": batch.teacher_input_ids.to(teacher_device),
                "attention_mask": batch.teacher_attention_mask.to(teacher_device),
            }
            s_kwargs = {
                "input_ids": batch.student_input_ids.to(student_device),
                "attention_mask": batch.student_attention_mask.to(student_device),
            }

            with torch.no_grad():
                t_out = teacher(**t_kwargs)
            s_out = student(**s_kwargs)

            shift_s_logits, shift_labels, shift_mask = shift_for_causal(
                s_out.logits,
                batch.labels.to(student_device),
                batch.student_attention_mask.to(student_device),
            )

            shift_t_logits = t_out.logits[:, :-1, :].contiguous()
            if shift_t_logits.device != shift_s_logits.device:
                shift_t_logits = shift_t_logits.to(shift_s_logits.device)

            loss_ce = ce_loss(shift_s_logits, shift_labels)
            loss_kl = kl_loss(
                shift_s_logits,
                shift_t_logits,
                shift_labels,
                shift_mask,
                temperature=args.temperature,
            )
            loss = args.alpha_ce * loss_ce + args.beta_kl * loss_kl
            loss = loss / args.grad_accum

            loss.backward()
            accum += 1
            running += loss.item()

            if accum % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                if global_step % args.log_every == 0:
                    avg = running / args.log_every
                    print(
                        f"step={global_step} loss={avg:.4f} ce={loss_ce.item():.4f} kl={loss_kl.item():.4f}"
                    )
                    save_metrics(
                        metrics_path,
                        {
                            "step": global_step,
                            "loss": float(avg),
                            "ce": float(loss_ce.item()),
                            "kl": float(loss_kl.item()),
                            "lr": float(scheduler.get_last_lr()[0]),
                            "epoch": epoch,
                        },
                    )
                    running = 0.0

                if global_step % args.save_every == 0:
                    ckpt = out_dir / f"checkpoint-{global_step}"
                    ckpt.mkdir(parents=True, exist_ok=True)
                    student.save_pretrained(ckpt)
                    student_processor.save_pretrained(ckpt)
                    print(f"Saved checkpoint to {ckpt}")

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(final_dir)
    student_processor.save_pretrained(final_dir)
    print(f"Final distilled student saved to: {final_dir}")


if __name__ == "__main__":
    main()
