#!/usr/bin/env python3
"""
distill_4b.py
Knowledge transfer (teacher->student) for Qwen3-VL style models.

- Teacher: frozen, typically 30B (optionally loaded in 4-bit for memory)
- Student: 4B (can be full-finetune or QLoRA via PEFT)
- Loss: alpha * CE(student, labels) + beta * KL(student || teacher, T)

Input dataset: JSONL with fields:
{
  "messages": [...],          # chat messages in Qwen format OR your own (see build_prompt)
  "image": "path/to/img.jpg"  # optional, can be null/absent
}

If your dataset currently stores "text" rather than "messages", adapt build_prompt().
"""

import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

# Optional QLoRA for student
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


# -------------------------
# Data
# -------------------------

class JsonlVLDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.items.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


def load_image_maybe(image_path: Optional[str]) -> Optional["Image.Image"]:
    if not image_path:
        return None
    if not PIL_AVAILABLE:
        raise RuntimeError("PIL not installed but dataset contains images. pip install pillow")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def build_prompt(sample: Dict[str, Any]) -> Any:
    """
    Returns messages structure suitable for processor.apply_chat_template(...).
    Adapt if your jsonl schema differs.
    """
    if "messages" in sample:
        return sample["messages"]

    # Fallback: if you have plain text
    if "text" in sample:
        # Put it into a single-user message
        return [{"role": "user", "content": [{"type": "text", "text": sample["text"]}]}]

    raise KeyError("Sample must contain either 'messages' or 'text'.")


@dataclass
class DistillBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    pixel_values: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None


class QwenVLDistillCollator:
    def __init__(
        self,
        processor,
        max_length: int,
        mask_think_tokens: bool = False,
        think_start: str = "<think>",
        think_end: str = "</think>",
    ):
        self.processor = processor
        self.max_length = max_length
        self.mask_think_tokens = mask_think_tokens
        self.think_start = think_start
        self.think_end = think_end

        # Token ids for masking (best-effort; depends on tokenizer)
        self._think_start_ids = None
        self._think_end_ids = None
        try:
            tok = processor.tokenizer
            self._think_start_ids = tok.encode(think_start, add_special_tokens=False)
            self._think_end_ids = tok.encode(think_end, add_special_tokens=False)
        except Exception:
            pass

    def _mask_think(self, labels: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.mask_think_tokens:
            return labels
        if not self._think_start_ids or not self._think_end_ids:
            return labels  # can't reliably mask

        # For each sequence, find first <think> ... </think> span and mask it out in labels.
        # This is intentionally conservative.
        bs, seqlen = input_ids.shape
        for i in range(bs):
            seq = input_ids[i].tolist()

            def find_subseq(hay, needle):
                n = len(needle)
                for j in range(0, len(hay) - n + 1):
                    if hay[j : j + n] == needle:
                        return j
                return -1

            s = find_subseq(seq, self._think_start_ids)
            if s == -1:
                continue
            e = find_subseq(seq[s + len(self._think_start_ids):], self._think_end_ids)
            if e == -1:
                continue
            e = s + len(self._think_start_ids) + e + len(self._think_end_ids)

            labels[i, s:e] = -100
        return labels

    def __call__(self, batch: List[Dict[str, Any]]) -> DistillBatch:
        # Build chat template text
        messages_list = [build_prompt(x) for x in batch]

        # Try to include images if present.
        # Qwen-VL processors typically accept a list of PIL images or None.
        images = []
        for x in batch:
            img = None
            if "image" in x and x["image"]:
                img = load_image_maybe(x["image"])
            images.append(img)

        # Apply chat template to get text prompt
        # If your processor doesn't support apply_chat_template, adapt here.
        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in messages_list
        ]

        # Processor handles multimodal packing.
        # Some Qwen-VL processors require images only for items that have them.
        # We pass full list with Nones.
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = input_ids.clone()

        labels = self._mask_think(labels, input_ids)

        # Pass through vision tensors if present
        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)

        return DistillBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )


# -------------------------
# Losses
# -------------------------

def kl_div_student_teacher(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    KL(student || teacher) on token distributions.
    - Both logits: [B, T, V]
    - Mask padding positions
    """
    T = temperature
    s = student_logits / T
    t = teacher_logits / T

    # log_probs student, probs teacher
    log_p_s = F.log_softmax(s, dim=-1)
    p_t = F.softmax(t, dim=-1)

    # token-wise KL: sum_v p_t * (log p_t - log p_s)
    # F.kl_div expects input=log_p_s, target=p_t
    kl = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1)  # [B, T]

    mask = attention_mask.to(kl.dtype)  # [B, T]
    kl = (kl * mask).sum() / mask.sum().clamp_min(1.0)

    # Scale by T^2 (standard distillation)
    return kl * (T * T)


def ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy, ignoring -100.
    logits: [B, T, V], labels: [B, T]
    """
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )


# -------------------------
# Main train
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=str, required=True, help="e.g., Qwen/Qwen3-VL-30B-A3B-Instruct")
    ap.add_argument("--student", type=str, required=True, help="e.g., Qwen/Qwen3-VL-4B-Instruct (or your 4B)")
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    # Training hyperparams
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=-1)

    # Distillation weights
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha_ce", type=float, default=0.2)
    ap.add_argument("--beta_kl", type=float, default=0.8)

    # Teacher loading options
    ap.add_argument("--teacher_4bit", action="store_true", help="load teacher in 4-bit to save VRAM")
    ap.add_argument("--student_4bit", action="store_true", help="(optional) load student in 4-bit + QLoRA")
    ap.add_argument("--use_qlora_student", action="store_true", help="train student via QLoRA adapters")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # Mask think
    ap.add_argument("--mask_think_tokens", action="store_true")

    # Misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=10)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Processor: use teacher's processor (usually same family)
    processor = AutoProcessor.from_pretrained(args.teacher, trust_remote_code=True)

    # Teacher
    t_quant = None
    if args.teacher_4bit:
        t_quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    teacher = AutoModelForVision2Seq.from_pretrained(
        args.teacher,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=t_quant,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Student
    s_quant = None
    if args.student_4bit:
        s_quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    student = AutoModelForVision2Seq.from_pretrained(
        args.student,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=s_quant,
    )

    if args.use_qlora_student:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft not installed. pip install peft")
        if not args.student_4bit:
            raise RuntimeError("For QLoRA student, pass --student_4bit")
        student = prepare_model_for_kbit_training(student)

        # NOTE: target_modules may differ by Qwen3-VL implementation;
        # these are common for Qwen-family transformers.
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        student = get_peft_model(student, lora_cfg)
        student.print_trainable_parameters()

    student.train()

    dataset = JsonlVLDataset(args.train_jsonl)
    collator = QwenVLDistillCollator(
        processor=processor,
        max_length=args.max_length,
        mask_think_tokens=args.mask_think_tokens,
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    # Build optimizer only on trainable params (important for QLoRA)
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # Steps
    total_update_steps = math.ceil(len(loader) / args.grad_accum) * args.epochs
    if args.max_steps > 0:
        total_update_steps = min(total_update_steps, args.max_steps)

    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)

    # Pick device: rely on model.device (device_map="auto" can place shards; we move inputs to student first device)
    # Best-effort: use the first parameter device for student forward inputs
    student_device = next(iter(student.parameters())).device

    global_step = 0
    accum = 0
    running = 0.0

    for epoch in range(args.epochs):
        for batch in loader:
            # Move batch to student device (for sharded models, HF will route internally)
            input_ids = batch.input_ids.to(student_device)
            attention_mask = batch.attention_mask.to(student_device)
            labels = batch.labels.to(student_device)

            kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            if batch.pixel_values is not None:
                kwargs["pixel_values"] = batch.pixel_values.to(student_device)
            if batch.image_grid_thw is not None:
                kwargs["image_grid_thw"] = batch.image_grid_thw.to(student_device)

            with torch.no_grad():
                t_out = teacher(**kwargs)
                teacher_logits = t_out.logits.detach()

            s_out = student(**kwargs)
            student_logits = s_out.logits

            loss_ce = ce_loss(student_logits, labels)
            loss_kl = kl_div_student_teacher(student_logits, teacher_logits, attention_mask, args.temperature)
            loss = args.alpha_ce * loss_ce + args.beta_kl * loss_kl
            loss = loss / args.grad_accum

            loss.backward()
            accum += 1
            running += loss.item()

            if accum % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                if global_step % args.log_every == 0:
                    avg = running / args.log_every
                    print(f"step={global_step} loss={avg:.4f} (ce={loss_ce.item():.4f} kl={loss_kl.item():.4f})")
                    running = 0.0

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # Save student (adapters if QLoRA; full if not)
    print("Saving...")
    student.save_pretrained(args.out_dir)
    processor.save_pretrained(args.out_dir)
    print(f"Done. Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
