#!/usr/bin/env python3
"""
distill_4b.py  –  Distil trained QLoRA teacher (checkpoint-800) into a 4B student.

Teacher:  Qwen/Qwen2.5-VL-32B-Instruct  +  LoRA adapter  (default: checkpoints/checkpoint-800)
          Loaded 4-bit NF4, pinned to each rank’s local GPU.
Student:  Qwen/Qwen2.5-VL-7B-Instruct  –  trained via QLoRA adapters.
Loss:     alpha * CE(student, labels) + beta * KL(student || teacher, T)
Data:     data_drivelm/train.jsonl  (DriveLM messages with inline file:// images)

Multi-GPU (2×A100-80GB) via ’accelerate’:
  Each rank loads its own teacher + student copy (4-bit NF4, ~18GB + ~5GB per GPU).
  Data is split across ranks via DistributedSampler (≈2× throughput).
  LoRA gradients are synchronised with all_reduce after each backward pass.
  Teacher is frozen – only LoRA adapter params of the student are updated.
  Inference and saving run only on rank 0.

All paths use /scratch/rbaskar5  (no /home writes).

Launch:
  source /scratch/rbaskar5/set.bash
  accelerate launch --config_file /scratch/rbaskar5/GLANCE/configs/accelerate_2gpu.yaml \\
      /scratch/rbaskar5/GLANCE/src/distill_4b.py \\
      --out_dir /scratch/rbaskar5/GLANCE/distilled_student
"""

import os
import json
import math
import argparse
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import datetime
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from accelerate import Accelerator, InitProcessGroupKwargs

# -------------------------------------------------------------------------
# CUBLAS fix – same workaround as train_teacher.py / infer_checkpoint200.py
# Must be set before any CUDA ops.
# -------------------------------------------------------------------------
torch.backends.cuda.preferred_blas_library("cublaslt")

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as _ModelClass
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as _ModelClass

# ---------------------------------------------------------------------------
# Monkey-patch: fix CUBLAS_STATUS_INVALID_VALUE in M-RoPE (same as train_teacher)
# ---------------------------------------------------------------------------
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _qwen25vl

_orig_rope_fwd = _qwen25vl.Qwen2_5_VLRotaryEmbedding.forward

def _patched_rope_forward(self, x, position_ids):
    inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(
        3, position_ids.shape[1], -1, 1
    )
    position_ids_expanded = position_ids[:, :, None, :].float()
    device_type = (
        x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    )
    with torch.amp.autocast(device_type=device_type, enabled=False):
        freqs = torch.einsum(
            "abcd,abde->abce", inv_freq_expanded, position_ids_expanded
        ).transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

_qwen25vl.Qwen2_5_VLRotaryEmbedding.forward = _patched_rope_forward

# Optional PEFT (QLoRA for student)
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent           # /scratch/rbaskar5/GLANCE


# -------------------------------------------------------------------------
# Image helpers
# -------------------------------------------------------------------------

def _load_image_robust(img_path: str) -> "PILImage.Image":
    """Load PIL image; handles 16-bit depth PNGs saved by compute_depth.py."""
    pil = PILImage.open(img_path)
    if pil.mode in ("I;16", "I"):
        arr = np.array(pil, dtype=np.float32)
        lo, hi = arr.min(), arr.max()
        if hi - lo > 1e-6:
            arr = (arr - lo) / (hi - lo) * 255.0
        arr = arr.astype(np.uint8)
        pil = PILImage.fromarray(np.stack([arr, arr, arr], axis=-1), mode="RGB")
    else:
        pil = pil.convert("RGB")
    return pil


def _extract_images_from_messages(messages: List[Dict]) -> List["PILImage.Image"]:
    """Walk message content list and load every image item (file:// or plain path)."""
    imgs = []
    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            path = item.get("image", "")
            if path.startswith("file://"):
                path = path[7:]
            if os.path.isfile(path):
                try:
                    imgs.append(_load_image_robust(path))
                except Exception as exc:
                    print(f"  [warn] could not load image {path}: {exc}")
    return imgs


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------

class JsonlVLDataset(Dataset):
    def __init__(self, path: str, max_samples: int = 0):
        self.items: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.items.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if max_samples > 0:
            self.items = self.items[:max_samples]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


def _get_messages(sample: Dict[str, Any]) -> List[Dict]:
    """Return the messages list from a sample regardless of storage format."""
    raw = sample.get("messages_json") or sample.get("messages")
    if raw is None:
        raise KeyError("Sample must contain 'messages' or 'messages_json'.")
    return json.loads(raw) if isinstance(raw, str) else raw


# -------------------------------------------------------------------------
# Collator
# -------------------------------------------------------------------------

@dataclass
class DistillBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    pixel_values: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None


class QwenVLDistillCollator:
    """
    Collator for DriveLM JSONL data → DistillBatch.

    Key fixes over the original version:
    - Images are extracted from within message content (file:// items), not a
      separate top-level "image" key (which DriveLM does not have).
    - Does NOT pass truncation=True to the processor – that clips input_ids
      after image token expansion and causes a pixel_values shape mismatch.
      Instead we manually truncate AFTER the processor runs.
    - Does NOT pass None items in the images list; only passes images kwarg when
      the batch actually contains images.
    - Masks padding tokens in labels (sets pad_token_id positions → -100).
    - Optionally masks <think>…</think> spans in labels.
    """

    def __init__(
        self,
        processor,
        max_length: int,
        mask_think_tokens: bool = False,
    ):
        self.processor = processor
        self.max_length = max_length
        self.mask_think_tokens = mask_think_tokens
        try:
            tok = processor.tokenizer
            self._think_end_ids = tok.encode("</think>", add_special_tokens=False)
        except Exception:
            self._think_end_ids = []

    def _find_think_end(self, ids: List[int]) -> int:
        marker = self._think_end_ids
        if not marker:
            return 0
        for i in range(len(ids) - len(marker) + 1):
            if ids[i : i + len(marker)] == marker:
                return i + len(marker)
        return 0

    def __call__(self, batch: List[Dict[str, Any]]) -> DistillBatch:
        texts: List[str] = []
        all_images: List["PILImage.Image"] = []
        has_images = False

        for sample in batch:
            messages = _get_messages(sample)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

            imgs = _extract_images_from_messages(messages)
            if imgs:
                all_images.extend(imgs)
                has_images = True

        # Build processor kwargs – do NOT set truncation=True here (causes
        # pixel_values/input_ids length mismatch after image token expansion).
        proc_kwargs: Dict[str, Any] = dict(
            text=texts,
            padding=True,
            return_tensors="pt",
        )
        if has_images:
            proc_kwargs["images"] = all_images

        inputs = self.processor(**proc_kwargs)

        # Manual truncation (safe because image resolution is already capped
        # via max_pixels on the processor, so visual tokens are bounded).
        seq_len = inputs["input_ids"].size(1)
        if seq_len > self.max_length:
            for key, val in inputs.items():
                if isinstance(val, torch.Tensor) and val.ndim >= 2 and val.size(1) == seq_len:
                    inputs[key] = val[:, : self.max_length]

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Labels = input_ids; mask padding positions
        labels = input_ids.clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        # Optionally mask <think>…</think> (only loss on answer tokens)
        if self.mask_think_tokens and self._think_end_ids:
            for i in range(labels.size(0)):
                end_idx = self._find_think_end(input_ids[i].tolist())
                if end_idx > 0:
                    labels[i, :end_idx] = -100

        return DistillBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
        )


# -------------------------------------------------------------------------
# Losses
# -------------------------------------------------------------------------

def kl_div_student_teacher(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
    top_k: int = 2048,
) -> torch.Tensor:
    """Sparse top-K KL(student || teacher) on token distributions.

    Computing softmax over the full vocab [B, T, 152064] requires ~2 GB per
    forward pass at bfloat16.  Restricting to the teacher's top-K tokens
    reduces that by ~75x while preserving training signal (the teacher's
    probability mass is concentrated in its top tokens).
    """
    T = temperature
    # Get top-K indices from teacher (the tokens it cares about most)
    with torch.no_grad():
        topk_vals, topk_idx = teacher_logits.topk(top_k, dim=-1)  # [B, T, K]

    # Gather matching student logits at those positions
    s_topk = student_logits.gather(-1, topk_idx)   # [B, T, K]
    t_topk = topk_vals                              # [B, T, K]

    # Softmax restricted to top-K (normalise within the subset)
    log_p_s = F.log_softmax(s_topk.float() / T, dim=-1)
    p_t     = F.softmax(t_topk.float() / T, dim=-1)

    kl = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1)  # [B, T]
    mask = attention_mask.to(kl.dtype)
    kl = (kl * mask).sum() / mask.sum().clamp_min(1.0)
    return kl * (T * T)


def ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy, ignoring -100.  logits: [B, T, V], labels: [B, T]."""
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )


# -------------------------------------------------------------------------
# Model loading helpers
# -------------------------------------------------------------------------

def _bnb_config_4bit() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_teacher(args, local_rank: int = 0) -> "_ModelClass":
    """Load base Qwen2.5-VL-32B (4-bit) + checkpoint-800 LoRA adapter on local_rank GPU."""
    print(f"[teacher] rank={local_rank}  Loading: {args.teacher}")
    quant_cfg = _bnb_config_4bit() if args.teacher_4bit else None
    model = _ModelClass.from_pretrained(
        args.teacher,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
        device_map={"":local_rank},     # pin to this rank’s GPU
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    if args.teacher_adapter and os.path.isdir(args.teacher_adapter):
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft not installed – needed to load LoRA adapter: pip install peft")
        print(f"[teacher] Loading LoRA adapter: {args.teacher_adapter}")
        model = PeftModel.from_pretrained(model, args.teacher_adapter)
    else:
        print(f"[teacher] WARNING: adapter path not found ({args.teacher_adapter}); using base weights.")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_student(args, local_rank: int = 0):
    """Load student model (4-bit + QLoRA adapters) on local_rank GPU."""
    print(f"[student] rank={local_rank}  Loading: {args.student}")
    quant_cfg = _bnb_config_4bit() if args.student_4bit else None
    model = _ModelClass.from_pretrained(
        args.student,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
        device_map={"":local_rank},     # pin to this rank’s GPU
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    if args.use_qlora_student:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft not installed. pip install peft")
        if not args.student_4bit:
            raise RuntimeError("--use_qlora_student requires --student_4bit")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        if args.student_4bit:
            model = prepare_model_for_kbit_training(model)
        model.enable_input_require_grads()

    return model


# -------------------------------------------------------------------------
# Inference helpers (run after distillation on val.jsonl)
# -------------------------------------------------------------------------

def run_post_distil_inference(student, processor, val_jsonl: str, out_dir: str,
                              num_samples: int = 20, max_new_tokens: int = 256,
                              temperature: float = 0.7, seed: int = 42,
                              step: Optional[int] = None):
    """Run inference with the distilled student on DriveLM val.jsonl and save results.

    If *step* is given the results are written to  infer_result_{step}.json
    (in addition to the generic distilled_inference_results.json).
    """
    import random
    random.seed(seed)

    if not os.path.isfile(val_jsonl):
        print(f"[infer] val.jsonl not found at {val_jsonl}, skipping inference.")
        return

    samples = []
    with open(val_jsonl, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f"[infer] Loaded {len(samples)} val samples")

    if num_samples > 0 and num_samples < len(samples):
        samples = random.sample(samples, num_samples)

    student.eval()
    student_device = next(p for p in student.parameters()).device

    results = []
    for i, sample in enumerate(samples, 1):
        messages = _get_messages(sample)

        # Build generation messages (strip assistant turns)
        gen_messages = []
        gt_answer = ""
        for msg in messages:
            if msg["role"] == "assistant":
                content = msg.get("content", "")
                gt_answer = content if isinstance(content, str) else " ".join(
                    it.get("text", "") if isinstance(it, dict) else str(it)
                    for it in content
                )
                continue
            gen_messages.append(msg)

        # Load images
        pil_images = _extract_images_from_messages(gen_messages)

        text = processor.apply_chat_template(gen_messages, tokenize=False, add_generation_prompt=True)
        proc_kw: Dict[str, Any] = dict(text=[text], return_tensors="pt", padding=True)
        if pil_images:
            proc_kw["images"] = pil_images

        inputs = processor(**proc_kw).to(student_device)

        with torch.no_grad():
            out_ids = student.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
            )

        generated = out_ids[0][inputs["input_ids"].shape[1]:]
        response = processor.decode(generated, skip_special_tokens=True)

        print(f"{'='*60}")
        print(f"  [{i}/{len(samples)}]  GT: {gt_answer[:120]}")
        print(f"  PRED: {response[:300]}")

        results.append({
            "sample_idx": i,
            "ground_truth": gt_answer,
            "model_response": response,
            "image_available": len(pil_images) > 0,
        })

    # Save results to the project-level results/ directory
    results_dir = str(PROJECT_ROOT / "results")
    os.makedirs(results_dir, exist_ok=True)

    out_path = os.path.join(results_dir, "distilled_inference_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[infer] Results saved → {out_path}")

    # Also save a step-specific copy so earlier results are never overwritten
    if step is not None:
        step_path = os.path.join(results_dir, f"infer_result_{step}.json")
        with open(step_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[infer] Step-specific results saved → {step_path}")

    # Quick metrics
    exact = sum(1 for r in results if r["model_response"].strip() == r["ground_truth"].strip())
    contains = sum(
        1 for r in results
        if r["ground_truth"].strip().lower() in r["model_response"].strip().lower()
    )
    n = len(results)
    print(f"[infer] Exact match:    {exact}/{n}  ({100*exact/max(n,1):.1f}%)")
    print(f"[infer] Contains match: {contains}/{n}  ({100*contains/max(n,1):.1f}%)")


# -------------------------------------------------------------------------
# Resume helpers
# -------------------------------------------------------------------------

def find_latest_checkpoint(out_dir: str) -> Optional[str]:
    """Return the path of the highest-numbered step-N checkpoint, or None."""
    ckpt_root = os.path.join(out_dir, "checkpoints")
    if not os.path.isdir(ckpt_root):
        return None
    candidates = [
        d for d in os.listdir(ckpt_root)
        if d.startswith("step-") and
           os.path.isfile(os.path.join(ckpt_root, d, "distill_state.json"))
    ]
    if not candidates:
        return None
    latest = max(candidates, key=lambda x: int(x.split("-")[1]))
    return os.path.join(ckpt_root, latest)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    # ---- Model paths ----
    ap.add_argument("--teacher", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct")
    ap.add_argument("--teacher_adapter", type=str,
                    default=str(PROJECT_ROOT / "checkpoints" / "checkpoint-800"))
    ap.add_argument("--student", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out_dir", type=str,
                    default=str(PROJECT_ROOT / "distilled_student"))

    # ---- Data ----
    ap.add_argument("--data_dir", type=str,
                    default=str(PROJECT_ROOT / "data_drivelm"))
    ap.add_argument("--train_jsonl", type=str, default="")
    ap.add_argument("--val_jsonl",   type=str, default="")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--max_length",  type=int, default=2048)

    # ---- Training ----
    ap.add_argument("--epochs",       type=int,   default=1)
    ap.add_argument("--lr",           type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--batch_size",   type=int,   default=1)
    ap.add_argument("--grad_accum",   type=int,   default=8)
    ap.add_argument("--max_steps",    type=int,   default=-1)

    # ---- Distillation ----
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha_ce",    type=float, default=0.2)
    ap.add_argument("--beta_kl",     type=float, default=0.8)

    # ---- Quantisation / PEFT ----
    ap.add_argument("--teacher_4bit",      action="store_true", default=True)
    ap.add_argument("--student_4bit",      action="store_true", default=True)
    ap.add_argument("--use_qlora_student", action="store_true", default=True)
    ap.add_argument("--lora_r",       type=int,   default=16)
    ap.add_argument("--lora_alpha",   type=int,   default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # ---- Checkpointing / Resume ----
    ap.add_argument("--save_steps",       type=int, default=200,
                    help="Save a checkpoint every N update steps (0 = only save at end)")
    ap.add_argument("--save_total_limit", type=int, default=3,
                    help="Keep only the N most recent checkpoints (oldest deleted)")
    ap.add_argument("--resume_from",      type=str, default="auto",
                    help="Path to a step-N checkpoint dir to resume from, "
                         "'auto' to find the latest automatically, or 'none' to start fresh")

    # ---- Misc ----
    ap.add_argument("--mask_think_tokens", action="store_true")
    ap.add_argument("--seed",               type=int,   default=42)
    ap.add_argument("--log_every",          type=int,   default=10)
    ap.add_argument("--infer_after",        type=int,   default=0,
                        help="Run inference every N update steps (0 = only after full training)")
    ap.add_argument("--infer_samples",      type=int,   default=20)
    ap.add_argument("--infer_max_new_tokens",type=int,  default=256)
    ap.add_argument("--infer_temperature",  type=float, default=0.7)

    args = ap.parse_args()

    # -----------------------------------------------------------------------
    # Accelerator  –  handles process-group init, rank, device assignment.
    # Launched via:  accelerate launch --config_file accelerate_2gpu.yaml ...
    # Each rank runs this full function independently.
    # -----------------------------------------------------------------------
    # Set a 2-hour NCCL timeout so barriers survive rank-0 inference runs
    # (default 600 s is too short for 40-sample inference at ~30-60 s each).
    process_group_kwargs = InitProcessGroupKwargs(
        timeout=datetime.timedelta(seconds=7200)
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="bf16",
        kwargs_handlers=[process_group_kwargs],
    )
    local_rank = accelerator.local_process_index   # 0 or 1
    is_main   = accelerator.is_main_process        # True on rank 0 only
    aprint    = accelerator.print                  # silenced on ranks > 0

    torch.manual_seed(args.seed + local_rank)
    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Resolve JSONL paths
    train_jsonl = args.train_jsonl or os.path.join(args.data_dir, "train.jsonl")
    val_jsonl   = args.val_jsonl   or os.path.join(args.data_dir, "val.jsonl")

    if is_main and not os.path.isfile(train_jsonl):
        aprint(f"[data] Building dataset from DriveLM JSON...")
        from data import main as _build_data
        import sys as _sys
        _sys.argv = [
            "data.py",
            "--data_json", "/scratch/rbaskar5/Dataset/DriveLM/v1_1_train_nus.json",
            "--data_root", "/scratch/rbaskar5/Dataset/DriveLM",
            "--output_dir", args.data_dir,
            "--val_ratio", "0.05",
            "--seed", str(args.seed),
            "--with_depth",
        ]
        _build_data()
    accelerator.wait_for_everyone()   # ranks 1+ wait for rank 0 to build data

    # ---- Processor (load on every rank, weights are tiny) ----
    aprint(f"[setup] Loading processor: {args.teacher}")
    processor = AutoProcessor.from_pretrained(
        args.teacher,
        trust_remote_code=True,
        padding_side="right",
        min_pixels=3136,
        max_pixels=401408,
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ---- Teacher: base model + checkpoint-800 adapter, pinned to local GPU ----
    teacher = load_teacher(args, local_rank)

    # ---- Student: 4-bit + QLoRA adapters, pinned to local GPU ----
    student = load_student(args, local_rank)
    student.train()
    student_device = next(p for p in student.parameters()).device

    # ---- Dataset: split across ranks via DistributedSampler ----
    dataset = JsonlVLDataset(train_jsonl, max_samples=args.max_samples)
    aprint(f"[data] Total training samples: {len(dataset)}")
    aprint(f"[data] Samples per rank: {math.ceil(len(dataset) / accelerator.num_processes)}")

    sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        seed=args.seed,
    )
    collator = QwenVLDistillCollator(
        processor=processor,
        max_length=args.max_length,
        mask_think_tokens=args.mask_think_tokens,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=2,
        pin_memory=False,   # quantised model uses custom memory layout
    )

    # ---- Optimiser: only LoRA params ----
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # Steps are per-rank (each rank sees len(dataset) / num_processes samples)
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_update_steps = steps_per_epoch * args.epochs
    if args.max_steps > 0:
        total_update_steps = min(total_update_steps, args.max_steps)
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)

    # Prepare optimizer + scheduler via accelerate
    # (model NOT prepared – device_map is set; DDP wrapping is done manually via all_reduce)
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    # ---- Resume from checkpoint ----
    resume_step = 0
    if args.resume_from.lower() != "none":
        resume_dir = (
            find_latest_checkpoint(args.out_dir)
            if args.resume_from.lower() == "auto"
            else args.resume_from
        )
        if resume_dir and os.path.isdir(resume_dir):
            state_path = os.path.join(resume_dir, "distill_state.json")
            with open(state_path) as _f:
                resume_state = json.load(_f)
            resume_step = resume_state["global_step"]
            aprint(f"[resume] Resuming from {resume_dir}  (global_step={resume_step})")
            # Load LoRA adapter weights into student
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file as sf_load
            adapter_bin = os.path.join(resume_dir, "adapter_model.safetensors")
            adapter_bin_legacy = os.path.join(resume_dir, "adapter_model.bin")
            if os.path.isfile(adapter_bin):
                adapter_weights = sf_load(adapter_bin, device=str(student_device))
                set_peft_model_state_dict(student, adapter_weights)
                aprint(f"[resume] Loaded adapter weights from {adapter_bin}")
            elif os.path.isfile(adapter_bin_legacy):
                adapter_weights = torch.load(adapter_bin_legacy, map_location=student_device)
                set_peft_model_state_dict(student, adapter_weights)
                aprint(f"[resume] Loaded adapter weights from {adapter_bin_legacy}")
            else:
                aprint(f"[resume] WARNING: no adapter_model file found in {resume_dir}")
            # Load optimizer state
            opt_path = os.path.join(resume_dir, "optimizer.pt")
            if os.path.isfile(opt_path):
                optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
                aprint(f"[resume] Loaded optimizer state")
            # Load scheduler state
            sch_path = os.path.join(resume_dir, "scheduler.pt")
            if os.path.isfile(sch_path):
                scheduler.load_state_dict(torch.load(sch_path, map_location="cpu"))
                aprint(f"[resume] Loaded scheduler state")
        else:
            aprint("[resume] No checkpoint found — starting from scratch")

    # ---- Training loop ----
    global_step  = resume_step
    accum        = 0
    running_loss = 0.0
    # Batches to skip at the start of epoch 0 to fast-forward past already-done steps
    skip_batches = resume_step * args.grad_accum

    aprint(f"\n[train] 2×A100-80GB  |  ranks={accelerator.num_processes}  "
           f"epochs={args.epochs}  update_steps={total_update_steps}  "
           f"resume_step={resume_step}\n")

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)   # re-shuffle differently each epoch

        for batch_idx, batch in enumerate(loader):
            # Skip batches already processed before the resume point
            if epoch == 0 and batch_idx < skip_batches:
                if batch_idx % 500 == 0 and batch_idx > 0:
                    aprint(f"[resume] Fast-forwarding... {batch_idx}/{skip_batches} batches skipped")
                continue
            input_ids      = batch.input_ids.to(student_device)
            attention_mask = batch.attention_mask.to(student_device)
            labels         = batch.labels.to(student_device)

            fwd_kwargs: Dict[str, Any] = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            if batch.pixel_values is not None:
                fwd_kwargs["pixel_values"] = batch.pixel_values.to(student_device)
            if batch.image_grid_thw is not None:
                fwd_kwargs["image_grid_thw"] = batch.image_grid_thw.to(student_device)

            # Teacher forward (frozen) – keep logits in bfloat16 to save VRAM
            with torch.no_grad():
                t_out = teacher(**fwd_kwargs)
                teacher_logits = t_out.logits.detach()
                del t_out

            # Student forward
            s_out = student(**fwd_kwargs)
            student_logits = s_out.logits
            del s_out

            loss_ce = ce_loss(student_logits, labels)
            loss_kl = kl_div_student_teacher(
                student_logits, teacher_logits, attention_mask, args.temperature
            )
            # Free full-vocab logit tensors immediately after KL computation
            del teacher_logits, student_logits
            torch.cuda.empty_cache()

            loss = (args.alpha_ce * loss_ce + args.beta_kl * loss_kl) / args.grad_accum
            loss.backward()

            accum        += 1
            running_loss += loss.item()

            if accum % args.grad_accum == 0:
                # Sync LoRA gradients across all ranks (replaces DDP all_reduce)
                if accelerator.num_processes > 1:
                    for p in trainable_params:
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                if global_step % args.log_every == 0:
                    # Log rank-0's local loss – avoids an extra all_reduce collective
                    # that can race with the infer/save barriers at milestone steps.
                    avg = running_loss / args.log_every
                    aprint(
                        f"  step={global_step:5d}  loss={avg:.4f}"
                        f"  ce={loss_ce.item():.4f}  kl={loss_kl.item():.4f}"
                        f"  gpu{local_rank}={torch.cuda.memory_reserved(local_rank)//1024**3}GB"
                    )
                    running_loss = 0.0

                # ---- Mid-training checkpoint (rank 0 only) ----
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    accelerator.wait_for_everyone()
                    if is_main:
                        ckpt_dir = os.path.join(args.out_dir, "checkpoints", f"step-{global_step}")
                        os.makedirs(ckpt_dir, exist_ok=True)
                        student.save_pretrained(ckpt_dir)
                        processor.save_pretrained(ckpt_dir)
                        # Save optimizer + scheduler state for resumability
                        torch.save(optimizer.state_dict(),  os.path.join(ckpt_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(),  os.path.join(ckpt_dir, "scheduler.pt"))
                        # Write a small metadata file
                        with open(os.path.join(ckpt_dir, "distill_state.json"), "w") as _f:
                            json.dump({"global_step": global_step, "epoch": epoch}, _f)
                        aprint(f"  [ckpt] Saved → {ckpt_dir}")
                        # Prune old checkpoints (keep save_total_limit newest)
                        if args.save_total_limit > 0:
                            ckpt_root = os.path.join(args.out_dir, "checkpoints")
                            all_ckpts = sorted(
                                [d for d in os.listdir(ckpt_root) if d.startswith("step-")],
                                key=lambda x: int(x.split("-")[1])
                            )
                            for old in all_ckpts[: -args.save_total_limit]:
                                import shutil
                                shutil.rmtree(os.path.join(ckpt_root, old), ignore_errors=True)
                                aprint(f"  [ckpt] Removed old checkpoint: {old}")
                    # Re-sync so rank 1 waits for rank 0 to finish saving before
                    # continuing the training loop (avoids all_reduce timeout).
                    accelerator.wait_for_everyone()

                # ---- Mid-training inference (rank 0 only) ----
                if args.infer_after > 0 and global_step % args.infer_after == 0:
                    accelerator.wait_for_everyone()
                    if is_main:
                        aprint(f"\n[infer] Mid-training inference at step {global_step}...")
                        run_post_distil_inference(
                            student=student,
                            processor=processor,
                            val_jsonl=val_jsonl,
                            out_dir=args.out_dir,
                            num_samples=args.infer_samples,
                            max_new_tokens=args.infer_max_new_tokens,
                            temperature=args.infer_temperature,
                            seed=args.seed,
                            step=global_step,
                        )
                    # Re-sync so rank 1 waits for rank 0 to finish inference before
                    # continuing the training loop (avoids NCCL all_reduce timeout).
                    accelerator.wait_for_everyone()

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # ---- Save (rank 0 only) ----
    accelerator.wait_for_everyone()
    if is_main:
        aprint(f"\n[save] Saving distilled student → {args.out_dir}")
        student.save_pretrained(args.out_dir)
        processor.save_pretrained(args.out_dir)
        aprint(f"[save] Done.")

    # ---- Post-distillation inference (rank 0 only) ----
    if is_main:
        aprint(f"\n[infer] Running post-distillation inference on val.jsonl...")
        run_post_distil_inference(
            student=student,
            processor=processor,
            val_jsonl=val_jsonl,
            out_dir=args.out_dir,
            num_samples=args.infer_samples,
            max_new_tokens=args.infer_max_new_tokens,
            temperature=args.infer_temperature,
            seed=args.seed,
            step=global_step,
        )


if __name__ == "__main__":
    main()
