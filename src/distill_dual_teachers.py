#!/usr/bin/env python3
"""
distill_dual_teachers.py
────────────────────────────────────────────────────────────────────────────
Dual-teacher logit distillation into Qwen2.5-VL-7B student.

Teachers
  • Vision teacher  : Qwen2.5-VL-32B-Instruct + Waymo LoRA adapter
                      (base + adapter; handles multimodal inputs)
  • Language teacher: QwQ-32B + reasoning QLoRA adapter
                      (text-only; distils structured reasoning)

Student
  • Qwen2.5-VL-7B-Instruct, wrapped in LoRA for parameter-efficient training

Loss
  L = α_v · KL(student ∥ vision_teacher; T)
    + α_l · KL(student ∥ language_teacher; T)
    + α_ce · CE(student, hard_labels)

Both teachers are loaded in 4-bit (QLoRA), frozen, eval-mode.
The student is trained in bf16 with LoRA.

Data
  Accepts two optional JSONL paths that are merged at load time:
  --waymo_jsonl   : multimodal Waymo trajectory records
                    messages contain [{"type":"image"}, {"type":"text",...}]
  --reason_jsonl  : text-only reasoning distillation records
                    messages contain [{"role":..., "content":str}, ...]

  Each record must have a "messages" key.  Image records must also have an
  "image_path" key (relative to --image_root) OR embed the image path inside
  the messages content list (auto-detected).

Usage
  python src/distill_dual_teachers.py \\
    --vision_base  Qwen/Qwen2.5-VL-32B-Instruct \\
    --vision_lora  checkpoints/waymo/qwen2.5-vl-32b-trajectory/final_lora \\
    --lang_base    Qwen/QwQ-32B \\
    --lang_lora    reasoning_distillation/checkpoints/teacher_qwq32b_qlora/teacher_lora_adapter \\
    --student      Qwen/Qwen2.5-VL-7B-Instruct \\
    --waymo_jsonl  data/waymo_trajectory/train.jsonl \\
    --reason_jsonl reasoning_distillation/data/distill_train_qwen25vl7b.jsonl \\
    --image_root   /scratch/rbaskar5/Dataset/waymo_front \\
    --out_dir      checkpoints/dual_distilled_7b \\
    --epochs 1 --batch_size 1 --grad_accum 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
import peft.tuners.lora.inc as _peft_inc
_peft_inc.is_inc_available = lambda: False  # neural_compressor installed but .torch submodule missing
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.optimization import get_cosine_schedule_with_warmup

# ── Qwen2.5-VL import (handles multiple transformers versions) ───────────
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as Qwen25VL
except ImportError:
    try:
        from transformers import AutoModelForVision2Seq as Qwen25VL  # type: ignore[assignment]
    except ImportError:
        from transformers import AutoModelForCausalLM as Qwen25VL  # type: ignore[assignment]

# ── Apply Qwen2.5-VL RoPE patch (CUBLAS workaround) ──────────────────────
torch.backends.cuda.preferred_blas_library("cublaslt")
os.environ.setdefault("TORCH_BLAS_PREFER_CUBLASLT", "1")

_ROPE_PATCHED = False


def _patch_qwen25vl_rope() -> None:
    global _ROPE_PATCHED
    if _ROPE_PATCHED:
        return
    try:
        import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _qmod

        def _patched(self, x, position_ids):
            inv_freq_expanded = (
                self.inv_freq[None, None, :, None]
                .float()
                .expand(3, position_ids.shape[1], -1, 1)
            )
            position_ids_expanded = position_ids[:, :, None, :].float()
            device_type = x.device.type if x.device.type != "mps" else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=False):
                freqs = torch.einsum(
                    "abcd,abde->abce",
                    inv_freq_expanded,
                    position_ids_expanded,
                ).transpose(2, 3)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos() * self.attention_scaling
                sin = emb.sin() * self.attention_scaling
            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

        _qmod.Qwen2_5_VLRotaryEmbedding.forward = _patched
        _ROPE_PATCHED = True
        print("[distill] Patched Qwen2.5-VL RoPE (CUBLAS einsum workaround)")
    except (ImportError, AttributeError) as exc:
        print(f"[distill] RoPE patch skipped: {exc}")


# ════════════════════════════════════════════════════════════════════════════
#  Dataset
# ════════════════════════════════════════════════════════════════════════════

class DualTeacherDataset(Dataset):
    """
    Merges Waymo trajectory JSONL (multimodal) and reasoning JSONL (text-only).

    Each record must have "messages".
    Waymo records additionally have "image_path" (relative to image_root).
    """

    def __init__(
        self,
        waymo_jsonl: Optional[str],
        reason_jsonl: Optional[str],
        image_root: str,
        max_waymo: int = 0,
        max_reason: int = 0,
        seed: int = 42,
    ) -> None:
        self.image_root = Path(image_root)
        self.items: List[Dict[str, Any]] = []

        if waymo_jsonl:
            waymo_rows = _load_jsonl(waymo_jsonl)
            if max_waymo > 0:
                waymo_rows = waymo_rows[:max_waymo]
            for row in waymo_rows:
                row["_has_image"] = True
            self.items.extend(waymo_rows)
            print(f"[dataset] Loaded {len(waymo_rows)} Waymo records")

        if reason_jsonl:
            reason_rows = _load_jsonl(reason_jsonl)
            if max_reason > 0:
                reason_rows = reason_rows[:max_reason]
            for row in reason_rows:
                row["_has_image"] = False
            self.items.extend(reason_rows)
            print(f"[dataset] Loaded {len(reason_rows)} reasoning records")

        rng = random.Random(seed)
        rng.shuffle(self.items)
        print(f"[dataset] Total: {len(self.items)} records")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON at line {i} in {path}") from exc
    return rows


# ════════════════════════════════════════════════════════════════════════════
#  Collator
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class DualDistillBatch:
    # Multimodal (student + vision teacher) — may include pixel_values
    vision_input_ids: torch.Tensor
    vision_attention_mask: torch.Tensor
    vision_pixel_values: Optional[torch.Tensor]          # None for text-only batches
    vision_image_grid_thw: Optional[torch.Tensor]        # Qwen2.5-VL grid info
    # Text-only (language teacher)
    lang_input_ids: torch.Tensor
    lang_attention_mask: torch.Tensor
    # Labels for CE loss (student sequence)
    labels: torch.Tensor
    # Which items in the batch have images
    has_image: torch.Tensor  # bool tensor [B]


class DualDistillCollator:
    """
    Builds three parallel encodings per batch:
      1. vision_* — processor encoding (with images when available) for student & vision teacher
      2. lang_*   — tokenizer-only encoding for the language teacher
    """

    def __init__(
        self,
        vision_processor,   # AutoProcessor for Qwen2.5-VL
        lang_tokenizer,     # AutoTokenizer for QwQ-32B
        image_root: Path,
        max_length: int = 2048,
    ) -> None:
        self.vision_proc = vision_processor
        self.lang_tok = lang_tokenizer
        self.image_root = image_root
        self.max_length = max_length

        # Cap image resolution so a single frame never exceeds ~1024 tokens.
        # Qwen2.5-VL uses 28×28 patches; 1024 tokens ≈ 784×784 px effective area.
        if hasattr(self.vision_proc, "image_processor"):
            self.vision_proc.image_processor.min_pixels = 256 * 28 * 28
            self.vision_proc.image_processor.max_pixels = 1024 * 28 * 28

        # Ensure pad tokens are set
        if self.lang_tok.pad_token is None:
            self.lang_tok.pad_token = self.lang_tok.eos_token
        vtok = self.vision_proc.tokenizer
        if vtok.pad_token is None:
            vtok.pad_token = vtok.eos_token

    def __call__(self, batch: List[Dict[str, Any]]) -> DualDistillBatch:
        vision_texts: List[str] = []
        lang_texts: List[str] = []
        images: List[Optional[Image.Image]] = []
        has_image: List[bool] = []

        for row in batch:
            messages = row.get("messages")
            if not messages:
                raise KeyError("Each record must contain 'messages'.")

            # ── Text string for the language teacher (strip image tokens) ─
            text_messages = _strip_image_from_messages(messages)
            lang_str = self.lang_tok.apply_chat_template(
                text_messages, tokenize=False, add_generation_prompt=False
            )
            lang_texts.append(lang_str)

            # ── Vision input: load image if present ────────────────────────
            img: Optional[Image.Image] = None
            if row.get("_has_image", False):
                img = _load_image(row, self.image_root)

            if img is not None:
                has_image.append(True)
                images.append(img)
                # Build Qwen2.5-VL style messages with image placeholder
                vision_msgs = _build_vision_messages(messages, img)
                vision_str = self.vision_proc.apply_chat_template(
                    vision_msgs, tokenize=False, add_generation_prompt=False
                )
            else:
                has_image.append(False)
                images.append(None)
                text_only_msgs = _strip_image_from_messages(messages)
                vision_str = self.vision_proc.tokenizer.apply_chat_template(
                    text_only_msgs, tokenize=False, add_generation_prompt=False
                )
            vision_texts.append(vision_str)

        # ── Vision/student encoding ────────────────────────────────────────
        non_none_images = [im for im in images if im is not None]
        if non_none_images:
            v_enc = self.vision_proc(
                text=vision_texts,
                images=non_none_images if non_none_images else None,
                return_tensors="pt",
                padding=True,
            )
        else:
            v_enc = self.vision_proc.tokenizer(
                vision_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )

        # ── Language teacher encoding ──────────────────────────────────────
        l_enc = self.lang_tok(
            lang_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # ── Labels for CE loss ─────────────────────────────────────────────
        labels = v_enc["input_ids"].clone()
        labels[v_enc["attention_mask"] == 0] = -100

        return DualDistillBatch(
            vision_input_ids=v_enc["input_ids"],
            vision_attention_mask=v_enc["attention_mask"],
            vision_pixel_values=v_enc.get("pixel_values"),
            vision_image_grid_thw=v_enc.get("image_grid_thw"),
            lang_input_ids=l_enc["input_ids"],
            lang_attention_mask=l_enc["attention_mask"],
            labels=labels,
            has_image=torch.tensor(has_image, dtype=torch.bool),
        )


def _strip_image_from_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return messages with image content items removed, keeping only text."""
    cleaned = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = [
                c["text"] for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            ]
            new_content = " ".join(text_parts).strip()
        else:
            new_content = str(content)
        cleaned.append({"role": msg["role"], "content": new_content})
    return cleaned


def _build_vision_messages(
    messages: List[Dict[str, Any]], img: Image.Image
) -> List[Dict[str, Any]]:
    """
    Reconstruct messages with a proper image placeholder for the vision processor.
    The image is injected into the first user message.
    """
    out = []
    image_injected = False
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "user" and not image_injected:
            # Build content list with image + text
            if isinstance(content, list):
                # Already structured; replace image dicts with the real image
                new_content: List[Any] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        new_content.append({"type": "image", "image": img})
                    else:
                        new_content.append(item)
                # If no image item found, prepend one
                if not any(
                    isinstance(c, dict) and c.get("type") == "image"
                    for c in new_content
                ):
                    new_content.insert(0, {"type": "image", "image": img})
            else:
                new_content = [
                    {"type": "image", "image": img},
                    {"type": "text", "text": str(content)},
                ]
            out.append({"role": role, "content": new_content})
            image_injected = True
        else:
            out.append(msg)
    return out


def _load_image(row: Dict[str, Any], image_root: Path) -> Optional[Image.Image]:
    """Try to load the image referenced in a Waymo record."""
    img_path_str = row.get("image_path", "")
    if not img_path_str:
        # Try to extract from messages content list
        for msg in row.get("messages", []):
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        img_path_str = item.get("image", "")
                        break
    if not img_path_str:
        return None
    full_path = image_root / img_path_str
    if not full_path.exists():
        return None
    try:
        return Image.open(full_path).convert("RGB")
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════════════
#  Loss functions
# ════════════════════════════════════════════════════════════════════════════

def _align_vocab(
    s_logits: torch.Tensor, t_logits: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trim both logit tensors to the smaller vocabulary."""
    v = min(s_logits.size(-1), t_logits.size(-1))
    return s_logits[..., :v], t_logits[..., :v]


def kl_distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Per-token KL divergence between student and teacher, temperature-scaled.
    Only tokens that are valid (mask>0 and label!=-100) contribute.
    Returns scalar loss.
    """
    s_log, t_log = _align_vocab(
        student_logits[:, :-1, :].contiguous(),
        teacher_logits[:, :-1, :].contiguous(),
    )
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = mask[:, 1:].contiguous()

    t = temperature
    log_p_s = F.log_softmax(s_log / t, dim=-1)
    p_t = F.softmax(t_log / t, dim=-1)
    token_kl = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1)

    valid = (shift_mask > 0) & (shift_labels != -100)
    denom = valid.float().sum().clamp_min(1.0)
    return ((token_kl * valid.float()).sum() / denom) * (t * t)


def ce_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    shift_logits = student_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


# ════════════════════════════════════════════════════════════════════════════
#  Model loading
# ════════════════════════════════════════════════════════════════════════════

def _bnb4bit() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_vision_teacher(
    base_id: str,
    adapter_path: str,
    use_4bit: bool = True,
) -> Tuple[Any, Any]:
    """Load Qwen2.5-VL-32B base + Waymo LoRA adapter in eval/frozen mode."""
    _patch_qwen25vl_rope()
    quant = _bnb4bit() if use_4bit else None
    kwargs: Dict[str, Any] = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
        device_map="auto",
        max_memory={0: "75GiB", 1: "0GiB"},
    )
    if quant:
        kwargs["quantization_config"] = quant

    print(f"[vision teacher] Loading base: {base_id}")
    model = Qwen25VL.from_pretrained(base_id, **kwargs)
    print(f"[vision teacher] Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(base_id, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    print("[vision teacher] Ready")
    return model, processor


def load_language_teacher(
    base_id: str,
    adapter_path: str,
    use_4bit: bool = True,
) -> Tuple[Any, Any]:
    """Load QwQ-32B base + reasoning QLoRA adapter in eval/frozen mode."""
    quant = _bnb4bit() if use_4bit else None
    kwargs: Dict[str, Any] = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_memory={0: "0GiB", 1: "75GiB"},
    )
    if quant:
        kwargs["quantization_config"] = quant

    print(f"[lang teacher] Loading base: {base_id}")
    model = AutoModelForCausalLM.from_pretrained(base_id, **kwargs)
    print(f"[lang teacher] Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("[lang teacher] Ready")
    return model, tokenizer


def load_student(
    base_id: str,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    use_4bit: bool = False,
) -> Tuple[Any, Any]:
    """Load Qwen2.5-VL-7B student with LoRA for training."""
    _patch_qwen25vl_rope()
    quant = _bnb4bit() if use_4bit else None
    kwargs: Dict[str, Any] = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
        device_map="auto",
    )
    if quant:
        kwargs["quantization_config"] = quant

    print(f"[student] Loading: {base_id}")
    model = Qwen25VL.from_pretrained(base_id, **kwargs)
    model.config.use_cache = False

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def _require_grad_hook(module, inp, out):
            out.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(_require_grad_hook)

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(base_id, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    print("[student] Ready")
    return model, processor


# ════════════════════════════════════════════════════════════════════════════
#  Training loop
# ════════════════════════════════════════════════════════════════════════════

def save_metrics(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    # ── Load models ──────────────────────────────────────────────────────
    vision_teacher, vision_proc = load_vision_teacher(
        args.vision_base, args.vision_lora, use_4bit=not args.no_4bit_teachers
    )
    lang_teacher, lang_tok = load_language_teacher(
        args.lang_base, args.lang_lora, use_4bit=not args.no_4bit_teachers
    )
    student, student_proc = load_student(
        args.student,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_4bit=args.student_4bit,
    )

    # ── Device references ─────────────────────────────────────────────────
    vision_dev = next(
        p for p in vision_teacher.parameters() if p.device.type != "cpu"
    ).device
    lang_dev = next(
        p for p in lang_teacher.parameters() if p.device.type != "cpu"
    ).device
    student_dev = next(
        p for p in student.parameters() if p.device.type != "cpu"
    ).device
    print(f"[devices] vision={vision_dev}  lang={lang_dev}  student={student_dev}")

    # ── Dataset & loader ─────────────────────────────────────────────────
    dataset = DualTeacherDataset(
        waymo_jsonl=args.waymo_jsonl,
        reason_jsonl=args.reason_jsonl,
        image_root=args.image_root,
        max_waymo=args.max_waymo,
        max_reason=args.max_reason,
        seed=args.seed,
    )
    collator = DualDistillCollator(
        vision_processor=student_proc,   # student and vision teacher share the same processor
        lang_tokenizer=lang_tok,
        image_root=Path(args.image_root),
        max_length=args.max_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,  # keep 0 to avoid multiprocessing issues with PIL images
    )

    # ── Optimiser & scheduler ─────────────────────────────────────────────
    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    total_updates = math.ceil(len(loader) / args.grad_accum) * args.epochs
    if args.max_steps > 0:
        total_updates = min(total_updates, args.max_steps)
    warmup_steps = max(1, int(total_updates * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_updates)

    print(f"[train] {len(dataset)} samples | {total_updates} optimizer steps")
    print(
        f"[train] α_vision={args.alpha_vision}  α_lang={args.alpha_lang}"
        f"  α_ce={args.alpha_ce}  T={args.temperature}"
    )

    global_step = 0
    accum = 0
    running_total = running_v = running_l = running_ce = 0.0
    student.train()

    for epoch in range(args.epochs):
        for batch in loader:
            # ── Vision teacher forward (frozen) ───────────────────────────
            v_kwargs: Dict[str, Any] = {
                "input_ids": batch.vision_input_ids.to(vision_dev),
                "attention_mask": batch.vision_attention_mask.to(vision_dev),
            }
            if batch.vision_pixel_values is not None:
                v_kwargs["pixel_values"] = batch.vision_pixel_values.to(vision_dev)
            if batch.vision_image_grid_thw is not None:
                v_kwargs["image_grid_thw"] = batch.vision_image_grid_thw.to(vision_dev)

            with torch.no_grad():
                v_out = vision_teacher(**v_kwargs)

            # ── Language teacher forward (frozen) ─────────────────────────
            l_kwargs: Dict[str, Any] = {
                "input_ids": batch.lang_input_ids.to(lang_dev),
                "attention_mask": batch.lang_attention_mask.to(lang_dev),
            }
            with torch.no_grad():
                l_out = lang_teacher(**l_kwargs)

            # ── Student forward ───────────────────────────────────────────
            s_kwargs: Dict[str, Any] = {
                "input_ids": batch.vision_input_ids.to(student_dev),
                "attention_mask": batch.vision_attention_mask.to(student_dev),
            }
            if batch.vision_pixel_values is not None:
                s_kwargs["pixel_values"] = batch.vision_pixel_values.to(student_dev)
            if batch.vision_image_grid_thw is not None:
                s_kwargs["image_grid_thw"] = batch.vision_image_grid_thw.to(student_dev)

            s_out = student(**s_kwargs)

            labels = batch.labels.to(student_dev)
            s_mask = batch.vision_attention_mask.to(student_dev)

            # ── Losses ────────────────────────────────────────────────────
            # Vision KL: student logits vs vision teacher logits
            v_teacher_logits = v_out.logits.to(student_dev)
            loss_v = kl_distill_loss(
                s_out.logits, v_teacher_logits, labels, s_mask, args.temperature
            )

            # Language KL: student logits vs language teacher logits
            # Align sequence length to the shorter of the two (student uses vision tokens,
            # lang teacher uses text-only tokens which may be shorter)
            l_teacher_logits = l_out.logits.to(student_dev)
            s_len = s_out.logits.size(1)
            l_len = l_teacher_logits.size(1)
            min_len = min(s_len, l_len)
            loss_l = kl_distill_loss(
                s_out.logits[:, :min_len, :],
                l_teacher_logits[:, :min_len, :],
                labels[:, :min_len],
                s_mask[:, :min_len],
                args.temperature,
            )

            # Hard-label CE
            loss_ce = ce_loss(s_out.logits, labels)

            loss = (
                args.alpha_vision * loss_v
                + args.alpha_lang * loss_l
                + args.alpha_ce * loss_ce
            ) / args.grad_accum

            loss.backward()
            accum += 1

            running_total += loss.item()
            running_v += loss_v.item()
            running_l += loss_l.item()
            running_ce += loss_ce.item()

            if accum % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0:
                    n = args.log_every
                    avg = running_total / n
                    print(
                        f"epoch={epoch} step={global_step}"
                        f" loss={avg * args.grad_accum:.4f}"
                        f" kl_v={running_v/n:.4f}"
                        f" kl_l={running_l/n:.4f}"
                        f" ce={running_ce/n:.4f}"
                        f" lr={scheduler.get_last_lr()[0]:.2e}"
                    )
                    save_metrics(
                        metrics_path,
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "loss": float(avg * args.grad_accum),
                            "kl_vision": float(running_v / n),
                            "kl_lang": float(running_l / n),
                            "ce": float(running_ce / n),
                            "lr": float(scheduler.get_last_lr()[0]),
                        },
                    )
                    running_total = running_v = running_l = running_ce = 0.0

                if global_step % args.save_every == 0:
                    ckpt = out_dir / f"checkpoint-{global_step}"
                    ckpt.mkdir(parents=True, exist_ok=True)
                    # Save only the LoRA adapter (no merge — avoids OOM on large base)
                    student.save_pretrained(ckpt)
                    student_proc.save_pretrained(ckpt)
                    print(f"[ckpt] Saved to {ckpt}")

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # ── Final save ────────────────────────────────────────────────────────
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(final_dir)
    student_proc.save_pretrained(final_dir)
    print(f"[done] Final student adapter saved to: {final_dir}")


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dual-teacher (vision + language) logit distillation into Qwen2.5-VL-7B"
    )

    # ── Teachers ──────────────────────────────────────────────────────────
    p.add_argument(
        "--vision_base", default="Qwen/Qwen2.5-VL-32B-Instruct",
        help="HF id for the vision teacher base model",
    )
    p.add_argument(
        "--vision_lora",
        default="/scratch/rbaskar5/GLANCE/checkpoints/waymo/qwen2.5-vl-32b-trajectory/final_lora",
        help="Path to the Waymo-trained LoRA adapter for the vision teacher",
    )
    p.add_argument(
        "--lang_base", default="Qwen/QwQ-32B",
        help="HF id for the language/reasoning teacher base model",
    )
    p.add_argument(
        "--lang_lora",
        default="/scratch/rbaskar5/GLANCE/reasoning_distillation/checkpoints/teacher_qwq32b_qlora/teacher_lora_adapter",
        help="Path to the reasoning QLoRA adapter for the language teacher",
    )

    # ── Student ───────────────────────────────────────────────────────────
    p.add_argument("--student", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--student_4bit", action="store_true",
                   help="Load student in 4-bit (saves GPU mem but slower gradients)")
    p.add_argument("--no_4bit_teachers", action="store_true",
                   help="Load teachers in bf16 instead of 4-bit (needs more VRAM)")

    # ── Data ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--waymo_jsonl",
        default="/scratch/rbaskar5/GLANCE/data/waymo_trajectory/train.jsonl",
        help="Waymo trajectory train JSONL (multimodal; set to '' to skip)",
    )
    p.add_argument(
        "--reason_jsonl",
        default="/scratch/rbaskar5/GLANCE/reasoning_distillation/data/distill_train_qwen25vl7b.jsonl",
        help="Reasoning distillation train JSONL (text-only; set to '' to skip)",
    )
    p.add_argument(
        "--image_root",
        default="/scratch/rbaskar5/Dataset/waymo_front",
        help="Root directory for Waymo images",
    )
    p.add_argument("--max_waymo", type=int, default=0, help="Cap Waymo samples (0=all)")
    p.add_argument("--max_reason", type=int, default=0, help="Cap reasoning samples (0=all)")

    # ── Output ────────────────────────────────────────────────────────────
    p.add_argument(
        "--out_dir",
        default="/scratch/rbaskar5/GLANCE/checkpoints/dual_distilled_7b",
    )

    # ── Training hyperparams ──────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)

    # ── Distillation hyperparams ──────────────────────────────────────────
    p.add_argument("--alpha_vision", type=float, default=0.5,
                   help="Weight for vision teacher KL loss")
    p.add_argument("--alpha_lang", type=float, default=0.3,
                   help="Weight for language teacher KL loss")
    p.add_argument("--alpha_ce", type=float, default=0.2,
                   help="Weight for hard-label CE loss")
    p.add_argument("--temperature", type=float, default=2.0,
                   help="Distillation temperature")

    # ── Misc ──────────────────────────────────────────────────────────────
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train(args)
