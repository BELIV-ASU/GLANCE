"""
train_teacher.py  –  Fine-tune Qwen2.5-VL-32B as a SafetyVLM Teacher
using LoRA on 2× A100-80GB.

Training paradigm:
  • Chain-of-Thought (CoT)    – <think>…</think> reasoning before answers
  • Grounding                 – [Source: …] citations to handbook sections
  • QLoRA 4-bit               – fits 30B+ params on 2× A100-80GB

Capabilities trained:
  • Rule reasoning            – traffic law analysis & justification
  • Multilingual knowledge    – non-English driving regulations
  • Edge cases                – trucks, motorcycles, emergencies, penalties
  • Simulated test scenarios  – exam-style Q&A generation
  • Vision grounding          – traffic sign/scenario image analysis

Usage:
  accelerate launch --num_processes 2 train_teacher.py [args]
  # or single-GPU:
  python train_teacher.py [args]
"""

import os, sys, json, logging, math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
# Force cublasLt backend – avoids CUBLAS_STATUS_INVALID_VALUE bugs
# caused by CUDA 12.8/12.9 version mismatch with PyTorch 2.10
torch.backends.cuda.preferred_blas_library("cublaslt")

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import Trainer
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Monkey-patch: fix CUBLAS_STATUS_INVALID_VALUE in rotary embeddings
#  PyTorch 2.10+cu128 has a bug in cublasSgemmStridedBatched with the
#  specific tensor shapes used by Qwen2.5-VL M-RoPE.
#  torch.einsum produces identical results but avoids the buggy kernel.
# ---------------------------------------------------------------------------
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _qwen25vl

_orig_rope_forward = _qwen25vl.Qwen2_5_VLRotaryEmbedding.forward

def _patched_rope_forward(self, x, position_ids):
    inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(
        3, position_ids.shape[1], -1, 1
    )
    position_ids_expanded = position_ids[:, :, None, :].float()
    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    with torch.amp.autocast(device_type=device_type, enabled=False):
        # Use einsum instead of @ to avoid cublasSgemmStridedBatched bug
        freqs = torch.einsum("abcd,abde->abce", inv_freq_expanded, position_ids_expanded).transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

_qwen25vl.Qwen2_5_VLRotaryEmbedding.forward = _patched_rope_forward
log.info("Patched Qwen2_5_VLRotaryEmbedding.forward (einsum workaround for CUBLAS bug)")


# ---------------------------------------------------------------------------
#  Training arguments
# ---------------------------------------------------------------------------

@dataclass
class TeacherArgs:
    # Model
    model_name: str = "Qwen/Qwen2.5-VL-32B-Instruct"
    trust_remote_code: bool = True
    # Use eager attention to avoid potential SDPA CUBLAS issues
    attn_implementation: str = "eager"

    # Quantisation – QLoRA 4-bit (NF4 + double quant)
    # Bypasses buggy cublas bf16 kernels (PyTorch 2.10+cu128)
    # and shrinks 32B model from ~64GB to ~10GB
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj"  # attention only – MoE experts use fused gate_up_proj
    lora_bias: str = "none"

    # Data
    data_dir: str = "/scratch/rbaskar5/GLANCE/data_drivelm"
    max_seq_length: int = 4096  # longer for CoT reasoning chains
    max_samples: int = 0  # 0 = all
    mask_think_tokens: bool = False  # if True, only compute loss after </think>

    # Training hyper-params
    output_dir: str = "/scratch/rbaskar5/GLANCE/checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Efficiency
    gradient_checkpointing: bool = True
    bf16: bool = True
    tf32: bool = True
    dataloader_num_workers: int = 1

    # Logging / saving
    logging_steps: int = 5
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 3
    eval_strategy: str = "steps"
    report_to: str = "none"

    # Misc
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None


def parse_args() -> TeacherArgs:
    """Simple argparse that maps --key value to TeacherArgs fields."""
    import argparse
    args = TeacherArgs()
    parser = argparse.ArgumentParser()
    for k, v in vars(args).items():
        t = type(v) if v is not None else str
        if t == bool:
            parser.add_argument(f"--{k}", action="store_true", default=v)
        else:
            parser.add_argument(f"--{k}", type=t, default=v)
    parsed = parser.parse_args()
    for k in vars(args):
        setattr(args, k, getattr(parsed, k))
    return args


# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL -> list of dicts (robust to encoding edge-cases)."""
    data = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines instead of crashing
                log.warning(f"Skipping malformed JSONL line {i} in {path}")
    return data


def make_hf_dataset(jsonl_path: str, max_samples: int = 0) -> Dataset:
    """Load JSONL as a HuggingFace Dataset.

    Messages contain mixed types (list-of-dicts for user content,
    plain strings for system/assistant) which PyArrow cannot handle
    natively.  We serialise each conversation as a single JSON string
    and deserialise in the collator.
    """
    raw = load_jsonl(jsonl_path)
    if max_samples > 0:
        raw = raw[:max_samples]

    # Store as a single JSON-string column to avoid Arrow schema issues
    rows = [{"messages_json": json.dumps(r["messages"], ensure_ascii=False)} for r in raw]
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
#  Model setup
# ---------------------------------------------------------------------------

def load_model_and_processor(args: TeacherArgs):
    """Load Qwen2.5-VL with LoRA adapters."""

    log.info(f"Loading model: {args.model_name}")
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

    # Build quantization config only if 4-bit is requested
    quant_kwargs = {}
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )
        quant_kwargs["quantization_config"] = bnb_config

    # Qwen2.5-VL-32B-Instruct
    log.info("Using Qwen2_5_VLForConditionalGeneration")
    # device_map={"":  0} streams weights directly to cuda:0 shard-by-shard
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        **quant_kwargs,
        torch_dtype=compute_dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    # Disable cache for training + gradient checkpointing
    model.config.use_cache = False

    # Prepare for k-bit training (only needed with quantization)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    else:
        # bf16 LoRA: enable gradient checkpointing manually
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        model.enable_input_require_grads()  # needed for LoRA

    # LoRA config
    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias=args.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Processor
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        padding_side="right",
        # Limit image resolution to prevent image tokens from exceeding
        # max_seq_length.  512 patches * 28 * 28 = 401408 pixels ≈ 512 visual tokens
        min_pixels=3136,     # minimum 4 patches (2×2)
        max_pixels=401408,   # ~512 visual tokens – leaves room for text in 2048 seq
    )
    # Ensure pad token
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return model, processor


# ---------------------------------------------------------------------------
#  Collator for VLM chat
# ---------------------------------------------------------------------------

class QwenVLCollator:
    """Custom collator that processes Qwen2.5-VL chat messages into model inputs.

    Handles both text-only and image+text conversations.
    Supports Chain-of-Thought aware label masking:
      - mask_think=False (default): train on ALL tokens including <think>…</think>
        so the model learns to produce CoT reasoning
      - mask_think=True: only compute loss on tokens AFTER </think>
        (useful for later distillation where you only want answer quality)
    """

    def __init__(self, processor, max_seq_length: int = 4096,
                 mask_think: bool = False):
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.mask_think = mask_think
        # Pre-encode the </think> marker for label masking
        self._think_end_ids = self.processor.tokenizer.encode(
            "</think>", add_special_tokens=False
        )

    def _find_think_end(self, input_ids: torch.Tensor) -> int:
        """Find the token index right after </think> in a 1-D tensor."""
        ids = input_ids.tolist()
        marker = self._think_end_ids
        for i in range(len(ids) - len(marker) + 1):
            if ids[i:i+len(marker)] == marker:
                return i + len(marker)
        return 0  # not found → don't mask anything

    def _load_images_from_messages(self, messages: List[Dict]) -> List:
        """Extract and load PIL images from the message content list.

        Handles both standard RGB images and 16-bit depth maps (saved as
        mode "I;16" PNG by compute_depth.py).  Depth maps are normalised
        to 8-bit and converted to 3-channel grayscale RGB so the VL
        processor receives a uniform (H, W, 3) uint8 tensor.
        """
        import numpy as np
        from PIL import Image as PILImage
        images = []
        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "image":
                    continue
                img_path = item.get("image", "")
                if img_path.startswith("file://"):
                    img_path = img_path[7:]
                if os.path.isfile(img_path):
                    try:
                        pil_img = PILImage.open(img_path)
                        # 16-bit depth maps need special handling
                        if pil_img.mode in ("I;16", "I"):
                            arr = np.array(pil_img, dtype=np.float32)
                            a_min, a_max = arr.min(), arr.max()
                            if a_max - a_min > 1e-6:
                                arr = (arr - a_min) / (a_max - a_min) * 255.0
                            arr = arr.astype(np.uint8)
                            pil_img = PILImage.fromarray(
                                np.stack([arr, arr, arr], axis=-1), mode="RGB"
                            )
                        else:
                            pil_img = pil_img.convert("RGB")
                        images.append(pil_img)
                    except Exception:
                        pass
        return images

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        texts = []
        all_images = []      # flat list of PIL images for the whole batch
        has_any_image = False

        for ex in examples:
            # Deserialise messages (stored as JSON string to dodge Arrow)
            raw = ex.get("messages_json") or ex.get("messages")
            messages = json.loads(raw) if isinstance(raw, str) else raw

            # Apply chat template to get the formatted text
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

            # Load images referenced in the conversation
            images = self._load_images_from_messages(messages)
            if images:
                all_images.extend(images)
                has_any_image = True

        # Process text + images through the Qwen VL processor
        # The processor handles image → pixel_values + image_grid_thw
        # NOTE: Do NOT pass truncation=True here — it truncates input_ids
        # after image token expansion, causing a mismatch with pixel_values.
        # Instead, we cap image resolution via max_pixels on the processor
        # and truncate manually below.
        proc_kwargs = dict(
            text=texts,
            padding=True,
            return_tensors="pt",
        )
        if has_any_image:
            proc_kwargs["images"] = all_images

        batch_inputs = self.processor(**proc_kwargs)

        # Manual truncation — safe because image tokens are already bounded
        seq_len = batch_inputs["input_ids"].size(1)
        if seq_len > self.max_seq_length:
            for key in batch_inputs:
                if isinstance(batch_inputs[key], torch.Tensor) and batch_inputs[key].ndim >= 2:
                    if batch_inputs[key].size(1) == seq_len:
                        batch_inputs[key] = batch_inputs[key][:, :self.max_seq_length]

        # Labels = input_ids (model shifts internally)
        batch_inputs["labels"] = batch_inputs["input_ids"].clone()

        # Mask padding tokens
        pad_token_id = self.processor.tokenizer.pad_token_id
        batch_inputs["labels"][batch_inputs["labels"] == pad_token_id] = -100

        # Optional: mask <think>…</think> tokens so loss is only on the answer
        if self.mask_think:
            for i in range(batch_inputs["labels"].size(0)):
                end_idx = self._find_think_end(batch_inputs["input_ids"][i])
                if end_idx > 0:
                    batch_inputs["labels"][i, :end_idx] = -100

        return batch_inputs


# ---------------------------------------------------------------------------
#  Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    log.info("=" * 60)
    log.info("  SafetyVLM Teacher Training – Qwen2.5-VL-32B + LoRA")
    log.info("=" * 60)

    # Ensure output dirs
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Data ----
    train_path = os.path.join(args.data_dir, "train.jsonl")
    val_path = os.path.join(args.data_dir, "val.jsonl")

    if not os.path.isfile(train_path):
        log.info("Training data not found – building dataset first...")
        from data import main as build_drivelm_data
        import sys
        sys.argv = [
            "data.py",
            "--data_json", "/scratch/rbaskar5/Dataset/DriveLM/v1_1_train_nus.json",
            "--data_root", "/scratch/rbaskar5/Dataset/DriveLM",
            "--output_dir", args.data_dir,
            "--val_ratio", "0.05",
            "--seed", str(args.seed),
            "--with_depth",
        ]
        build_drivelm_data()

    log.info(f"Loading training data from {train_path}")
    train_dataset = make_hf_dataset(train_path, args.max_samples)
    val_dataset = make_hf_dataset(val_path)
    log.info(f"  Train: {len(train_dataset)}  Val: {len(val_dataset)}")

    # ---- Model ----
    model, processor = load_model_and_processor(args)

    # ---- Collator (CoT-aware) ----
    collator = QwenVLCollator(
        processor,
        max_seq_length=args.max_seq_length,
        mask_think=args.mask_think_tokens,
    )

    # ---- Training config ----
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        tf32=args.tf32,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy=args.eval_strategy,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=args.report_to,
        seed=args.seed,
        remove_unused_columns=False,
    )

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # ---- Train ----
    log.info("Starting training...")
    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    # ---- Save ----
    log.info("Saving final model...")
    final_dir = os.path.join(args.output_dir, "final_lora")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    log.info(f"  LoRA adapter saved to {final_dir}")

    # Also save merged if we have enough memory
    try:
        log.info("Attempting to merge and save full model...")
        merged_model = model.merge_and_unload()
        merged_dir = os.path.join(args.output_dir, "merged_model")
        merged_model.save_pretrained(merged_dir, safe_serialization=True)
        processor.save_pretrained(merged_dir)
        log.info(f"  Merged model saved to {merged_dir}")
    except Exception as e:
        log.warning(f"  Could not save merged model (likely OOM): {e}")
        log.info("  Use the LoRA adapter with the base model instead.")

    log.info("=" * 60)
    log.info("  Training complete!")
    log.info(f"  Checkpoints: {args.output_dir}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
