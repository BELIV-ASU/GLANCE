"""
train_teacher.py  –  Fine-tune Qwen3-VL as a SafetyVLM Teacher
using QLoRA (4-bit NF4) + LoRA on 2× A100-80GB.

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
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
)

from transformers import Qwen3VLMoeForConditionalGeneration

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import Trainer
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Training arguments
# ---------------------------------------------------------------------------

@dataclass
class TeacherArgs:
    # Model
    model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    trust_remote_code: bool = True
    attn_implementation: str = "sdpa"

    # Quantisation – disabled: bf16 LoRA fits on 2×A100-80GB (30B×2B=60GB)
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = False

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj"  # attention only – MoE experts use fused gate_up_proj
    lora_bias: str = "none"

    # Data
    data_dir: str = "/scratch/jnolas77/SafetyVLM/Qwen-3-VL/data"
    max_seq_length: int = 4096  # longer for CoT reasoning chains
    max_samples: int = 0  # 0 = all
    mask_think_tokens: bool = False  # if True, only compute loss after </think>

    # Training hyper-params
    output_dir: str = "/scratch/jnolas77/SafetyVLM/Qwen-3-VL/checkpoints"
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
    dataloader_num_workers: int = 4

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
    """Load Qwen3-VL-30B-A3B (MoE) with LoRA adapters in bf16."""

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

    # Qwen3-VL-30B-A3B is MoE → must use the MoE class
    log.info("Using Qwen3VLMoeForConditionalGeneration (MoE architecture)")
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        args.model_name,
        **quant_kwargs,
        torch_dtype=compute_dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        device_map="auto",
    )

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
    )
    # Ensure pad token
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return model, processor


# ---------------------------------------------------------------------------
#  Collator for VLM chat
# ---------------------------------------------------------------------------

class Qwen3VLCollator:
    """Custom collator that processes Qwen3-VL chat messages into model inputs.

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
        """Extract and load PIL images from the message content list."""
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
                        images.append(PILImage.open(img_path).convert("RGB"))
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
        proc_kwargs = dict(
            text=texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        if has_any_image:
            proc_kwargs["images"] = all_images

        batch_inputs = self.processor(**proc_kwargs)

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
    log.info("  SafetyVLM Teacher Training – Qwen3-VL-32B + QLoRA")
    log.info("=" * 60)

    # Ensure output dirs
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Data ----
    train_path = os.path.join(args.data_dir, "train.jsonl")
    val_path = os.path.join(args.data_dir, "val.jsonl")

    if not os.path.isfile(train_path):
        log.info("Training data not found – building dataset first...")
        from safety_data import build_dataset, save_dataset
        ds = build_dataset(
            data_json="/scratch/jnolas77/SafetyVLM/driving_handbook_data/data.json",
            data_root="/scratch/jnolas77/SafetyVLM/driving_handbook_data",
            seed=args.seed,
        )
        save_dataset(ds, args.data_dir)

    log.info(f"Loading training data from {train_path}")
    train_dataset = make_hf_dataset(train_path, args.max_samples)
    val_dataset = make_hf_dataset(val_path)
    log.info(f"  Train: {len(train_dataset)}  Val: {len(val_dataset)}")

    # ---- Model ----
    model, processor = load_model_and_processor(args)

    # ---- Collator (CoT-aware) ----
    collator = Qwen3VLCollator(
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
        processing_class=processor.tokenizer,
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
