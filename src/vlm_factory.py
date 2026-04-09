#!/usr/bin/env python3
"""
vlm_factory.py  –  Unified factory for loading 7B-class VLMs
═══════════════════════════════════════════════════════════════════════════════

Provides a single entry-point:

    model, processor = get_vlm_and_processor(model_name, cfg)

Supported models:
    1. Qwen/Qwen2.5-VL-7B-Instruct
    2. nvidia/Cosmos-Reason1-7B          (Cosmos R1-7B)
    3. XiaomiMiMo/MiMo-VL-7B-RL         (MiMo-VL-7B)
    4. openvla/openvla-7b                (OpenVLA-7B)
    5. Efficient-Large-Model/VILA1.5-7B  (VILA-7B)

All models are loaded in bf16 on CUDA with the CUBLAS work-arounds required
for PyTorch 2.10 + CUDA 12.8.

Usage:
    from vlm_factory import get_vlm_and_processor, apply_lora
    model, processor = get_vlm_and_processor("Qwen/Qwen2.5-VL-7B-Instruct", cfg)
    model = apply_lora(model, cfg["lora"])
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

# Import torch
import torch
# ── CUBLAS workaround ────────────────────────────────────────────────────
# Must be called before *any* model instantiation.
torch.backends.cuda.preferred_blas_library("cublaslt")
os.environ.setdefault("TORCH_BLAS_PREFER_CUBLASLT", "1")

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)

# AutoModelForVision2Seq may not exist in all transformers versions
try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    AutoModelForVision2Seq = None  # type: ignore[assignment,misc]

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Qwen2.5-VL rotary-embedding monkey-patch
#
#  PyTorch 2.10+cu128 cublasSgemmStridedBatched bug triggers
#  CUBLAS_STATUS_INVALID_VALUE on the specific tensor shapes
#  used by Qwen2.5-VL M-RoPE.  torch.einsum avoids the kernel.
# ═══════════════════════════════════════════════════════════════════════════

_QWEN_PATCHED = False


def _patch_qwen25vl_rope() -> None:
    """Apply the rope monkey-patch exactly once."""
    global _QWEN_PATCHED
    if _QWEN_PATCHED:
        return

    try:
        import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _qmod

        _orig = _qmod.Qwen2_5_VLRotaryEmbedding.forward

        def _patched(self, x, position_ids):
            inv_freq_expanded = (
                self.inv_freq[None, None, :, None]
                .float()
                .expand(3, position_ids.shape[1], -1, 1)
            )
            position_ids_expanded = position_ids[:, :, None, :].float()
            device_type = (
                x.device.type
                if isinstance(x.device.type, str) and x.device.type != "mps"
                else "cpu"
            )
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
        _QWEN_PATCHED = True
        log.info(
            "Patched Qwen2_5_VLRotaryEmbedding.forward "
            "(einsum workaround for CUBLAS bug)"
        )
    except (ImportError, AttributeError) as exc:
        log.debug("Qwen2.5-VL rope patch skipped: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════
#  Model-specific loaders
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_dtype(name: str) -> torch.dtype:
    """Map config string → torch dtype."""
    return {
        "bfloat16": torch.bfloat16,
        "bf16":     torch.bfloat16,
        "float16":  torch.float16,
        "fp16":     torch.float16,
        "float32":  torch.float32,
        "fp32":     torch.float32,
    }.get(name.lower(), torch.bfloat16)


def _quantization_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build optional bitsandbytes quantization kwargs from config."""
    model_cfg = cfg.get("model", {})
    if not model_cfg.get("load_in_4bit", False):
        return {}

    compute_dtype = _resolve_dtype(model_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
    return {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True),
        )
    }


def _common_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Shared kwargs for all `from_pretrained` calls."""
    model_cfg = cfg.get("model", {})
    kwargs = dict(
        torch_dtype=_resolve_dtype(model_cfg.get("dtype", "bfloat16")),
        attn_implementation=model_cfg.get("attn_implementation", "eager"),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    # Only set device_map="auto" if not running distributed
    if not (
        int(os.environ.get("WORLD_SIZE", "1")) > 1 or
        os.environ.get("ACCELERATE_USE_DISTRIBUTED", "") == "1" or
        os.environ.get("LOCAL_RANK") is not None
    ):
        kwargs["device_map"] = "auto"
    kwargs.update(_quantization_kwargs(cfg))
    return kwargs


# ── 1. Qwen2.5-VL-7B ────────────────────────────────────────────────────

def _load_qwen25vl(name: str, cfg: Dict[str, Any]) -> Tuple[PreTrainedModel, Any]:
    """
    Load Qwen/Qwen2.5-VL-7B-Instruct.

    Uses the dedicated `Qwen2_5_VLForConditionalGeneration` class for proper
    image-token handling and the M-RoPE monkey-patch.
    """
    from transformers import Qwen2_5_VLForConditionalGeneration

    _patch_qwen25vl_rope()

    kwargs = _common_kwargs(cfg)
    log.info("Loading Qwen2.5-VL: %s  dtype=%s", name, kwargs["torch_dtype"])

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(name, **kwargs)
    processor = AutoProcessor.from_pretrained(
        name,
        trust_remote_code=True,
        padding_side="left",
    )

    # Ensure pad_token is set
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return model, processor


# ── 2. Cosmos R1-7B (nvidia/Cosmos-Reason1-7B) ───────────────────────────

def _load_cosmos_r1(name: str, cfg: Dict[str, Any]) -> Tuple[PreTrainedModel, Any]:
    """
    Load NVIDIA Cosmos Reason1-7B.

    Cosmos R1 follows a LLaVA-style architecture.  The model is loaded via
    AutoModelForCausalLM with trust_remote_code because it may use custom
    modelling code hosted on the Hub.

    NOTE: This loader stub may need adjustment once the exact Hub weights
    are publicly released.  Check the model card for processor specifics.
    """
    kwargs = _common_kwargs(cfg)
    log.info("Loading Cosmos R1-7B: %s  dtype=%s", name, kwargs["torch_dtype"])

    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)

    if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return model, processor


# ── 3. MiMo-VL-7B (XiaomiMiMo/MiMo-VL-7B-RL) ──────────────────────────

def _load_mimo_vl(name: str, cfg: Dict[str, Any]) -> Tuple[PreTrainedModel, Any]:
    """
    Load Xiaomi MiMo-VL-7B-RL.

    MiMo-VL is built on Qwen2.5-VL backbone with RL-based training.
    It uses the same Qwen2.5-VL architecture, so the rope patch is needed.
    """
    from transformers import Qwen2_5_VLForConditionalGeneration

    _patch_qwen25vl_rope()

    kwargs = _common_kwargs(cfg)
    log.info("Loading MiMo-VL-7B: %s  dtype=%s", name, kwargs["torch_dtype"])

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(name, **kwargs)
    processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)

    if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return model, processor


# ── 4. OpenVLA-7B (openvla/openvla-7b) ──────────────────────────────────

def _load_openvla(name: str, cfg: Dict[str, Any]) -> Tuple[PreTrainedModel, Any]:
    """
    Load OpenVLA-7B — a Prismatic VLM for robotic action prediction.

    OpenVLA uses a custom modelling class (Prismatic), so trust_remote_code
    is mandatory.  The processor combines a SigLIP image encoder with a
    Llama-2-7B LLM backbone.

    For downstream fine-tuning on traffic violation tasks, you may want
    to replace the action-token head with a text-generation head.  This
    stub loads the base model as-is.

    NOTE: OpenVLA's architecture is different from typical VLMs. It may
    require a custom collation strategy in WaymoCollator.
    """
    kwargs = _common_kwargs(cfg)
    log.info("Loading OpenVLA-7B: %s  dtype=%s", name, kwargs["torch_dtype"])

    if AutoModelForVision2Seq is not None:
        model = AutoModelForVision2Seq.from_pretrained(name, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)

    return model, processor


# ── 5. VILA-7B (Efficient-Large-Model/VILA1.5-7B) ───────────────────────

def _load_vila(name: str, cfg: Dict[str, Any]) -> Tuple[PreTrainedModel, Any]:
    """
    Load VILA 1.5-7B (NVidia Efficient-Large-Model).

    VILA uses a LLaVA-Next-style architecture with a Llama backbone.
    It requires trust_remote_code for the custom modelling files on Hub.

    NOTE: VILA-1.5 may need specific image pre-processing (e.g., anyres
    dynamic resolution).  Check the model card for details.
    """
    kwargs = _common_kwargs(cfg)
    log.info("Loading VILA-7B: %s  dtype=%s", name, kwargs["torch_dtype"])

    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)

    # VILA may have a custom processor or just a tokenizer + image processor
    try:
        processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
    except Exception:
        log.warning(
            "AutoProcessor failed for %s; falling back to tokenizer + "
            "manual image preprocessing.", name,
        )
        processor = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

    if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    elif hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token

    return model, processor


# ═══════════════════════════════════════════════════════════════════════════
#  Model registry
# ═══════════════════════════════════════════════════════════════════════════

# Maps canonical short names (lowercase) → (default HF hub name, loader fn)
MODEL_REGISTRY: Dict[str, Tuple[str, Any]] = {
    # ── Qwen variants ────────────────────────────────────────────────────
    "qwen2.5-vl-7b":              ("Qwen/Qwen2.5-VL-7B-Instruct",          _load_qwen25vl),
    "qwen/qwen2.5-vl-7b-instruct": ("Qwen/Qwen2.5-VL-7B-Instruct",        _load_qwen25vl),
    "qwen2.5-vl-32b":             ("Qwen/Qwen2.5-VL-32B-Instruct",         _load_qwen25vl),
    "qwen/qwen2.5-vl-32b-instruct": ("Qwen/Qwen2.5-VL-32B-Instruct",       _load_qwen25vl),

    # ── Cosmos ────────────────────────────────────────────────────────────
    "cosmos-r1-7b":               ("nvidia/Cosmos-Reason1-7B",               _load_cosmos_r1),
    "nvidia/cosmos-reason1-7b":   ("nvidia/Cosmos-Reason1-7B",               _load_cosmos_r1),

    # ── MiMo-VL ──────────────────────────────────────────────────────────
    "mimo-vl-7b":                 ("XiaomiMiMo/MiMo-VL-7B-RL",              _load_mimo_vl),
    "xiaomimimo/mimo-vl-7b-rl":   ("XiaomiMiMo/MiMo-VL-7B-RL",             _load_mimo_vl),

    # ── OpenVLA ──────────────────────────────────────────────────────────
    "openvla-7b":                 ("openvla/openvla-7b",                     _load_openvla),
    "openvla/openvla-7b":         ("openvla/openvla-7b",                     _load_openvla),

    # ── VILA ─────────────────────────────────────────────────────────────
    "vila-7b":                    ("Efficient-Large-Model/VILA1.5-7B",       _load_vila),
    "efficient-large-model/vila1.5-7b": ("Efficient-Large-Model/VILA1.5-7B", _load_vila),
}


def get_vlm_and_processor(
    model_name: str,
    cfg: Dict[str, Any],
) -> Tuple[PreTrainedModel, Any]:
    """
    Unified entry-point: load any supported 7B VLM + its processor.

    Parameters
    ----------
    model_name : str
        Either a short name (e.g. "qwen2.5-vl-7b") or a full HF hub id
        (e.g. "Qwen/Qwen2.5-VL-7B-Instruct").
    cfg : dict
        Full config dict (must contain at least a "model" sub-dict).

    Returns
    -------
    (model, processor)
    """
    key = model_name.lower().strip()

    if key in MODEL_REGISTRY:
        hub_name, loader = MODEL_REGISTRY[key]
        return loader(hub_name, cfg)

    # ── Fall through: treat model_name as a HF hub id ────────────────────
    log.warning(
        "Model '%s' not in registry. Attempting generic AutoModel load.", model_name
    )
    kwargs = _common_kwargs(cfg)
    if AutoModelForVision2Seq is not None:
        try:
            model = AutoModelForVision2Seq.from_pretrained(model_name, **kwargs)
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return model, processor


# ═══════════════════════════════════════════════════════════════════════════
#  LoRA adapter application
# ═══════════════════════════════════════════════════════════════════════════

def apply_lora(
    model: PreTrainedModel,
    lora_cfg: Dict[str, Any],
    model_name: str = "",
) -> PreTrainedModel:
    """
    Wrap `model` with PEFT LoRA adapters.

    Parameters
    ----------
    model : PreTrainedModel
        The base VLM (already loaded in bf16).
    lora_cfg : dict
        The `lora:` section from waymo_finetune_config.yaml.
    model_name : str
        Used to determine the correct PEFT TaskType.

    Returns
    -------
    PeftModel
    """

    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    target_modules = lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    if isinstance(target_modules, str):
        target_modules = [m.strip() for m in target_modules.split(",")]

    # Determine task type based on model architecture
    task_type = TaskType.CAUSAL_LM  # default

    # If model is quantized (QLoRA), prepare for k-bit training
    if hasattr(model, "quantization_method") or hasattr(model, "is_loaded_in_4bit") or hasattr(model, "is_loaded_in_8bit"):
        if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
            model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("alpha", 128),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=target_modules,
        bias=lora_cfg.get("bias", "none"),
        task_type=task_type,
    )

    # Enable gradient for input embeddings (needed for some VLMs)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def _make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)

    peft_model = get_peft_model(model, config)

    trainable, total = 0, 0
    for p in peft_model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    log.info(
        "LoRA applied: trainable=%.2fM / total=%.2fM  (%.2f%%)",
        trainable / 1e6, total / 1e6, 100.0 * trainable / total,
    )
    peft_model.print_trainable_parameters()
    return peft_model


# ═══════════════════════════════════════════════════════════════════════════
#  Utility: list available models
# ═══════════════════════════════════════════════════════════════════════════

def list_models() -> None:
    """Pretty-print the model registry."""
    print("\n  Supported VLMs")
    print("  " + "─" * 60)
    seen = set()
    for short_name, (hub_name, _) in MODEL_REGISTRY.items():
        if hub_name not in seen:
            seen.add(hub_name)
            print(f"    {short_name:<38s}  →  {hub_name}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
#  CLI self-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    parser = argparse.ArgumentParser(description="VLM Factory self-test")
    parser.add_argument(
        "--model", type=str, default="qwen2.5-vl-7b",
        help="Model short name or HF hub id",
    )
    parser.add_argument("--list", action="store_true", help="List available models")
    args = parser.parse_args()

    if args.list:
        list_models()
    else:
        cfg = {
            "model": {
                "dtype": "bfloat16",
                "attn_implementation": "eager",
                "trust_remote_code": True,
            }
        }
        print(f"\nLoading: {args.model}")
        model, processor = get_vlm_and_processor(args.model, cfg)
        print(f"  Model class:     {type(model).__name__}")
        print(f"  Processor class: {type(processor).__name__}")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters:      {n_params / 1e9:.2f}B")
        print("  ✓ Load OK\n")
