#!/usr/bin/env python3
"""
export_gguf.py  –  Merge LoRA adapter into base model and export to GGUF.

Run this on the WORKSTATION (x86 + GPU) to produce GGUF files that can
then be copied to the Jetson AGX Orin.

Steps:
  1. Load Qwen2.5-VL-7B-Instruct (fp16)
  2. Merge LoRA adapter weights into base model
  3. Save merged model to HuggingFace format
  4. Convert to GGUF using llama.cpp's convert script
  5. Quantize to Q4_K_M (best quality/size for Jetson 32-64GB)

Usage:
  python export_gguf.py                              # Defaults: Q4_K_M
  python export_gguf.py --quant Q4_K_S               # Smaller (lower quality)
  python export_gguf.py --quant Q3_K_M               # Even smaller for 32GB Orin
  python export_gguf.py --skip_convert                # Only merge, skip GGUF

Output:
  exports/merged/               — merged HF model (fp16)
  exports/model-f16.gguf        — full precision GGUF
  exports/model-Q4_K_M.gguf    — quantized GGUF (deploy this)

Requirements:
  pip install transformers peft torch accelerate
  git clone https://github.com/ggml-org/llama.cpp.git (for convert scripts)
"""

import os, sys, argparse, shutil, subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_BASE_MODEL  = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_ADAPTER     = str(SCRIPT_DIR / "checkpoints")
DEFAULT_EXPORT_DIR  = str(SCRIPT_DIR / "exports")

# GGUF quantization types — ranked by size (smallest to largest)
QUANT_TYPES = ["Q2_K", "Q3_K_S", "Q3_K_M", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"]

# Approximate sizes for Qwen2.5-VL-7B
SIZE_ESTIMATES = {
    "Q3_K_M": "~3.5 GB  (tight fit for 32GB Orin shared RAM)",
    "Q4_K_S": "~4.0 GB  (good balance)",
    "Q4_K_M": "~4.5 GB  (recommended — best quality/size)",
    "Q5_K_M": "~5.5 GB  (higher quality, fits 64GB Orin easily)",
    "Q8_0":   "~8.0 GB  (near lossless, 64GB Orin only)",
}


def merge_lora(base_model: str, adapter_path: str, output_dir: str):
    """Merge LoRA adapter into base model and save as HF format."""
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from peft import PeftModel

    print(f"\n[1/4] Loading base model: {base_model}")
    print(f"      Adapter: {adapter_path}")

    # Load in fp16 on CPU to avoid GPU memory issues during export
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration as ModelClass

    model = ModelClass.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="cpu",
    )

    print(f"[2/4] Loading and merging LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    merged_dir = os.path.join(output_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)

    print(f"[3/4] Saving merged model to {merged_dir}")
    model.save_pretrained(merged_dir, safe_serialization=True)

    # Also save processor/tokenizer
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    processor.save_pretrained(merged_dir)

    print(f"      Saved merged model ({sum(f.stat().st_size for f in Path(merged_dir).rglob('*') if f.is_file()) / 1e9:.1f} GB)")
    return merged_dir


def find_llama_cpp():
    """Find or suggest llama.cpp installation."""
    # Check common locations
    candidates = [
        Path.home() / "llama.cpp",
        SCRIPT_DIR / "llama.cpp",
        Path("/opt/llama.cpp"),
    ]
    for p in candidates:
        convert_script = p / "convert_hf_to_gguf.py"
        if convert_script.exists():
            return p

    # Check if llama-quantize is on PATH
    if shutil.which("llama-quantize"):
        return None  # quantize binary available, no source dir needed

    return None


def convert_to_gguf(merged_dir: str, output_dir: str, llama_cpp_dir: Path = None):
    """Convert merged HF model to GGUF format."""
    f16_gguf = os.path.join(output_dir, "model-f16.gguf")

    if llama_cpp_dir:
        convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
        print(f"\n[3.5/4] Converting to GGUF (f16)...")
        cmd = [
            sys.executable, str(convert_script),
            merged_dir,
            "--outtype", "f16",
            "--outfile", f16_gguf,
        ]
        subprocess.run(cmd, check=True)
    else:
        # Try using the python package
        print(f"\n[3.5/4] Converting to GGUF using llama-cpp-python...")
        try:
            cmd = [
                sys.executable, "-m", "llama_cpp.convert",
                merged_dir,
                "--outtype", "f16",
                "--outfile", f16_gguf,
            ]
            subprocess.run(cmd, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("\n[!] Cannot find llama.cpp conversion tools.")
            print("    Please install llama.cpp:")
            print("      git clone https://github.com/ggml-org/llama.cpp.git")
            print("      cd llama.cpp && pip install -r requirements.txt")
            print(f"\n    Then re-run with: python export_gguf.py --llama_cpp_dir <path>")
            print(f"\n    Merged model saved at: {merged_dir}")
            return None

    print(f"      f16 GGUF: {f16_gguf} ({os.path.getsize(f16_gguf)/1e9:.1f} GB)")
    return f16_gguf


def quantize_gguf(f16_gguf: str, output_dir: str, quant_type: str,
                   llama_cpp_dir: Path = None):
    """Quantize f16 GGUF to target quantization type."""
    quant_gguf = os.path.join(output_dir, f"model-{quant_type}.gguf")

    # Find quantize binary
    quantize_bin = None
    if llama_cpp_dir:
        for name in ["llama-quantize", "quantize"]:
            p = llama_cpp_dir / "build" / "bin" / name
            if p.exists():
                quantize_bin = str(p)
                break
            p = llama_cpp_dir / name
            if p.exists():
                quantize_bin = str(p)
                break
    if not quantize_bin:
        quantize_bin = shutil.which("llama-quantize") or shutil.which("quantize")

    if not quantize_bin:
        print(f"\n[!] Cannot find llama-quantize binary.")
        print(f"    Build llama.cpp first:")
        print(f"      cd llama.cpp && cmake -B build && cmake --build build -j")
        print(f"\n    f16 GGUF available at: {f16_gguf}")
        print(f"    Quantize manually: llama-quantize {f16_gguf} {quant_gguf} {quant_type}")
        return None

    print(f"\n[4/4] Quantizing to {quant_type}...")
    if quant_type in SIZE_ESTIMATES:
        print(f"      Expected size: {SIZE_ESTIMATES[quant_type]}")

    subprocess.run([quantize_bin, f16_gguf, quant_gguf, quant_type], check=True)

    size_gb = os.path.getsize(quant_gguf) / 1e9
    print(f"      Quantized GGUF: {quant_gguf} ({size_gb:.1f} GB)")
    return quant_gguf


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA + export to GGUF for Jetson deployment")
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter_path", default=DEFAULT_ADAPTER)
    parser.add_argument("--output_dir", default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--quant", default="Q4_K_M", choices=QUANT_TYPES,
                        help="GGUF quantization type")
    parser.add_argument("--llama_cpp_dir", default=None,
                        help="Path to llama.cpp source")
    parser.add_argument("--skip_merge", action="store_true",
                        help="Skip merge, use existing merged dir")
    parser.add_argument("--skip_convert", action="store_true",
                        help="Only merge LoRA, skip GGUF conversion")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1-3: Merge LoRA
    merged_dir = os.path.join(args.output_dir, "merged")
    if args.skip_merge and os.path.isdir(merged_dir):
        print(f"[*] Using existing merged model: {merged_dir}")
    else:
        merged_dir = merge_lora(args.base_model, args.adapter_path,
                                args.output_dir)

    if args.skip_convert:
        print(f"\n[*] Merge complete. Merged model at: {merged_dir}")
        return

    # Find llama.cpp
    llama_cpp = None
    if args.llama_cpp_dir:
        llama_cpp = Path(args.llama_cpp_dir)
    else:
        llama_cpp = find_llama_cpp()

    # Step 3.5: Convert to f16 GGUF
    f16_gguf = convert_to_gguf(merged_dir, args.output_dir, llama_cpp)
    if not f16_gguf:
        return

    # Step 4: Quantize
    quant_gguf = quantize_gguf(f16_gguf, args.output_dir, args.quant, llama_cpp)

    # Summary
    print(f"\n{'='*60}")
    print(f"  EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"  Merged HF model: {merged_dir}")
    print(f"  f16 GGUF:        {f16_gguf}")
    if quant_gguf:
        print(f"  {args.quant} GGUF:    {quant_gguf}")
    print(f"\n  Deploy to Jetson AGX Orin:")
    print(f"    scp {quant_gguf or f16_gguf} jetson:/path/to/models/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
