#!/usr/bin/env python3
"""
infer_jetson.py  –  Lightweight inference for Jetson AGX Orin

Uses Qwen2.5-VL-7B with aggressive memory optimizations for Jetson's
shared CPU/GPU memory (32GB or 64GB).

Key differences from workstation version:
  - No bitsandbytes (not available on aarch64/Jetson)
  - Uses fp16 with aggressive torch.cuda memory management
  - Smaller image resolution to reduce vision token count
  - Optional GGUF path via llama-cpp-python (even lower VRAM)

Usage:
  # HuggingFace fp16 mode (needs ~14GB shared memory)
  python infer_jetson.py --image_dir images/

  # With merged LoRA model (skip adapter loading)
  python infer_jetson.py --base_model exports/merged --no_adapter --image_dir images/

  # GGUF mode via llama-cpp-python (needs ~5GB, fastest)
  python infer_jetson.py --gguf model-Q4_K_M.gguf --mmproj mmproj.gguf --image_dir images/
"""

import os, sys, json, argparse, time, gc
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

SYSTEM_PROMPT = (
    "You are SafetyVLM, an expert driving instructor and traffic-rule analyst. "
    "When shown a traffic scene, identify signs, markings, and signals. "
    "Explain the rules that apply and state correct driver behaviour. "
    "Be concise."
)

IMAGE_PROMPT = (
    "Identify any traffic signs, road markings, signals or driving-relevant "
    "elements visible. Describe what you see, explain the rules, and state "
    "correct driver behaviour."
)


# =========================================================================
#  Mode 1: HuggingFace Transformers (fp16, no bitsandbytes)
# =========================================================================

def load_model_hf(args):
    """Load Qwen2.5-VL-7B (fp16) for Jetson."""
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration as ModelClass

    print(f"[*] Loading model: {args.base_model}")
    print(f"[*] Mode: fp16 (Jetson AGX Orin)")

    # fp16 on Jetson AGX Orin memory budget:
    #   fp16 model: ~14GB + KV cache ~0.5GB + vision ~0.5GB ≈ 15GB
    #   Leaves ~17GB for OS + apps on 32GB Orin
    #   or ~49GB on 64GB Orin
    model = ModelClass.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # Load and merge LoRA if available
    if not args.no_adapter and os.path.isdir(args.adapter_path):
        from peft import PeftModel
        print(f"[*] Loading & merging LoRA from {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.eval()

    proc_name = args.base_model
    processor = AutoProcessor.from_pretrained(
        proc_name,
        trust_remote_code=True,
        padding_side="left",
        # Aggressive image downsizing for Jetson
        min_pixels=128 * 28 * 28,     # ~100K pixels
        max_pixels=512 * 28 * 28,     # ~400K pixels (half of workstation)
    )

    return model, processor


def generate_hf(model, processor, prompt, image_path=None,
                max_new_tokens=256, temperature=0.7):
    """Generate with HF model on Jetson."""
    import torch
    from PIL import Image

    user_content = []
    img = None

    if image_path:
        img = Image.open(image_path).convert("RGB")
        # Resize large images to save memory
        max_dim = 720
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        user_content.append({"type": "image", "image": img})

    user_content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    text = processor.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True)

    if image_path:
        inputs = processor(text=[text], images=[img],
                           return_tensors="pt", padding=True).to(model.device)
    else:
        inputs = processor(text=[text], return_tensors="pt",
                           padding=True).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(generated, skip_special_tokens=True)

    # Aggressively free memory after each image
    del inputs, output_ids
    if img:
        del img
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return response


# =========================================================================
#  Mode 2: GGUF via llama-cpp-python (lowest memory)
# =========================================================================

def generate_gguf(args, image_paths, prompt):
    """Generate using GGUF model via llama-cpp-python — lowest memory."""
    try:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen2VLChatHandler
    except ImportError:
        print("[!] llama-cpp-python not installed.")
        print("    On Jetson: pip install llama-cpp-python --extra-index-url "
              "https://abetlen.github.io/llama-cpp-python/whl/cu122")
        sys.exit(1)

    print(f"[*] Loading GGUF model: {args.gguf}")
    chat_handler = Qwen2VLChatHandler(clip_model_path=args.mmproj)

    llm = Llama(
        model_path=args.gguf,
        chat_handler=chat_handler,
        n_gpu_layers=-1,       # Offload all layers
        n_ctx=4096,            # Smaller context for Jetson
        n_batch=512,
        flash_attn=True,
        verbose=False,
    )

    results = []
    for i, img_path in enumerate(image_paths, 1):
        print(f"\n[*] Image {i}/{len(image_paths)}: {img_path.name}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"file://{img_path}"}},
                {"type": "text", "text": prompt},
            ]},
        ]

        t0 = time.time()
        resp = llm.create_chat_completion(
            messages=messages,
            max_tokens=256,
            temperature=0.7,
        )
        elapsed = time.time() - t0

        answer = resp["choices"][0]["message"]["content"]
        tokens = resp["usage"]["completion_tokens"]
        print(f"    {tokens} tokens in {elapsed:.1f}s ({tokens/elapsed:.1f} t/s)")
        print(f"    {answer[:200]}...")

        results.append({
            "image": img_path.name,
            "prompt": prompt,
            "model_response": answer,
            "tokens": tokens,
            "time_s": round(elapsed, 2),
        })

    return results


# =========================================================================
#  Visual report (same as save_results.py logic)
# =========================================================================

def render_card(image_path, prompt, response, index, total):
    """Render a visual card with image + prompt + answer."""
    import textwrap
    from PIL import Image, ImageDraw, ImageFont

    def get_font(size):
        for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                   "/usr/share/fonts/TTF/DejaVuSans.ttf"]:
            if os.path.isfile(p):
                return ImageFont.truetype(p, size)
        return ImageFont.load_default()

    pad = 20
    font_title = get_font(24)
    font_label = get_font(18)
    font_body  = get_font(14)

    # Load and scale image
    src = Image.open(image_path).convert("RGB")
    img_w = 800
    scale = img_w / src.width
    img_h = min(int(src.height * scale), 500)
    src_resized = src.resize((img_w, img_h), Image.LANCZOS)

    # Text layout
    prompt_lines = textwrap.wrap(prompt, width=45)
    resp_lines = textwrap.wrap(response, width=95)
    line_h = 20

    right_h = max(img_h, len(prompt_lines) * line_h + 50)
    resp_h = len(resp_lines) * line_h + 50
    title_h = 40
    card_w = 1600
    total_h = title_h + pad + right_h + pad + resp_h + pad

    card = Image.new("RGB", (card_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(card)

    # Title
    draw.rectangle([(0, 0), (card_w, title_h)], fill=(20, 80, 50))
    draw.text((pad, 8), f"  Image {index}/{total} — {Path(image_path).name}",
              fill=(255, 255, 255), font=font_title)

    y = title_h + pad
    card.paste(src_resized, (pad, y))

    # Prompt
    rx = img_w + 2 * pad
    draw.text((rx, y), "PROMPT:", fill=(20, 80, 50), font=font_label)
    ty = y + 28
    for line in prompt_lines:
        draw.text((rx, ty), line, fill=(60, 60, 60), font=font_body)
        ty += line_h

    # Response
    div_y = y + right_h + pad // 2
    draw.line([(pad, div_y), (card_w - pad, div_y)], fill=(180, 180, 180))
    ry = div_y + pad
    draw.text((pad, ry), "MODEL RESPONSE:", fill=(20, 80, 50), font=font_label)
    ry += 28
    for line in resp_lines:
        draw.text((pad + 10, ry), line, fill=(40, 40, 40), font=font_body)
        ry += line_h

    return card


# =========================================================================
#  Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Jetson AGX Orin inference for SafetyVLM")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter_path", default=str(SCRIPT_DIR / "checkpoints"))
    parser.add_argument("--no_adapter", action="store_true",
                        help="Skip LoRA adapter (use with pre-merged model)")
    parser.add_argument("--image_dir", default=str(SCRIPT_DIR / "images"))
    parser.add_argument("--output_dir", default=str(SCRIPT_DIR / "results_jetson"))
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max tokens (lower = less memory, default 256)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--prompt", default=None)
    # GGUF mode
    parser.add_argument("--gguf", default=None,
                        help="Path to GGUF model (lowest memory mode)")
    parser.add_argument("--mmproj", default=None,
                        help="Path to mmproj GGUF (required with --gguf)")
    args = parser.parse_args()

    prompt = args.prompt or IMAGE_PROMPT

    # Collect images
    img_dir = Path(args.image_dir)
    patterns = ["*.png", "*.jpg", "*.jpeg"]
    image_paths = sorted(p for pat in patterns for p in img_dir.glob(pat))

    if not image_paths:
        print(f"[!] No images found in {img_dir}")
        return

    print(f"[*] Found {len(image_paths)} images")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── GGUF mode ──
    if args.gguf:
        if not args.mmproj:
            print("[!] --mmproj required with --gguf")
            return
        results = generate_gguf(args, image_paths, prompt)

    # ── HuggingFace mode ──
    else:
        model, processor = load_model_hf(args)

        # Print memory after model load
        try:
            import torch
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1e9
                print(f"[*] GPU memory after model load: {mem:.1f} GB")
        except:
            pass

        results = []
        for i, img_path in enumerate(image_paths, 1):
            print(f"\n[*] Image {i}/{len(image_paths)}: {img_path.name}")
            t0 = time.time()

            response = generate_hf(
                model, processor, prompt,
                image_path=str(img_path),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            elapsed = time.time() - t0

            print(f"    {elapsed:.1f}s — {response[:150]}...")
            results.append({
                "image": img_path.name,
                "image_path": str(img_path),
                "prompt": prompt,
                "model_response": response,
                "time_s": round(elapsed, 2),
            })

    # ── Save JSON ──
    json_out = out_dir / "results.json"
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[*] JSON saved to {json_out}")

    # ── Save visual cards ──
    try:
        from PIL import Image
        cards = []
        for i, r in enumerate(results, 1):
            ip = Path(r.get("image_path", img_dir / r["image"]))
            if not ip.exists():
                ip = img_dir / r["image"]
            card = render_card(str(ip), r["prompt"], r["model_response"],
                               i, len(results))
            card_path = out_dir / f"card_{i:02d}_{ip.stem}.png"
            card.save(card_path, quality=90)
            cards.append(card)
            print(f"[*] Saved {card_path.name}")

        # Stitch report
        gap = 8
        report_w = max(c.width for c in cards)
        report_h = sum(c.height for c in cards) + gap * (len(cards) - 1)
        report = Image.new("RGB", (report_w, report_h), (240, 240, 240))
        y = 0
        for c in cards:
            report.paste(c, (0, y))
            y += c.height + gap
        report_path = out_dir / "report.png"
        report.save(report_path, quality=90)
        print(f"[*] Report: {report_path}")
    except Exception as e:
        print(f"[!] Card generation skipped: {e}")

    # ── Summary ──
    total_time = sum(r.get("time_s", 0) for r in results)
    print(f"\n{'='*55}")
    print(f"  JETSON INFERENCE COMPLETE")
    print(f"{'='*55}")
    print(f"  Images:     {len(results)}")
    print(f"  Total time: {total_time:.1f}s ({total_time/len(results):.1f}s/image)")
    print(f"  Output:     {out_dir}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
