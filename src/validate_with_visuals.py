#!/usr/bin/env python3
"""
validate_with_visuals.py

Run quick validation inference on image+prompt samples and save:
1) JSONL results with model outputs
2) Visualization PNGs with image + text panels (larger font)

Example:
  python src/validate_with_visuals.py \
    --base_model Qwen/Qwen2.5-VL-7B-Instruct \
    --adapter_path /scratch/rbaskar5/GLANCE/checkpoints/dual_distilled_7b/final_lora \
    --input_jsonl /scratch/rbaskar5/GLANCE/data/waymo_trajectory/val.jsonl \
    --image_root /scratch/rbaskar5/Dataset/waymo_front \
    --output_dir /scratch/rbaskar5/GLANCE/evals/val_visuals \
    --max_samples 64 \
    --font_size 34
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate model and save image+output visualizations")
    parser.add_argument("--base_model", required=True, help="HF base model id")
    parser.add_argument("--adapter_path", default="", help="Optional PEFT adapter path")
    parser.add_argument("--input_jsonl", required=True, help="Validation JSONL")
    parser.add_argument("--image_root", default="", help="Optional root for relative image paths")
    parser.add_argument("--output_dir", required=True, help="Directory for JSONL + PNG outputs")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--font_size", type=int, default=34, help="Font size for output text in PNG")
    parser.add_argument("--panel_width", type=int, default=1200, help="Right-side text panel width")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def resolve_image_path(sample: Dict[str, Any], image_root: Optional[Path]) -> Optional[Path]:
    image_path = sample.get("image_path", "")
    if not image_path:
        return None

    p = Path(image_path)
    if p.is_file():
        return p

    if image_root is not None:
        p2 = (image_root / image_path).resolve()
        if p2.is_file():
            return p2
    return None


def get_prompt_and_gt(sample: Dict[str, Any]) -> Tuple[str, str]:
    prompt = sample.get("prompt", "").strip()
    gt = sample.get("answer", "").strip()

    if prompt and gt:
        return prompt, gt

    # Fallback for chat-format entries.
    messages = sample.get("messages", [])
    user_text = ""
    assistant_text = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")
        if role == "user":
            if isinstance(content, str):
                user_text = content
            elif isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                user_text = "\n".join([p for p in parts if p])
        elif role == "assistant":
            if isinstance(content, str):
                assistant_text = content

    prompt = prompt or user_text or "Describe this driving scene and trajectory."
    gt = gt or assistant_text
    return prompt, gt


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for fp in candidates:
        p = Path(fp)
        if p.is_file():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    lines: List[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append("")
            continue

        words = paragraph.split()
        current = words[0]
        for word in words[1:]:
            trial = f"{current} {word}"
            left, top, right, bottom = draw.textbbox((0, 0), trial, font=font)
            if (right - left) <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def render_visual(
    image: Image.Image,
    prompt: str,
    gt: str,
    pred: str,
    output_path: Path,
    font_size: int,
    panel_width: int,
    index: int,
) -> None:
    img = image.convert("RGB")
    left_w, left_h = img.size

    # Create a canvas with right text panel.
    canvas = Image.new("RGB", (left_w + panel_width, left_h), color=(255, 255, 255))
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    title_font = load_font(font_size + 6)
    body_font = load_font(font_size)

    margin = 20
    x = left_w + margin
    y = margin
    text_width = panel_width - (2 * margin)

    sections = [
        (f"Sample {index}", "", title_font, (0, 0, 0)),
        ("Prompt", prompt, body_font, (20, 20, 20)),
        ("Ground Truth", gt or "(no ground truth in input)", body_font, (0, 70, 140)),
        ("Model Output", pred, body_font, (140, 0, 0)),
    ]

    for title, body, font, color in sections:
        draw.text((x, y), title, fill=color, font=title_font)
        y += int((font_size + 6) * 1.35)

        if body:
            lines = wrap_text(draw, body, font, text_width)
            for line in lines:
                draw.text((x, y), line, fill=color, font=font)
                y += int(font_size * 1.35)
        y += int(font_size * 0.8)

        if y > left_h - margin:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


@torch.no_grad()
def generate_one(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
    device: str,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
    )

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = output_ids[0][prompt_len:]
    return processor.decode(gen_ids, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    out_dir = Path(args.output_dir)
    vis_dir = out_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    image_root = Path(args.image_root) if args.image_root else None

    print(f"[model] loading base: {args.base_model}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=device,
    )

    if args.adapter_path:
        print(f"[model] applying adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)

    rows = load_jsonl(Path(args.input_jsonl))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    results_path = out_dir / "validation_results.jsonl"

    with results_path.open("w", encoding="utf-8") as f:
        for i, sample in enumerate(rows, start=1):
            img_path = resolve_image_path(sample, image_root)
            if img_path is None:
                continue

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as exc:
                print(f"[warn] failed to open image {img_path}: {exc}")
                continue

            prompt, gt = get_prompt_and_gt(sample)
            pred = generate_one(
                model=model,
                processor=processor,
                image=image,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )

            row = {
                "index": i,
                "image_path": str(img_path),
                "prompt": prompt,
                "ground_truth": gt,
                "prediction": pred,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            vis_path = vis_dir / f"sample_{i:05d}.png"
            render_visual(
                image=image,
                prompt=prompt,
                gt=gt,
                pred=pred,
                output_path=vis_path,
                font_size=args.font_size,
                panel_width=args.panel_width,
                index=i,
            )

            if i % 10 == 0:
                print(f"[progress] processed {i}/{len(rows)}")

    print(f"[done] wrote results: {results_path}")
    print(f"[done] wrote visualizations: {vis_dir}")


if __name__ == "__main__":
    main()
