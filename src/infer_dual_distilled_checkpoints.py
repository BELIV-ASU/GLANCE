#!/usr/bin/env python3
"""
Run sequential inference for dual-distillation checkpoints and save per-checkpoint
results + visualizations.

Outputs are written under:
  results/dualdistilation/checkpoint_<step>/

Example:
  python src/infer_dual_distilled_checkpoints.py \
    --checkpoint_root checkpoints/dual_distilled_7b_v2 \
    --val_jsonl data/waymo_trajectory/val.jsonl \
    --image_root /scratch/rbaskar5/Dataset/waymo_front \
    --num_samples 5
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from peft import PeftModel

from vlm_factory import get_vlm_and_processor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inference over all dual-distillation checkpoints (sequential)."
    )
    p.add_argument(
        "--checkpoint_root",
        default="/scratch/rbaskar5/GLANCE/checkpoints/dual_distilled_7b_v2",
        help="Root directory that contains checkpoint-<N>/ and/or final/",
    )
    p.add_argument(
        "--config",
        default="/scratch/rbaskar5/GLANCE/configs/waymo_finetune_qwen32b_trajectory.yaml",
        help="YAML config used to read model name and defaults",
    )
    p.add_argument(
        "--val_jsonl",
        default="/scratch/rbaskar5/GLANCE/data/waymo_trajectory/val.jsonl",
        help="Validation JSONL for inference",
    )
    p.add_argument(
        "--image_root",
        default="/scratch/rbaskar5/Dataset/waymo_front",
        help="Image root joined with each row image_path",
    )
    p.add_argument(
        "--output_root",
        default="/scratch/rbaskar5/GLANCE/results/dualdistilation",
        help="Root output directory (per-checkpoint outputs go here)",
    )
    p.add_argument("--num_samples", type=int, default=5, help="Samples per checkpoint")
    p.add_argument("--seed", type=int, default=42, help="Sampling seed")
    p.add_argument("--max_new_tokens", type=int, default=192)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Use 0.0 for deterministic evaluation",
    )
    p.add_argument(
        "--include_final",
        action="store_true",
        help="Also evaluate final/ adapter (saved as checkpoint_final)",
    )
    return p.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_rows(rows: List[Dict[str, Any]], k: int, seed: int) -> List[Dict[str, Any]]:
    if k <= 0 or k >= len(rows):
        return rows
    rng = random.Random(seed)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    pick = sorted(idxs[:k])
    return [rows[i] for i in pick]


def _step_of(p: Path) -> int:
    m = re.search(r"checkpoint-(\d+)$", p.name)
    if m:
        return int(m.group(1))
    return 10**9


def discover_checkpoints(root: Path, include_final: bool) -> List[Path]:
    ckpts = [
        p
        for p in root.iterdir()
        if p.is_dir() and re.match(r"checkpoint-\d+$", p.name)
    ]
    ckpts.sort(key=_step_of)
    if include_final:
        final = root / "final"
        if final.is_dir():
            ckpts.append(final)
    return ckpts


def base_model_from_adapter(ckpt_dir: Path) -> str | None:
    cfg_path = ckpt_dir / "adapter_config.json"
    if not cfg_path.is_file():
        return None
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    base = data.get("base_model_name_or_path")
    return str(base) if base else None


def make_messages(prompt: str) -> List[Dict[str, Any]]:
    system_prompt = (
        "You are a trajectory analysis assistant for autonomous driving. "
        "Given a Waymo driving image, infer scene layout, likely ego motion, "
        "main agents, and safety risk. Return concise structured analysis."
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def run_one(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    messages = make_messages(prompt)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=0.9,
        )

    generated = out_ids[0][inputs["input_ids"].shape[1] :]
    return processor.decode(generated, skip_special_tokens=True).strip()


def _font() -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ):
        p = Path(path)
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), 15)
            except Exception:
                continue
    return ImageFont.load_default()


def render_visual(image: Image.Image, output_text: str, out_png: Path, title: str) -> None:
    img = image.convert("RGB")
    max_h = 680
    if img.height > max_h:
        ratio = max_h / img.height
        img = img.resize((int(img.width * ratio), max_h))

    pad = 14
    panel_w = 860
    canvas_w = img.width + panel_w + pad * 3
    canvas_h = max(760, img.height + pad * 2)

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(243, 245, 248))
    draw = ImageDraw.Draw(canvas)
    font = _font()

    canvas.paste(img, (pad, pad))

    x0 = img.width + 2 * pad
    draw.rectangle(
        [x0 - 8, pad - 8, canvas_w - pad, canvas_h - pad],
        fill=(255, 255, 255),
        outline=(205, 210, 218),
        width=2,
    )
    draw.text((x0, pad), title, fill=(23, 28, 38), font=font)

    y = pad + 26
    wrapped: List[str] = []
    for para in output_text.split("\n"):
        if not para.strip():
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(para, width=88))

    line_h = 16
    max_lines = int((canvas_h - y - pad) / line_h)
    for line in wrapped[:max_lines]:
        draw.text((x0, y), line, fill=(40, 43, 52), font=font)
        y += line_h

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


def ckpt_label(ckpt_dir: Path) -> str:
    if ckpt_dir.name.startswith("checkpoint-"):
        step = ckpt_dir.name.split("-")[-1]
        return f"checkpoint_{step}"
    return "checkpoint_final"


def main() -> None:
    args = parse_args()

    ckpt_root = Path(args.checkpoint_root)
    base_cfg = load_yaml(Path(args.config))

    rows = load_jsonl(Path(args.val_jsonl))
    rows = select_rows(rows, args.num_samples, args.seed)
    if not rows:
        raise SystemExit("No validation rows found.")

    ckpts = discover_checkpoints(ckpt_root, args.include_final)
    if not ckpts:
        raise SystemExit(f"No checkpoints found under {ckpt_root}")

    print("=" * 80)
    print("Dual-distillation checkpoint inference")
    print(f"Checkpoint root : {ckpt_root}")
    print(f"Found checkpoints: {[p.name for p in ckpts]}")
    print(f"Samples/checkpoint: {len(rows)}")
    print(f"Results root: {args.output_root}")
    print("=" * 80)

    image_root = Path(args.image_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary: List[Dict[str, Any]] = []

    for ckpt in ckpts:
        label = ckpt_label(ckpt)
        run_dir = output_root / label
        viz_dir = run_dir / "visualized"
        result_jsonl = run_dir / "inference_results.jsonl"
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Running {ckpt.name} -> {run_dir} ---")

        cfg = json.loads(json.dumps(base_cfg))
        model_name = base_model_from_adapter(ckpt) or cfg.get("model", {}).get("name", "qwen2.5-vl-7b")
        cfg.setdefault("model", {})["name"] = model_name
        print(f"Using base model: {model_name}")

        # Load base + adapter fresh for each checkpoint so tests are isolated/clear.
        model, processor = get_vlm_and_processor(model_name, cfg)
        model = PeftModel.from_pretrained(model, str(ckpt))
        model.eval()

        count = 0
        non_empty = 0
        with result_jsonl.open("w", encoding="utf-8") as f:
            for i, row in enumerate(rows, start=1):
                rel_path = row.get("image_path", "")
                img_path = image_root / rel_path
                if not img_path.is_file():
                    print(f"[warn] missing image: {img_path}")
                    continue

                image = Image.open(img_path).convert("RGB")
                prompt = row.get("prompt") or (
                    "Analyze this Waymo driving scene and produce a structured trajectory analysis "
                    "with scene_summary, key_objects, ego_trajectory, agent_trajectories, safety_assessment."
                )
                gt = row.get("answer", "")

                output = run_one(
                    model=model,
                    processor=processor,
                    image=image,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )

                rec = {
                    "checkpoint": ckpt.name,
                    "index": i,
                    "image_path": rel_path,
                    "prompt": prompt,
                    "model_output": output,
                    "ground_truth": gt,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                title = f"{ckpt.name} | sample {i} | {rel_path}"
                render_visual(image, output, viz_dir / f"sample_{i:03d}.png", title)

                count += 1
                if output.strip():
                    non_empty += 1
                print(f"[{ckpt.name}] sample {i}/{len(rows)} done")

        summary.append(
            {
                "checkpoint": ckpt.name,
                "output_dir": str(run_dir),
                "samples_attempted": len(rows),
                "samples_written": count,
                "non_empty_outputs": non_empty,
            }
        )

        # Free GPU memory before next checkpoint.
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDone. Summary:")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
