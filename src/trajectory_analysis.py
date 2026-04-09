#!/usr/bin/env python3
"""
trajectory_analysis.py  –  Waymo image trajectory analysis with Qwen2.5-VL
===============================================================================

Runs Qwen2.5-VL over Waymo images and writes training-ready annotations
that can be consumed by `waymo_dataset.py` / `train_waymo.py`.

Outputs JSONL rows with:
  - image_path
  - camera
  - prompt
  - answer  (model-generated trajectory analysis)
  - messages (chat-format conversation)

Typical usage:
  python src/trajectory_analysis.py \
      --image_root /scratch/rbaskar5/Dataset/waymo_front \
      --output /scratch/rbaskar5/GLANCE/data/waymo_trajectory_analysis.jsonl

If you already have a training annotation JSON/JSONL, pass --annotations_json
instead of --image_root.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from PIL import Image

from vlm_factory import get_vlm_and_processor
from waymo_dataset import discover_images, load_annotations, load_from_tfrecords

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a trajectory analysis assistant for autonomous driving. "
    "Given a Waymo driving image, infer the scene layout, the most likely ego "
    "vehicle motion, the main surrounding agents, and the safety risk. "
    "Return a concise structured analysis that can be used as training data."
)

USER_PROMPT = (
    "Analyze this Waymo driving scene and produce a structured trajectory analysis.\n\n"
    "Include the following fields in your answer:\n"
    "1. scene_summary: short description of the traffic scene\n"
    "2. key_objects: the important vehicles, pedestrians, signals, or lane features\n"
    "3. ego_trajectory: the likely next motion of the ego vehicle over the next few seconds\n"
    "4. agent_trajectories: likely motions of the other visible agents\n"
    "5. safety_assessment: safe / caution / unsafe with one-sentence justification\n\n"
    "Keep the output concise and grounded in the visible image."
)


def _load_config(config_path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read the config file") from exc

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_image(sample: Dict[str, Any], image_root: Optional[Path]) -> Optional[Image.Image]:
    if "pil_image" in sample and sample["pil_image"] is not None:
        return sample["pil_image"]

    image_path = sample.get("image_path")
    if not image_path:
        return None

    candidate = Path(image_path)
    if not candidate.is_file() and image_root is not None:
        candidate = (image_root / image_path).resolve()

    if not candidate.is_file():
        return None

    return Image.open(candidate).convert("RGB")


def _maybe_resize_image(image: Image.Image, image_max_side: int) -> Image.Image:
    if image_max_side <= 0:
        return image

    w, h = image.size
    longest = max(w, h)
    if longest <= image_max_side:
        return image

    scale = image_max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _collect_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    max_samples = args.max_samples

    if args.annotations_json:
        samples = load_annotations(args.annotations_json, max_samples=max_samples)
        return samples

    if args.tfrecord_dir:
        return load_from_tfrecords(
            args.tfrecord_dir,
            camera_name=args.camera,
            max_samples=max_samples,
        )

    if args.image_root:
        return discover_images(args.image_root, max_samples=max_samples)

    raise ValueError("Provide either --annotations_json, --tfrecord_dir, or --image_root")


def _apply_sharding(samples: List[Dict[str, Any]], num_shards: int, shard_id: int) -> List[Dict[str, Any]]:
    if num_shards <= 1:
        return samples
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"Invalid shard_id={shard_id} for num_shards={num_shards}")
    return [s for i, s in enumerate(samples) if i % num_shards == shard_id]


def _sample_key(sample: Dict[str, Any], camera_default: str) -> str:
    return f"{sample.get('image_path', '')}::{sample.get('camera', camera_default)}"


def _load_existing_keys(output_path: Path) -> set[str]:
    seen: set[str] = set()
    if not output_path.exists():
        return seen
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen.add(f"{row.get('image_path', '')}::{row.get('camera', '')}")
    return seen


def _build_messages(prompt: str, image: Image.Image) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def _generate_one(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    messages = _build_messages(prompt, image)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=0.9,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def _generate_batch(
    model: Any,
    processor: Any,
    images: List[Image.Image],
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
) -> List[str]:
    messages_batch = [_build_messages(p, im) for p, im in zip(prompts, images)]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_batch]

    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=0.9,
        )

    prompt_len = inputs["input_ids"].shape[1]
    outputs: List[str] = []
    for i in range(output_ids.shape[0]):
        generated = output_ids[i][prompt_len:]
        outputs.append(processor.decode(generated, skip_special_tokens=True).strip())
    return outputs


def _default_output_prompt(sample: Dict[str, Any]) -> str:
    prompt = sample.get("prompt")
    if prompt:
        return prompt
    return USER_PROMPT


def run(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    model_name = args.model_name or cfg.get("model", {}).get("name", "qwen2.5-vl-32b")
    cfg.setdefault("model", {})["name"] = model_name

    model, processor = get_vlm_and_processor(model_name, cfg)
    model.eval()

    samples = _collect_samples(args)
    samples = _apply_sharding(samples, args.num_shards, args.shard_id)
    if not samples:
        log.warning("No Waymo samples found; nothing to analyze.")
        return

    image_root = Path(args.image_root) if args.image_root else None
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_keys = _load_existing_keys(output_path) if (args.resume and output_path.exists()) else set()
    if seen_keys:
        log.info("Resume enabled: loaded %d existing rows from %s", len(seen_keys), output_path)

    mode = "a" if (args.resume and output_path.exists()) else "w"
    processed = len(seen_keys)
    written_now = 0

    with output_path.open(mode, encoding="utf-8") as out_f:
        for batch_start in range(0, len(samples), args.batch_size):
            batch = samples[batch_start: batch_start + args.batch_size]

            valid_items: List[Dict[str, Any]] = []
            batch_images: List[Image.Image] = []
            batch_prompts: List[str] = []

            for sample in batch:
                key = _sample_key(sample, args.camera)
                if key in seen_keys:
                    continue

                image = _resolve_image(sample, image_root)
                if image is None:
                    log.warning("Skipping sample because the image could not be loaded")
                    continue
                image = _maybe_resize_image(image, args.image_max_side)
                prompt = _default_output_prompt(sample)
                valid_items.append(sample)
                batch_images.append(image)
                batch_prompts.append(prompt)

            if not valid_items:
                continue

            answers = _generate_batch(
                model=model,
                processor=processor,
                images=batch_images,
                prompts=batch_prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            for sample, prompt, answer in zip(valid_items, batch_prompts, answers):
                processed += 1
                written_now += 1
                row = {
                    "source": "waymo_trajectory_analysis",
                    "sample_index": processed,
                    "camera": sample.get("camera", args.camera),
                    "image_path": sample.get("image_path", ""),
                    "prompt": prompt,
                    "answer": answer,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": prompt},
                            ],
                        },
                        {"role": "assistant", "content": answer},
                    ],
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                seen_keys.add(_sample_key(sample, args.camera))

            out_f.flush()

            if processed % args.log_every == 0 or processed == len(samples):
                log.info("Processed %d/%d samples", processed, len(samples))

    print(f"Wrote {written_now:,} new trajectory-analysis samples to {output_path} (total rows now ~{processed:,})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Waymo trajectory analysis annotations with Qwen2.5-VL")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "configs" / "waymo_finetune_config.yaml"))
    parser.add_argument("--model_name", default=None, help="Override the model name from config")
    parser.add_argument("--image_root", default="/scratch/rbaskar5/Dataset/waymo_front", help="Root directory containing extracted Waymo images")
    parser.add_argument("--annotations_json", default="", help="Optional existing annotations JSON or JSONL")
    parser.add_argument("--tfrecord_dir", default="", help="Optional raw Waymo TFRecord directory")
    parser.add_argument("--camera", default="FRONT", help="Camera to extract when using TFRecords")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent.parent / "data" / "waymo_trajectory_analysis.jsonl"))
    parser.add_argument("--max_samples", type=int, default=0, help="Limit the number of samples (0 = all)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--image_max_side", type=int, default=0, help="Resize images so longest side is <= this value (0 = keep original)")
    parser.add_argument("--num_shards", type=int, default=1, help="Number of data shards for multi-process runs")
    parser.add_argument("--shard_id", type=int, default=0, help="Current shard index [0, num_shards)")
    parser.add_argument("--resume", action="store_true", help="Resume by appending to existing output and skipping existing rows")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--log_every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
